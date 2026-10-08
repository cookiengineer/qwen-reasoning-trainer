package train

import (
	"math"
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
)

// hostModelRef builds the whole model on the host autograd graph using the
// DeviceModel's adapters, and returns the logits plus every adapter's params.
func hostModelRef(cfg *qwen38.Config, w *qwen38.Weights, m *DeviceModel, x *autograd.Value) (*autograd.Value, map[*qwen38.Weight][2]*autograd.Value) {
	all := map[*qwen38.Weight][2]*autograd.Value{}
	merge := func(p map[*qwen38.Weight][2]*autograd.Value) {
		for k, v := range p {
			all[k] = v
		}
	}
	for il, l := range m.layers {
		lw := &w.Layers[il]
		if l.attn != nil {
			enc := map[*qwen38.Weight]*LoRA{
				lw.AttnQ: l.attn.q.lora, lw.AttnK: l.attn.k.lora, lw.AttnV: l.attn.v.lora,
				lw.AttnOutput: l.attn.o.lora, lw.FfnGate: l.attn.fg.lora,
				lw.FfnUp: l.attn.fu.lora, lw.FfnDown: l.attn.fd.lora,
			}
			y, p := hostAttnRef(cfg, lw, enc, x)
			merge(p)
			x = y
		} else {
			enc := map[*qwen38.Weight]*LoRA{
				lw.AttnQKV: l.gdn.qkv.lora, lw.AttnGate: l.gdn.gate.lora,
				lw.SSMOut: l.gdn.out.lora, lw.FfnGate: l.gdn.fg.lora,
				lw.FfnUp: l.gdn.fu.lora, lw.FfnDown: l.gdn.fd.lora,
			}
			y, p := hostGdnRef(cfg, lw, enc, x)
			merge(p)
			x = y
		}
	}
	xn := autograd.VRMSNorm(x, w.OutputNorm, cfg.RmsEps)
	logits := autograd.VMatMulWeight(w.Output, xn)
	return logits, all
}

func TestDeviceModelMatchesHost(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 99)
	loCfg := LoRAConfig{Rank: 3, Alpha: 6, Targets: DefaultLoRA().Targets}
	m, err := NewDeviceModel(v, cfg, w, loCfg, 300)
	if err != nil {
		t.Fatal(err)
	}

	tokens := []int32{1, 2, 3, 4}
	T := len(tokens)

	// Forward parity: with zero-init B the adapter is a no-op, so the resident
	// forward must equal the base inference forward.
	ref := qwen38.NewModel(w)
	res, err := ref.Forward(tokens, qwen38.ForwardOptions{})
	if err != nil {
		t.Fatal(err)
	}
	logits, ctx, err := m.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	gotLogits, err := v.Download(logits)
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiff(gotLogits.F32, res.Logits.F32); diff > 1e-3 {
		t.Fatalf("forward mismatch: max diff %g", diff)
	}

	// Backward with a fixed covector loss: loss = sum(logits * C), dLogits = C.
	c := autograd.Random(400, 0.5, cfg.NVocab, T)
	cb, err := v.Upload(c)
	if err != nil {
		t.Fatal(err)
	}
	grads, err := m.Backward(ctx, cb)
	if err != nil {
		t.Fatal(err)
	}

	// Host reference: rebuild the host graph from the actual embedding lookup
	// result so the inputs match the device exactly.
	emb, err := w.TokenEmbd.Dequant()
	if err != nil {
		t.Fatal(err)
	}
	xin := compute.NewF32(cfg.NEmbd, T)
	for i, tk := range tokens {
		copy(xin.F32[i*cfg.NEmbd:], emb.F32[int(tk)*cfg.NEmbd:(int(tk)+1)*cfg.NEmbd])
	}
	xinV := autograd.Param(xin)
	cV := autograd.Param(c)
	hostLogits, params := hostModelRef(cfg, w, m, xinV)
	loss := autograd.VSum(autograd.VMul(hostLogits, cV))
	loss.Backward()

	checked := 0
	for _, ag := range grads {
		p := params[ag.lin.w]
		if p[0] == nil {
			t.Fatalf("no host param for weight")
		}
		dA, err := v.Download(ag.dA)
		if err != nil {
			t.Fatal(err)
		}
		dB, err := v.Download(ag.dB)
		if err != nil {
			t.Fatal(err)
		}
		if diff := maxAbsDiff(dA.F32, p[0].Grad.F32); diff > 5e-3 {
			t.Fatalf("dA mismatch: max diff %g", diff)
		}
		if diff := maxAbsDiff(dB.F32, p[1].Grad.F32); diff > 5e-3 {
			t.Fatalf("dB mismatch: max diff %g", diff)
		}
		checked++
	}
	if checked < 25 {
		t.Fatalf("only %d adapters checked", checked)
	}
	t.Logf("verified %d adapters across %d layers", checked, len(m.layers))
}

func uploadReplace(v *vulkan.Backend, old compute.Buffer, t *compute.Tensor) (compute.Buffer, error) {
	nb, err := v.Upload(t)
	if err != nil {
		return old, err
	}
	v.Free(old)
	return nb, nil
}

func TestDeviceModelStepReducesLoss(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 123)
	loCfg := LoRAConfig{Rank: 3, Alpha: 6, Targets: DefaultLoRA().Targets}
	m, err := NewDeviceModel(v, cfg, w, loCfg, 400)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{1, 2, 3, 4, 5}
	T := len(tokens)
	c := autograd.Random(500, 0.5, cfg.NVocab, T)
	cb, _ := v.Upload(c)

	lossOf := func(logits compute.Buffer) float32 {
		lt, err := v.Download(logits)
		if err != nil {
			t.Fatal(err)
		}
		var s float32
		for i := range lt.F32 {
			s += lt.F32[i] * c.F32[i]
		}
		return s
	}

	logits, ctx, err := m.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	l0 := lossOf(logits)

	const lr = float32(0.02)
	for step := 0; step < 5; step++ {
		grads, err := m.Backward(ctx, cb)
		if err != nil {
			t.Fatal(err)
		}
		for _, ag := range grads {
			dA, err := v.Download(ag.dA)
			if err != nil {
				t.Fatal(err)
			}
			dB, err := v.Download(ag.dB)
			if err != nil {
				t.Fatal(err)
			}
			for i := range ag.lin.lora.A.F32 {
				ag.lin.lora.A.F32[i] -= lr * dA.F32[i]
			}
			for i := range ag.lin.lora.B.F32 {
				ag.lin.lora.B.F32[i] -= lr * dB.F32[i]
			}
			if ag.lin.a, err = uploadReplace(v, ag.lin.a, ag.lin.lora.A); err != nil {
				t.Fatal(err)
			}
			if ag.lin.b, err = uploadReplace(v, ag.lin.b, ag.lin.lora.B); err != nil {
				t.Fatal(err)
			}
		}
		logits, ctx, err = m.Forward(tokens)
		if err != nil {
			t.Fatal(err)
		}
	}
	l1 := lossOf(logits)
	t.Logf("loss: %.5f -> %.5f", l0, l1)
	if !(l1 < l0) {
		t.Fatalf("loss did not decrease: %g -> %g", l0, l1)
	}
}

func TestDeviceModelTiming(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := midCfg()
	w := qwen38.NewRandom(cfg, 321)
	loCfg := LoRAConfig{Rank: 8, Alpha: 16, Targets: DefaultLoRA().Targets}
	m, err := NewDeviceModel(v, cfg, w, loCfg, 500)
	if err != nil {
		t.Fatal(err)
	}
	const T = 32
	tokens := make([]int32, T)
	for i := range tokens {
		tokens[i] = int32(1 + i%16)
	}
	c := autograd.Random(600, 0.3, cfg.NVocab, T)
	cb, _ := v.Upload(c)

	logits, ctx, err := m.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Backward(ctx, cb); err != nil {
		t.Fatal(err)
	}
	v.Sync()
	v.ResetStats()

	const iters = 5
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		logits, ctx, err = m.Forward(tokens)
		if err != nil {
			t.Fatal(err)
		}
		grads, err := m.Backward(ctx, cb)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := v.Download(logits); err != nil {
			t.Fatal(err)
		}
		_ = grads
	}
	devDur := time.Since(t0) / iters
	st := v.Stats()

	// Host reference timing.
	emb, _ := w.TokenEmbd.Dequant()
	xin := compute.NewF32(cfg.NEmbd, T)
	for i, tk := range tokens {
		copy(xin.F32[i*cfg.NEmbd:], emb.F32[int(tk)*cfg.NEmbd:(int(tk)+1)*cfg.NEmbd])
	}
	t1 := time.Now()
	for i := 0; i < iters; i++ {
		xV := autograd.Param(xin)
		cV := autograd.Param(c)
		hl, _ := hostModelRef(cfg, w, m, xV)
		autograd.VSum(autograd.VMul(hl, cV)).Backward()
	}
	hostDur := time.Since(t1) / iters

	t.Logf("model (%d layers, D=%d, T=%d): device %v submits=%d dispatches=%d",
		len(m.layers), cfg.NEmbd, T, devDur.Round(time.Microsecond), st.Submits, st.Dispatches)
	t.Logf("host autograd: %v  (%.1fx)", hostDur.Round(time.Microsecond), float64(hostDur)/float64(devDur))
}

func TestDeviceModelLossMatchesHost(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 150)
	loCfg := LoRAConfig{Rank: 3, Alpha: 6, Targets: DefaultLoRA().Targets}
	m, err := NewDeviceModel(v, cfg, w, loCfg, 700)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int32{1, 2, 3, 4}
	targets := []int32{2, 3, 4, 0}
	T := len(tokens)

	logits, ctx, err := m.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	devLoss, dLogits, err := m.Loss(logits, targets, -100)
	if err != nil {
		t.Fatal(err)
	}
	grads, err := m.Backward(ctx, dLogits)
	if err != nil {
		t.Fatal(err)
	}

	// Host reference.
	emb, err := w.TokenEmbd.Dequant()
	if err != nil {
		t.Fatal(err)
	}
	xin := compute.NewF32(cfg.NEmbd, T)
	for i, tk := range tokens {
		copy(xin.F32[i*cfg.NEmbd:], emb.F32[int(tk)*cfg.NEmbd:(int(tk)+1)*cfg.NEmbd])
	}
	xinV := autograd.Param(xin)
	hostLogits, params := hostModelRef(cfg, w, m, xinV)
	intTargets := make([]int, T)
	for i, tg := range targets {
		intTargets[i] = int(tg)
	}
	hostLossV := autograd.VCrossEntropy(hostLogits, intTargets, -100)
	hostLossV.Backward()
	if d := maxAbsDiff([]float32{devLoss}, hostLossV.Data.F32); d > 1e-4 {
		t.Fatalf("loss mismatch: %g vs %g", devLoss, hostLossV.Data.F32[0])
	}
	for _, ag := range grads {
		p := params[ag.lin.w]
		dA, _ := v.Download(ag.dA)
		dB, _ := v.Download(ag.dB)
		if diff := maxAbsDiff(dA.F32, p[0].Grad.F32); diff > 5e-3 {
			t.Fatalf("dA mismatch: %g", diff)
		}
		if diff := maxAbsDiff(dB.F32, p[1].Grad.F32); diff > 5e-3 {
			t.Fatalf("dB mismatch: %g", diff)
		}
	}
}

func deviceModelGrads(t *testing.T, m *DeviceModel, v *vulkan.Backend, tokens, targets []int32) [][2][]float32 {
	t.Helper()
	logits, ctx, err := m.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	_, dLogits, err := m.Loss(logits, targets, -100)
	if err != nil {
		t.Fatal(err)
	}
	grads, err := m.Backward(ctx, dLogits)
	if err != nil {
		t.Fatal(err)
	}
	out := make([][2][]float32, len(grads))
	for i, ag := range grads {
		dA, _ := v.Download(ag.dA)
		dB, _ := v.Download(ag.dB)
		out[i] = [2][]float32{dA.F32, dB.F32}
	}
	return out
}

func TestDeviceModelCheckpoint(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 160)
	loCfg := LoRAConfig{Rank: 3, Alpha: 6, Targets: DefaultLoRA().Targets}
	plain, err := NewDeviceModel(v, cfg, w, loCfg, 800)
	if err != nil {
		t.Fatal(err)
	}
	ckpt, err := NewDeviceModel(v, cfg, w, loCfg, 800)
	if err != nil {
		t.Fatal(err)
	}
	ckpt.Checkpoint = true

	tokens := []int32{1, 2, 3, 4, 5}
	targets := []int32{2, 3, 4, 0, 1}
	gp := deviceModelGrads(t, plain, v, tokens, targets)
	gc := deviceModelGrads(t, ckpt, v, tokens, targets)
	if len(gp) != len(gc) {
		t.Fatalf("grad count %d vs %d", len(gp), len(gc))
	}
	for i := range gp {
		if d := maxAbsDiff(gp[i][0], gc[i][0]); d > 1e-4 {
			t.Fatalf("checkpointed dA[%d] mismatch: %g", i, d)
		}
		if d := maxAbsDiff(gp[i][1], gc[i][1]); d > 1e-4 {
			t.Fatalf("checkpointed dB[%d] mismatch: %g", i, d)
		}
	}
}

func shortCfg64() *qwen38.Config {
	cfg := tinyCfg()
	cfg.NLayer = 64
	types := make([]modelcfg.LayerType, 64)
	for i := range types {
		if (i+1)%4 == 0 {
			types[i] = modelcfg.LayerFullAttention
		} else {
			types[i] = modelcfg.LayerLinearAttention
		}
	}
	cfg.LayerTypes = types
	return cfg
}

func TestDeviceModel64Layers(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := shortCfg64()
	w := qwen38.NewRandom(cfg, 170)
	loCfg := LoRAConfig{Rank: 3, Alpha: 6, Targets: DefaultLoRA().Targets}
	m, err := NewDeviceModel(v, cfg, w, loCfg, 900)
	if err != nil {
		t.Fatal(err)
	}
	m.Checkpoint = true
	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	targets := make([]int32, len(tokens))
	for i := range targets {
		targets[i] = int32((i + 2) % cfg.NVocab)
	}
	logits, ctx, err := m.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	loss, dLogits, err := m.Loss(logits, targets, -100)
	if err != nil {
		t.Fatal(err)
	}
	grads, err := m.Backward(ctx, dLogits)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("bad loss %g", loss)
	}
	if len(grads) != 48*6+16*7 {
		t.Fatalf("expected %d adapters, got %d", 48*6+16*7, len(grads))
	}
	t.Logf("64-layer model: loss=%.4f adapters=%d device_bytes=%.0fMB", loss, len(grads), float64(v.Stats().DeviceBytes)/1e6)
}
