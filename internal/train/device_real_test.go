package train

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
)

// TestRealWeightFusedAB compares the fused and row-blocked-scratch quantized
// weight GEMMs on the real model's weight bytes at M=512. Opt-in:
//
//	QWEN38_REAL_DEVICE=1 go test ./internal/train -run TestRealWeightFusedAB -v
func TestRealWeightFusedAB(t *testing.T) {
	if os.Getenv("QWEN38_REAL_DEVICE") != "1" {
		t.Skip("set QWEN38_REAL_DEVICE=1 to run the real-weight fused A/B")
	}
	path := os.Getenv("QWEN38_MODEL")
	if path == "" {
		for _, cand := range []string{
			"models/Qwen3.8-27B-UD-Q4_K_M.gguf",
			"../../models/Qwen3.8-27B-UD-Q4_K_M.gguf",
		} {
			if _, err := os.Stat(cand); err == nil {
				path = cand
				break
			}
		}
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Skipf("model not found (%v)", err)
	}
	defer g.Close()
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()

	const M = 512
	names := []string{
		"output.weight",
		"blk.60.ssm_out.weight", "blk.60.attn_qkv.weight", "blk.60.attn_gate.weight",
		"blk.60.ffn_down.weight", "blk.60.ssm_alpha.weight", "blk.3.attn_output.weight",
	}
	for _, name := range names {
		ti, ok := g.Tensor(name)
		if !ok {
			t.Errorf("%s: missing", name)
			continue
		}
		raw, err := g.ReadTensor(ti)
		if err != nil {
			t.Fatalf("%s: read: %v", name, err)
		}
		K, N := int(ti.Dims[0]), int(ti.Dims[1])
		wb, err := be.UploadWeight(ti.Type, raw, []int{K, N})
		if err != nil {
			t.Fatalf("%s: upload: %v", name, err)
		}
		x := make([]float32, K*M)
		for i := range x {
			x[i] = float32(math.Cos(float64(i)*0.013)) * 0.5
		}
		xb, err := be.Upload(&compute.Tensor{Dims: []int{K, M}, F32: x})
		if err != nil {
			t.Fatal(err)
		}

		fused, err := be.MatMulWeight(wb, xb)
		if err != nil {
			t.Fatalf("%s: fused fwd: %v", name, err)
		}
		be.SetNoFusedWeight(true)
		scratch, err := be.MatMulWeight(wb, xb)
		be.SetNoFusedWeight(false)
		if err != nil {
			t.Fatalf("%s: scratch fwd: %v", name, err)
		}
		tf, _ := be.Download(fused)
		ts, _ := be.Download(scratch)
		dFwd, iFwd := maxDiff(tf.F32, ts.F32)

		dy := make([]float32, N*M)
		for i := range dy {
			dy[i] = float32(math.Sin(float64(i)*0.021)) * 0.4
		}
		dyb, err := be.Upload(&compute.Tensor{Dims: []int{N, M}, F32: dy})
		if err != nil {
			t.Fatal(err)
		}
		fusedT, err := be.MatMulWeightTranspose(wb, dyb)
		if err != nil {
			t.Fatalf("%s: fused dx: %v", name, err)
		}
		be.SetNoFusedWeight(true)
		scratchT, err := be.MatMulWeightTranspose(wb, dyb)
		be.SetNoFusedWeight(false)
		if err != nil {
			t.Fatalf("%s: scratch dx: %v", name, err)
		}
		ttf, _ := be.Download(fusedT)
		tts, _ := be.Download(scratchT)
		dT, iT := maxDiff(ttf.F32, tts.F32)

		t.Logf("%-28s %-6s [%d,%d]  fwd maxdiff %.3g@%d  dx maxdiff %.3g@%d", name, ti.Type, K, N, dFwd, iFwd, dT, iT)
	}
}

// TestRealAllWeightsFusedAB runs the fused-vs-scratch A/B over every 2D weight
// tensor in the real model and reports any that differ. Opt-in:
//
//	QWEN38_REAL_DEVICE=1 go test ./internal/train -run TestRealAllWeightsFusedAB -v
func TestRealAllWeightsFusedAB(t *testing.T) {
	if os.Getenv("QWEN38_REAL_DEVICE") != "1" {
		t.Skip("set QWEN38_REAL_DEVICE=1")
	}
	path := os.Getenv("QWEN38_MODEL")
	if path == "" {
		for _, cand := range []string{
			"models/Qwen3.8-27B-UD-Q4_K_M.gguf",
			"../../models/Qwen3.8-27B-UD-Q4_K_M.gguf",
		} {
			if _, err := os.Stat(cand); err == nil {
				path = cand
				break
			}
		}
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Skipf("model not found (%v)", err)
	}
	defer g.Close()
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()

	const M = 8
	bad := 0
	for _, ti := range g.Tensors {
		if len(ti.Dims) != 2 {
			continue
		}
		K, N := int(ti.Dims[0]), int(ti.Dims[1])
		raw, err := g.ReadTensor(ti)
		if err != nil {
			t.Fatalf("%s: read: %v", ti.Name, err)
		}
		wb, err := be.UploadWeight(ti.Type, raw, []int{K, N})
		if err != nil {
			t.Errorf("%s: upload: %v", ti.Name, err)
			continue
		}
		x := make([]float32, K*M)
		for i := range x {
			x[i] = float32(math.Cos(float64(i)*0.013)) * 0.5
		}
		xb, err := be.Upload(&compute.Tensor{Dims: []int{K, M}, F32: x})
		if err != nil {
			t.Fatal(err)
		}
		fused, err := be.MatMulWeight(wb, xb)
		if err != nil {
			t.Errorf("%s: fused: %v", ti.Name, err)
			continue
		}
		be.SetNoFusedWeight(true)
		scratch, err := be.MatMulWeight(wb, xb)
		be.SetNoFusedWeight(false)
		if err != nil {
			t.Errorf("%s: scratch: %v", ti.Name, err)
			continue
		}
		tf, _ := be.Download(fused)
		ts, _ := be.Download(scratch)
		d, idx := maxDiff(tf.F32, ts.F32)
		if d > 1e-5 {
			t.Errorf("%s %s [%d,%d]: fwd maxdiff %.3g @%d", ti.Name, ti.Type, K, N, d, idx)
			bad++
		}
		be.Free(wb)
		be.Free(xb)
		be.Free(fused)
		be.Free(scratch)
	}
	if bad > 0 {
		t.Fatalf("%d tensors differ", bad)
	}
	t.Logf("all 2D weights: fused == scratch")
}

// TestRealForwardFusedParity compares the full-model forward logits between the
// fused and scratch weight GEMMs with no backward in between. Opt-in.
func TestRealForwardFusedParity(t *testing.T) {
	if os.Getenv("QWEN38_REAL_DEVICE") != "1" {
		t.Skip("set QWEN38_REAL_DEVICE=1")
	}
	path := os.Getenv("QWEN38_MODEL")
	if path == "" {
		for _, cand := range []string{
			"models/Qwen3.8-27B-UD-Q4_K_M.gguf",
			"../../models/Qwen3.8-27B-UD-Q4_K_M.gguf",
		} {
			if _, err := os.Stat(cand); err == nil {
				path = cand
				break
			}
		}
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Skipf("model not found (%v)", err)
	}
	defer g.Close()
	mc, err := modelcfg.FromGGUF(g)
	if err != nil {
		t.Fatal(err)
	}
	w, err := qwen38.LoadFromGGUF(g, mc)
	if err != nil {
		t.Fatal(err)
	}
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()
	m, err := NewDeviceModel(be, w.Cfg, w, DefaultLoRA(), 1)
	if err != nil {
		t.Fatal(err)
	}
	m.Checkpoint = true
	if err := be.Sync(); err != nil {
		t.Fatal(err)
	}
	T := 8
	if s := os.Getenv("QWEN38_REAL_T"); s != "" {
		fmt.Sscanf(s, "%d", &T)
	}
	tokens := make([]int32, T)
	for i := range tokens {
		tokens[i] = int32(1 + (i*7919+13)%(w.Cfg.NVocab-1))
	}

	lgF, ctxF, err := m.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	tf, _ := be.Download(lgF)
	be.Free(lgF)
	xsF := snapshotBuffers(t, be, ctxF.xs)

	be.SetNoFusedFwd(true)
	lgS, ctxS, err := m.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	ts, _ := be.Download(lgS)
	be.Free(lgS)
	xsS := snapshotBuffers(t, be, ctxS.xs)

	if len(xsF) != len(xsS) {
		t.Fatalf("ctx.xs length %d != %d", len(xsF), len(xsS))
	}
	worst := float64(0)
	worstLayer := -1
	for i := range xsF {
		for k := range xsF[i] {
			d := math.Abs(float64(xsF[i][k] - xsS[i][k]))
			if d > worst {
				worst, worstLayer = d, i
			}
		}
	}
	t.Logf("ctx.xs: max diff %.3g at layer %d", worst, worstLayer)

	// Pool vs no-pool: both fused.
	be.SetNoFusedFwd(false)
	be.SetNoPool(true)
	lgP, ctxP, err := m.Forward(tokens)
	if err != nil {
		t.Fatal(err)
	}
	tp, _ := be.Download(lgP)
	be.Free(lgP)
	xsP := snapshotBuffers(t, be, ctxP.xs)
	firstDiff := -1
	for i := range xsF {
		for k := range xsF[i] {
			if xsF[i][k] != xsP[i][k] {
				firstDiff = i
				break
			}
		}
		if firstDiff >= 0 {
			break
		}
	}
	var poolLogMax float32
	for k := range tf.F32 {
		if d := float32(math.Abs(float64(tf.F32[k] - tp.F32[k]))); d > poolLogMax {
			poolLogMax = d
		}
	}
	t.Logf("pool vs nopool: first differing ctx.xs layer %d, logits maxdiff %.3g", firstDiff, poolLogMax)

	var maxAbs float32
	var nDiff, nExact int
	for i := range tf.F32 {
		if tf.F32[i] != ts.F32[i] {
			nExact++
		}
		d := float32(math.Abs(float64(tf.F32[i] - ts.F32[i])))
		if d > maxAbs {
			maxAbs = d
		}
		if d > 1e-6 {
			nDiff++
		}
	}
	t.Logf("T=%d logits: %d/%d differ bitwise, %d differ >1e-6, maxabs %.3g", T, nExact, len(tf.F32), nDiff, maxAbs)
}

func snapshotBuffers(t *testing.T, be *vulkan.Backend, bufs []compute.Buffer) [][]float32 {
	t.Helper()
	out := make([][]float32, len(bufs))
	for i, b := range bufs {
		tn, err := be.Download(b)
		if err != nil {
			t.Fatal(err)
		}
		out[i] = append([]float32(nil), tn.F32...)
	}
	return out
}

func maxDiff(a, b []float32) (float32, int) {
	var m float32
	idx := -1
	for i := range a {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > m {
			m, idx = d, i
		}
	}
	return m, idx
}

// TestRealDeviceStep runs a real Qwen3.8-27B resident forward + loss + backward
// step at a long sequence length on the Vulkan device. It is opt-in and heavy
// (loads ~16 GiB and runs many dispatches), so it is excluded from make check:
//
//	QWEN38_REAL_DEVICE=1 QWEN38_REAL_T=512 go test ./internal/train -run TestRealDeviceStep -v
//
// QWEN38_MODEL overrides the GGUF path; QWEN38_REAL_T overrides the sequence
// length (default 512).
func TestRealDeviceStep(t *testing.T) {
	if os.Getenv("QWEN38_REAL_DEVICE") != "1" {
		t.Skip("set QWEN38_REAL_DEVICE=1 to run the real-model device step")
	}
	path := os.Getenv("QWEN38_MODEL")
	if path == "" {
		// `go test` runs with CWD at the package directory, so look from the
		// repo root (../../) as well as the conventional models/ dir.
		for _, cand := range []string{
			"models/Qwen3.8-27B-UD-Q4_K_M.gguf",
			"../../models/Qwen3.8-27B-UD-Q4_K_M.gguf",
		} {
			if _, err := os.Stat(cand); err == nil {
				path = cand
				break
			}
		}
	}
	if path == "" {
		path = "models/Qwen3.8-27B-UD-Q4_K_M.gguf"
	}
	T := 512
	if s := os.Getenv("QWEN38_REAL_T"); s != "" {
		if _, err := fmt.Sscanf(s, "%d", &T); err != nil {
			t.Fatalf("bad QWEN38_REAL_T=%q", s)
		}
	}

	g, err := gguf.Open(path)
	if err != nil {
		t.Skipf("model not found (%v)", err)
	}
	defer g.Close()
	mc, err := modelcfg.FromGGUF(g)
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	w, err := qwen38.LoadFromGGUF(g, mc)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded weights in %s", time.Since(t0))

	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()
	// QWEN38_NO_FUSED forces the row-blocked float32 scratch fallback so the
	// fused dequantize+GEMM path can be A/B compared. QWEN38_NO_FUSED_FWD /
	// QWEN38_NO_FUSED_T disable only the forward / transposed fused GEMM.
	if os.Getenv("QWEN38_NO_FUSED") == "1" {
		be.SetNoFusedWeight(true)
	}
	if os.Getenv("QWEN38_NO_FUSED_FWD") == "1" {
		be.SetNoFusedFwd(true)
	}
	if os.Getenv("QWEN38_NO_FUSED_T") == "1" {
		be.SetNoFusedT(true)
	}
	if os.Getenv("QWEN38_NO_POOL") == "1" {
		be.SetNoPool(true)
	}
	caps := be.Capabilities()
	t.Logf("device: %s (%d bytes)", caps.Name, caps.MemoryBytes)

	loCfg := DefaultLoRA()
	t0 = time.Now()
	m, err := NewDeviceModel(be, w.Cfg, w, loCfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := be.Sync(); err != nil {
		t.Fatal(err)
	}
	t.Logf("built %d-layer device model with %d adapters in %s", w.Cfg.TrunkLayers(), len(m.AdapterLins()), time.Since(t0))

	m.Checkpoint = true
	tokens := make([]int32, T)
	targets := make([]int32, T)
	for i := range tokens {
		tokens[i] = int32(1 + (i*7919+13)%(w.Cfg.NVocab-1))
		targets[i] = int32(1 + ((i+1)*7919+13)%(w.Cfg.NVocab-1))
	}

	tF := time.Now()
	logits, ctx, err := m.Forward(tokens)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	loss, dlogits, err := m.Loss(logits, targets, -100)
	if err != nil {
		t.Fatalf("loss: %v", err)
	}
	be.Free(logits)
	tFL := time.Since(tF)

	tB := time.Now()
	grads, err := m.Backward(ctx, dlogits)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}
	be.Free(dlogits)
	if err := be.Sync(); err != nil {
		t.Fatal(err)
	}
	tBL := time.Since(tB)

	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("non-finite loss %g", loss)
	}
	// Gradients must be finite and not all zero. dA is zero on the first step
	// (B is zero-initialized), so the non-zero signal is dB.
	var maxAbs float32
	for _, gr := range grads {
		for _, buf := range []struct {
			name string
			b    compute.Buffer
		}{{"dA", gr.dA}, {"dB", gr.dB}} {
			gt, err := be.Download(buf.b)
			if err != nil {
				t.Fatal(err)
			}
			for _, f := range gt.F32 {
				if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
					t.Fatalf("non-finite %s gradient", buf.name)
				}
				if a := float32(math.Abs(float64(f))); a > maxAbs {
					maxAbs = a
				}
			}
		}
	}
	t.Logf("T=%d forward+loss %s, backward %s (total %s), loss %.4f, %d adapter grads, max|grad| %.3g",
		T, tFL, tBL, tFL+tBL, loss, len(grads), maxAbs)
	if maxAbs == 0 {
		t.Fatal("all-zero adapter gradient")
	}

	// Optional A/B: recompute the same step with the fused weight GEMMs disabled
	// and compare per-adapter gradients. Diagnoses fused-vs-scratch divergence.
	if os.Getenv("QWEN38_FUSED_AB") == "1" {
		be.SetNoFusedWeight(true)
		lg2, cx2, err := m.Forward(tokens)
		if err != nil {
			t.Fatal(err)
		}
		ls2, dl2, err := m.Loss(lg2, targets, -100)
		if err != nil {
			t.Fatal(err)
		}
		be.Free(lg2)
		g2, err := m.Backward(cx2, dl2)
		if err != nil {
			t.Fatal(err)
		}
		be.Free(dl2)
		if err := be.Sync(); err != nil {
			t.Fatal(err)
		}
		t.Logf("A/B loss: fused %.6f vs scratch %.6f", loss, ls2)
		lines := m.AdapterLins()
		worst, worstName := float32(0), ""
		for i := range grads {
			for _, pair := range []struct {
				a, b compute.Buffer
			}{{grads[i].dA, g2[i].dA}, {grads[i].dB, g2[i].dB}} {
				ta, err := be.Download(pair.a)
				if err != nil {
					t.Fatal(err)
				}
				tb, err := be.Download(pair.b)
				if err != nil {
					t.Fatal(err)
				}
				for k := range ta.F32 {
					d := float32(math.Abs(float64(ta.F32[k] - tb.F32[k])))
					if d > worst {
						worst = d
						worstName = lines[i].name
					}
				}
			}
		}
		t.Logf("A/B max adapter gradient diff %.3g on %s", worst, worstName)
	}
}
