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

// hostAttnRef builds the full-attention block on the host autograd graph and
// returns the output value plus the LoRA A/B parameter values in component
// order (q,k,v,o,fg,fu,fd).
func hostAttnRef(cfg *qwen38.Config, lw *qwen38.LayerWeights, hostEnc map[*qwen38.Weight]*LoRA, x *autograd.Value) (*autograd.Value, map[*qwen38.Weight][2]*autograd.Value) {
	be := lw
	_ = be
	hd := cfg.HeadDim
	T := x.Data.Ne(1)
	params := map[*qwen38.Weight][2]*autograd.Value{}

	mm := func(w *qwen38.Weight, xv *autograd.Value) *autograd.Value {
		y := autograd.VMatMulWeight(w, xv)
		if l := hostEnc[w]; l != nil {
			a := autograd.Param(l.A)
			b := autograd.Param(l.B)
			params[w] = [2]*autograd.Value{a, b}
			y = autograd.VAdd(y, autograd.VScale(autograd.VMatMul(b, autograd.VMatMul(a, xv)), l.Scale))
		}
		return y
	}

	h := autograd.VRMSNorm(x, lw.AttnNorm, cfg.RmsEps)
	qg := mm(lw.AttnQ, h)
	q, gate := autograd.VSplitQG(qg, hd, cfg.NHead, T)
	gateSig := autograd.VSigmoid(gate)
	kBuf := mm(lw.AttnK, h)
	vBuf := mm(lw.AttnV, h)
	k3 := autograd.VReshape(kBuf, hd, cfg.NHeadKV, T)
	v3 := autograd.VReshape(vBuf, hd, cfg.NHeadKV, T)
	qN := autograd.VRMSNorm(q, lw.AttnQNorm, cfg.RmsEps)
	kN := autograd.VRMSNorm(k3, lw.AttnKNorm, cfg.RmsEps)
	positions := int32Positions(T)
	qR := autograd.VRoPENeoX(qN, positions, cfg.RopeTheta, cfg.NRot)
	kR := autograd.VRoPENeoX(kN, positions, cfg.RopeTheta, cfg.NRot)
	scale := float32(1 / math.Sqrt(float64(hd)))
	attn0 := autograd.VAttention(qR, kR, v3, cfg.NHead, cfg.NHeadKV, scale, true)
	attn := autograd.VMul(attn0, gateSig)
	attnFlat := autograd.VReshape(attn, cfg.NHead*hd, T)
	oV := mm(lw.AttnOutput, attnFlat)
	x1 := autograd.VAdd(x, oV)
	h2 := autograd.VRMSNorm(x1, lw.PostAttnNorm, cfg.RmsEps)
	act := autograd.VMul(autograd.VSilu(mm(lw.FfnGate, h2)), mm(lw.FfnUp, h2))
	ff := mm(lw.FfnDown, act)
	y := autograd.VAdd(x1, ff)
	return y, params
}

func int32Positions(n int) []int32 {
	p := make([]int32, n)
	for i := range p {
		p[i] = int32(i)
	}
	return p
}

func TestDeviceAttentionMatchesHost(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 42)
	lw := &w.Layers[3] // full-attention layer
	loCfg := LoRAConfig{Rank: 3, Alpha: 6, Targets: DefaultLoRA().Targets}

	d, err := NewDeviceAttention(v, cfg, lw, loCfg, 100)
	if err != nil {
		t.Fatal(err)
	}
	// Host-side adapters must be the same tensors the device uploaded.
	hostEnc := map[*qwen38.Weight]*LoRA{
		lw.AttnQ: d.q.lora, lw.AttnK: d.k.lora, lw.AttnV: d.v.lora,
		lw.AttnOutput: d.o.lora, lw.FfnGate: d.fg.lora, lw.FfnUp: d.fu.lora,
		lw.FfnDown: d.fd.lora,
	}

	const T = 3
	x := autograd.Random(200, 0.5, cfg.NEmbd, T)
	c := autograd.Random(201, 0.5, cfg.NEmbd, T)
	positions := int32Positions(T)

	// Host forward/backward.
	xV := autograd.Param(x)
	cV := autograd.Param(c)
	yV, params := hostAttnRef(cfg, lw, hostEnc, xV)
	loss := autograd.VSum(autograd.VMul(yV, cV))
	loss.Backward()

	// Device forward/backward.
	xb, _ := v.Upload(x)
	cb, _ := v.Upload(c)
	ctx, err := d.Forward(xb, positions)
	if err != nil {
		t.Fatal(err)
	}
	dHostY, err := v.Download(ctx.y)
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiff(dHostY.F32, yV.Data.F32); diff > 1e-4 {
		t.Fatalf("forward mismatch: max diff %g", diff)
	}

	g, err := d.Backward(ctx, cb, positions)
	if err != nil {
		t.Fatal(err)
	}
	check := func(name string, dev compute.Buffer, host *autograd.Value) {
		dt, err := v.Download(dev)
		if err != nil {
			t.Fatal(err)
		}
		if diff := maxAbsDiff(dt.F32, host.Grad.F32); diff > 2e-3 {
			t.Fatalf("%s mismatch: max diff %g", name, diff)
		}
	}
	check("dAq", g.DQ[0], params[lw.AttnQ][0])
	check("dBq", g.DQ[1], params[lw.AttnQ][1])
	check("dAk", g.DK[0], params[lw.AttnK][0])
	check("dBk", g.DK[1], params[lw.AttnK][1])
	check("dAv", g.DV[0], params[lw.AttnV][0])
	check("dBv", g.DV[1], params[lw.AttnV][1])
	check("dAo", g.DO[0], params[lw.AttnOutput][0])
	check("dBo", g.DO[1], params[lw.AttnOutput][1])
	check("dAfg", g.DFG[0], params[lw.FfnGate][0])
	check("dBfg", g.DFG[1], params[lw.FfnGate][1])
	check("dAfu", g.DFU[0], params[lw.FfnUp][0])
	check("dBfu", g.DFU[1], params[lw.FfnUp][1])
	check("dAfd", g.DFD[0], params[lw.FfnDown][0])
	check("dBfd", g.DFD[1], params[lw.FfnDown][1])

	dt, err := v.Download(g.DX)
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiff(dt.F32, xV.Grad.F32); diff > 2e-3 {
		t.Fatalf("dX mismatch: max diff %g", diff)
	}
}

func midCfg() *qwen38.Config {
	return &qwen38.Config{
		NEmbd: 256, NHead: 8, NHeadKV: 2, HeadDim: 32, NRot: 32, NFf: 512, NLayer: 4, NVocab: 64,
		RopeTheta: 10000, RmsEps: 1e-6,
		DConv: 4, DState: 32, DInner: 256, DtRank: 8, GroupCount: 4,
		FullAttnInterval: 4, NextNPredict: 0,
		LayerTypes: []modelcfg.LayerType{
			modelcfg.LayerLinearAttention, modelcfg.LayerLinearAttention,
			modelcfg.LayerLinearAttention, modelcfg.LayerFullAttention,
		},
	}
}

func TestDeviceAttentionTiming(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := midCfg()
	w := qwen38.NewRandom(cfg, 7)
	lw := &w.Layers[3]
	loCfg := LoRAConfig{Rank: 8, Alpha: 16, Targets: DefaultLoRA().Targets}
	d, err := NewDeviceAttention(v, cfg, lw, loCfg, 100)
	if err != nil {
		t.Fatal(err)
	}
	hostEnc := map[*qwen38.Weight]*LoRA{
		lw.AttnQ: d.q.lora, lw.AttnK: d.k.lora, lw.AttnV: d.v.lora,
		lw.AttnOutput: d.o.lora, lw.FfnGate: d.fg.lora, lw.FfnUp: d.fu.lora, lw.FfnDown: d.fd.lora,
	}
	const T = 64
	x := autograd.Random(300, 0.3, cfg.NEmbd, T)
	c := autograd.Random(301, 0.3, cfg.NEmbd, T)
	positions := int32Positions(T)
	xb, _ := v.Upload(x)
	cb, _ := v.Upload(c)

	ctx, err := d.Forward(xb, positions)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.Backward(ctx, cb, positions); err != nil {
		t.Fatal(err)
	}
	v.Sync()
	v.ResetStats()

	const iters = 10
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		ctx, err := d.Forward(xb, positions)
		if err != nil {
			t.Fatal(err)
		}
		g, err := d.Backward(ctx, cb, positions)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := v.Download(g.DX); err != nil {
			t.Fatal(err)
		}
	}
	devDur := time.Since(t0) / iters
	st := v.Stats()

	t1 := time.Now()
	for i := 0; i < iters; i++ {
		xV := autograd.Param(x)
		cV := autograd.Param(c)
		yV, _ := hostAttnRef(cfg, lw, hostEnc, xV)
		loss := autograd.VSum(autograd.VMul(yV, cV))
		loss.Backward()
	}
	hostDur := time.Since(t1) / iters

	t.Logf("attention layer (D=%d T=%d): device %v submits=%d dispatches=%d",
		cfg.NEmbd, T, devDur.Round(time.Microsecond), st.Submits, st.Dispatches)
	t.Logf("host autograd: %v  (%.1fx)", hostDur.Round(time.Microsecond), float64(hostDur)/float64(devDur))
}
