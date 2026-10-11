package train

import (
	"math"
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

func hostGdnRef(cfg *qwen38.Config, lw *qwen38.LayerWeights, hostEnc map[*qwen38.Weight]*LoRA, x *autograd.Value) (*autograd.Value, map[*qwen38.Weight][2]*autograd.Value) {
	hd := cfg.HeadDim
	_ = hd
	T := x.Data.Ne(1)
	headVDim := cfg.HeadVDim()
	nV := cfg.DtRank
	nK := cfg.GroupCount
	headKDim := cfg.DState
	keyDim := cfg.KeyDim()
	convDim := cfg.ConvDim()
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
	qkv := mm(lw.AttnQKV, h)
	z := mm(lw.AttnGate, h)
	beta := autograd.VSigmoid(autograd.VMatMulWeight(lw.SSMBeta, h))
	alpha := autograd.VMatMulWeight(lw.SSMAlpha, h)

	dtT := make([]float32, nV*T)
	aT := make([]float32, nV*T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nV; hh++ {
			dtT[hh+t*nV] = lw.SSMDt[hh]
			aT[hh+t*nV] = lw.SSMA[hh]
		}
	}
	pre := autograd.VAdd(alpha, autograd.Const(&compute.Tensor{Dims: []int{nV, T}, F32: dtT}))
	gate := autograd.VReshape(autograd.VMul(autograd.VSoftplus(pre), autograd.Const(&compute.Tensor{Dims: []int{nV, T}, F32: aT})), 1, nV, T)
	betaT := autograd.VReshape(beta, 1, nV, T)

	convIn := autograd.VConvInput(qkv, cfg.DConv, convDim, T)
	convOut := autograd.VSilu(autograd.VSSMConv(convIn, autograd.Const(lw.SSMConv1d)))
	qPre := autograd.VGatherHeads(convOut, 0, headKDim, nK, T, convDim)
	kPre := autograd.VGatherHeads(convOut, keyDim, headKDim, nK, T, convDim)
	v := autograd.VGatherHeads(convOut, 2*keyDim, headVDim, nV, T, convDim)
	q := autograd.VL2Norm(qPre, 1e-6)
	k := autograd.VL2Norm(kPre, 1e-6)
	q48 := autograd.VRepeatHeads(q, headKDim, nK, nV, T)
	k48 := autograd.VRepeatHeads(k, headKDim, nK, nV, T)
	state := autograd.Const(compute.NewF32(headVDim, headVDim, nV))
	out, _ := autograd.VGatedDeltaNet(q48, k48, v, gate, betaT, state)
	norm := autograd.VRMSNorm(out, lw.SSMNorm, cfg.RmsEps)
	zr := autograd.VReshape(z, headVDim, nV, T)
	gated := autograd.VMul(norm, autograd.VSilu(zr))
	flat := autograd.VReshape(gated, cfg.ValueDim(), T)
	linOut := mm(lw.SSMOut, flat)
	x1 := autograd.VAdd(x, linOut)
	h2 := autograd.VRMSNorm(x1, lw.PostAttnNorm, cfg.RmsEps)
	act := autograd.VMul(autograd.VSilu(mm(lw.FfnGate, h2)), mm(lw.FfnUp, h2))
	y := autograd.VAdd(x1, mm(lw.FfnDown, act))
	return y, params
}

func TestDeviceGDNMatchesHost(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 55)
	lw := &w.Layers[0] // linear-attention layer
	loCfg := LoRAConfig{Rank: 3, Alpha: 6, Targets: DefaultLoRA().Targets}

	d, err := NewDeviceGDN(v, cfg, lw, loCfg, 200)
	if err != nil {
		t.Fatal(err)
	}
	hostEnc := map[*qwen38.Weight]*LoRA{
		lw.AttnQKV: d.qkv.lora, lw.AttnGate: d.gate.lora, lw.SSMOut: d.out.lora,
		lw.FfnGate: d.fg.lora, lw.FfnUp: d.fu.lora, lw.FfnDown: d.fd.lora,
	}

	const T = 4
	x := autograd.Random(210, 0.5, cfg.NEmbd, T)
	c := autograd.Random(211, 0.5, cfg.NEmbd, T)

	xV := autograd.Param(x)
	cV := autograd.Param(c)
	yV, params := hostGdnRef(cfg, lw, hostEnc, xV)
	loss := autograd.VSum(autograd.VMul(yV, cV))
	loss.Backward()

	xb, _ := v.Upload(x)
	cb, _ := v.Upload(c)
	ctx, err := d.Forward(xb, nil)
	if err != nil {
		t.Fatal(err)
	}
	dy, err := v.Download(ctx.y)
	if err != nil {
		t.Fatal(err)
	}
	if diff := maxAbsDiff(dy.F32, yV.Data.F32); diff > 1e-4 {
		t.Fatalf("forward mismatch: max diff %g", diff)
	}
	g, err := d.Backward(ctx, cb)
	if err != nil {
		t.Fatal(err)
	}
	check := func(name string, dev compute.Buffer, host *autograd.Value) {
		dt, err := v.Download(dev)
		if err != nil {
			t.Fatal(err)
		}
		if diff := maxAbsDiff(dt.F32, host.Grad.F32); diff > 3e-3 {
			t.Fatalf("%s mismatch: max diff %g", name, diff)
		}
	}
	check("dAqkv", g.DQKV[0], params[lw.AttnQKV][0])
	check("dBqkv", g.DQKV[1], params[lw.AttnQKV][1])
	check("dAgate", g.DGate[0], params[lw.AttnGate][0])
	check("dBgate", g.DGate[1], params[lw.AttnGate][1])
	check("dAout", g.DOUT[0], params[lw.SSMOut][0])
	check("dBout", g.DOUT[1], params[lw.SSMOut][1])
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
	if diff := maxAbsDiff(dt.F32, xV.Grad.F32); diff > 3e-3 {
		t.Fatalf("dX mismatch: max diff %g", diff)
	}
}

var _ = math.Abs

func TestDeviceGDNTiming(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cfg := midCfg()
	w := qwen38.NewRandom(cfg, 9)
	lw := &w.Layers[0] // linear attention
	loCfg := LoRAConfig{Rank: 8, Alpha: 16, Targets: DefaultLoRA().Targets}
	d, err := NewDeviceGDN(v, cfg, lw, loCfg, 200)
	if err != nil {
		t.Fatal(err)
	}
	hostEnc := map[*qwen38.Weight]*LoRA{
		lw.AttnQKV: d.qkv.lora, lw.AttnGate: d.gate.lora, lw.SSMOut: d.out.lora,
		lw.FfnGate: d.fg.lora, lw.FfnUp: d.fu.lora, lw.FfnDown: d.fd.lora,
	}
	const T = 64
	x := autograd.Random(310, 0.3, cfg.NEmbd, T)
	c := autograd.Random(311, 0.3, cfg.NEmbd, T)
	xb, _ := v.Upload(x)
	cb, _ := v.Upload(c)

	ctx, err := d.Forward(xb, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.Backward(ctx, cb); err != nil {
		t.Fatal(err)
	}
	v.Sync()
	v.ResetStats()

	const iters = 10
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		ctx, err := d.Forward(xb, nil)
		if err != nil {
			t.Fatal(err)
		}
		g, err := d.Backward(ctx, cb)
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
		yV, _ := hostGdnRef(cfg, lw, hostEnc, xV)
		loss := autograd.VSum(autograd.VMul(yV, cV))
		loss.Backward()
	}
	hostDur := time.Since(t1) / iters

	t.Logf("gdn layer (D=%d T=%d): device %v submits=%d dispatches=%d",
		cfg.NEmbd, T, devDur.Round(time.Microsecond), st.Submits, st.Dispatches)
	t.Logf("host autograd: %v  (%.1fx)", hostDur.Round(time.Microsecond), float64(hostDur)/float64(devDur))
}
