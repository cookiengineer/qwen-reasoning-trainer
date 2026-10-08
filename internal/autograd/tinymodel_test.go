package autograd

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// tinyDims fixes the geometry of the tiny hybrid model used to validate that
// the backward ops compose correctly across residual streams, normalization,
// full attention and GatedDeltaNet. It mirrors internal/model/qwen38 but with
// small dimensions.
type tinyDims struct {
	D, V, T         int
	H, hd, KV, NRot int
	FF              int
	kd, hLin, sv    int // linear attention: key head dim, value heads, value head dim
	DInner, dconv   int
	eps             float32
	ropeTheta       float64
}

func newTinyDims() tinyDims {
	return tinyDims{
		D: 6, V: 7, T: 3, H: 2, hd: 3, KV: 1, NRot: 2, FF: 8,
		kd: 3, hLin: 1, sv: 3, DInner: 3, dconv: 2,
		eps: 1e-6, ropeTheta: 10000,
	}
}

func (d tinyDims) normVec(seed uint64, n int) []float32 {
	return Random(seed, 1.0, n).F32
}

// tinyParams returns the model parameters in a fixed order. The list is the
// source of truth for both the analytic and finite-difference gradients.
func tinyParams(d tinyDims) []*compute.Tensor {
	keyDim := d.kd * d.hLin
	return []*compute.Tensor{
		Random(100, 0.3, d.D, d.V),                   // 0 emb
		Random(101, 0.3, d.D, d.H*d.hd),              // 1 Wq
		Random(102, 0.3, d.D, d.KV*d.hd),             // 2 Wk
		Random(103, 0.3, d.D, d.KV*d.hd),             // 3 Wv
		Random(104, 0.3, d.D, d.H*d.hd),              // 4 Wgate
		Random(105, 0.3, d.H*d.hd, d.D),              // 5 Wo
		Random(106, 0.3, d.D, d.FF),                  // 6 WffnG
		Random(107, 0.3, d.D, d.FF),                  // 7 WffnU
		Random(108, 0.3, d.FF, d.D),                  // 8 WffnD
		Random(109, 0.3, d.D, keyDim),                // 9 Wq2
		Random(110, 0.3, d.D, keyDim),                // 10 Wk2
		Random(111, 0.5, d.T+d.dconv-1, d.DInner, 1), // 11 convIn
		Random(112, 0.3, d.dconv, d.DInner),          // 12 Wconv
		Random(113, 0.3, d.D, 1),                     // 13 Wb2
		Random(114, 0.3, d.D, 1),                     // 14 Wg2
		Random(115, 0.3, d.D, d.DInner),              // 15 Wz2
		Random(116, 0.3, d.DInner, d.D),              // 16 Wo2
		Random(117, 0.3, d.D, d.FF),                  // 17 Wffn2G
		Random(118, 0.3, d.D, d.FF),                  // 18 Wffn2U
		Random(119, 0.3, d.FF, d.D),                  // 19 Wffn2D
		Random(120, 0.2, d.sv, d.sv, d.hLin),         // 20 state
		Random(121, 0.3, d.D, d.V),                   // 21 Wlm
	}
}

// tinyForward builds the graph from a flat parameter slice and returns the
// scalar loss plus the parameter leaves in the same order.
func tinyForward(ps []*compute.Tensor) (*Value, []*Value) {
	d := newTinyDims()
	leaves := make([]*Value, len(ps))
	for i, p := range ps {
		leaves[i] = Param(p)
	}
	L := func(i int) *Value { return leaves[i] }

	positions := []int32{0, 1, 2}
	tokens := []int32{1, 2, 3}
	scale := float32(1 / math.Sqrt(float64(d.hd)))

	qnW := d.normVec(200, d.hd)
	knW := d.normVec(201, d.hd)
	lnAttn := d.normVec(202, d.D)
	lnPost := d.normVec(203, d.D)
	lnAttn2 := d.normVec(204, d.D)
	lnPost2 := d.normVec(205, d.D)
	lnFinal := d.normVec(206, d.D)

	x := VGetRows(L(0), tokens) // [D,T]

	// Full-attention block.
	hn := VRMSNorm(x, lnAttn, d.eps)
	q := VRoPENeoX(VRMSNorm(VReshape(VMatMul(L(1), hn), d.hd, d.H, d.T), qnW, d.eps), positions, d.ropeTheta, d.NRot)
	k := VRoPENeoX(VRMSNorm(VReshape(VMatMul(L(2), hn), d.hd, d.KV, d.T), knW, d.eps), positions, d.ropeTheta, d.NRot)
	v := VReshape(VMatMul(L(3), hn), d.hd, d.KV, d.T)
	attn := VAttention(q, k, v, d.H, d.KV, scale, true) // [hd,H,T]
	gate := VSigmoid(VMatMul(L(4), hn))                 // [H*hd,T]
	attn = VMul(attn, gate)
	o := VMatMul(L(5), VReshape(attn, d.H*d.hd, d.T))
	x = VAdd(x, o)

	h2 := VRMSNorm(x, lnPost, d.eps)
	ff := VMatMul(L(8), VMul(VSilu(VMatMul(L(6), h2)), VMatMul(L(7), h2)))
	x = VAdd(x, ff)

	// GatedDeltaNet block.
	hn2 := VRMSNorm(x, lnAttn2, d.eps)
	convOut := VSilu(VSSMConv(L(11), L(12))) // [DInner,T,1]
	vL := VReshape(convOut, d.sv, d.hLin, d.T)
	qL := VL2Norm(VReshape(VMatMul(L(9), hn2), d.kd, d.hLin, d.T), d.eps)
	kL := VL2Norm(VReshape(VMatMul(L(10), hn2), d.kd, d.hLin, d.T), d.eps)
	beta := VSigmoid(VReshape(VMatMul(L(13), hn2), 1, d.hLin, d.T))
	gL := VScale(VSoftplus(VReshape(VMatMul(L(14), hn2), 1, d.hLin, d.T)), 0.3)
	gdnOut, newState := VGatedDeltaNet(qL, kL, vL, gL, beta, L(20))
	nrm := VRMSNorm(gdnOut, nil, d.eps)
	z := VSilu(VReshape(VMatMul(L(15), hn2), d.DInner, d.hLin, d.T))
	y := VReshape(VMul(nrm, z), d.DInner, d.T)
	oL := VMatMul(L(16), y)
	x = VAdd(x, oL)

	h3 := VRMSNorm(x, lnPost2, d.eps)
	ff2 := VMatMul(L(19), VMul(VSilu(VMatMul(L(17), h3)), VMatMul(L(18), h3)))
	x = VAdd(x, ff2)

	// Loss: assistant-style cross-entropy plus a small state term so the
	// GatedDeltaNet state gradient is exercised.
	xn := VRMSNorm(x, lnFinal, d.eps)
	logits := VMatMul(L(21), xn) // [V,T]
	loss := VCrossEntropy(logits, []int{1, 2, 3}, -100)
	loss = VAdd(loss, VScale(VSum(newState), 0.01))
	return loss, leaves
}

func TestTinyModelGradients(t *testing.T) {
	ps := tinyParams(newTinyDims())

	loss, leaves := tinyForward(ps)
	loss.Backward()
	grads := make([]*compute.Tensor, len(leaves))
	for i, l := range leaves {
		if l.Grad == nil {
			t.Fatalf("parameter %d received no gradient", i)
		}
		grads[i] = l.Grad
	}

	obj := func(in []*compute.Tensor) float64 {
		l, _ := tinyForward(in)
		return float64(l.Data.F32[0])
	}

	worst, where, err := MaxGradError(ps, grads, obj, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("tiny model worst relative grad error: %.3g", worst)
	if worst > 5e-2 {
		t.Fatalf("tiny model grad error %.3g too high at %s", worst, where)
	}
	if math.IsNaN(worst) {
		t.Fatal("NaN gradient error")
	}
}

func TestTinyModelLossDecreases(t *testing.T) {
	// A single step of gradient descent on the loss must reduce it, which
	// exercises the full backward composition beyond a local grad check.
	ps := tinyParams(newTinyDims())
	loss0, leaves := tinyForward(ps)
	loss0.Backward()
	before := loss0.Data.F32[0]
	const lr = 0.05
	for i, l := range leaves {
		if l.Grad == nil {
			continue
		}
		for j := range ps[i].F32 {
			ps[i].F32[j] -= lr * l.Grad.F32[j]
		}
	}
	loss1, _ := tinyForward(ps)
	after := loss1.Data.F32[0]
	if !(after < before) {
		t.Fatalf("loss did not decrease: before=%g after=%g", before, after)
	}
	if math.IsNaN(float64(after)) {
		t.Fatal("NaN loss after step")
	}
}
