package autograd

import "github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"

// VRMSNorm is the differentiable RMSNorm. Only the input gradient is
// propagated; the per-row weight is treated as a frozen constant (the
// QLoRA setup freezes all normalization weights).
func VRMSNorm(x *Value, weight []float32, eps float32) *Value {
	data := compute.RMSNorm(x.Data, weight, eps)
	return newVal(data, []*Value{x}, func(g *compute.Tensor) {
		dX, _ := RMSNormBackward(x.Data, g, weight, eps)
		x.AddToGrad(dX)
	})
}

// VL2Norm is the differentiable row L2 normalization.
func VL2Norm(x *Value, eps float32) *Value {
	data := compute.L2Norm(x.Data, eps)
	return newVal(data, []*Value{x}, func(g *compute.Tensor) {
		x.AddToGrad(L2NormBackward(x.Data, g, eps))
	})
}

// VSoftmax is the differentiable softmax.
func VSoftmax(x *Value) *Value {
	out := x.Data.Clone()
	compute.Softmax(out)
	return newVal(out, []*Value{x}, func(g *compute.Tensor) {
		x.AddToGrad(SoftmaxBackward(out, g))
	})
}

// VRoPENeoX is the differentiable GPT-NeoX rotary embedding.
func VRoPENeoX(x *Value, positions []int32, theta float64, nDims int) *Value {
	data, err := compute.RoPENeoX(x.Data, positions, theta, nDims)
	if err != nil {
		panic(err)
	}
	return newVal(data, []*Value{x}, func(g *compute.Tensor) {
		x.AddToGrad(RoPENeoXBackward(x.Data, g, positions, theta, nDims))
	})
}

// VGetRows is the differentiable row gather.
func VGetRows(table *Value, indices []int32) *Value {
	data, err := compute.GetRows(table.Data, indices)
	if err != nil {
		panic(err)
	}
	rowLen := table.Data.Ne(0)
	nRows := table.Data.Ne(1)
	return newVal(data, []*Value{table}, func(g *compute.Tensor) {
		table.AddToGrad(GetRowsBackward(rowLen, nRows, indices, g))
	})
}

// VAttention is the differentiable grouped-query attention.
func VAttention(q, k, v *Value, nHead, nHeadKV int, scale float32, causal bool) *Value {
	data, err := compute.Attention(q.Data, k.Data, v.Data, nHead, nHeadKV, scale, causal)
	if err != nil {
		panic(err)
	}
	return newVal(data, []*Value{q, k, v}, func(g *compute.Tensor) {
		dQ, dK, dV := AttentionBackward(q.Data, k.Data, v.Data, g, nHead, nHeadKV, scale, causal)
		q.AddToGrad(dQ)
		k.AddToGrad(dK)
		v.AddToGrad(dV)
	})
}

// VSSMConv is the differentiable causal depthwise conv1d.
func VSSMConv(sx, c *Value) *Value {
	data, err := compute.SSMConv(sx.Data, c.Data)
	if err != nil {
		panic(err)
	}
	return newVal(data, []*Value{sx, c}, func(g *compute.Tensor) {
		dSx, dC, err := SSMConvBackward(sx.Data, c.Data, g)
		if err != nil {
			panic(err)
		}
		sx.AddToGrad(dSx)
		c.AddToGrad(dC)
	})
}

// VGatedDeltaNet is the differentiable gated delta rule. It returns the
// attention output and the updated state as Values.
func VGatedDeltaNet(q, k, v, g, beta, state *Value) (out, newState *Value) {
	o, ns, err := compute.GatedDeltaNet(q.Data, k.Data, v.Data, g.Data, beta.Data, state.Data)
	if err != nil {
		panic(err)
	}
	out = newVal(o, []*Value{q, k, v, g, beta, state}, func(dOut *compute.Tensor) {
		// dNewState is supplied by the newState node below; the output
		// gradient alone flows through this closure.
		dQ, dK, dV, dG, dBeta, dState, err := GatedDeltaNetBackward(q.Data, k.Data, v.Data, g.Data, beta.Data, state.Data, dOut, zeroLike(ns))
		if err != nil {
			panic(err)
		}
		q.AddToGrad(dQ)
		k.AddToGrad(dK)
		v.AddToGrad(dV)
		g.AddToGrad(dG)
		beta.AddToGrad(dBeta)
		state.AddToGrad(dState)
	})
	newState = newVal(ns, []*Value{q, k, v, g, beta, state}, func(dNS *compute.Tensor) {
		dQ, dK, dV, dG, dBeta, dState, err := GatedDeltaNetBackward(q.Data, k.Data, v.Data, g.Data, beta.Data, state.Data, zeroLike(o), dNS)
		if err != nil {
			panic(err)
		}
		q.AddToGrad(dQ)
		k.AddToGrad(dK)
		v.AddToGrad(dV)
		g.AddToGrad(dG)
		beta.AddToGrad(dBeta)
		state.AddToGrad(dState)
	})
	return out, newState
}

// VCrossEntropy is the differentiable mean cross-entropy loss, returning a
// scalar Value.
func VCrossEntropy(logits *Value, targets []int, ignoreIndex int) *Value {
	loss, err := compute.CrossEntropy(logits.Data, targets, ignoreIndex)
	if err != nil {
		panic(err)
	}
	data := compute.NewF32(1)
	data.F32[0] = loss
	return newVal(data, []*Value{logits}, func(g *compute.Tensor) {
		d := CrossEntropyBackward(logits.Data, targets, ignoreIndex)
		s := g.F32[0]
		for i := range d.F32 {
			d.F32[i] *= s
		}
		logits.AddToGrad(d)
	})
}

func zeroLike(t *compute.Tensor) *compute.Tensor {
	return compute.NewF32(t.Dims...)
}
