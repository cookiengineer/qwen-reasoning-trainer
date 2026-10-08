package autograd

import "github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"

// Weight abstracts a frozen model parameter for the autograd graph. Large
// weights stay quantized on disk; only the activation gradient is needed
// because the base is frozen (LoRA/adapters are trained separately).
type Weight interface {
	In() int
	Out() int
	// MatMul computes W * x: W [K,N], x [K,M] -> [N,M].
	MatMul(x *compute.Tensor) (*compute.Tensor, error)
	// MatMulTranspose computes W^T * dY: W [K,N], dY [N,M] -> [K,M].
	MatMulTranspose(dY *compute.Tensor) (*compute.Tensor, error)
	// GetRows gathers selected output rows (embedding lookup).
	GetRows(indices []int32) (*compute.Tensor, error)
}

// VMatMulWeight is the differentiable matmul against a frozen (quantized)
// weight. Only the activation gradient flows.
func VMatMulWeight(w Weight, x *Value) *Value {
	data, err := w.MatMul(x.Data)
	if err != nil {
		panic(err)
	}
	return newVal(data, []*Value{x}, func(g *compute.Tensor) {
		dx, err := w.MatMulTranspose(g)
		if err != nil {
			panic(err)
		}
		x.AddToGrad(dx)
	})
}

// VGetRowsWeight is the embedding gather against a frozen (quantized) table.
// The table does not receive a gradient.
func VGetRowsWeight(w Weight, indices []int32) *Value {
	data, err := w.GetRows(indices)
	if err != nil {
		panic(err)
	}
	return newVal(data, nil, nil)
}

// VSplitQG splits the fused Q+gate projection. qg has Dims [2*hd*nHead, T];
// each (token, head) contributes hd query values then hd gate values. The
// returned q has Dims [hd,nHead,T] and gate has [nHead*hd,T].
func VSplitQG(qg *Value, hd, nHead, T int) (q, gate *Value) {
	qgN := 2 * hd
	row := nHead * qgN
	qData := compute.NewF32(hd, nHead, T)
	gData := compute.NewF32(nHead*hd, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nHead; hh++ {
			base := t*row + hh*qgN
			for d := 0; d < hd; d++ {
				qData.F32[d+hh*hd+t*(nHead*hd)] = qg.Data.F32[base+d]
				gData.F32[hh*hd+d+t*(nHead*hd)] = qg.Data.F32[base+hd+d]
			}
		}
	}
	q = newVal(qData, []*Value{qg}, func(g *compute.Tensor) {
		dqg := compute.NewF32(qg.Data.Dims...)
		for t := 0; t < T; t++ {
			for hh := 0; hh < nHead; hh++ {
				base := t*row + hh*qgN
				for d := 0; d < hd; d++ {
					dqg.F32[base+d] += g.F32[d+hh*hd+t*(nHead*hd)]
				}
			}
		}
		qg.AddToGrad(dqg)
	})
	gate = newVal(gData, []*Value{qg}, func(g *compute.Tensor) {
		dqg := compute.NewF32(qg.Data.Dims...)
		for t := 0; t < T; t++ {
			for hh := 0; hh < nHead; hh++ {
				base := t*row + hh*qgN
				for d := 0; d < hd; d++ {
					dqg.F32[base+hd+d] += g.F32[hh*hd+d+t*(nHead*hd)]
				}
			}
		}
		qg.AddToGrad(dqg)
	})
	return q, gate
}

// VGatherHeads extracts head-structured channels from a conv output of Dims
// [convDim,T], producing [headDim,nHead,T]. Channel c = offset + head*headDim + d.
func VGatherHeads(convOut *Value, offset, headDim, nHead, T, convDim int) *Value {
	data := compute.NewF32(headDim, nHead, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nHead; hh++ {
			base := offset + hh*headDim
			for d := 0; d < headDim; d++ {
				data.F32[d+hh*headDim+t*(nHead*headDim)] = convOut.Data.F32[(base+d)+t*convDim]
			}
		}
	}
	return newVal(data, []*Value{convOut}, func(g *compute.Tensor) {
		d := compute.NewF32(convOut.Data.Dims...)
		for t := 0; t < T; t++ {
			for hh := 0; hh < nHead; hh++ {
				base := offset + hh*headDim
				for dd := 0; dd < headDim; dd++ {
					d.F32[(base+dd)+t*convDim] += g.F32[dd+hh*headDim+t*(nHead*headDim)]
				}
			}
		}
		convOut.AddToGrad(d)
	})
}

// VRepeatHeads repeats q/k heads from nIn to nOut so that output head j takes
// input head j%nIn.
func VRepeatHeads(x *Value, headDim, nIn, nOut, T int) *Value {
	data := compute.NewF32(headDim, nOut, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nOut; hh++ {
			src := hh % nIn
			for d := 0; d < headDim; d++ {
				data.F32[d+hh*headDim+t*(nOut*headDim)] = x.Data.F32[d+src*headDim+t*(nIn*headDim)]
			}
		}
	}
	return newVal(data, []*Value{x}, func(g *compute.Tensor) {
		d := compute.NewF32(x.Data.Dims...)
		for t := 0; t < T; t++ {
			for hh := 0; hh < nOut; hh++ {
				src := hh % nIn
				for dd := 0; dd < headDim; dd++ {
					d.F32[dd+src*headDim+t*(nIn*headDim)] += g.F32[dd+hh*headDim+t*(nOut*headDim)]
				}
			}
		}
		x.AddToGrad(d)
	})
}

// VConvInput builds the causal-conv input for a full (training) sequence: DConv-1
// zero history rows followed by the qkv projection [convDim,T], laid out as
// [DConv-1+T, convDim, 1].
func VConvInput(qkv *Value, dConv, convDim, T int) *Value {
	ncs := dConv - 1 + T
	data := compute.NewF32(ncs, convDim, 1)
	for t := 0; t < T; t++ {
		for ch := 0; ch < convDim; ch++ {
			data.F32[(dConv-1+t)+ch*ncs] = qkv.Data.F32[ch+t*convDim]
		}
	}
	return newVal(data, []*Value{qkv}, func(g *compute.Tensor) {
		d := compute.NewF32(qkv.Data.Dims...)
		for t := 0; t < T; t++ {
			for ch := 0; ch < convDim; ch++ {
				d.F32[ch+t*convDim] += g.F32[(dConv-1+t)+ch*ncs]
			}
		}
		qkv.AddToGrad(d)
	})
}
