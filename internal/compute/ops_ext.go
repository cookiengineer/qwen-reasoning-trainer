package compute

import (
	"errors"
	"math"
)

// ErrNotImplemented reports an op that a backend does not support.
var ErrNotImplemented = errors.New("compute: op not implemented")

// GetRows selects rows from table, whose Dims are [rowLen, nRows]. indices
// must be in [0, nRows). The result has Dims [rowLen, len(indices)], matching
// ggml_get_rows.
func GetRows(table *Tensor, indices []int32) (*Tensor, error) {
	rowLen := table.Ne(0)
	nRows := table.Ne(1)
	out := NewF32(rowLen, len(indices))
	for i, idx := range indices {
		if idx < 0 || int(idx) >= nRows {
			return nil, ErrShape
		}
		copy(out.F32[i*rowLen:(i+1)*rowLen], table.F32[int(idx)*rowLen:(int(idx)+1)*rowLen])
	}
	return out, nil
}

// RoPENeoX applies rotary position embeddings in the GPT-NeoX layout (pairs
// i and i+nDims/2) over the first nDims elements of each head. x has Dims
// [headDim, nHead, nTokens]; positions has length nTokens.
func RoPENeoX(x *Tensor, positions []int32, theta float64, nDims int) (*Tensor, error) {
	headDim := x.Ne(0)
	nHead := x.Ne(1)
	nTok := x.Ne(2)
	if len(positions) != nTok {
		return nil, ErrShape
	}
	if nDims <= 0 || nDims > headDim {
		return nil, ErrShape
	}
	if nDims%2 != 0 {
		return nil, ErrShape
	}
	half := nDims / 2
	out := x.Clone()
	for t := 0; t < nTok; t++ {
		pos := float64(positions[t])
		for h := 0; h < nHead; h++ {
			base := (t*nHead + h) * headDim
			for i := 0; i < half; i++ {
				freq := math.Pow(theta, -2*float64(i)/float64(nDims))
				ang := pos * freq
				c, s := math.Cos(ang), math.Sin(ang)
				x0 := float64(out.F32[base+i])
				x1 := float64(out.F32[base+i+half])
				out.F32[base+i] = float32(x0*c - x1*s)
				out.F32[base+i+half] = float32(x0*s + x1*c)
			}
		}
	}
	return out, nil
}

// L2Norm normalizes each contiguous row of length Dims[0] to unit L2 norm
// (with epsilon), as used by the GatedDeltaNet q/k projection.
func L2Norm(x *Tensor, eps float32) *Tensor {
	row := x.Ne(0)
	rows := len(x.F32) / row
	out := NewF32(x.Dims...)
	for r := 0; r < rows; r++ {
		base := r * row
		var ss float64
		for i := 0; i < row; i++ {
			v := float64(x.F32[base+i])
			ss += v * v
		}
		inv := float32(1 / math.Sqrt(ss+float64(eps)))
		for i := 0; i < row; i++ {
			out.F32[base+i] = x.F32[base+i] * inv
		}
	}
	return out
}

// Attention computes scaled dot-product attention with grouped-query support.
// q has Dims [headDim, nHead, nQ], k and v have [headDim, nHeadKV, nKV]. scale
// is applied to the logits before softmax. When causal is true and nQ == nKV,
// query token i attends only to key token j <= i. When nQ != nKV (incremental
// decode against a cache) every query attends to every key. The result has
// Dims [headDim, nHead, nQ].
func Attention(q, k, v *Tensor, nHead, nHeadKV int, scale float32, causal bool) (*Tensor, error) {
	headDim := q.Ne(0)
	nQ := q.Ne(2)
	nKV := k.Ne(2)
	if k.Ne(0) != headDim || v.Ne(0) != headDim || k.Ne(2) != nKV || v.Ne(2) != nKV {
		return nil, ErrShape
	}
	if nHeadKV <= 0 || nHead%nHeadKV != 0 {
		return nil, ErrShape
	}
	group := nHead / nHeadKV
	out := NewF32(headDim, nHead, nQ)
	logits := make([]float64, nKV)
	qBase := func(h, t int) int { return (t*nHead + h) * headDim }
	kBase := func(h, t int) int { return (t*nHeadKV + h) * headDim }

	for h := 0; h < nHead; h++ {
		kv := h / group
		for i := 0; i < nQ; i++ {
			qOff := qBase(h, i)
			maxJ := nKV
			if causal && nQ == nKV {
				maxJ = i + 1
			}
			max := math.Inf(-1)
			for j := 0; j < maxJ; j++ {
				kOff := kBase(kv, j)
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(q.F32[qOff+d]) * float64(k.F32[kOff+d])
				}
				dot *= float64(scale)
				logits[j] = dot
				if dot > max {
					max = dot
				}
			}
			var sum float64
			for j := 0; j < maxJ; j++ {
				e := math.Exp(logits[j] - max)
				logits[j] = e
				sum += e
			}
			outOff := qOff
			for d := 0; d < headDim; d++ {
				var acc float64
				for j := 0; j < maxJ; j++ {
					vOff := (j*nHeadKV + kv) * headDim
					acc += logits[j] * float64(v.F32[vOff+d])
				}
				out.F32[outOff+d] = float32(acc / sum)
			}
		}
	}
	return out, nil
}

// SSMConv applies a causal depthwise conv1d, mirroring ggml_ssm_conv. sx has
// Dims [dConv-1+nT, dInner, nSeq], c has [dConv, dInner], and the result has
// [dInner, nT, nSeq].
func SSMConv(sx, c *Tensor) (*Tensor, error) {
	dConv := c.Ne(0)
	dInner := c.Ne(1)
	ncs := sx.Ne(0)
	nT := ncs - dConv + 1
	nSeq := sx.Ne(2)
	if sx.Ne(1) != dInner || nT < 0 {
		return nil, ErrShape
	}
	out := NewF32(dInner, nT, nSeq)
	for s := 0; s < nSeq; s++ {
		for t := 0; t < nT; t++ {
			for ch := 0; ch < dInner; ch++ {
				var sum float32
				sOff := ch*ncs + t
				cOff := ch * dConv
				for i := 0; i < dConv; i++ {
					sum += sx.F32[sOff+i] * c.F32[cOff+i]
				}
				out.F32[(s*nT+t)*dInner+ch] = sum
			}
		}
	}
	return out, nil
}

// GatedDeltaNet runs the gated delta rule recurrence for one sequence,
// mirroring ggml_gated_delta_net (K=1). Inputs use Dims:
// q,k,v [S_v,nHead,nTok], g [1 or S_v,nHead,nTok], beta [1,nHead,nTok],
// state [S_v,S_v,nHead]. It returns the attention output [S_v,nHead,nTok] and
// the updated state [S_v,S_v,nHead]. The state is stored transposed:
// state[j*S_v+i] = S[i][j].
func GatedDeltaNet(q, k, v, g, beta, state *Tensor) (out, newState *Tensor, err error) {
	sv := v.Ne(0)
	h := v.Ne(1)
	nTok := v.Ne(2)
	if q.Ne(0) != sv || k.Ne(0) != sv || q.Ne(1) != h || k.Ne(1) != h {
		return nil, nil, ErrShape
	}
	if state.Ne(0) != sv || state.Ne(1) != sv || state.Ne(2) != h {
		return nil, nil, ErrShape
	}
	kda := g.Ne(0) == sv
	if g.Ne(0) != 1 && !kda {
		return nil, nil, ErrShape
	}
	scale := float32(1 / math.Sqrt(float64(sv)))

	out = NewF32(sv, h, nTok)
	newState = NewF32(sv, sv, h)
	work := make([]float64, sv*sv)
	decay := make([]float64, sv)
	delta := make([]float64, sv)

	for hh := 0; hh < h; hh++ {
		// Copy the incoming state for this head, transposed layout.
		for i := 0; i < sv*sv; i++ {
			work[i] = float64(state.F32[hh*sv*sv+i])
		}
		for t := 0; t < nTok; t++ {
			qOff := (t*h + hh) * sv
			kOff := qOff
			vOff := qOff
			gOff := (t*h + hh) * g.Ne(0)
			bOff := (t*h + hh) * beta.Ne(0)

			if kda {
				for i := 0; i < sv; i++ {
					decay[i] = math.Exp(float64(g.F32[gOff+i]))
				}
				for j := 0; j < sv; j++ {
					row := work[j*sv : j*sv+sv]
					for i := 0; i < sv; i++ {
						row[i] *= decay[i]
					}
				}
			} else {
				dec := math.Exp(float64(g.F32[gOff]))
				for i := range work {
					work[i] *= dec
				}
			}

			b := float64(beta.F32[bOff])
			for j := 0; j < sv; j++ {
				var sum float64
				row := work[j*sv : j*sv+sv]
				for i := 0; i < sv; i++ {
					sum += row[i] * float64(k.F32[kOff+i])
				}
				delta[j] = (float64(v.F32[vOff+j]) - sum) * b
			}
			for j := 0; j < sv; j++ {
				dj := delta[j]
				row := work[j*sv : j*sv+sv]
				for i := 0; i < sv; i++ {
					row[i] += float64(k.F32[kOff+i]) * dj
				}
			}
			for j := 0; j < sv; j++ {
				var sum float64
				row := work[j*sv : j*sv+sv]
				for i := 0; i < sv; i++ {
					sum += row[i] * float64(q.F32[qOff+i])
				}
				out.F32[(t*h+hh)*sv+j] = float32(sum) * scale
			}
		}
		for i := 0; i < sv*sv; i++ {
			newState.F32[hh*sv*sv+i] = float32(work[i])
		}
	}
	return out, newState, nil
}
