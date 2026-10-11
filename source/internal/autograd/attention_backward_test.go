package autograd

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func TestSoftplusBackward(t *testing.T) {
	x := Random(40, 2.0, 6, 3)
	obj := func(in []*compute.Tensor) float64 {
		var s float64
		for _, v := range in[0].F32 {
			s += math.Log1p(math.Exp(float64(v)))
		}
		return s
	}
	check(t, []*compute.Tensor{x}, []*compute.Tensor{SoftplusBackward(x, Ones(6, 3))}, obj, 5e-3)
}

func TestAttentionBackwardCausalGQA(t *testing.T) {
	headDim, nHead, nKV, nTok := 2, 2, 1, 3
	q := Random(41, 1.0, headDim, nHead, nTok)
	k := Random(42, 1.0, headDim, nKV, nTok)
	v := Random(43, 1.0, headDim, nKV, nTok)
	scale := float32(1 / math.Sqrt(float64(headDim)))

	obj := func(in []*compute.Tensor) float64 {
		out, err := compute.Attention(in[0], in[1], in[2], nHead, nKV, scale, true)
		if err != nil {
			panic(err)
		}
		return sumTensors(out)
	}
	dQ, dK, dV := AttentionBackward(q, k, v, Ones(headDim, nHead, nTok), nHead, nKV, scale, true)
	check(t, []*compute.Tensor{q, k, v}, []*compute.Tensor{dQ, dK, dV}, obj, 5e-3)
}

func TestAttentionBackwardNonCausalGQA(t *testing.T) {
	headDim, nHead, nKV := 2, 2, 1
	nQ, nKVLen := 2, 3
	q := Random(44, 1.0, headDim, nHead, nQ)
	k := Random(45, 1.0, headDim, nKV, nKVLen)
	v := Random(46, 1.0, headDim, nKV, nKVLen)
	scale := float32(1 / math.Sqrt(float64(headDim)))

	obj := func(in []*compute.Tensor) float64 {
		out, err := compute.Attention(in[0], in[1], in[2], nHead, nKV, scale, false)
		if err != nil {
			panic(err)
		}
		return sumTensors(out)
	}
	dQ, dK, dV := AttentionBackward(q, k, v, Ones(headDim, nHead, nQ), nHead, nKV, scale, false)
	check(t, []*compute.Tensor{q, k, v}, []*compute.Tensor{dQ, dK, dV}, obj, 5e-3)
}
