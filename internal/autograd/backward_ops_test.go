package autograd

import (
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func TestMatMulBackward(t *testing.T) {
	a := Random(30, 1.0, 3, 4)
	b := Random(31, 1.0, 3, 5)
	obj := func(in []*compute.Tensor) float64 {
		out, err := compute.MatMul(in[0], in[1])
		if err != nil {
			panic(err)
		}
		return sumTensors(out)
	}
	dA, dB := MatMulBackward(a, b, Ones(4, 5))
	check(t, []*compute.Tensor{a, b}, []*compute.Tensor{dA, dB}, obj, 5e-3)
}

func TestRMSNormBackward(t *testing.T) {
	x := Random(32, 1.5, 4, 3)
	w := Random(33, 1.0, 4)
	obj := func(in []*compute.Tensor) float64 {
		return sumTensors(compute.RMSNorm(in[0], in[1].F32, 1e-6))
	}
	dX, dW := RMSNormBackward(x, Ones(4, 3), w.F32, 1e-6)
	dWT := &compute.Tensor{Dims: []int{4}, F32: dW}
	check(t, []*compute.Tensor{x, w}, []*compute.Tensor{dX, dWT}, obj, 5e-3)
}

func TestSiluBackward(t *testing.T) {
	x := Random(34, 2.0, 6, 3)
	obj := func(in []*compute.Tensor) float64 {
		act := in[0].Clone()
		for i, v := range act.F32 {
			act.F32[i] = compute.Silu(v)
		}
		return sumTensors(act)
	}
	check(t, []*compute.Tensor{x}, []*compute.Tensor{SiluBackward(x, Ones(6, 3))}, obj, 5e-3)
}

func TestSoftmaxBackward(t *testing.T) {
	x := Random(35, 2.0, 5, 4)
	obj := func(in []*compute.Tensor) float64 {
		sm := in[0].Clone()
		compute.Softmax(sm)
		return sumTensors(sm)
	}
	sm := x.Clone()
	compute.Softmax(sm)
	check(t, []*compute.Tensor{x}, []*compute.Tensor{SoftmaxBackward(sm, Ones(5, 4))}, obj, 5e-3)
}

func TestRoPENeoXBackward(t *testing.T) {
	x := Random(36, 1.0, 4, 2, 3)
	positions := []int32{0, 2, 5}
	obj := func(in []*compute.Tensor) float64 {
		out, err := compute.RoPENeoX(in[0], positions, 10000, 4)
		if err != nil {
			panic(err)
		}
		return sumTensors(out)
	}
	dx := RoPENeoXBackward(x, Ones(4, 2, 3), positions, 10000, 4)
	check(t, []*compute.Tensor{x}, []*compute.Tensor{dx}, obj, 5e-3)
}

func TestGetRowsBackward(t *testing.T) {
	table := Random(37, 1.0, 4, 5)
	indices := []int32{2, 0, 2}
	obj := func(in []*compute.Tensor) float64 {
		out, err := compute.GetRows(in[0], indices)
		if err != nil {
			panic(err)
		}
		return sumTensors(out)
	}
	dTable := GetRowsBackward(4, 5, indices, Ones(4, 3))
	check(t, []*compute.Tensor{table}, []*compute.Tensor{dTable}, obj, 5e-3)
}

func TestCrossEntropyBackward(t *testing.T) {
	logits := Random(38, 2.0, 6, 4)
	targets := []int{1, -100, 5, 0}
	obj := func(in []*compute.Tensor) float64 {
		l, err := compute.CrossEntropy(in[0], targets, -100)
		if err != nil {
			panic(err)
		}
		return float64(l)
	}
	dLogits := CrossEntropyBackward(logits, targets, -100)
	check(t, []*compute.Tensor{logits}, []*compute.Tensor{dLogits}, obj, 5e-3)
}
