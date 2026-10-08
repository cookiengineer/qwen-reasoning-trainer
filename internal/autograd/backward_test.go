package autograd

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func sumTensors(ts ...*compute.Tensor) float64 {
	var s float64
	for _, t := range ts {
		for _, v := range t.F32 {
			s += float64(v)
		}
	}
	return s
}

// check verifies analytic gradients against central differences and fails if the
// worst relative error exceeds tol.
func check(t *testing.T, inputs, grads []*compute.Tensor, obj Objective, tol float64) {
	t.Helper()
	worst, where, err := MaxGradError(inputs, grads, obj, 1e-3)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("worst relative grad error: %.3g", worst)
	if worst > tol {
		t.Fatalf("worst relative grad error %.3g > %.3g at %s", worst, tol, where)
	}
}

func TestL2NormBackward(t *testing.T) {
	x := Random(1, 2, 4, 3)
	obj := func(in []*compute.Tensor) float64 {
		return sumTensors(compute.L2Norm(in[0], 1e-6))
	}
	dx := L2NormBackward(x, Ones(4, 3), 1e-6)
	check(t, []*compute.Tensor{x}, []*compute.Tensor{dx}, obj, 5e-3)
}

func TestSSMConvBackward(t *testing.T) {
	sx := Random(2, 1.5, 5, 3, 1)
	c := Random(3, 1.5, 2, 3)
	obj := func(in []*compute.Tensor) float64 {
		out, err := compute.SSMConv(in[0], in[1])
		if err != nil {
			panic(err)
		}
		return sumTensors(out)
	}
	dSx, dC, err := SSMConvBackward(sx, c, Ones(3, 4, 1))
	if err != nil {
		t.Fatal(err)
	}
	check(t, []*compute.Tensor{sx, c}, []*compute.Tensor{dSx, dC}, obj, 5e-3)
}

func TestSSMConvBackwardMultipleChannels(t *testing.T) {
	sx := Random(4, 1.0, 6, 4, 1)
	c := Random(5, 1.0, 3, 4)
	obj := func(in []*compute.Tensor) float64 {
		out, err := compute.SSMConv(in[0], in[1])
		if err != nil {
			panic(err)
		}
		return sumTensors(out)
	}
	dSx, dC, err := SSMConvBackward(sx, c, Ones(4, 4, 1))
	if err != nil {
		t.Fatal(err)
	}
	check(t, []*compute.Tensor{sx, c}, []*compute.Tensor{dSx, dC}, obj, 5e-3)
}

func TestGatedDeltaNetBackwardScalarGate(t *testing.T) {
	sv, h, nTok := 2, 1, 3
	q := Random(6, 1.0, sv, h, nTok)
	k := Random(7, 1.0, sv, h, nTok)
	v := Random(8, 1.0, sv, h, nTok)
	g := Random(9, 0.3, 1, h, nTok)
	beta := Random(10, 0.5, 1, h, nTok)
	state := Random(11, 0.5, sv, sv, h)

	obj := func(in []*compute.Tensor) float64 {
		out, ns, err := compute.GatedDeltaNet(in[0], in[1], in[2], in[3], in[4], in[5])
		if err != nil {
			panic(err)
		}
		return sumTensors(out, ns)
	}
	dQ, dK, dV, dG, dBeta, dState, err := GatedDeltaNetBackward(q, k, v, g, beta, state, Ones(sv, h, nTok), Ones(sv, sv, h))
	if err != nil {
		t.Fatal(err)
	}
	inputs := []*compute.Tensor{q, k, v, g, beta, state}
	grads := []*compute.Tensor{dQ, dK, dV, dG, dBeta, dState}
	check(t, inputs, grads, obj, 5e-3)
}

func TestGatedDeltaNetBackwardKDAGate(t *testing.T) {
	sv, h, nTok := 3, 2, 3
	q := Random(16, 1.0, sv, h, nTok)
	k := Random(17, 1.0, sv, h, nTok)
	v := Random(18, 1.0, sv, h, nTok)
	g := Random(19, 0.3, sv, h, nTok)
	beta := Random(20, 0.5, 1, h, nTok)
	state := Random(21, 0.5, sv, sv, h)

	obj := func(in []*compute.Tensor) float64 {
		out, ns, err := compute.GatedDeltaNet(in[0], in[1], in[2], in[3], in[4], in[5])
		if err != nil {
			panic(err)
		}
		return sumTensors(out, ns)
	}
	dQ, dK, dV, dG, dBeta, dState, err := GatedDeltaNetBackward(q, k, v, g, beta, state, Ones(sv, h, nTok), Ones(sv, sv, h))
	if err != nil {
		t.Fatal(err)
	}
	inputs := []*compute.Tensor{q, k, v, g, beta, state}
	grads := []*compute.Tensor{dQ, dK, dV, dG, dBeta, dState}
	check(t, inputs, grads, obj, 5e-3)
}

func closeAll(t *testing.T, name string, a, b *compute.Tensor) {
	t.Helper()
	if a == nil || b == nil || len(a.F32) != len(b.F32) {
		t.Fatalf("%s: shape mismatch", name)
	}
	for i := range a.F32 {
		d := math.Abs(float64(a.F32[i] - b.F32[i]))
		if d > 1e-4+1e-4*math.Abs(float64(b.F32[i])) {
			t.Fatalf("%s[%d]: chunked %g vs naive %g", name, i, a.F32[i], b.F32[i])
		}
	}
}

func TestGatedDeltaNetBackwardChunked(t *testing.T) {
	for _, kda := range []bool{false, true} {
		sv, h, nTok := 3, 2, 7
		q := Random(70, 1.0, sv, h, nTok)
		k := Random(71, 1.0, sv, h, nTok)
		v := Random(72, 1.0, sv, h, nTok)
		var g *compute.Tensor
		if kda {
			g = Random(73, 0.3, sv, h, nTok)
		} else {
			g = Random(73, 0.3, 1, h, nTok)
		}
		beta := Random(74, 0.5, 1, h, nTok)
		state := Random(75, 0.5, sv, sv, h)
		dOut := Random(76, 1.0, sv, h, nTok)
		dNew := Random(77, 1.0, sv, sv, h)

		nQ, nK, nV, nG, nB, nS, err := GatedDeltaNetBackward(q, k, v, g, beta, state, dOut, dNew)
		if err != nil {
			t.Fatal(err)
		}
		for _, chunk := range []int{1, 2, 3, 4, 7, 100} {
			cQ, cK, cV, cG, cB, cS, err := GatedDeltaNetBackwardChunked(q, k, v, g, beta, state, dOut, dNew, chunk)
			if err != nil {
				t.Fatal(err)
			}
			closeAll(t, "dQ", cQ, nQ)
			closeAll(t, "dK", cK, nK)
			closeAll(t, "dV", cV, nV)
			closeAll(t, "dG", cG, nG)
			closeAll(t, "dBeta", cB, nB)
			closeAll(t, "dState", cS, nS)
		}
	}
}
