package compute

import (
	"math"
	"testing"
)

func TestGetRows(t *testing.T) {
	table := &Tensor{Dims: []int{2, 3}, F32: []float32{1, 2, 3, 4, 5, 6}}
	out, err := GetRows(table, []int32{2, 0})
	if err != nil {
		t.Fatal(err)
	}
	if out.Dims[0] != 2 || out.Dims[1] != 2 {
		t.Fatalf("dims = %v", out.Dims)
	}
	want := []float32{5, 6, 1, 2}
	for i, w := range want {
		if out.F32[i] != w {
			t.Errorf("out[%d] = %v, want %v", i, out.F32[i], w)
		}
	}
	if _, err := GetRows(table, []int32{3}); err == nil {
		t.Error("expected out-of-range error")
	}
}

func TestRoPENeoX(t *testing.T) {
	// headDim 2, nDims 2, theta 1 -> freq 1. pos 1 rotates (1,0) to (cos1,sin1).
	x := &Tensor{Dims: []int{2, 1, 1}, F32: []float32{1, 0}}
	out, err := RoPENeoX(x, []int32{1}, 1.0, 2)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, out.F32[0], float32(math.Cos(1)), 1e-6, "cos")
	approx(t, out.F32[1], float32(math.Sin(1)), 1e-6, "sin")

	// Position 0 is the identity.
	id, err := RoPENeoX(x, []int32{0}, 10000, 2)
	if err != nil {
		t.Fatal(err)
	}
	if id.F32[0] != 1 || id.F32[1] != 0 {
		t.Errorf("identity = %v", id.F32)
	}

	// Rotation preserves the pair norm.
	y := &Tensor{Dims: []int{4, 1, 1}, F32: []float32{3, 4, 1, -2}}
	r, err := RoPENeoX(y, []int32{7}, 10000, 4)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, r.F32[0]*r.F32[0]+r.F32[2]*r.F32[2], 3*3+1*1, 1e-4, "norm0")
	approx(t, r.F32[1]*r.F32[1]+r.F32[3]*r.F32[3], 4*4+2*2, 1e-4, "norm1")
}

func TestL2Norm(t *testing.T) {
	x := &Tensor{Dims: []int{2}, F32: []float32{3, 4}}
	out := L2Norm(x, 0)
	approx(t, out.F32[0], 0.6, 1e-6, "x")
	approx(t, out.F32[1], 0.8, 1e-6, "y")
}

func TestAttentionCausal(t *testing.T) {
	q := &Tensor{Dims: []int{1, 1, 2}, F32: []float32{1, 1}}
	k := &Tensor{Dims: []int{1, 1, 2}, F32: []float32{1, 0}}
	v := &Tensor{Dims: []int{1, 1, 2}, F32: []float32{10, 20}}
	out, err := Attention(q, k, v, 1, 1, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	// Token 0 attends only to itself.
	approx(t, out.F32[0], 10, 1e-5, "tok0")
	// Token 1: softmax([1,0]) -> 0.7311*10 + 0.2689*20.
	approx(t, out.F32[1], 12.689, 1e-3, "tok1")
}

func TestAttentionGQA(t *testing.T) {
	// 2 query heads share 1 KV head.
	q := &Tensor{Dims: []int{1, 2, 1}, F32: []float32{1, 1}}
	k := &Tensor{Dims: []int{1, 1, 1}, F32: []float32{1}}
	v := &Tensor{Dims: []int{1, 1, 1}, F32: []float32{7}}
	out, err := Attention(q, k, v, 2, 1, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if out.F32[0] != 7 || out.F32[1] != 7 {
		t.Errorf("out = %v, want [7 7]", out.F32)
	}
}

func TestSSMConv(t *testing.T) {
	sx := &Tensor{Dims: []int{3, 1, 1}, F32: []float32{1, 2, 3}}
	c := &Tensor{Dims: []int{2, 1}, F32: []float32{1, 1}}
	out, err := SSMConv(sx, c)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dims[0] != 1 || out.Dims[1] != 2 {
		t.Fatalf("dims = %v", out.Dims)
	}
	approx(t, out.F32[0], 3, 1e-6, "o0")
	approx(t, out.F32[1], 5, 1e-6, "o1")
}

func TestGatedDeltaNetSingleToken(t *testing.T) {
	// S_v=2, H=1, T=1. Zero state, scalar gate exp(0)=1, beta=1,
	// k=[1,0], v=[1,1], q=[1,0].
	q := &Tensor{Dims: []int{2, 1, 1}, F32: []float32{1, 0}}
	k := &Tensor{Dims: []int{2, 1, 1}, F32: []float32{1, 0}}
	v := &Tensor{Dims: []int{2, 1, 1}, F32: []float32{1, 1}}
	g := &Tensor{Dims: []int{1, 1, 1}, F32: []float32{0}}
	beta := &Tensor{Dims: []int{1, 1, 1}, F32: []float32{1}}
	state := NewF32(2, 2, 1)

	out, ns, err := GatedDeltaNet(q, k, v, g, beta, state)
	if err != nil {
		t.Fatal(err)
	}
	inv := float32(1 / math.Sqrt(2))
	approx(t, out.F32[0], 1*inv, 1e-5, "o0")
	approx(t, out.F32[1], 1*inv, 1e-5, "o1")
	// New state transposed layout [j*S_v+i] = S[i][j] = [[1,1],[0,0]] -> [1,0,1,0].
	want := []float32{1, 0, 1, 0}
	for i, w := range want {
		approx(t, ns.F32[i], w, 1e-5, "state")
	}
}

func TestGatedDeltaNetDecay(t *testing.T) {
	// With a strong scalar gate, the contribution of the previous state decays.
	q := &Tensor{Dims: []int{1, 1, 2}, F32: []float32{1, 1}}
	k := &Tensor{Dims: []int{1, 1, 2}, F32: []float32{1, 1}}
	v := &Tensor{Dims: []int{1, 1, 2}, F32: []float32{1, 1}}
	g := &Tensor{Dims: []int{1, 1, 2}, F32: []float32{-1000, 0}}
	beta := &Tensor{Dims: []int{1, 1, 2}, F32: []float32{1, 1}}
	state := NewF32(1, 1, 1)
	out, _, err := GatedDeltaNet(q, k, v, g, beta, state)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(float64(out.F32[1])) {
		t.Fatal("NaN output with strong decay")
	}
}
