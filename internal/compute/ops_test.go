package compute

import (
	"math"
	"testing"
)

func approx(t *testing.T, got, want, tol float32, msg string) {
	t.Helper()
	if float32(math.Abs(float64(got-want))) > tol {
		t.Errorf("%s: got %v, want %v", msg, got, want)
	}
}

func TestMatMul(t *testing.T) {
	a := &Tensor{Dims: []int{2, 2}, F32: []float32{1, 2, 3, 4}}
	b := &Tensor{Dims: []int{2, 2}, F32: []float32{5, 6, 7, 8}}
	c, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{17, 23, 39, 53}
	for i, w := range want {
		if c.F32[i] != w {
			t.Errorf("c[%d] = %v, want %v", i, c.F32[i], w)
		}
	}
	if c.Dims[0] != 2 || c.Dims[1] != 2 {
		t.Errorf("dims = %v", c.Dims)
	}
}

func TestMatMulVector(t *testing.T) {
	// x [K=3] times W [K=3, M=2]. GGML order: each of the M columns is K long.
	x := &Tensor{Dims: []int{3}, F32: []float32{1, 2, 3}}
	w := &Tensor{Dims: []int{3, 2}, F32: []float32{1, 0, 0, 1, 1, 1}}
	out, err := MatMul(x, w)
	if err != nil {
		t.Fatal(err)
	}
	// m0 = 1*1+2*0+3*0 = 1; m1 = 1*1+2*1+3*1 = 6.
	if out.F32[0] != 1 || out.F32[1] != 6 {
		t.Errorf("out = %v, want [1 6]", out.F32)
	}
}

func TestRMSNorm(t *testing.T) {
	x := &Tensor{Dims: []int{2}, F32: []float32{3, 4}}
	y := RMSNorm(x, nil, 0)
	inv := float32(1 / math.Sqrt(12.5))
	approx(t, y.F32[0], 3*inv, 1e-5, "y0")
	approx(t, y.F32[1], 4*inv, 1e-5, "y1")

	z := RMSNorm(x, []float32{1, 2}, 0)
	approx(t, z.F32[1], 8*inv, 1e-5, "z1")
}

func TestSoftmax(t *testing.T) {
	x := &Tensor{Dims: []int{3}, F32: []float32{1, 2, 3}}
	Softmax(x)
	approx(t, x.F32[0], 0.09003, 1e-4, "p0")
	approx(t, x.F32[1], 0.24473, 1e-4, "p1")
	approx(t, x.F32[2], 0.66524, 1e-4, "p2")
	var sum float32
	for _, v := range x.F32 {
		sum += v
	}
	approx(t, sum, 1, 1e-5, "sum")
}

func TestCrossEntropy(t *testing.T) {
	logits := &Tensor{Dims: []int{3, 1}, F32: []float32{0, 0, 0}}
	loss, err := CrossEntropy(logits, []int{0}, -100)
	if err != nil {
		t.Fatal(err)
	}
	approx(t, loss, float32(math.Log(3)), 1e-5, "loss")
}

func TestCrossEntropyIgnore(t *testing.T) {
	logits := &Tensor{Dims: []int{2, 2}, F32: []float32{10, 0, 0, 10}}
	loss, err := CrossEntropy(logits, []int{0, -100}, -100)
	if err != nil {
		t.Fatal(err)
	}
	if loss < 0 || loss > 1e-3 {
		t.Errorf("loss = %v, want ~0", loss)
	}
}

func TestElementwise(t *testing.T) {
	a := &Tensor{Dims: []int{2}, F32: []float32{1, 2}}
	b := &Tensor{Dims: []int{2}, F32: []float32{3, 4}}
	s, _ := Add(a, b)
	if s.F32[0] != 4 || s.F32[1] != 6 {
		t.Errorf("add = %v", s.F32)
	}
	m, _ := Mul(a, b)
	if m.F32[0] != 3 || m.F32[1] != 8 {
		t.Errorf("mul = %v", m.F32)
	}
	sc := Scale(a, 2)
	if sc.F32[0] != 2 || sc.F32[1] != 4 {
		t.Errorf("scale = %v", sc.F32)
	}
}

func TestSilu(t *testing.T) {
	approx(t, Silu(0), 0, 1e-6, "silu0")
	approx(t, Silu(1), 0.7310586, 1e-6, "silu1")
}
