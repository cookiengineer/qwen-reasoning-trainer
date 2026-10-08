package vulkan_test

import (
	"math"
	"strings"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/cpu"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
)

func up(t *testing.T, b compute.Backend, dims []int, vals []float32) compute.Buffer {
	t.Helper()
	buf, err := b.Upload(&compute.Tensor{Dims: dims, F32: vals})
	if err != nil {
		t.Fatal(err)
	}
	return buf
}

func down(t *testing.T, b compute.Backend, buf compute.Buffer) []float32 {
	t.Helper()
	tens, err := b.Download(buf)
	if err != nil {
		t.Fatal(err)
	}
	return tens.F32
}

func compare(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", name, len(got), len(want))
	}
	for i := range want {
		tol := 1e-4 + 1e-3*float32(math.Abs(float64(want[i])))
		if float32(math.Abs(float64(got[i]-want[i]))) > tol {
			t.Fatalf("%s[%d]: got %v want %v", name, i, got[i], want[i])
		}
	}
}

func newBackends(t *testing.T) (compute.Backend, compute.Backend) {
	t.Helper()
	c := cpu.New()
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	if !strings.Contains(v.Capabilities().Name, "vulkan") {
		t.Fatalf("unexpected capabilities: %+v", v.Capabilities())
	}
	t.Cleanup(func() { v.Close() })
	return c, v
}

func TestVulkanUnaryBinary(t *testing.T) {
	c, v := newBackends(t)
	in := []float32{-2, -1, 0, 1, 2, 3, 0.5, -0.5}
	for _, op := range []compute.UnaryOp{compute.UnarySilu, compute.UnarySigmoid, compute.UnaryNeg, compute.UnaryExp, compute.UnarySqr} {
		cf := func(b compute.Backend) (compute.Buffer, error) { return b.Unary(op, up(t, b, []int{len(in)}, in)) }
		co, _ := cf(c)
		vo, _ := cf(v)
		compare(t, "unary:"+op.String(), down(t, v, vo), down(t, c, co))
	}
	other := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	for _, op := range []compute.BinaryOp{compute.BinaryAdd, compute.BinarySub, compute.BinaryMul, compute.BinaryDiv} {
		cf := func(b compute.Backend) (compute.Buffer, error) {
			return b.Binary(op, up(t, b, []int{len(in)}, in), up(t, b, []int{len(other)}, other))
		}
		co, _ := cf(c)
		vo, _ := cf(v)
		compare(t, "binary:"+op.String(), down(t, v, vo), down(t, c, co))
	}
	cf := func(b compute.Backend) (compute.Buffer, error) { return b.Scale(up(t, b, []int{len(in)}, in), 2.5) }
	co, _ := cf(c)
	vo, _ := cf(v)
	compare(t, "scale", down(t, v, vo), down(t, c, co))
}

func TestVulkanMatMul(t *testing.T) {
	c, v := newBackends(t)
	a := []float32{1, 2, 3, 4}
	m := []float32{5, 6, 7, 8}
	cf := func(b compute.Backend) (compute.Buffer, error) {
		return b.MatMul(up(t, b, []int{2, 2}, a), up(t, b, []int{2, 2}, m))
	}
	co, _ := cf(c)
	vo, _ := cf(v)
	compare(t, "matmul", down(t, v, vo), down(t, c, co))

	// A non-square case [K=3,N=2] x [K=3,M=4].
	aa := make([]float32, 6)
	for i := range aa {
		aa[i] = float32(i) * 0.5
	}
	mm := make([]float32, 12)
	for i := range mm {
		mm[i] = float32(12-i) * 0.25
	}
	cf2 := func(b compute.Backend) (compute.Buffer, error) {
		return b.MatMul(up(t, b, []int{3, 2}, aa), up(t, b, []int{3, 4}, mm))
	}
	co2, _ := cf2(c)
	vo2, _ := cf2(v)
	compare(t, "matmul2", down(t, v, vo2), down(t, c, co2))
}

func TestVulkanNormSoftmax(t *testing.T) {
	c, v := newBackends(t)
	x := []float32{3, 4, 0, 1, -1, 2}
	w := []float32{1, 2, 3}
	cf := func(b compute.Backend) (compute.Buffer, error) {
		return b.RMSNorm(up(t, b, []int{3, 2}, x), up(t, b, []int{3}, w), 1e-6)
	}
	co, _ := cf(c)
	vo, _ := cf(v)
	compare(t, "rmsnorm", down(t, v, vo), down(t, c, co))

	sm := []float32{1, 2, 3, 0, 0, 0}
	cf2 := func(b compute.Backend) (compute.Buffer, error) { return b.Softmax(up(t, b, []int{3, 2}, sm)) }
	co2, _ := cf2(c)
	vo2, _ := cf2(v)
	compare(t, "softmax", down(t, v, vo2), down(t, c, co2))
}

func TestVulkanGetRowsRoPE(t *testing.T) {
	c, v := newBackends(t)
	table := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	cf := func(b compute.Backend) (compute.Buffer, error) {
		return b.GetRows(up(t, b, []int{3, 3}, table), []int32{2, 0})
	}
	co, _ := cf(c)
	vo, _ := cf(v)
	compare(t, "getrows", down(t, v, vo), down(t, c, co))

	rx := []float32{1, 0, 0, 1, 2, -1, -2, 1}
	cf2 := func(b compute.Backend) (compute.Buffer, error) {
		return b.RoPE(up(t, b, []int{4, 1, 2}, rx), []int32{0, 3}, 10000, 4)
	}
	co2, _ := cf2(c)
	vo2, _ := cf2(v)
	compare(t, "rope", down(t, v, vo2), down(t, c, co2))
}

func TestVulkanAttention(t *testing.T) {
	c, v := newBackends(t)
	q := []float32{1, 1}
	k := []float32{1, 0}
	vv := []float32{10, 20}
	cf := func(b compute.Backend) (compute.Buffer, error) {
		return b.Attention(up(t, b, []int{1, 1, 2}, q), up(t, b, []int{1, 1, 2}, k), up(t, b, []int{1, 1, 2}, vv), 1, 1, 1, true)
	}
	co, _ := cf(c)
	vo, _ := cf(v)
	compare(t, "attention", down(t, v, vo), down(t, c, co))
}

func TestVulkanGatedDeltaNetFallback(t *testing.T) {
	c, v := newBackends(t)
	q := []float32{1, 0}
	k := []float32{1, 0}
	vv := []float32{1, 1}
	g := []float32{0}
	beta := []float32{1}
	state := []float32{0, 0, 0, 0}
	cf := func(b compute.Backend) (compute.Buffer, error) {
		out, _, err := b.GatedDeltaNet(
			up(t, b, []int{2, 1, 1}, q), up(t, b, []int{2, 1, 1}, k), up(t, b, []int{2, 1, 1}, vv),
			up(t, b, []int{1, 1, 1}, g), up(t, b, []int{1, 1, 1}, beta), up(t, b, []int{2, 2, 1}, state))
		return out, err
	}
	co, _ := cf(c)
	vo, _ := cf(v)
	compare(t, "gdn", down(t, v, vo), down(t, c, co))
}
