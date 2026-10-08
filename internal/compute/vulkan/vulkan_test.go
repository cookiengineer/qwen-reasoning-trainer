package vulkan_test

import (
	"math"
	"strings"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/cpu"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
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

// buildRaw creates pseudo-random storage bytes for a quantized type, with the
// scale fields overwritten by finite values so the dequant result is finite.
func buildRaw(typ quant.Type, blocks int, seed uint32, scaleOffsets []int) []byte {
	ts := typ.TypeSize()
	raw := make([]byte, blocks*ts)
	x := seed
	for i := range raw {
		x = x*1664525 + 1013904223
		raw[i] = byte(x >> 16)
	}
	for b := 0; b < blocks; b++ {
		for k, off := range scaleOffsets {
			var h float32 = 1.0
			if k%2 == 1 {
				h = 0.1
			}
			u := quant.F32ToF16(h)
			raw[b*ts+off] = byte(u)
			raw[b*ts+off+1] = byte(u >> 8)
		}
	}
	return raw
}

func dequantShaderCase(typ quant.Type) (scaleOffsets []int, ok bool) {
	switch typ {
	case quant.TypeQ8_0, quant.TypeIQ4_XS, quant.TypeIQ4_NL, quant.TypeIQ3_S:
		return []int{0}, true
	case quant.TypeQ4_K, quant.TypeQ5_K:
		return []int{0, 2}, true
	case quant.TypeQ6_K:
		return []int{208}, true
	case quant.TypeQ3_K:
		return []int{108}, true
	}
	return nil, false
}

func TestVulkanDequant(t *testing.T) {
	_, v := newBackends(t)
	types := []quant.Type{
		quant.TypeQ8_0, quant.TypeQ4_K, quant.TypeQ5_K, quant.TypeQ6_K,
		quant.TypeQ3_K, quant.TypeIQ4_XS, quant.TypeIQ4_NL, quant.TypeIQ3_S,
	}
	for _, typ := range types {
		offs, ok := dequantShaderCase(typ)
		if !ok {
			continue
		}
		blocks := 3
		raw := buildRaw(typ, blocks, uint32(typ)+1, offs)
		n := int64(blocks * typ.BlockSize())
		want, err := quant.Dequant(typ, raw, n)
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		wb, err := v.UploadWeight(typ, raw, []int{int(n)})
		if err != nil {
			t.Fatalf("%s: upload: %v", typ, err)
		}
		db, err := v.DequantWeight(wb)
		if err != nil {
			t.Fatalf("%s: dequant: %v", typ, err)
		}
		got := down(t, v, db)
		v.Free(wb)
		v.Free(db)
		if len(got) != len(want) {
			t.Fatalf("%s: len %d != %d", typ, len(got), len(want))
		}
		for i := range want {
			tol := 1e-3 + 1e-3*float32(math.Abs(float64(want[i])))
			if float32(math.Abs(float64(got[i]-want[i]))) > tol {
				t.Fatalf("%s[%d]: got %v want %v", typ, i, got[i], want[i])
			}
		}
	}
}

func TestVulkanMatMulWeight(t *testing.T) {
	c, v := newBackends(t)
	k, n, m := 32, 8, 4
	data := make([]float32, k*n)
	for i := range data {
		data[i] = float32(math.Sin(float64(i))) * 2
	}
	raw, err := quant.Quantize(quant.TypeQ8_0, data)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, k*m)
	for i := range x {
		x[i] = float32(math.Cos(float64(i)))
	}

	wb, err := v.UploadWeight(quant.TypeQ8_0, raw, []int{k, n})
	if err != nil {
		t.Fatal(err)
	}
	xb := up(t, v, []int{k, m}, x)
	got, err := v.MatMulWeight(wb, xb)
	if err != nil {
		t.Fatal(err)
	}

	// CPU reference: dequantize then matmul.
	dq, _ := quant.Dequant(quant.TypeQ8_0, raw, int64(k*n))
	wt := &compute.Tensor{Dims: []int{k, n}, F32: dq}
	wantT, err := compute.MatMul(wt, &compute.Tensor{Dims: []int{k, m}, F32: x})
	if err != nil {
		t.Fatal(err)
	}
	_ = c
	compare(t, "matmulweight", down(t, v, got), wantT.F32)
}

func TestVulkanMatMulWeightBlocked(t *testing.T) {
	if !vulkan.Available() {
		t.Skip("vulkan unavailable")
	}
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()
	// Force a tiny scratch so the weight is processed in many row blocks.
	v.SetMaxScratchFloats(64)

	k, n, m := 32, 200, 3
	data := make([]float32, k*n)
	for i := range data {
		data[i] = float32(math.Sin(float64(i)*0.3)) * 2
	}
	raw, err := quant.Quantize(quant.TypeQ8_0, data)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, k*m)
	for i := range x {
		x[i] = float32(math.Cos(float64(i) * 0.7))
	}

	wb, err := v.UploadWeight(quant.TypeQ8_0, raw, []int{k, n})
	if err != nil {
		t.Fatal(err)
	}
	xb := up(t, v, []int{k, m}, x)
	got, err := v.MatMulWeight(wb, xb)
	if err != nil {
		t.Fatal(err)
	}

	dq, _ := quant.Dequant(quant.TypeQ8_0, raw, int64(k*n))
	wantT, err := compute.MatMul(&compute.Tensor{Dims: []int{k, n}, F32: dq}, &compute.Tensor{Dims: []int{k, m}, F32: x})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "matmulweight-blocked", down(t, v, got), wantT.F32)
}
