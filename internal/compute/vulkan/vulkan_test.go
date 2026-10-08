package vulkan_test

import (
	"math"
	"strings"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
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

func TestVulkanGemvFused(t *testing.T) {
	_, v := newBackends(t)
	types := []quant.Type{quant.TypeQ4_K, quant.TypeQ5_K, quant.TypeQ6_K, quant.TypeIQ4_XS}
	for _, typ := range types {
		offs, _ := dequantShaderCase(typ)
		k, n := 256, 8
		raw := buildRaw(typ, n, uint32(typ)+17, offs)
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.11)) * 0.5
		}
		// Expected: dequantize the weight then matmul with x.
		dq, err := quant.Dequant(typ, raw, int64(k*n))
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		wantT, err := compute.MatMul(&compute.Tensor{Dims: []int{k, n}, F32: dq}, &compute.Tensor{Dims: []int{k, 1}, F32: x})
		if err != nil {
			t.Fatal(err)
		}
		wb, err := v.UploadWeight(typ, raw, []int{k, n})
		if err != nil {
			t.Fatal(err)
		}
		xb := up(t, v, []int{k, 1}, x)
		got, err := v.MatMulWeight(wb, xb)
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		vgot := down(t, v, got)
		if len(vgot) != n {
			t.Fatalf("%s: len %d", typ, len(vgot))
		}
		for i := 0; i < n; i++ {
			tol := 1e-3 + 1e-3*float32(math.Abs(float64(wantT.F32[i])))
			if float32(math.Abs(float64(vgot[i]-wantT.F32[i]))) > tol {
				t.Fatalf("%s[%d]: got %v want %v", typ, i, vgot[i], wantT.F32[i])
			}
		}
	}
}

func TestVulkanStagingRoundTrip(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()
	// Force non-host-visible device-local allocation when the device exposes
	// such a type; otherwise this still validates the host-visible path.
	v.SetForceStaging(true)

	dims := []int{8, 5}
	vals := make([]float32, 40)
	for i := range vals {
		vals[i] = float32(i) * 0.25
	}
	buf, err := v.Upload(&compute.Tensor{Dims: dims, F32: append([]float32(nil), vals...)})
	if err != nil {
		t.Fatal(err)
	}
	sum, err := v.Binary(compute.BinaryAdd, buf, buf)
	if err != nil {
		t.Fatal(err)
	}
	got, err := v.Download(sum)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float32, len(vals))
	for i := range vals {
		want[i] = vals[i] * 2
	}
	compare(t, "staging-add", got.F32, want)

	// Weight upload through staging plus a dequant dispatch.
	K, N := 32, 4
	wf := make([]float32, K*N)
	for i := range wf {
		wf[i] = float32(i%7) - 3
	}
	raw, err := quant.Quantize(quant.TypeQ8_0, wf)
	if err != nil {
		t.Fatal(err)
	}
	wb, err := v.UploadWeight(quant.TypeQ8_0, raw, []int{K, N})
	if err != nil {
		t.Fatal(err)
	}
	dq, err := v.DequantWeight(wb)
	if err != nil {
		t.Fatal(err)
	}
	gotW, err := v.Download(dq)
	if err != nil {
		t.Fatal(err)
	}
	refW, err := quant.Dequant(quant.TypeQ8_0, raw, int64(K*N))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "staging-dequant", gotW.F32, refW)
}

// hostTransposeReference computes dX = W^T dY on the host from a dequantized
// weight, independent of both backends.
func hostTransposeReference(wf []float32, dy *compute.Tensor, k, n, m int) []float32 {
	out := make([]float32, k*m)
	for ni := 0; ni < n; ni++ {
		for mi := 0; mi < m; mi++ {
			g := dy.F32[ni+mi*n]
			if g == 0 {
				continue
			}
			base := ni * k
			for ki := 0; ki < k; ki++ {
				out[ki+mi*k] += wf[base+ki] * g
			}
		}
	}
	return out
}

func TestVulkanMatMulWeightTranspose(t *testing.T) {
	c, v := newBackends(t)
	const K, N, M = 256, 8, 3
	wf := make([]float32, K*N)
	for i := range wf {
		wf[i] = float32((i%13)-6) * 0.1
	}
	dy := compute.NewF32(N, M)
	for i := range dy.F32 {
		dy.F32[i] = float32((i%7)-3) * 0.2
	}

	types := []quant.Type{
		quant.TypeF32, quant.TypeF16, quant.TypeQ8_0,
		quant.TypeQ4_K, quant.TypeQ5_K, quant.TypeQ6_K, quant.TypeQ3_K,
		quant.TypeIQ4_XS, quant.TypeIQ4_NL, quant.TypeIQ3_S,
	}
	for _, typ := range types {
		raw, err := quant.Quantize(typ, wf)
		if err != nil {
			t.Fatalf("%s: quantize: %v", typ, err)
		}
		cw, err := c.UploadWeight(typ, raw, []int{K, N})
		if err != nil {
			t.Fatalf("%s: cpu upload: %v", typ, err)
		}
		vw, err := v.UploadWeight(typ, raw, []int{K, N})
		if err != nil {
			t.Fatalf("%s: vk upload: %v", typ, err)
		}
		cdy := up(t, c, []int{N, M}, append([]float32(nil), dy.F32...))
		vdy := up(t, v, []int{N, M}, append([]float32(nil), dy.F32...))

		cout, err := c.MatMulWeightTranspose(cw, cdy)
		if err != nil {
			t.Fatalf("%s: cpu: %v", typ, err)
		}
		vout, err := v.MatMulWeightTranspose(vw, vdy)
		if err != nil {
			t.Fatalf("%s: vk: %v", typ, err)
		}
		got := down(t, v, vout)
		want := down(t, c, cout)
		compare(t, "dX "+typ.String(), got, want)

		deq, err := quant.Dequant(typ, raw, int64(K*N))
		if err != nil {
			t.Fatalf("%s: dequant: %v", typ, err)
		}
		ref := hostTransposeReference(deq, dy, K, N, M)
		compare(t, "dX-host "+typ.String(), got, ref)
	}
}

func TestVulkanMatMulWeightTransposeBlocked(t *testing.T) {
	const K, N, M = 256, 16, 4
	c := cpu.New()
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()
	// Force small row blocks so the accumulation path runs several times.
	v.SetMaxScratchFloats(K * 2)

	wf := make([]float32, K*N)
	for i := range wf {
		wf[i] = float32((i%11)-5) * 0.13
	}
	dy := compute.NewF32(N, M)
	for i := range dy.F32 {
		dy.F32[i] = float32((i%9)-4) * 0.21
	}
	for _, typ := range []quant.Type{quant.TypeQ8_0, quant.TypeQ4_K, quant.TypeIQ4_XS} {
		raw, err := quant.Quantize(typ, wf)
		if err != nil {
			t.Fatal(err)
		}
		cw, _ := c.UploadWeight(typ, raw, []int{K, N})
		vw, _ := v.UploadWeight(typ, raw, []int{K, N})
		cdy := up(t, c, []int{N, M}, append([]float32(nil), dy.F32...))
		vdy := up(t, v, []int{N, M}, append([]float32(nil), dy.F32...))
		cout, err := c.MatMulWeightTranspose(cw, cdy)
		if err != nil {
			t.Fatal(err)
		}
		vout, err := v.MatMulWeightTranspose(vw, vdy)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "blocked dX "+typ.String(), down(t, v, vout), down(t, c, cout))
	}
}

func vecDot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func TestVulkanSiluBack(t *testing.T) {
	c, v := newBackends(t)
	const n = 37
	x := make([]float32, n)
	dout := make([]float32, n)
	for i := range x {
		x[i] = float32((i%9)-4) * 0.4
		dout[i] = float32((i%5)-2) * 0.3
	}
	cv, err := c.SiluBack(up(t, c, []int{n}, append([]float32(nil), x...)), up(t, c, []int{n}, append([]float32(nil), dout...)))
	if err != nil {
		t.Fatal(err)
	}
	vv, err := v.SiluBack(up(t, v, []int{n}, append([]float32(nil), x...)), up(t, v, []int{n}, append([]float32(nil), dout...)))
	if err != nil {
		t.Fatal(err)
	}
	got := down(t, v, vv)
	compare(t, "silu_back", got, down(t, c, cv))

	// GPU finite differences: f(x) = sum(silu(x) * dOut).
	const eps = 1e-2
	for j := 0; j < n; j++ {
		xp := append([]float32(nil), x...)
		xp[j] += eps
		xm := append([]float32(nil), x...)
		xm[j] -= eps
		fp, _ := v.Unary(compute.UnarySilu, up(t, v, []int{n}, xp))
		fm, _ := v.Unary(compute.UnarySilu, up(t, v, []int{n}, xm))
		num := (vecDot(down(t, v, fp), dout) - vecDot(down(t, v, fm), dout)) / (2 * eps)
		if d := math.Abs(num - float64(got[j])); d > 1e-2+1e-2*math.Abs(num) {
			t.Fatalf("silu_back FD[%d]: numeric %g vs kernel %g", j, num, got[j])
		}
	}
}

func TestVulkanRMSNormBack(t *testing.T) {
	c, v := newBackends(t)
	const D, rows = 8, 3
	n := D * rows
	x := make([]float32, n)
	w := make([]float32, D)
	dout := make([]float32, n)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.3
	}
	for i := range w {
		w[i] = 0.5 + 0.1*float32(i)
	}
	for i := range dout {
		dout[i] = float32((i%6)-2) * 0.25
	}
	const eps = 1e-6

	cv, err := c.RMSNormBack(up(t, c, []int{D, rows}, append([]float32(nil), x...)), up(t, c, []int{D}, w), up(t, c, []int{D, rows}, append([]float32(nil), dout...)), eps)
	if err != nil {
		t.Fatal(err)
	}
	vv, err := v.RMSNormBack(up(t, v, []int{D, rows}, append([]float32(nil), x...)), up(t, v, []int{D}, w), up(t, v, []int{D, rows}, append([]float32(nil), dout...)), eps)
	if err != nil {
		t.Fatal(err)
	}
	got := down(t, v, vv)
	compare(t, "rms_norm_back", got, down(t, c, cv))

	// nil weight path parity.
	cv2, _ := c.RMSNormBack(up(t, c, []int{D, rows}, append([]float32(nil), x...)), nil, up(t, c, []int{D, rows}, append([]float32(nil), dout...)), eps)
	vv2, _ := v.RMSNormBack(up(t, v, []int{D, rows}, append([]float32(nil), x...)), nil, up(t, v, []int{D, rows}, append([]float32(nil), dout...)), eps)
	compare(t, "rms_norm_back_nilw", down(t, v, vv2), down(t, c, cv2))

	// GPU finite differences on the weighted case.
	step := float32(1e-2)
	for j := 0; j < n; j++ {
		xp := append([]float32(nil), x...)
		xp[j] += step
		xm := append([]float32(nil), x...)
		xm[j] -= step
		op, _ := v.RMSNorm(up(t, v, []int{D, rows}, xp), up(t, v, []int{D}, w), eps)
		om, _ := v.RMSNorm(up(t, v, []int{D, rows}, xm), up(t, v, []int{D}, w), eps)
		num := (vecDot(down(t, v, op), dout) - vecDot(down(t, v, om), dout)) / float64(2*step)
		if d := math.Abs(num - float64(got[j])); d > 2e-2+2e-2*math.Abs(num) {
			t.Fatalf("rms_norm_back FD[%d]: numeric %g vs kernel %g", j, num, got[j])
		}
	}
}

func TestVulkanL2NormBack(t *testing.T) {
	c, v := newBackends(t)
	const D, rows = 6, 3
	n := D * rows
	x := make([]float32, n)
	dout := make([]float32, n)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.5
	}
	for i := range dout {
		dout[i] = float32((i%5)-2) * 0.4
	}
	const eps = 1e-6

	// Forward parity.
	fc, err := c.L2Norm(up(t, c, []int{D, rows}, append([]float32(nil), x...)), eps)
	if err != nil {
		t.Fatal(err)
	}
	fv, err := v.L2Norm(up(t, v, []int{D, rows}, append([]float32(nil), x...)), eps)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "l2_norm", down(t, v, fv), down(t, c, fc))

	// Backward parity.
	bc, _ := c.L2NormBack(up(t, c, []int{D, rows}, append([]float32(nil), x...)), up(t, c, []int{D, rows}, append([]float32(nil), dout...)), eps)
	bv, _ := v.L2NormBack(up(t, v, []int{D, rows}, append([]float32(nil), x...)), up(t, v, []int{D, rows}, append([]float32(nil), dout...)), eps)
	got := down(t, v, bv)
	compare(t, "l2_norm_back", got, down(t, c, bc))

	// GPU finite differences.
	step := float32(1e-2)
	for j := 0; j < n; j++ {
		xp := append([]float32(nil), x...)
		xp[j] += step
		xm := append([]float32(nil), x...)
		xm[j] -= step
		op, _ := v.L2Norm(up(t, v, []int{D, rows}, xp), eps)
		om, _ := v.L2Norm(up(t, v, []int{D, rows}, xm), eps)
		num := (vecDot(down(t, v, op), dout) - vecDot(down(t, v, om), dout)) / float64(2*step)
		if d := math.Abs(num - float64(got[j])); d > 2e-2+2e-2*math.Abs(num) {
			t.Fatalf("l2_norm_back FD[%d]: numeric %g vs kernel %g", j, num, got[j])
		}
	}
}

func TestVulkanSSMConvBack(t *testing.T) {
	c, v := newBackends(t)
	const dConv, dInner, nT = 3, 4, 5
	ncs := dConv - 1 + nT
	sx := make([]float32, ncs*dInner)
	cc := make([]float32, dConv*dInner)
	for i := range sx {
		sx[i] = float32((i%7)-3) * 0.3
	}
	for i := range cc {
		cc[i] = float32((i%5)-2) * 0.4
	}
	dout := make([]float32, dInner*nT)
	for i := range dout {
		dout[i] = float32((i%6)-2) * 0.5
	}
	sxDims := []int{ncs, dInner, 1}
	cDims := []int{dConv, dInner}
	outDims := []int{dInner, nT, 1}

	fc, err := c.SSMConv(up(t, c, sxDims, append([]float32(nil), sx...)), up(t, c, cDims, cc))
	if err != nil {
		t.Fatal(err)
	}
	fv, err := v.SSMConv(up(t, v, sxDims, append([]float32(nil), sx...)), up(t, v, cDims, cc))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "ssm_conv", down(t, v, fv), down(t, c, fc))

	bsxC, bcC, err := c.SSMConvBack(up(t, c, sxDims, append([]float32(nil), sx...)), up(t, c, cDims, cc), up(t, c, outDims, dout))
	if err != nil {
		t.Fatal(err)
	}
	bsxV, bcV, err := v.SSMConvBack(up(t, v, sxDims, append([]float32(nil), sx...)), up(t, v, cDims, cc), up(t, v, outDims, dout))
	if err != nil {
		t.Fatal(err)
	}
	dsx := down(t, v, bsxV)
	dc := down(t, v, bcV)
	compare(t, "ssm_conv_back_sx", dsx, down(t, c, bsxC))
	compare(t, "ssm_conv_back_c", dc, down(t, c, bcC))

	// GPU finite differences.
	step := float32(1e-2)
	for j := range sx {
		xp := append([]float32(nil), sx...)
		xp[j] += step
		xm := append([]float32(nil), sx...)
		xm[j] -= step
		op, _ := v.SSMConv(up(t, v, sxDims, xp), up(t, v, cDims, cc))
		om, _ := v.SSMConv(up(t, v, sxDims, xm), up(t, v, cDims, cc))
		num := (vecDot(down(t, v, op), dout) - vecDot(down(t, v, om), dout)) / float64(2*step)
		if dd := math.Abs(num - float64(dsx[j])); dd > 2e-2+2e-2*math.Abs(num) {
			t.Fatalf("ssm_conv_back_sx FD[%d]: numeric %g vs kernel %g", j, num, dsx[j])
		}
	}
	for j := range cc {
		cp := append([]float32(nil), cc...)
		cp[j] += step
		cm := append([]float32(nil), cc...)
		cm[j] -= step
		op, _ := v.SSMConv(up(t, v, sxDims, sx), up(t, v, cDims, cp))
		om, _ := v.SSMConv(up(t, v, sxDims, sx), up(t, v, cDims, cm))
		num := (vecDot(down(t, v, op), dout) - vecDot(down(t, v, om), dout)) / float64(2*step)
		if dd := math.Abs(num - float64(dc[j])); dd > 2e-2+2e-2*math.Abs(num) {
			t.Fatalf("ssm_conv_back_c FD[%d]: numeric %g vs kernel %g", j, num, dc[j])
		}
	}
}

func TestVulkanGatedDeltaNetForward(t *testing.T) {
	c, v := newBackends(t)
	const sv, h, nTok = 3, 2, 4
	q := make([]float32, sv*h*nTok)
	k := make([]float32, sv*h*nTok)
	vv := make([]float32, sv*h*nTok)
	for i := range q {
		q[i] = float32((i%7)-3) * 0.3
		k[i] = float32((i%5)-2) * 0.25
		vv[i] = float32((i%9)-4) * 0.2
	}
	beta := make([]float32, h*nTok)
	for i := range beta {
		beta[i] = 0.3 + 0.1*float32(i%4)
	}
	state := make([]float32, sv*sv*h)
	for i := range state {
		state[i] = float32((i%11)-5) * 0.05
	}

	run := func(be compute.Backend, gate []float32, gdims []int) ([]float32, []float32) {
		qb := up(t, be, []int{sv, h, nTok}, append([]float32(nil), q...))
		kb := up(t, be, []int{sv, h, nTok}, append([]float32(nil), k...))
		vb := up(t, be, []int{sv, h, nTok}, append([]float32(nil), vv...))
		gb := up(t, be, gdims, append([]float32(nil), gate...))
		beb := up(t, be, []int{1, h, nTok}, beta)
		sb := up(t, be, []int{sv, sv, h}, append([]float32(nil), state...))
		out, ns, err := be.GatedDeltaNet(qb, kb, vb, gb, beb, sb)
		if err != nil {
			t.Fatal(err)
		}
		return down(t, be, out), down(t, be, ns)
	}

	// Scalar gate.
	scalar := make([]float32, h*nTok)
	for i := range scalar {
		scalar[i] = float32(i%3) * 0.2
	}
	oc, nc := run(c, scalar, []int{1, h, nTok})
	ov, nv := run(v, scalar, []int{1, h, nTok})
	compare(t, "gdn-out-scalar", ov, oc)
	compare(t, "gdn-state-scalar", nv, nc)

	// Per-element (KDA) gate.
	kda := make([]float32, sv*h*nTok)
	for i := range kda {
		kda[i] = float32((i%4)-1) * 0.15
	}
	oc2, nc2 := run(c, kda, []int{sv, h, nTok})
	ov2, nv2 := run(v, kda, []int{sv, h, nTok})
	compare(t, "gdn-out-kda", ov2, oc2)
	compare(t, "gdn-state-kda", nv2, nc2)
}

func TestVulkanSoftmaxBack(t *testing.T) {
	c, v := newBackends(t)
	const row, rows = 5, 3
	n := row * rows
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.6
	}
	dout := make([]float32, n)
	for i := range dout {
		dout[i] = float32((i%5)-2) * 0.4
	}
	// Forward softmax on both backends to get `out`.
	smC, _ := c.Softmax(up(t, c, []int{row, rows}, append([]float32(nil), x...)))
	smV, _ := v.Softmax(up(t, v, []int{row, rows}, append([]float32(nil), x...)))
	bc, err := c.SoftmaxBack(smC, up(t, c, []int{row, rows}, append([]float32(nil), dout...)))
	if err != nil {
		t.Fatal(err)
	}
	bv, err := v.SoftmaxBack(smV, up(t, v, []int{row, rows}, append([]float32(nil), dout...)))
	if err != nil {
		t.Fatal(err)
	}
	got := down(t, v, bv)
	compare(t, "soft_max_back", got, down(t, c, bc))

	// GPU finite differences: f(x) = sum(softmax(x) * dOut).
	step := float32(1e-2)
	for j := 0; j < n; j++ {
		xp := append([]float32(nil), x...)
		xp[j] += step
		xm := append([]float32(nil), x...)
		xm[j] -= step
		op, _ := v.Softmax(up(t, v, []int{row, rows}, xp))
		om, _ := v.Softmax(up(t, v, []int{row, rows}, xm))
		num := (vecDot(down(t, v, op), dout) - vecDot(down(t, v, om), dout)) / float64(2*step)
		if d := math.Abs(num - float64(got[j])); d > 2e-2+2e-2*math.Abs(num) {
			t.Fatalf("soft_max_back FD[%d]: numeric %g vs kernel %g", j, num, got[j])
		}
	}
}

func TestVulkanMatMulWeightGrad(t *testing.T) {
	c, v := newBackends(t)
	const In, Out, m = 5, 4, 3
	x := make([]float32, In*m)
	w := make([]float32, In*Out)
	dout := make([]float32, Out*m)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.3
	}
	for i := range w {
		w[i] = float32((i%5)-2) * 0.2
	}
	for i := range dout {
		dout[i] = float32((i%6)-2) * 0.4
	}
	cv, err := c.MatMulWeightGrad(up(t, c, []int{In, m}, append([]float32(nil), x...)), up(t, c, []int{Out, m}, append([]float32(nil), dout...)))
	if err != nil {
		t.Fatal(err)
	}
	vv, err := v.MatMulWeightGrad(up(t, v, []int{In, m}, append([]float32(nil), x...)), up(t, v, []int{Out, m}, append([]float32(nil), dout...)))
	if err != nil {
		t.Fatal(err)
	}
	got := down(t, v, vv)
	compare(t, "matmul_wg", got, down(t, c, cv))

	// GPU finite differences: f(W) = sum(W x * dOut), W [In,Out].
	step := float32(1e-2)
	for j := range w {
		wp := append([]float32(nil), w...)
		wp[j] += step
		wm := append([]float32(nil), w...)
		wm[j] -= step
		yp, _ := v.MatMul(up(t, v, []int{In, Out}, wp), up(t, v, []int{In, m}, x))
		ym, _ := v.MatMul(up(t, v, []int{In, Out}, wm), up(t, v, []int{In, m}, x))
		num := (vecDot(down(t, v, yp), dout) - vecDot(down(t, v, ym), dout)) / float64(2*step)
		if d := math.Abs(num - float64(got[j])); d > 2e-2+2e-2*math.Abs(num) {
			t.Fatalf("matmul_wg FD[%d]: numeric %g vs kernel %g", j, num, got[j])
		}
	}
}

func TestVulkanAttentionBackward(t *testing.T) {
	c, v := newBackends(t)
	scale := float32(1.0 / math.Sqrt(3))

	run := func(be compute.Backend, hd, nHead, nHeadKV, nQ, nKV int, causal bool) (dq, dk, dv []float32) {
		q := autograd.Random(50, 1.0, hd, nHead, nQ)
		k := autograd.Random(51, 1.0, hd, nHeadKV, nKV)
		vv := autograd.Random(52, 1.0, hd, nHeadKV, nKV)
		dO := autograd.Random(53, 1.0, hd, nHead, nQ)
		a, b, d, err := be.AttentionBackward(
			up(t, be, q.Dims, q.F32), up(t, be, k.Dims, k.F32),
			up(t, be, vv.Dims, vv.F32), up(t, be, dO.Dims, dO.F32),
			nHead, nHeadKV, scale, causal)
		if err != nil {
			t.Fatal(err)
		}
		return down(t, be, a), down(t, be, b), down(t, be, d)
	}

	// Causal GQA.
	cq, ck, cv := run(c, 3, 2, 1, 3, 3, true)
	vq, vk, vv := run(v, 3, 2, 1, 3, 3, true)
	compare(t, "attn_back dQ causal", vq, cq)
	compare(t, "attn_back dK causal", vk, ck)
	compare(t, "attn_back dV causal", vv, cv)

	// Non-causal GQA (nQ != nKV).
	cq2, ck2, cv2 := run(c, 2, 2, 1, 2, 3, false)
	vq2, vk2, vv2 := run(v, 2, 2, 1, 2, 3, false)
	compare(t, "attn_back dQ noncausal", vq2, cq2)
	compare(t, "attn_back dK noncausal", vk2, ck2)
	compare(t, "attn_back dV noncausal", vv2, cv2)

	// Independent oracle: autograd.AttentionBackward on the same causal inputs
	// (seeds match the first run above).
	q := autograd.Random(50, 1.0, 3, 2, 3)
	k := autograd.Random(51, 1.0, 3, 1, 3)
	vvv := autograd.Random(52, 1.0, 3, 1, 3)
	dO := autograd.Random(53, 1.0, 3, 2, 3)
	hdq, hdk, hdv := autograd.AttentionBackward(q, k, vvv, dO, 2, 1, scale, true)
	if maxAbsDiffLocal(vq, hdq.F32) > 1e-4 || maxAbsDiffLocal(vk, hdk.F32) > 1e-4 || maxAbsDiffLocal(vv, hdv.F32) > 1e-4 {
		t.Fatalf("device attention backward disagrees with autograd oracle: dQ %g dK %g dV %g",
			maxAbsDiffLocal(vq, hdq.F32), maxAbsDiffLocal(vk, hdk.F32), maxAbsDiffLocal(vv, hdv.F32))
	}
}

func maxAbsDiffLocal(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

func TestVulkanGatedDeltaNetBackward(t *testing.T) {
	c, v := newBackends(t)
	if !v.Capabilities().Float32Atomics {
		t.Skip("vulkan device lacks float32 atomics")
	}
	const sv, h, nTok = 3, 2, 5
	q := autograd.Random(80, 1.0, sv, h, nTok)
	k := autograd.Random(81, 1.0, sv, h, nTok)
	vv := autograd.Random(82, 1.0, sv, h, nTok)
	beta := autograd.Random(84, 0.5, 1, h, nTok)
	state := autograd.Random(85, 0.5, sv, sv, h)
	dOut := autograd.Random(86, 1.0, sv, h, nTok)
	dNew := autograd.Random(87, 1.0, sv, sv, h)

	for _, kda := range []bool{false, true} {
		var g *compute.Tensor
		if kda {
			g = autograd.Random(83, 0.3, sv, h, nTok)
		} else {
			g = autograd.Random(83, 0.3, 1, h, nTok)
		}
		name := "scalar"
		if kda {
			name = "kda"
		}
		cq, ck, cvv, cg, cb, cs, err := c.GatedDeltaNetBackward(
			up(t, c, q.Dims, q.F32), up(t, c, k.Dims, k.F32), up(t, c, vv.Dims, vv.F32),
			up(t, c, g.Dims, g.F32), up(t, c, beta.Dims, beta.F32), up(t, c, state.Dims, state.F32),
			up(t, c, dOut.Dims, dOut.F32), up(t, c, dNew.Dims, dNew.F32))
		if err != nil {
			t.Fatal(err)
		}
		vq, vk, vvv, vg, vb, vs, err := v.GatedDeltaNetBackward(
			up(t, v, q.Dims, q.F32), up(t, v, k.Dims, k.F32), up(t, v, vv.Dims, vv.F32),
			up(t, v, g.Dims, g.F32), up(t, v, beta.Dims, beta.F32), up(t, v, state.Dims, state.F32),
			up(t, v, dOut.Dims, dOut.F32), up(t, v, dNew.Dims, dNew.F32))
		if err != nil {
			t.Fatal(err)
		}
		compare(t, name+" dQ", down(t, v, vq), down(t, c, cq))
		compare(t, name+" dK", down(t, v, vk), down(t, c, ck))
		compare(t, name+" dV", down(t, v, vvv), down(t, c, cvv))
		compare(t, name+" dG", down(t, v, vg), down(t, c, cg))
		compare(t, name+" dBeta", down(t, v, vb), down(t, c, cb))
		compare(t, name+" dState", down(t, v, vs), down(t, c, cs))

		// Independent oracle.
		hq, hk, hvv, hg, hb, hs, err := autograd.GatedDeltaNetBackward(q, k, vv, g, beta, state, dOut, dNew)
		if err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			n    string
			got  []float32
			want *compute.Tensor
		}{
			{name + " dQ", down(t, v, vq), hq}, {name + " dK", down(t, v, vk), hk},
			{name + " dV", down(t, v, vvv), hvv}, {name + " dG", down(t, v, vg), hg},
			{name + " dBeta", down(t, v, vb), hb}, {name + " dState", down(t, v, vs), hs},
		} {
			if d := maxAbsDiffLocal(tc.got, tc.want.F32); d > 1e-3 {
				t.Fatalf("%s: device vs autograd max diff %g", tc.n, d)
			}
		}
	}
}

func TestVulkanHeadViewOps(t *testing.T) {
	c, v := newBackends(t)
	const dConv, convDim, T = 3, 8, 4
	ncs := dConv - 1 + T
	qkv := autograd.Random(90, 1.0, convDim, T)
	cv := autograd.Random(91, 1.0, convDim, T)

	// ConvInput / back.
	ciC, _ := c.ConvInput(up(t, c, []int{convDim, T}, qkv.F32), dConv, convDim, T)
	ciV, _ := v.ConvInput(up(t, v, []int{convDim, T}, qkv.F32), dConv, convDim, T)
	compare(t, "conv_input", down(t, v, ciV), down(t, c, ciC))
	cbC, _ := c.ConvInputBack(up(t, c, []int{ncs, convDim, 1}, autograd.Random(95, 1.0, ncs, convDim, 1).F32), dConv, convDim, T)
	cbV, _ := v.ConvInputBack(up(t, v, []int{ncs, convDim, 1}, autograd.Random(95, 1.0, ncs, convDim, 1).F32), dConv, convDim, T)
	compare(t, "conv_input_back", down(t, v, cbV), down(t, c, cbC))

	// GatherHeads / back.
	const offset, headDim, nHead = 2, 2, 3
	gC, _ := c.GatherHeads(up(t, c, []int{convDim, T}, cv.F32), offset, headDim, nHead, T, convDim)
	gV, _ := v.GatherHeads(up(t, v, []int{convDim, T}, cv.F32), offset, headDim, nHead, T, convDim)
	compare(t, "gather_heads", down(t, v, gV), down(t, c, gC))
	dg := autograd.Random(92, 1.0, headDim, nHead, T)
	gbC, _ := c.GatherHeadsBack(up(t, c, []int{headDim, nHead, T}, dg.F32), offset, headDim, nHead, T, convDim)
	gbV, _ := v.GatherHeadsBack(up(t, v, []int{headDim, nHead, T}, dg.F32), offset, headDim, nHead, T, convDim)
	compare(t, "gather_heads_back", down(t, v, gbV), down(t, c, gbC))

	// RepeatHeads / back.
	const hd, nIn, nOut = 2, 2, 4
	rx := autograd.Random(93, 1.0, hd, nIn, T)
	rC, _ := c.RepeatHeads(up(t, c, []int{hd, nIn, T}, rx.F32), hd, nIn, nOut, T)
	rV, _ := v.RepeatHeads(up(t, v, []int{hd, nIn, T}, rx.F32), hd, nIn, nOut, T)
	compare(t, "repeat_heads", down(t, v, rV), down(t, c, rC))
	drx := autograd.Random(94, 1.0, hd, nOut, T)
	rbC, _ := c.RepeatHeadsBack(up(t, c, []int{hd, nOut, T}, drx.F32), hd, nIn, nOut, T)
	rbV, _ := v.RepeatHeadsBack(up(t, v, []int{hd, nOut, T}, drx.F32), hd, nIn, nOut, T)
	compare(t, "repeat_heads_back", down(t, v, rbV), down(t, c, rbC))
}

func TestVulkanGetRowsWeight(t *testing.T) {
	c, v := newBackends(t)
	const in, nRows = 256, 5
	wf := make([]float32, in*nRows)
	for i := range wf {
		wf[i] = float32((i%19)-9) * 0.2
	}
	indices := []int32{3, 0, 4, 3}
	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ5_K, quant.TypeQ6_K, quant.TypeIQ4_XS} {
		raw, err := quant.Quantize(typ, wf)
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		cw, _ := c.UploadWeight(typ, raw, []int{in, nRows})
		vw, _ := v.UploadWeight(typ, raw, []int{in, nRows})
		co, err := c.GetRowsWeight(cw, indices)
		if err != nil {
			t.Fatal(err)
		}
		vo, err := v.GetRowsWeight(vw, indices)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "get_rows_weight "+typ.String(), down(t, v, vo), down(t, c, co))
	}
}

func TestVulkanCrossEntropy(t *testing.T) {
	c, v := newBackends(t)
	const V, T = 37, 4
	logits := autograd.Random(700, 2.0, V, T)
	targets := []int32{3, -100, 10, 36}

	cl, cd, err := c.CrossEntropy(up(t, c, []int{V, T}, logits.F32), targets, -100)
	if err != nil {
		t.Fatal(err)
	}
	vl, vd, err := v.CrossEntropy(up(t, v, []int{V, T}, logits.F32), targets, -100)
	if err != nil {
		t.Fatal(err)
	}
	gotL := down(t, v, vl)
	wantL := down(t, c, cl)
	if d := math.Abs(float64(gotL[0] - wantL[0])); d > 1e-5 {
		t.Fatalf("loss %g vs %g", gotL[0], wantL[0])
	}
	compare(t, "cross_entropy dLogits", down(t, v, vd), down(t, c, cd))

	// Independent oracle: compute.CrossEntropy + autograd.CrossEntropyBackward.
	refLoss, err := compute.CrossEntropy(logits, []int{3, -100, 10, 36}, -100)
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Abs(float64(gotL[0] - refLoss)); d > 1e-5 {
		t.Fatalf("loss vs oracle: %g vs %g", gotL[0], refLoss)
	}
	refD := autograd.CrossEntropyBackward(logits, []int{3, -100, 10, 36}, -100)
	if d := maxAbsDiffLocal(down(t, v, vd), refD.F32); d > 1e-5 {
		t.Fatalf("dLogits vs oracle: max diff %g", d)
	}
}

func TestVulkanAdamWStep(t *testing.T) {
	c, v := newBackends(t)
	const n = 137
	param := make([]float32, n)
	grad := make([]float32, n)
	for i := range param {
		param[i] = float32(i)*0.01 - 0.5
		grad[i] = float32(math.Sin(float64(i))) * 0.3
	}
	p := compute.AdamWParams{
		Alpha: 1e-3, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 0.01,
		Beta1Hat: float32(1 / (1 - 0.9)), Beta2Hat: float32(1 / (1 - 0.999)),
	}
	run := func(b compute.Backend) (paramOut, m, vv []float32) {
		pb := up(t, b, []int{n}, append([]float32(nil), param...))
		gb := up(t, b, []int{n}, append([]float32(nil), grad...))
		mb, err := b.Upload(compute.NewF32(n))
		if err != nil {
			t.Fatal(err)
		}
		vb, err := b.Upload(compute.NewF32(n))
		if err != nil {
			t.Fatal(err)
		}
		if err := b.AdamWStep(pb, gb, mb, vb, p); err != nil {
			t.Fatal(err)
		}
		return down(t, b, pb), down(t, b, mb), down(t, b, vb)
	}
	cP, cM, cV := run(c)
	vP, vM, vV := run(v)
	compare(t, "adamw param", vP, cP)
	compare(t, "adamw m", vM, cM)
	compare(t, "adamw v", vV, cV)
}

func TestVulkanSumSquares(t *testing.T) {
	c, v := newBackends(t)
	for _, n := range []int{1, 255, 256, 1000, 4097, 65536} {
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(math.Sin(float64(i) * 0.1))
		}
		run := func(b compute.Backend) float32 {
			src := up(t, b, []int{n}, data)
			dst, err := b.Upload(compute.NewF32(1))
			if err != nil {
				t.Fatal(err)
			}
			if err := b.SumSquares(dst, src); err != nil {
				t.Fatal(err)
			}
			return down(t, b, dst)[0]
		}
		got := run(v)
		want := run(c)
		tol := 1e-3 * float32(math.Abs(float64(want)))
		if tol < 1e-4 {
			tol = 1e-4
		}
		if d := float32(math.Abs(float64(got - want))); d > tol {
			t.Fatalf("sum_squares n=%d: got %v want %v (diff %v)", n, got, want, d)
		}
	}
}
