package vulkan_test

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// TestVulkanGEMVRowParity checks the row-cooperative GEMV shaders against the
// CPU dequant+matmul oracle, including a row count that requires multiple
// dispatch chunks.
func TestVulkanGEMVRowParity(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	cases := []struct {
		typ  quant.Type
		k, n int
	}{
		{quant.TypeQ4_K, 5120, 257},
		{quant.TypeQ8_0, 5120, 257},
		{quant.TypeQ5_K, 5120, 257},
		{quant.TypeQ6_K, 5120, 257},
		{quant.TypeIQ4_XS, 5120, 257},
		{quant.TypeQ4_K, 256, 65540},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.typ.String(), func(t *testing.T) {
			data := make([]float32, tc.k*tc.n)
			for i := range data {
				data[i] = float32(math.Sin(float64(i))) * 2
			}
			raw, err := quant.Quantize(tc.typ, data)
			if err != nil {
				t.Fatal(err)
			}
			x := make([]float32, tc.k)
			for i := range x {
				x[i] = float32(math.Cos(float64(i))) * 0.5
			}

			wb, err := v.UploadWeight(tc.typ, raw, []int{tc.k, tc.n})
			if err != nil {
				t.Fatal(err)
			}
			xb := up(t, v, []int{tc.k, 1}, x)
			got, err := v.MatMulWeight(wb, xb)
			if err != nil {
				t.Fatal(err)
			}
			dequant, err := quant.Dequant(tc.typ, raw, int64(tc.k*tc.n))
			if err != nil {
				t.Fatal(err)
			}
			want, err := compute.MatMul(
				&compute.Tensor{Dims: []int{tc.k, tc.n}, F32: dequant},
				&compute.Tensor{Dims: []int{tc.k, 1}, F32: x},
			)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, tc.typ.String(), down(t, v, got), want.F32)
			v.Free(wb)
			v.Free(xb)
			v.Free(got)
		})
	}
}

func benchGEMV(b *testing.B, typ quant.Type, k, n int) {
	b.Helper()
	v := realBackend(b)
	defer v.Close()
	data := make([]float32, k*n)
	for i := range data {
		data[i] = float32(math.Sin(float64(i))) * 2
	}
	raw, err := quant.Quantize(typ, data)
	if err != nil {
		b.Fatal(err)
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(math.Cos(float64(i))) * 0.5
	}
	wb, err := v.UploadWeight(typ, raw, []int{k, n})
	if err != nil {
		b.Fatal(err)
	}
	xb, err := v.Upload(&compute.Tensor{Dims: []int{k, 1}, F32: x})
	if err != nil {
		b.Fatal(err)
	}
	if err := v.Sync(); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := v.MatMulWeight(wb, xb)
		if err != nil {
			b.Fatal(err)
		}
		v.Free(out)
		if err := v.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGEMVReal uses real projection shapes (FFN gate/up size).
func BenchmarkGEMVQ4KReal(b *testing.B)   { benchGEMV(b, quant.TypeQ4_K, 5120, 8192) }
func BenchmarkGEMVQ8_0Real(b *testing.B)  { benchGEMV(b, quant.TypeQ8_0, 5120, 8192) }
func BenchmarkGEMVQ4KFFN(b *testing.B)    { benchGEMV(b, quant.TypeQ4_K, 5120, 17408) }
func BenchmarkGEMVQ8_0FFN(b *testing.B)   { benchGEMV(b, quant.TypeQ8_0, 5120, 17408) }
func BenchmarkGEMVQ4KLMHead(b *testing.B) { benchGEMV(b, quant.TypeQ4_K, 5120, 151936) }

// BenchmarkGEMVTypesFFN covers the mixed quant types used by the Q4_K_M model.
func BenchmarkGEMVTypesFFN(b *testing.B) {
	types := []quant.Type{
		quant.TypeQ5_K,
		quant.TypeQ6_K,
		quant.TypeQ3_K,
		quant.TypeIQ4_XS,
		quant.TypeIQ4_NL,
		quant.TypeIQ3_S,
	}
	for _, typ := range types {
		typ := typ
		b.Run(typ.String(), func(b *testing.B) {
			benchGEMV(b, typ, 5120, 17408)
		})
	}
}

// TestVulkanGEMVMultiParity checks the small-M multi-token GEMV kernels. It
// also covers a type without a multi kernel (Q5_K), which must transparently
// fall back to the tiled GEMM.
func TestVulkanGEMVMultiParity(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	types := []quant.Type{quant.TypeQ4_K, quant.TypeQ5_K, quant.TypeQ6_K, quant.TypeQ8_0, quant.TypeIQ4_XS}
	for _, typ := range types {
		for _, m := range []int{2, 5, 8} {
			typ, m := typ, m
			t.Run(typ.String(), func(t *testing.T) {
				const k, n = 512, 37
				data := make([]float32, k*n)
				for i := range data {
					data[i] = float32(math.Sin(float64(i))) * 2
				}
				raw, err := quant.Quantize(typ, data)
				if err != nil {
					t.Fatal(err)
				}
				x := make([]float32, k*m)
				for i := range x {
					x[i] = float32(math.Cos(float64(i))) * 0.5
				}
				wb, err := v.UploadWeight(typ, raw, []int{k, n})
				if err != nil {
					t.Fatal(err)
				}
				xb := up(t, v, []int{k, m}, x)
				got, err := v.MatMulWeight(wb, xb)
				if err != nil {
					t.Fatal(err)
				}
				dequant, err := quant.Dequant(typ, raw, int64(k*n))
				if err != nil {
					t.Fatal(err)
				}
				want, err := compute.MatMul(
					&compute.Tensor{Dims: []int{k, n}, F32: dequant},
					&compute.Tensor{Dims: []int{k, m}, F32: x},
				)
				if err != nil {
					t.Fatal(err)
				}
				compare(t, typ.String(), down(t, v, got), want.F32)
				v.Free(wb)
				v.Free(xb)
				v.Free(got)
			})
		}
	}
}
