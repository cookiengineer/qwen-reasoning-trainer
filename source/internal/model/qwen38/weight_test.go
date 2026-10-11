package qwen38

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func randF32(n int, seed uint32) []float32 {
	out := make([]float32, n)
	x := seed
	for i := range out {
		x = x*1664525 + 1013904223
		out[i] = (float32(x>>8)/float32(1<<24) - 0.5) * 2
	}
	return out
}

// TestWeightMatMulQuantized checks that the blocked, dequantizing matmul
// matches a plain matmul on the dequantized weight.
func TestWeightMatMulQuantized(t *testing.T) {
	for _, typ := range []quant.Type{quant.TypeF32, quant.TypeF16, quant.TypeQ8_0, quant.TypeQ4_0} {
		k, n, m := 128, 40, 5
		data := randF32(k*n, uint32(typ)+7)
		raw, err := quant.Quantize(typ, data)
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		w := &Weight{Typ: typ, Dims: []int{k, n}, Raw: raw}
		wf := w.Tensor()

		x := &compute.Tensor{Dims: []int{k, m}, F32: randF32(k*m, 99)}
		got, err := w.MatMul(x)
		if err != nil {
			t.Fatal(err)
		}
		want, err := compute.MatMul(wf, x)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want.F32 {
			if math.Abs(float64(got.F32[i]-want.F32[i])) > 1e-6 {
				t.Fatalf("%s: matmul[%d] = %v, want %v", typ, i, got.F32[i], want.F32[i])
			}
		}
	}
}

// TestWeightMatMulBlocked exercises the N-blocking path with more than one block.
func TestWeightMatMulBlocked(t *testing.T) {
	k, n, m := 32, 2500, 3
	data := randF32(k*n, 11)
	w := NewWeightF32([]int{k, n}, data)
	x := &compute.Tensor{Dims: []int{k, m}, F32: randF32(k*m, 3)}
	got, err := w.MatMul(x)
	if err != nil {
		t.Fatal(err)
	}
	want, err := compute.MatMul(w.Tensor(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want.F32 {
		if got.F32[i] != want.F32[i] {
			t.Fatalf("blocked matmul[%d] = %v, want %v", i, got.F32[i], want.F32[i])
		}
	}
}

func TestWeightGetRows(t *testing.T) {
	k, n := 8, 6
	data := randF32(k*n, 5)
	w := NewWeightF32([]int{k, n}, data)
	out, err := w.GetRows([]int32{4, 0})
	if err != nil {
		t.Fatal(err)
	}
	if out.Ne(0) != k || out.Ne(1) != 2 {
		t.Fatalf("dims = %v", out.Dims)
	}
	for i := 0; i < k; i++ {
		if out.F32[i] != data[4*k+i] {
			t.Errorf("row0[%d] = %v, want %v", i, out.F32[i], data[4*k+i])
		}
		if out.F32[k+i] != data[i] {
			t.Errorf("row1[%d] = %v, want %v", i, out.F32[k+i], data[i])
		}
	}
	if _, err := w.GetRows([]int32{6}); err == nil {
		t.Error("expected out-of-range error")
	}
}
