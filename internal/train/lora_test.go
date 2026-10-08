package train

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func TestLoRAMergeEquivalence(t *testing.T) {
	K, N, M, r := 8, 6, 3, 4
	W := autograd.Random(1, 0.5, K, N)
	x := autograd.Random(2, 0.5, K, M)
	l := NewLoRA(3, K, N, r, 2*float32(r))
	// Non-zero B so the adapter actually contributes.
	for i := range l.B.F32 {
		l.B.F32[i] = 0.1 * float32((i%5)-2)
	}

	base, err := compute.MatMul(W, x)
	if err != nil {
		t.Fatal(err)
	}
	upd, err := l.Update(x)
	if err != nil {
		t.Fatal(err)
	}
	applied, err := compute.Add(base, upd)
	if err != nil {
		t.Fatal(err)
	}

	Wm := W.Clone()
	l.MergeInto(Wm)
	merged, err := compute.MatMul(Wm, x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range applied.F32 {
		if d := math.Abs(float64(applied.F32[i] - merged.F32[i])); d > 1e-4 {
			t.Fatalf("element %d: apply=%g merge=%g diff=%g", i, applied.F32[i], merged.F32[i], d)
		}
	}
}

func TestLoRAZeroInitIsNoOp(t *testing.T) {
	K, N, M, r := 5, 4, 2, 3
	x := autograd.Random(5, 0.5, K, M)
	l := NewLoRA(6, K, N, r, 16)
	upd, err := l.Update(x)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range upd.F32 {
		if v != 0 {
			t.Fatalf("zero-init adapter produced %g at %d", v, i)
		}
	}
}

func TestLoRAMergeToQuant(t *testing.T) {
	K, N, r := 32, 8, 4
	W := autograd.Random(7, 0.5, K, N)
	baseRaw, err := quant.Quantize(quant.TypeQ8_0, W.F32)
	if err != nil {
		t.Fatal(err)
	}
	l := NewLoRA(8, K, N, r, 2*float32(r))
	for i := range l.B.F32 {
		l.B.F32[i] = 0.05 * float32((i%3)-1)
	}
	outType, raw, err := l.MergeToQuant(quant.TypeQ8_0, baseRaw, []int{K, N})
	if err != nil {
		t.Fatal(err)
	}
	if outType != quant.TypeQ8_0 {
		t.Fatalf("type = %v, want Q8_0", outType)
	}
	got, err := quant.Dequant(outType, raw, int64(K*N))
	if err != nil {
		t.Fatal(err)
	}
	want := W.Clone()
	l.MergeInto(want)
	// Q8_0 requantization tolerance.
	for i := range want.F32 {
		if d := math.Abs(float64(got[i] - want.F32[i])); d > 0.02*math.Max(1, math.Abs(float64(want.F32[i]))) {
			t.Fatalf("merged quant element %d = %g, want %g", i, got[i], want.F32[i])
		}
	}
}
