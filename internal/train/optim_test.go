package train

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func TestAdamWQuadratic(t *testing.T) {
	// Minimize sum_i (p_i - 3)^2; gradient 2(p_i-3).
	p := compute.NewF32(5)
	for i := range p.F32 {
		p.F32[i] = float32(i) // 0..4
	}
	opt := NewAdamW(DefaultAdamW(0.1), []*compute.Tensor{p})
	for step := 0; step < 500; step++ {
		g := compute.NewF32(5)
		for i := range g.F32 {
			g.F32[i] = 2 * (p.F32[i] - 3)
		}
		opt.Step([]*compute.Tensor{p}, []*compute.Tensor{g})
	}
	for i, v := range p.F32 {
		if math.Abs(float64(v)-3) > 1e-2 {
			t.Fatalf("param %d = %g, want ~3", i, v)
		}
	}
}

func TestCosineSchedule(t *testing.T) {
	sched := CosineSchedule(1.0, 0.1, 10, 100)
	if got := sched(0); math.Abs(float64(got)-0.1) > 1e-6 {
		t.Errorf("warmup step 0 = %g, want 0.1", got)
	}
	if got := sched(9); got > 1.0 || got <= sched(0) {
		t.Errorf("warmup should increase: step9=%g step0=%g", got, sched(0))
	}
	if got := sched(100); math.Abs(float64(got)-0.1) > 1e-3 {
		t.Errorf("final = %g, want ~0.1", got)
	}
	if sched(50) >= sched(10) {
		t.Errorf("cosine should decay: step10=%g step50=%g", sched(10), sched(50))
	}
}

func TestClipGradGlobalNorm(t *testing.T) {
	g := &compute.Tensor{Dims: []int{3}, F32: []float32{3, 4, 0}} // norm 5
	norm := ClipGradGlobalNorm([]*compute.Tensor{g}, 1.0)
	if math.Abs(float64(norm)-5) > 1e-5 {
		t.Fatalf("norm = %g, want 5", norm)
	}
	var ss float64
	for _, v := range g.F32 {
		ss += float64(v) * float64(v)
	}
	if math.Abs(math.Sqrt(ss)-1) > 1e-5 {
		t.Fatalf("clipped norm = %g, want 1", math.Sqrt(ss))
	}
	if norm := ClipGradGlobalNorm([]*compute.Tensor{g}, 10); norm > 1.01 {
		t.Fatalf("no-clip case returned %g", norm)
	}
}
