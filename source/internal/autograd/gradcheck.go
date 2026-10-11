// Package autograd holds the reference (CPU) gradients for the Qwen3.8 forward
// ops and a central finite-difference gradient checker. The Go gradients are
// the numerical oracle for the Vulkan training kernels: every backward kernel
// is validated against these before a shader is written.
//
// The backward functions operate on host compute.Tensor values in the same GGML
// layout as the forward ops they differentiate. They are intentionally simple
// and exact; the device implementations must match them.
package autograd

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// Objective is a scalar function of a set of input tensors. It must not mutate
// its inputs.
type Objective func(inputs []*compute.Tensor) float64

// defaultEps is the central-difference step. float32 inputs lose precision for
// much smaller steps; this value balances truncation and rounding error.
const defaultEps = 1e-3

// defaultTol is the relative tolerance applied to max(1, |numeric|, |analytic|).
const defaultTol = 2e-2

// GradCheck compares the analytic gradients of a scalar objective against
// central finite differences and returns an error describing the worst
// mismatch, or nil if they agree within tolerance.
//
// inputs and analytic must have matching length and shapes: analytic[i] is the
// gradient of obj with respect to inputs[i]. obj is evaluated many times and
// must be side-effect free.
func GradCheck(inputs, analytic []*compute.Tensor, obj Objective, eps float64) error {
	return GradCheckTol(inputs, analytic, obj, eps, defaultTol)
}

// GradCheckTol is GradCheck with an explicit relative tolerance.
func GradCheckTol(inputs, analytic []*compute.Tensor, obj Objective, eps, tol float64) error {
	worst, where, err := MaxGradError(inputs, analytic, obj, eps)
	if err != nil {
		return err
	}
	if worst > tol {
		return fmt.Errorf("gradcheck: worst relative error %.3g > %.3g at %s", worst, tol, where)
	}
	return nil
}

// MaxGradError returns the worst relative gradient error and a human-readable
// description of where it occurred. It is used for diagnostics and by the
// backend parity suite.
func MaxGradError(inputs, analytic []*compute.Tensor, obj Objective, eps float64) (float64, string, error) {
	if len(inputs) != len(analytic) {
		return 0, "", fmt.Errorf("gradcheck: %d inputs but %d analytic gradients", len(inputs), len(analytic))
	}
	if eps <= 0 {
		eps = defaultEps
	}

	var worst float64
	var worstWhere string
	for i := range inputs {
		in := inputs[i]
		ana := analytic[i]
		if len(in.F32) != len(ana.F32) {
			return 0, "", fmt.Errorf("gradcheck: input %d has %d elements but analytic has %d", i, len(in.F32), len(ana.F32))
		}
		for j := range in.F32 {
			orig := in.F32[j]
			in.F32[j] = orig + float32(eps)
			fp := obj(inputs)
			in.F32[j] = orig - float32(eps)
			fm := obj(inputs)
			in.F32[j] = orig

			num := (fp - fm) / (2 * eps)
			a := float64(ana.F32[j])
			diff := math.Abs(num - a)
			scale := math.Max(1, math.Max(math.Abs(num), math.Abs(a)))
			rel := diff / scale
			if rel > worst {
				worst = rel
				worstWhere = fmt.Sprintf("input %d element %d (dims %v): analytic=%g numeric=%g", i, j, in.Dims, a, num)
			}
		}
	}
	return worst, worstWhere, nil
}

// Ones returns an all-ones tensor with the given dimensions.
func Ones(dims ...int) *compute.Tensor {
	t := compute.NewF32(dims...)
	for i := range t.F32 {
		t.F32[i] = 1
	}
	return t
}

// Random returns a tensor of the given dims with deterministic pseudo-random
// values in [-scale, scale].
func Random(seed uint64, scale float32, dims ...int) *compute.Tensor {
	t := compute.NewF32(dims...)
	s := seed
	for i := range t.F32 {
		s = s*6364136223846793005 + 1442695040888963407
		u := float32((s>>11)&((1<<53)-1)) / float32(1<<53)
		t.F32[i] = (u*2 - 1) * scale
	}
	return t
}
