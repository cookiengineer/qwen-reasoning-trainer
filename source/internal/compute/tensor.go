// Package compute defines the tensor and backend abstractions shared by the
// CPU reference implementation and the Vulkan backend.
package compute

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// Tensor is a dense, row-major tensor in GGML dimension order: Dims[0] is the
// fastest-varying (row) dimension, matching the ne[] convention.
type Tensor struct {
	Dims []int
	// Type is the storage type. The CPU reference backend stores all tensor
	// data as float32 regardless of Type, which records the logical type.
	Type quant.Type
	// F32 holds the elements. Its length equals NumElements.
	F32 []float32
}

// NewF32 allocates a float32 tensor with the given GGML-order dimensions.
func NewF32(dims ...int) *Tensor {
	n := 1
	for _, d := range dims {
		if d < 0 {
			panic("compute: negative dimension")
		}
		n *= d
	}
	return &Tensor{Dims: append([]int(nil), dims...), Type: quant.TypeF32, F32: make([]float32, n)}
}

// NumElements returns the product of the dimensions.
func (t *Tensor) NumElements() int {
	n := 1
	for _, d := range t.Dims {
		n *= d
	}
	return n
}

// Ne returns Dims[i] or 1 when i is out of range, mirroring ggml's ne().
func (t *Tensor) Ne(i int) int {
	if i < 0 || i >= len(t.Dims) {
		return 1
	}
	return t.Dims[i]
}

// Clone returns a deep copy.
func (t *Tensor) Clone() *Tensor {
	return &Tensor{Dims: append([]int(nil), t.Dims...), Type: t.Type, F32: append([]float32(nil), t.F32...)}
}

// Reshape returns a view sharing the same backing storage with new dimensions
// whose element count must match.
func (t *Tensor) Reshape(dims ...int) (*Tensor, error) {
	n := 1
	for _, d := range dims {
		n *= d
	}
	if n != len(t.F32) {
		return nil, fmt.Errorf("compute: reshape %v to %v changes element count", t.Dims, dims)
	}
	return &Tensor{Dims: append([]int(nil), dims...), Type: t.Type, F32: t.F32}, nil
}

// V returns the single element for a rank-1 tensor of length 1.
func (t *Tensor) V() float32 {
	if len(t.F32) == 0 {
		return 0
	}
	return t.F32[0]
}

// Sigmoid computes 1/(1+exp(-x)).
func Sigmoid(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

// Silu computes x*sigmoid(x).
func Silu(x float32) float32 { return x * Sigmoid(x) }
