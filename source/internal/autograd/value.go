package autograd

import (
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func softplus(v float32) float32 { return float32(math.Log1p(math.Exp(float64(v)))) }

// Value is a node in the host autograd graph. It is the reference composition
// mechanism used to gradient-check whole models; it is NOT the device graph.
// The device implementation is an explicit, static forward+backward mirroring
// the model architecture.
//
// Every op appends its operands to prev and installs a backward closure that
// distributes the incoming gradient to those operands. Backward runs a reverse
// topological sweep.
type Value struct {
	Data *compute.Tensor
	Grad *compute.Tensor
	prev []*Value
	bwd  func(g *compute.Tensor)
}

// Param creates a differentiable leaf wrapping an existing tensor.
func Param(t *compute.Tensor) *Value { return &Value{Data: t} }

// Const creates a non-differentiable leaf.
func Const(t *compute.Tensor) *Value { return &Value{Data: t} }

func newVal(data *compute.Tensor, prev []*Value, bwd func(*compute.Tensor)) *Value {
	return &Value{Data: data, prev: prev, bwd: bwd}
}

// AddToGrad accumulates g into v's gradient, allocating on first use.
func (v *Value) AddToGrad(g *compute.Tensor) {
	if v.Grad == nil {
		v.Grad = compute.NewF32(v.Data.Dims...)
	}
	for i := range v.Grad.F32 {
		v.Grad.F32[i] += g.F32[i]
	}
}

// ZeroGrad clears accumulated gradients in the graph rooted at v.
func (v *Value) ZeroGrad() {
	seen := map[*Value]bool{}
	var visit func(n *Value)
	visit = func(n *Value) {
		if seen[n] {
			return
		}
		seen[n] = true
		n.Grad = nil
		for _, p := range n.prev {
			visit(p)
		}
	}
	visit(v)
}

// Backward computes gradients of the scalar objective v with respect to all
// leaves reachable from it. The root gradient defaults to ones, so v should be
// a scalar (Dims [1]).
func (v *Value) Backward() {
	order := make([]*Value, 0, 16)
	seen := map[*Value]bool{}
	var visit func(n *Value)
	visit = func(n *Value) {
		if seen[n] {
			return
		}
		seen[n] = true
		for _, p := range n.prev {
			visit(p)
		}
		order = append(order, n)
	}
	visit(v)

	if v.Grad == nil {
		v.Grad = compute.NewF32(v.Data.Dims...)
		for i := range v.Grad.F32 {
			v.Grad.F32[i] = 1
		}
	}
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		if n.Grad == nil || n.bwd == nil {
			continue
		}
		n.bwd(n.Grad)
	}
}

// VMatMul is the differentiable compute.MatMul: a [K,N], b [K,M] -> [N,M].
func VMatMul(a, b *Value) *Value {
	data, err := compute.MatMul(a.Data, b.Data)
	if err != nil {
		panic(err)
	}
	return newVal(data, []*Value{a, b}, func(g *compute.Tensor) {
		dA, dB := MatMulBackward(a.Data, b.Data, g)
		a.AddToGrad(dA)
		b.AddToGrad(dB)
	})
}

// VAdd is the differentiable elementwise add.
func VAdd(a, b *Value) *Value {
	data, err := compute.Add(a.Data, b.Data)
	if err != nil {
		panic(err)
	}
	return newVal(data, []*Value{a, b}, func(g *compute.Tensor) {
		a.AddToGrad(g)
		b.AddToGrad(g)
	})
}

// VMul is the differentiable elementwise multiply.
func VMul(a, b *Value) *Value {
	data, err := compute.Mul(a.Data, b.Data)
	if err != nil {
		panic(err)
	}
	return newVal(data, []*Value{a, b}, func(g *compute.Tensor) {
		da := compute.NewF32(a.Data.Dims...)
		db := compute.NewF32(b.Data.Dims...)
		for i := range g.F32 {
			da.F32[i] = g.F32[i] * b.Data.F32[i]
			db.F32[i] = g.F32[i] * a.Data.F32[i]
		}
		a.AddToGrad(da)
		b.AddToGrad(db)
	})
}

// VScale multiplies by a constant.
func VScale(a *Value, s float32) *Value {
	data := compute.Scale(a.Data, s)
	return newVal(data, []*Value{a}, func(g *compute.Tensor) {
		d := compute.Scale(g, s)
		a.AddToGrad(d)
	})
}

// VSum returns the sum of all elements as a scalar.
func VSum(x *Value) *Value {
	var s float32
	for _, v := range x.Data.F32 {
		s += v
	}
	data := compute.NewF32(1)
	data.F32[0] = s
	return newVal(data, []*Value{x}, func(g *compute.Tensor) {
		d := compute.NewF32(x.Data.Dims...)
		for i := range d.F32 {
			d.F32[i] = g.F32[0]
		}
		x.AddToGrad(d)
	})
}

// VReshape changes the view of a tensor without moving data.
func VReshape(x *Value, dims ...int) *Value {
	data, err := x.Data.Reshape(dims...)
	if err != nil {
		panic(err)
	}
	return newVal(data, []*Value{x}, func(g *compute.Tensor) {
		x.AddToGrad(g)
	})
}

// elementwiseUnary builds a Value for an elementwise function with a known
// local derivative d/dx given the forward output.
func elementwiseUnary(x *Value, fwd func(float32) float32, deriv func(out, x float32) float32) *Value {
	data := compute.NewF32(x.Data.Dims...)
	for i, v := range x.Data.F32 {
		data.F32[i] = fwd(v)
	}
	return newVal(data, []*Value{x}, func(g *compute.Tensor) {
		d := compute.NewF32(x.Data.Dims...)
		for i := range g.F32 {
			d.F32[i] = g.F32[i] * deriv(data.F32[i], x.Data.F32[i])
		}
		x.AddToGrad(d)
	})
}

// VSilu is the differentiable SiLU.
func VSilu(x *Value) *Value {
	return elementwiseUnary(x, compute.Silu, func(out, in float32) float32 {
		s := compute.Sigmoid(in)
		return s * (1 + in*(1-s))
	})
}

// VSigmoid is the differentiable sigmoid.
func VSigmoid(x *Value) *Value {
	return elementwiseUnary(x, compute.Sigmoid, func(out, in float32) float32 {
		return out * (1 - out)
	})
}

// VSoftplus is the differentiable softplus.
func VSoftplus(x *Value) *Value {
	return elementwiseUnary(x, func(v float32) float32 {
		return softplus(v)
	}, func(out, in float32) float32 {
		return compute.Sigmoid(in)
	})
}
