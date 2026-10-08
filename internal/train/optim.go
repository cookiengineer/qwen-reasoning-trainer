// Package train implements the QLoRA training loop: AdamW, a cosine learning
// rate schedule, LoRA adapters on a frozen quantized base, gradient
// accumulation/clipping, and checkpointing.
package train

import (
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// AdamWConfig holds the optimizer hyperparameters.
type AdamWConfig struct {
	LR          float32
	Beta1       float32
	Beta2       float32
	Eps         float32
	WeightDecay float32
}

// DefaultAdamW is the standard configuration used for LoRA training.
func DefaultAdamW(lr float32) AdamWConfig {
	return AdamWConfig{LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 0.0}
}

// AdamW is a decoupled-weight-decay Adam optimizer over a fixed parameter list.
type AdamW struct {
	cfg AdamWConfig
	m   [][]float32
	v   [][]float32
	t   int
}

// NewAdamW allocates optimizer state for the given parameters.
func NewAdamW(cfg AdamWConfig, params []*compute.Tensor) *AdamW {
	o := &AdamW{cfg: cfg, m: make([][]float32, len(params)), v: make([][]float32, len(params))}
	for i, p := range params {
		o.m[i] = make([]float32, len(p.F32))
		o.v[i] = make([]float32, len(p.F32))
	}
	return o
}

// Step applies one AdamW update. grads may contain nil entries for parameters
// without a gradient, which are skipped.
func (o *AdamW) Step(params, grads []*compute.Tensor) {
	if len(params) != len(grads) {
		panic("train: AdamW param/grad length mismatch")
	}
	o.t++
	b1, b2 := float64(o.cfg.Beta1), float64(o.cfg.Beta2)
	bc1 := 1 - math.Pow(b1, float64(o.t))
	bc2 := 1 - math.Pow(b2, float64(o.t))
	eps := float64(o.cfg.Eps)
	lr := float64(o.cfg.LR)
	wd := float64(o.cfg.WeightDecay)
	for i, p := range params {
		g := grads[i]
		if g == nil {
			continue
		}
		m, v := o.m[i], o.v[i]
		for j := range p.F32 {
			gj := float64(g.F32[j])
			m[j] = float32(b1*float64(m[j]) + (1-b1)*gj)
			v[j] = float32(b2*float64(v[j]) + (1-b2)*gj*gj)
			mhat := float64(m[j]) / bc1
			vhat := float64(v[j]) / bc2
			upd := mhat / (math.Sqrt(vhat) + eps)
			if wd != 0 {
				upd += wd * float64(p.F32[j])
			}
			p.F32[j] -= float32(lr * upd)
		}
	}
}

// CosineSchedule returns a function mapping a global step (0-based) to a
// learning rate: linear warmup over warmupSteps, then cosine decay to minLR
// over totalSteps.
func CosineSchedule(base, minLR float32, warmupSteps, totalSteps int) func(step int) float32 {
	return func(step int) float32 {
		if warmupSteps > 0 && step < warmupSteps {
			return base * float32(step+1) / float32(warmupSteps)
		}
		if totalSteps <= warmupSteps {
			return minLR
		}
		prog := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
		if prog < 0 {
			prog = 0
		}
		if prog > 1 {
			prog = 1
		}
		factor := 0.5 * (1 + math.Cos(math.Pi*prog))
		return minLR + (base-minLR)*float32(factor)
	}
}

// ClipGradGlobalNorm scales gradients in place so their global L2 norm does not
// exceed maxNorm. It returns the pre-clip global norm.
func ClipGradGlobalNorm(grads []*compute.Tensor, maxNorm float32) float32 {
	var ss float64
	for _, g := range grads {
		if g == nil {
			continue
		}
		for _, x := range g.F32 {
			ss += float64(x) * float64(x)
		}
	}
	norm := float32(math.Sqrt(ss))
	if maxNorm > 0 && norm > maxNorm {
		s := maxNorm / norm
		for _, g := range grads {
			if g == nil {
				continue
			}
			for i := range g.F32 {
				g.F32[i] *= s
			}
		}
	}
	return norm
}
