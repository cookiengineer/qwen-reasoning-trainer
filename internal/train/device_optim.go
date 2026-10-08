package train

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// adamMoments holds the first/second moment buffers for one adapter's two
// factors, resident on the device.
type adamMoments struct {
	mA, vA, mB, vB compute.Buffer
}

// DeviceAdamW keeps the AdamW optimizer state for every LoRA adapter on the
// device and applies fully-resident steps, so a training step never round-trips
// the params or gradients through the host.
type DeviceAdamW struct {
	be  compute.Backend
	cfg AdamWConfig
	t   int
	m   map[*devLin]*adamMoments
}

// NewDeviceAdamW allocates zeroed moment buffers for every adapter.
func NewDeviceAdamW(be compute.Backend, cfg AdamWConfig, lins []*devLin) (*DeviceAdamW, error) {
	o := &DeviceAdamW{be: be, cfg: cfg, m: make(map[*devLin]*adamMoments, len(lins))}
	for _, l := range lins {
		mA, err := be.Upload(compute.NewF32(l.lora.A.Dims...))
		if err != nil {
			return nil, err
		}
		vA, err := be.Upload(compute.NewF32(l.lora.A.Dims...))
		if err != nil {
			return nil, err
		}
		mB, err := be.Upload(compute.NewF32(l.lora.B.Dims...))
		if err != nil {
			return nil, err
		}
		vB, err := be.Upload(compute.NewF32(l.lora.B.Dims...))
		if err != nil {
			return nil, err
		}
		o.m[l] = &adamMoments{mA: mA, vA: vA, mB: mB, vB: vB}
	}
	return o, nil
}

// Step applies one AdamW update to every adapter using the given gradients.
func (o *DeviceAdamW) Step(grads []adapterGrad, lr float32) error {
	o.t++
	p := compute.AdamWParams{
		Alpha:       lr,
		Beta1:       o.cfg.Beta1,
		Beta2:       o.cfg.Beta2,
		Eps:         o.cfg.Eps,
		WeightDecay: o.cfg.WeightDecay,
		Beta1Hat:    float32(1 / (1 - math.Pow(float64(o.cfg.Beta1), float64(o.t)))),
		Beta2Hat:    float32(1 / (1 - math.Pow(float64(o.cfg.Beta2), float64(o.t)))),
	}
	for _, g := range grads {
		mom := o.m[g.lin]
		if mom == nil {
			continue
		}
		if err := o.be.AdamWStep(g.lin.a, g.dA, mom.mA, mom.vA, p); err != nil {
			return err
		}
		if err := o.be.AdamWStep(g.lin.b, g.dB, mom.mB, mom.vB, p); err != nil {
			return err
		}
	}
	return nil
}

// scaleReplace returns b*s and frees b.
func scaleReplace(be compute.Backend, b compute.Buffer, s float32) (compute.Buffer, error) {
	nb, err := be.Scale(b, s)
	if err != nil {
		return nil, err
	}
	be.Free(b)
	return nb, nil
}

// accumulateGrads adds grads into acc (device-resident). It initializes acc
// from grads on the first call. The adapter order must match.
func accumulateGrads(be compute.Backend, acc, grads []adapterGrad) ([]adapterGrad, error) {
	if acc == nil {
		out := make([]adapterGrad, len(grads))
		copy(out, grads)
		return out, nil
	}
	if len(acc) != len(grads) {
		return nil, fmt.Errorf("train: gradient accumulation shape mismatch")
	}
	for i := range grads {
		if acc[i].lin != grads[i].lin {
			return nil, fmt.Errorf("train: gradient accumulation adapter mismatch")
		}
		da, err := be.Binary(compute.BinaryAdd, acc[i].dA, grads[i].dA)
		if err != nil {
			return nil, err
		}
		be.Free(acc[i].dA)
		db, err := be.Binary(compute.BinaryAdd, acc[i].dB, grads[i].dB)
		if err != nil {
			return nil, err
		}
		be.Free(acc[i].dB)
		acc[i].dA = da
		acc[i].dB = db
	}
	return acc, nil
}

// clipGrads computes the global L2 norm of the accumulated grads on the device
// and, if it exceeds maxNorm, rescales every gradient. It returns the pre-clip
// norm. Only the single-element norm accumulator is read back.
func clipGrads(be compute.Backend, grads []adapterGrad, maxNorm float32) (float32, error) {
	sum, err := be.Upload(compute.NewF32(1))
	if err != nil {
		return 0, err
	}
	for _, g := range grads {
		if err := be.SumSquares(sum, g.dA); err != nil {
			return 0, err
		}
		if err := be.SumSquares(sum, g.dB); err != nil {
			return 0, err
		}
	}
	t, err := be.Download(sum)
	if err != nil {
		return 0, err
	}
	norm := float32(math.Sqrt(float64(t.F32[0])))
	if maxNorm <= 0 || norm <= maxNorm {
		return norm, nil
	}
	s := maxNorm / norm
	for i := range grads {
		da, err := scaleReplace(be, grads[i].dA, s)
		if err != nil {
			return norm, err
		}
		db, err := scaleReplace(be, grads[i].dB, s)
		if err != nil {
			return norm, err
		}
		grads[i].dA = da
		grads[i].dB = db
	}
	return norm, nil
}

// exampleTargets converts a training example into the token array and the
// next-token target array used by DeviceModel.Loss (masked positions are
// ignoreIndex). It mirrors QwenModel.ForwardBackward.
func exampleTargets(ex Example) (tokens, targets []int32) {
	T := len(ex.Tokens)
	tokens = make([]int32, T)
	copy(tokens, ex.Tokens)
	targets = make([]int32, T)
	for i := range targets {
		targets[i] = -100
	}
	for t := 0; t < T-1; t++ {
		if len(ex.LossMask) == T && !ex.LossMask[t+1] {
			continue
		}
		targets[t] = ex.Tokens[t+1]
	}
	return tokens, targets
}

// DeviceTrainerConfig controls the resident device training loop.
type DeviceTrainerConfig struct {
	Steps       int
	Accum       int
	MaxGradNorm float32
	LogEvery    int
}

// DeviceTrainer drives a DeviceModel with a device-resident DeviceAdamW.
type DeviceTrainer struct {
	Model *DeviceModel
	Opt   *DeviceAdamW
	Cfg   DeviceTrainerConfig
}

// Run executes the resident training loop, returning the logged mean losses.
func (t *DeviceTrainer) Run(data []Example, sched func(step int) float32, onStep func(step int, loss float32)) ([]float32, error) {
	if t.Model == nil || t.Opt == nil {
		return nil, fmt.Errorf("train: device trainer requires model and optimizer")
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("train: empty dataset")
	}
	accum := t.Cfg.Accum
	if accum < 1 {
		accum = 1
	}
	be := t.Model.be
	var history []float32
	var err error
	for step := 0; step < t.Cfg.Steps; step++ {
		var acc []adapterGrad
		accLoss := float32(0)
		for a := 0; a < accum; a++ {
			ex := data[(step*accum+a)%len(data)]
			tokens, targets := exampleTargets(ex)
			if len(tokens) < 2 {
				continue
			}
			logits, ctx, err := t.Model.Forward(tokens)
			if err != nil {
				return history, err
			}
			loss, dLogits, err := t.Model.Loss(logits, targets, -100)
			if err != nil {
				return history, err
			}
			be.Free(logits)
			grads, err := t.Model.Backward(ctx, dLogits)
			if err != nil {
				return history, err
			}
			be.Free(dLogits)
			accLoss += loss
			if acc, err = accumulateGrads(be, acc, grads); err != nil {
				return history, err
			}
		}
		meanLoss := accLoss / float32(accum)
		if acc != nil {
			if accum > 1 {
				inv := 1 / float32(accum)
				for i := range acc {
					if acc[i].dA, err = scaleReplace(be, acc[i].dA, inv); err != nil {
						return history, err
					}
					if acc[i].dB, err = scaleReplace(be, acc[i].dB, inv); err != nil {
						return history, err
					}
				}
			}
			if t.Cfg.MaxGradNorm > 0 {
				if _, err := clipGrads(be, acc, t.Cfg.MaxGradNorm); err != nil {
					return history, err
				}
			}
			if err := t.Opt.Step(acc, sched(step)); err != nil {
				return history, err
			}
			for i := range acc {
				be.Free(acc[i].dA)
				be.Free(acc[i].dB)
			}
		}
		history = append(history, meanLoss)
		if onStep != nil {
			onStep(step, meanLoss)
		}
	}
	return history, nil
}

// EvalLoss returns the mean cross-entropy loss over the examples without
// updating any parameters. It is used for the validation split.
func (m *DeviceModel) EvalLoss(data []Example) (float32, error) {
	if len(data) == 0 {
		return 0, nil
	}
	var sum float32
	var n int
	for _, ex := range data {
		tokens, targets := exampleTargets(ex)
		if len(tokens) < 2 {
			continue
		}
		logits, ctx, err := m.Forward(tokens)
		if err != nil {
			return 0, err
		}
		loss, dLogits, err := m.Loss(logits, targets, -100)
		if err != nil {
			return 0, err
		}
		m.be.Free(logits)
		m.be.Free(dLogits)
		ctx.release(m.be)
		sum += loss
		n++
	}
	if n == 0 {
		return 0, nil
	}
	return sum / float32(n), nil
}
