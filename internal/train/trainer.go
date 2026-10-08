package train

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// Example is one training sequence: token ids plus a per-token loss mask
// (true where the assistant target should be scored).
type Example struct {
	Tokens   []int32
	LossMask []bool
}

// Model is a trainable graph. ForwardBackward runs one forward+backward over an
// example and returns the mean loss and the parameter gradients (nil entries
// allowed for parameters without a gradient this step).
type Model interface {
	Params() []*compute.Tensor
	ForwardBackward(ex Example) (loss float32, grads []*compute.Tensor, err error)
}

// TrainerConfig controls the training loop.
type TrainerConfig struct {
	Steps       int
	Accum       int     // gradient accumulation steps
	MaxGradNorm float32 // 0 disables clipping
	LogEvery    int
	Seed        uint64
}

// Trainer drives a Model with an optimizer and learning-rate schedule.
type Trainer struct {
	Cfg TrainerConfig
}

// Run executes the training loop, returning the logged losses. LR returns the
// learning rate for a given optimizer step.
func (t *Trainer) Run(m Model, opt *AdamW, sched func(step int) float32, data []Example, onStep func(step int, loss float32)) ([]float32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("train: empty dataset")
	}
	accum := t.Cfg.Accum
	if accum < 1 {
		accum = 1
	}
	params := m.Params()
	var history []float32
	var accumGrads []*compute.Tensor
	accLoss := float32(0)
	for step := 0; step < t.Cfg.Steps; step++ {
		accLoss = 0
		for a := 0; a < accum; a++ {
			ex := data[(step*accum+a)%len(data)]
			loss, grads, err := m.ForwardBackward(ex)
			if err != nil {
				return history, err
			}
			accLoss += loss
			if accumGrads == nil {
				accumGrads = newZeroGrads(grads)
			}
			addGrads(accumGrads, grads)
		}
		meanLoss := accLoss / float32(accum)
		// Normalize accumulated gradients by the accumulation count.
		scaleGrads(accumGrads, 1/float32(accum))
		if t.Cfg.MaxGradNorm > 0 {
			ClipGradGlobalNorm(accumGrads, t.Cfg.MaxGradNorm)
		}
		opt.cfg.LR = sched(step)
		opt.Step(params, accumGrads)
		accumGrads = nil
		history = append(history, meanLoss)
		if onStep != nil {
			onStep(step, meanLoss)
		}
	}
	return history, nil
}

func newZeroGrads(grads []*compute.Tensor) []*compute.Tensor {
	out := make([]*compute.Tensor, len(grads))
	for i, g := range grads {
		if g == nil {
			continue
		}
		out[i] = compute.NewF32(g.Dims...)
	}
	return out
}

func addGrads(dst, src []*compute.Tensor) {
	for i := range dst {
		if dst[i] == nil || src[i] == nil {
			continue
		}
		for j := range dst[i].F32 {
			dst[i].F32[j] += src[i].F32[j]
		}
	}
}

func scaleGrads(grads []*compute.Tensor, s float32) {
	for _, g := range grads {
		if g == nil {
			continue
		}
		for i := range g.F32 {
			g.F32[i] *= s
		}
	}
}

// SaveCheckpoint writes the parameter tensors to path using gob.
func SaveCheckpoint(path string, params []*compute.Tensor) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	return enc.Encode(params)
}

// LoadCheckpoint reads parameter tensors previously written by
// SaveCheckpoint, replacing the contents of the provided tensors in order.
func LoadCheckpoint(path string, params []*compute.Tensor) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var loaded []*compute.Tensor
	if err := gob.NewDecoder(f).Decode(&loaded); err != nil {
		return err
	}
	if len(loaded) != len(params) {
		return fmt.Errorf("train: checkpoint has %d tensors, want %d", len(loaded), len(params))
	}
	for i := range params {
		if len(loaded[i].F32) != len(params[i].F32) {
			return fmt.Errorf("train: tensor %d size mismatch", i)
		}
		copy(params[i].F32, loaded[i].F32)
	}
	return nil
}
