package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// linearModel is a minimal test model: loss = mean((Xw - y)^2).
type linearModel struct {
	w *compute.Tensor
	X []float32
	y []float32
	K int
}

func (m *linearModel) Params() []*compute.Tensor { return []*compute.Tensor{m.w} }

func (m *linearModel) ForwardBackward(ex Example) (float32, []*compute.Tensor, error) {
	K := m.K
	n := len(m.X) / K
	pred := make([]float64, n)
	var loss float64
	for i := 0; i < n; i++ {
		var s float64
		for j := 0; j < K; j++ {
			s += float64(m.X[i*K+j]) * float64(m.w.F32[j])
		}
		pred[i] = s
		d := s - float64(m.y[i])
		loss += d * d
	}
	loss /= float64(n)
	grad := compute.NewF32(K)
	inv := 2 / float64(n)
	for j := 0; j < K; j++ {
		var g float64
		for i := 0; i < n; i++ {
			g += (pred[i] - float64(m.y[i])) * float64(m.X[i*K+j])
		}
		grad.F32[j] = float32(g * inv)
	}
	return float32(loss), []*compute.Tensor{grad}, nil
}

func TestTrainerReducesLoss(t *testing.T) {
	K := 4
	m := &linearModel{w: compute.NewF32(K), K: K}
	// Fit y = X w* with w* = [1, -2, 3, 0.5].
	target := []float32{1, -2, 3, 0.5}
	for i := 0; i < 20; i++ {
		for j := 0; j < K; j++ {
			m.X = append(m.X, float32((i*7+j*3)%5)-2)
		}
		var s float32
		for j := 0; j < K; j++ {
			s += m.X[i*K+j] * target[j]
		}
		m.y = append(m.y, s)
	}
	data := []Example{{Tokens: []int32{0}}}

	opt := NewAdamW(DefaultAdamW(0.05), m.Params())
	tr := &Trainer{Cfg: TrainerConfig{Steps: 300, Accum: 1, MaxGradNorm: 1}}
	hist, err := tr.Run(m, opt, func(step int) float32 { return 0.05 }, data, nil)
	if err != nil {
		t.Fatal(err)
	}
	if hist[len(hist)-1] >= hist[0] {
		t.Fatalf("loss did not decrease: first=%g last=%g", hist[0], hist[len(hist)-1])
	}
	for j := 0; j < K; j++ {
		if math.Abs(float64(m.w.F32[j]-target[j])) > 0.05 {
			t.Fatalf("w[%d]=%g want %g", j, m.w.F32[j], target[j])
		}
	}
}

func TestCheckpointRoundTrip(t *testing.T) {
	params := []*compute.Tensor{compute.NewF32(2, 3), compute.NewF32(4)}
	for i := range params[0].F32 {
		params[0].F32[i] = float32(i) + 0.5
	}
	for i := range params[1].F32 {
		params[1].F32[i] = float32(-i)
	}
	path := filepath.Join(t.TempDir(), "ckpt.gob")
	if err := SaveCheckpoint(path, params); err != nil {
		t.Fatal(err)
	}
	restored := []*compute.Tensor{compute.NewF32(2, 3), compute.NewF32(4)}
	if err := LoadCheckpoint(path, restored); err != nil {
		t.Fatal(err)
	}
	for i := range params[0].F32 {
		if restored[0].F32[i] != params[0].F32[i] {
			t.Fatalf("mismatch at %d", i)
		}
	}
	for i := range params[1].F32 {
		if restored[1].F32[i] != params[1].F32[i] {
			t.Fatalf("mismatch at %d", i)
		}
	}
	_ = os.Remove(path)
}
