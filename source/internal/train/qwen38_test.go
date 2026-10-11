package train

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func tinyCfg() *qwen38.Config {
	return &qwen38.Config{
		NEmbd: 16, NHead: 4, NHeadKV: 2, HeadDim: 4, NRot: 4, NFf: 24, NLayer: 4, NVocab: 20,
		RopeTheta: 10000, RmsEps: 1e-6,
		DConv: 4, DState: 4, DInner: 16, DtRank: 4, GroupCount: 2,
		FullAttnInterval: 4, NextNPredict: 0,
		LayerTypes: []modelcfg.LayerType{
			modelcfg.LayerLinearAttention,
			modelcfg.LayerLinearAttention,
			modelcfg.LayerLinearAttention,
			modelcfg.LayerFullAttention,
		},
	}
}

func TestTrainableForwardMatchesInference(t *testing.T) {
	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 1)
	tokens := []int32{1, 2, 3, 4}

	ref := qwen38.NewModel(w)
	res, err := ref.Forward(tokens, qwen38.ForwardOptions{})
	if err != nil {
		t.Fatal(err)
	}

	tm := NewQwenModel(cfg, w, DefaultLoRA(), 7)
	got := tm.Logits(tokens)

	if len(got.F32) != len(res.Logits.F32) {
		t.Fatalf("logit size %d != %d", len(got.F32), len(res.Logits.F32))
	}
	var maxDiff float64
	for i := range got.F32 {
		d := math.Abs(float64(got.F32[i] - res.Logits.F32[i]))
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-4 {
		t.Fatalf("trainable forward differs from inference by %g", maxDiff)
	}
}

func TestTrainableOneStepReducesLoss(t *testing.T) {
	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 2)
	tm := NewQwenModel(cfg, w, DefaultLoRA(), 11)
	ex := Example{Tokens: []int32{1, 2, 3, 4, 5}, LossMask: []bool{false, true, true, true, true}}

	loss0, grads, err := tm.ForwardBackward(ex)
	if err != nil {
		t.Fatal(err)
	}
	if loss0 <= 0 || math.IsNaN(float64(loss0)) {
		t.Fatalf("bad initial loss %g", loss0)
	}
	opt := NewAdamW(DefaultAdamW(0.01), tm.Params())
	opt.Step(tm.Params(), grads)

	loss1, _, err := tm.ForwardBackward(ex)
	if err != nil {
		t.Fatal(err)
	}
	if !(loss1 < loss0) {
		t.Fatalf("loss did not decrease: before=%g after=%g", loss0, loss1)
	}
}

func TestTrainableGradientsAreNonZero(t *testing.T) {
	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 3)
	tm := NewQwenModel(cfg, w, DefaultLoRA(), 13)
	ex := Example{Tokens: []int32{1, 2, 3, 4}, LossMask: []bool{false, true, true, true}}
	_, grads, err := tm.ForwardBackward(ex)
	if err != nil {
		t.Fatal(err)
	}
	nonzero := 0
	for _, g := range grads {
		if g == nil {
			continue
		}
		for _, v := range g.F32 {
			if v != 0 {
				nonzero++
				break
			}
		}
	}
	// Attention/MLP adapters on both layer types must receive gradients.
	if nonzero < 6 {
		t.Fatalf("only %d/%d adapter tensors received a gradient", nonzero, len(grads))
	}
}

func TestTrainableWeightQuantizedParity(t *testing.T) {
	// The base weights need not be f32: a quantized (f16) base must still run.
	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 4)
	for il := range w.Layers {
		lw := &w.Layers[il]
		for _, ptr := range []**qwen38.Weight{&lw.FfnGate, &lw.FfnUp, &lw.FfnDown} {
			f32, err := (*ptr).Dequant()
			if err != nil {
				t.Fatal(err)
			}
			raw, err := quant.Quantize(quant.TypeF16, f32.F32)
			if err != nil {
				t.Fatal(err)
			}
			*ptr = &qwen38.Weight{Typ: quant.TypeF16, Dims: (*ptr).Dims, Raw: raw}
		}
	}
	tm := NewQwenModel(cfg, w, DefaultLoRA(), 17)
	got := tm.Logits([]int32{1, 2, 3, 4})
	if len(got.F32) == 0 {
		t.Fatal("no logits from quantized base")
	}
	for i, v := range got.F32 {
		if math.IsNaN(float64(v)) {
			t.Fatalf("NaN logit %d", i)
		}
	}
}
