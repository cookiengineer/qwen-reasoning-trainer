package qwen38

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
)

func tinyConfig() *Config {
	return &Config{
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

func TestForwardShapes(t *testing.T) {
	cfg := tinyConfig()
	m := NewModel(NewRandom(cfg, 1))
	res, err := m.Forward([]int32{1, 2, 3, 4}, ForwardOptions{RecordHidden: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Logits.Ne(0) != cfg.NVocab || res.Logits.Ne(1) != 4 {
		t.Fatalf("logits dims = %v", res.Logits.Dims)
	}
	for i, v := range res.Logits.F32 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logits[%d] = %v", i, v)
		}
	}
	if len(res.Hidden) != cfg.TrunkLayers()+1 {
		t.Fatalf("hidden count = %d, want %d", len(res.Hidden), cfg.TrunkLayers()+1)
	}
	for i, h := range res.Hidden {
		if h.Ne(0) != cfg.NEmbd || h.Ne(1) != 4 {
			t.Errorf("hidden[%d] dims = %v", i, h.Dims)
		}
	}
}

func TestForwardDeterministic(t *testing.T) {
	cfg := tinyConfig()
	a, err := NewModel(NewRandom(cfg, 7)).Forward([]int32{3, 1, 4, 1, 5}, ForwardOptions{})
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewModel(NewRandom(cfg, 7)).Forward([]int32{3, 1, 4, 1, 5}, ForwardOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.Logits.F32 {
		if a.Logits.F32[i] != b.Logits.F32[i] {
			t.Fatalf("nondeterministic at %d: %v != %v", i, a.Logits.F32[i], b.Logits.F32[i])
		}
	}
}

// TestIncrementalMatchesFull verifies the KV cache and recurrent state: feeding
// tokens one at a time must reproduce the per-token logits of a full-prompt pass.
func TestIncrementalMatchesFull(t *testing.T) {
	cfg := tinyConfig()
	m := NewModel(NewRandom(cfg, 42))
	tokens := []int32{1, 5, 2, 7, 3, 9}

	full, err := m.Forward(tokens, ForwardOptions{})
	if err != nil {
		t.Fatal(err)
	}

	st := NewState(cfg)
	for i, tok := range tokens {
		res, err := m.Forward([]int32{tok}, ForwardOptions{State: st, Positions: []int32{int32(i)}})
		if err != nil {
			t.Fatal(err)
		}
		vocab := cfg.NVocab
		for v := 0; v < vocab; v++ {
			want := full.Logits.F32[v+i*vocab]
			got := res.Logits.F32[v]
			if math.Abs(float64(got-want)) > 2e-3+2e-3*math.Abs(float64(want)) {
				t.Fatalf("token %d logit %d: incremental %v != full %v", i, v, got, want)
			}
		}
	}
}

func TestRandomFromGGUFConfigShape(t *testing.T) {
	// Ensure FromModelcfg maps the real hyperparameters consistently.
	mc := &modelcfg.Config{
		EmbeddingLength: 5120, HeadCount: 24, HeadCountKV: 4, KeyLength: 256,
		RopeDimCount: 64, FeedForwardLength: 17408, BlockCount: 65, VocabSize: 248320,
		RopeTheta: 1e7, RMSNormEps: 1e-6, SSMConvKernel: 4, SSMStateSize: 128,
		SSMInnerSize: 6144, SSMTimeStepRank: 48, SSMGroupCount: 16,
		FullAttentionInterval: 4, NextNPredictLayers: 1,
	}
	c, err := FromModelcfg(mc)
	if err != nil {
		t.Fatal(err)
	}
	if c.TrunkLayers() != 64 || c.KeyDim() != 2048 || c.ValueDim() != 6144 || c.ConvDim() != 10240 {
		t.Fatalf("derived config wrong: %+v keyDim=%d valueDim=%d convDim=%d", c, c.KeyDim(), c.ValueDim(), c.ConvDim())
	}
	if c.HeadVDim() != 128 {
		t.Fatalf("headVDim = %d, want 128", c.HeadVDim())
	}
	if c.IsRecurrent(0) != true || c.IsRecurrent(3) != false {
		t.Fatal("layer classification wrong")
	}
}
