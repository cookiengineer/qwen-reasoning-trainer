package evaluate

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
)

func testConfig() *qwen38.Config {
	return &qwen38.Config{
		NEmbd: 32, NHead: 4, NHeadKV: 2, HeadDim: 8, NRot: 8, NFf: 48,
		NLayer: 4, NVocab: 50, RopeTheta: 10000, RmsEps: 1e-6,
		DConv: 4, DState: 8, DInner: 24, DtRank: 3, GroupCount: 2,
		FullAttnInterval: 4,
		LayerTypes: []modelcfg.LayerType{
			modelcfg.LayerLinearAttention, modelcfg.LayerLinearAttention,
			modelcfg.LayerLinearAttention, modelcfg.LayerFullAttention,
		},
	}
}

func testVocab() *tokenizer.Vocab {
	toks := make([]string, 50)
	for i := range toks {
		toks[i] = string(rune('a' + i%26))
	}
	return tokenizer.NewVocab(toks)
}

func TestKLDivergenceIdentity(t *testing.T) {
	cfg := testConfig()
	m := qwen38.NewModel(qwen38.NewRandom(cfg, 3))
	prompts := [][]int32{{1, 2, 3}, {4, 5}}
	kl, err := KLDivergence(m, m, prompts)
	if err != nil {
		t.Fatal(err)
	}
	if kl > 1e-9 {
		t.Fatalf("KL(x,x) = %v, want 0", kl)
	}
}

func TestRefusalRateRange(t *testing.T) {
	cfg := testConfig()
	m := qwen38.NewModel(qwen38.NewRandom(cfg, 4))
	v := testVocab()
	rate, err := RefusalRate(m, v, [][]int32{{1, 2, 3}}, 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(rate) || rate < 0 || rate > 1 {
		t.Fatalf("refusal rate out of range: %v", rate)
	}
}
