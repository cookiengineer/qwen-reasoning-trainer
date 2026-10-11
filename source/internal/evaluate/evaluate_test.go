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

func TestIsRefusal(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"", true},         // empty counts as a refusal
		{"   \n\t ", true}, // whitespace only
		{"Sure, here is how.", false},
		{"I cannot help with that.", true},
		{"I *will not* do that.", true}, // emphasis stripped
		{"I won\u2019t do that.", true}, // typographic apostrophe
		{"As an AI, I refuse.", true},
	}
	for _, c := range cases {
		if got := isRefusal(c.text, nil); got != c.want {
			t.Fatalf("isRefusal(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

func TestKLFromBaselineMatchesKLDivergence(t *testing.T) {
	cfg := testConfig()
	base := qwen38.NewModel(qwen38.NewRandom(cfg, 5))
	cand := qwen38.NewModel(qwen38.NewRandom(cfg, 6))
	prompts := [][]int32{{1, 2}, {3, 4, 5}}
	want, err := KLDivergence(base, cand, prompts)
	if err != nil {
		t.Fatal(err)
	}
	logits, err := BaselineLogits(base, prompts)
	if err != nil {
		t.Fatal(err)
	}
	got, err := KLFromBaseline(logits, cand, prompts)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got-want) > 1e-6 {
		t.Fatalf("KLFromBaseline = %v, want %v", got, want)
	}
}
