package evaluate

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

func TestLogProbAt(t *testing.T) {
	uniform := []float32{0, 0, 0}
	for target := 0; target < 3; target++ {
		got := logProbAt(uniform, 3, 0, target)
		want := -math.Log(3)
		if math.Abs(got-want) > 1e-12 {
			t.Fatalf("uniform logProbAt(target=%d) = %v, want %v", target, got, want)
		}
	}
	peaked := []float32{1, 0, 0}
	got := logProbAt(peaked, 3, 0, 0)
	want := 1 - math.Log(math.Exp(1)+2)
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("peaked logProbAt = %v, want %v", got, want)
	}
	if lower := logProbAt(peaked, 3, 0, 1); lower >= got {
		t.Fatalf("non-target log-prob %v should be below target %v", lower, got)
	}
}

func TestPerplexityUniformIsVocab(t *testing.T) {
	cfg := testConfig()
	w := qwen38.NewRandom(cfg, 11)
	w.Output = qwen38.NewWeightF32(w.Output.Dims, make([]float32, w.Output.NumElements()))
	m := qwen38.NewModel(w)

	tokens := []int32{0, 1, 2, 3, 4, 1, 2}
	res, err := PerplexityWindowed(m, tokens, 0)
	if err != nil {
		t.Fatal(err)
	}
	if res.Tokens != len(tokens)-1 {
		t.Fatalf("counted %d tokens, want %d", res.Tokens, len(tokens)-1)
	}
	if math.Abs(res.Perplexity-float64(cfg.NVocab)) > 1e-6 {
		t.Fatalf("uniform perplexity = %v, want %d", res.Perplexity, cfg.NVocab)
	}
	wantNLL := float64(len(tokens)-1) * math.Log(float64(cfg.NVocab))
	if math.Abs(res.NLL-wantNLL) > 1e-9 {
		t.Fatalf("NLL = %v, want %v", res.NLL, wantNLL)
	}
}

func TestPerplexityWindowedMatchesSingleWindow(t *testing.T) {
	cfg := testConfig()
	m := qwen38.NewModel(qwen38.NewRandom(cfg, 12))
	tokens := []int32{5, 4, 3, 2, 1, 0, 1, 2, 3, 4}

	all, err := PerplexityWindowed(m, tokens, 0)
	if err != nil {
		t.Fatal(err)
	}
	full, err := PerplexityWindowed(m, tokens, len(tokens))
	if err != nil {
		t.Fatal(err)
	}
	if all != full {
		t.Fatalf("window=0 %+v != window=len %+v", all, full)
	}
	if math.Abs(all.Perplexity-math.Exp(all.NLL/float64(all.Tokens))) > 1e-9 {
		t.Fatalf("perplexity is not exp(NLL/tokens): %+v", all)
	}

	// A smaller window resets context per window; each window contributes
	// len(window)-1 predictions (the first token has no predecessor to score).
	windows := []int{4, 4, 2}
	want := 0
	for _, n := range windows {
		want += n - 1
	}
	chunked, err := PerplexityWindowed(m, tokens, 4)
	if err != nil {
		t.Fatal(err)
	}
	if chunked.Tokens != want {
		t.Fatalf("chunked counted %d tokens, want %d", chunked.Tokens, want)
	}
}

func TestPerplexityTooShort(t *testing.T) {
	cfg := testConfig()
	m := qwen38.NewModel(qwen38.NewRandom(cfg, 13))
	if _, err := PerplexityWindowed(m, []int32{7}, 0); err == nil {
		t.Fatal("expected error for a single token")
	}
	if _, err := PerplexityWindowed(m, []int32{1, 2, 3}, 1); err == nil {
		t.Fatal("expected error for window=1")
	}
}
