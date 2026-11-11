package evaluate

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

func TestPerplexityCachedMatchesWindowed(t *testing.T) {
	cfg := testConfig()
	m := qwen38.NewModel(qwen38.NewRandom(cfg, 31))
	tokens := []int32{5, 4, 3, 2, 1, 0, 1, 2, 3, 4}

	windowed, err := PerplexityWindowed(m, tokens, len(tokens))
	if err != nil {
		t.Fatal(err)
	}
	cached, err := PerplexityCached(m, tokens, CachedPPLOptions{Chunk: len(tokens)})
	if err != nil {
		t.Fatal(err)
	}
	if cached.Tokens != windowed.Tokens {
		t.Fatalf("cached counted %d tokens, want %d", cached.Tokens, windowed.Tokens)
	}
	if math.Abs(cached.NLL-windowed.NLL) > 1e-6 {
		t.Fatalf("cached NLL = %v, want %v", cached.NLL, windowed.NLL)
	}
	if math.Abs(cached.Perplexity-windowed.Perplexity) > 1e-6 {
		t.Fatalf("cached perplexity = %v, want %v", cached.Perplexity, windowed.Perplexity)
	}
}

func TestPerplexityCachedSkipsChunkBoundaries(t *testing.T) {
	cfg := testConfig()
	m := qwen38.NewModel(qwen38.NewRandom(cfg, 32))
	tokens := []int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

	res, err := PerplexityCached(m, tokens, CachedPPLOptions{Chunk: 4})
	if err != nil {
		t.Fatal(err)
	}
	// Chunks are [0:4], [4:8], [8:10], each contributing len-1 predictions.
	want := 3 + 3 + 1
	if res.Tokens != want {
		t.Fatalf("counted %d tokens, want %d", res.Tokens, want)
	}
}

func TestPerplexityCachedSamples(t *testing.T) {
	cfg := testConfig()
	m := qwen38.NewModel(qwen38.NewRandom(cfg, 33))
	tokens := []int32{0, 1, 2, 3, 4, 5, 6, 7}

	var positions []int
	res, err := PerplexityCached(m, tokens, CachedPPLOptions{
		Chunk:        len(tokens),
		SampleStride: 2,
		OnSample: func(position int, logits []float32) {
			if len(logits) != cfg.NVocab {
				t.Fatalf("sample logits length %d, want %d", len(logits), cfg.NVocab)
			}
			positions = append(positions, position)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	wantPositions := []int{0, 2, 4, 6}
	if len(positions) != len(wantPositions) {
		t.Fatalf("sampled positions %v, want %v", positions, wantPositions)
	}
	for index := range positions {
		if positions[index] != wantPositions[index] {
			t.Fatalf("sampled positions %v, want %v", positions, wantPositions)
		}
	}
	if res.Tokens != len(tokens)-1 {
		t.Fatalf("counted %d tokens, want %d", res.Tokens, len(tokens)-1)
	}
}
