package inference

import (
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/evaluate"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// TestRealKVQuality compares Q4_K and Q8_0 KV cache quality on WikiText-2 by
// scoring teacher-forced next tokens through the real device KV path.
//
//	QWEN38_REAL_DEVICE=1 QWEN38_REAL_QUALITY=1 QWEN38_PPL_TEXT=datasets/wikitext2-test.txt \
//	  go test ./internal/inference -run TestRealKVQuality -v
func TestRealKVQuality(t *testing.T) {
	requireEnv(t, "QWEN38_REAL_DEVICE")
	requireEnv(t, "QWEN38_REAL_QUALITY")

	corpus, err := os.ReadFile(realCorpusPath(t))
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	maxTokens := envInt(t, "QWEN38_QUALITY_TOKENS", 4096)
	chunkOverride := envInt(t, "QWEN38_QUALITY_CHUNK", 0)
	sampleStride := envInt(t, "QWEN38_QUALITY_SAMPLE_STRIDE", 128)
	// The engine context selects YaRN independently of the scored token count.
	contextLength := envContext(t, "QWEN38_QUALITY_CONTEXT", strconv.Itoa(maxTokens))

	engine, _ := loadRealEngine(t, contextLength, "q4_k")
	defer engine.Close()

	tokens := engine.Vocab.Encode(string(corpus))
	if len(tokens) < 2 {
		t.Fatalf("corpus encoded to %d tokens", len(tokens))
	}
	if len(tokens) > maxTokens {
		tokens = tokens[:maxTokens]
	}
	chunk := engine.PrefillChunk()
	if chunkOverride > 0 {
		chunk = chunkOverride
	}
	if chunk > len(tokens) {
		chunk = len(tokens)
	}

	samples := map[quant.Type]map[int][]float32{}
	results := map[quant.Type]evaluate.PPLResult{}
	for _, typ := range []quant.Type{quant.TypeQ8_0, quant.TypeQ4_K} {
		typ := typ
		t.Run(kvTypeName(typ), func(t *testing.T) {
			positions := make(map[int][]float32)
			started := time.Now()
			result, err := evaluate.PerplexityCached(engine.Model, tokens, evaluate.CachedPPLOptions{
				Backend:       engine.Backend,
				KVType:        typ,
				DeviceForward: true,
				Capacity:      len(tokens),
				Chunk:         chunk,
				SampleStride:  sampleStride,
				OnSample: func(position int, logits []float32) {
					positions[position] = append([]float32(nil), logits...)
				},
			})
			if err != nil {
				t.Fatalf("perplexity: %v", err)
			}
			elapsed := time.Since(started)
			results[typ] = result
			samples[typ] = positions
			logReal(t, "kv_quality", map[string]any{
				"kv_type":           kvTypeName(typ),
				"tokens":            result.Tokens,
				"scored_tokens":     len(tokens),
				"chunk":             chunk,
				"context":           contextLength,
				"nll":               result.NLL,
				"perplexity":        result.Perplexity,
				"seconds":           elapsed.Seconds(),
				"tokens_per_sec":    float64(result.Tokens) / elapsed.Seconds(),
				"sampled_positions": len(positions),
				"rope_scaling":      engine.Model.Cfg.RopeScaling,
				"rope_factor":       engine.Model.Cfg.RopeScaleFactor,
				"attention_scale":   engine.Model.Cfg.AttnScale,
			})
		})
	}

	q8 := results[quant.TypeQ8_0]
	q4 := results[quant.TypeQ4_K]
	if q8.Tokens == 0 || q4.Tokens == 0 {
		t.Fatal("missing KV quality result")
	}
	// KL(Q8 || Q4): Q8_0 is the higher-quality reference.
	var klSum float64
	var klCount int
	for position, q8Logits := range samples[quant.TypeQ8_0] {
		q4Logits, ok := samples[quant.TypeQ4_K][position]
		if !ok {
			continue
		}
		klSum += evaluate.KLFromLogits(q8Logits, q4Logits)
		klCount++
	}
	meanKL := 0.0
	if klCount > 0 {
		meanKL = klSum / float64(klCount)
	}
	logReal(t, "kv_quality_compare", map[string]any{
		"perplexity_q4_k":  q4.Perplexity,
		"perplexity_q8_0":  q8.Perplexity,
		"perplexity_delta": q4.Perplexity - q8.Perplexity,
		"perplexity_ratio": q4.Perplexity / q8.Perplexity,
		"mean_kl_q8_q4":    meanKL,
		"kl_samples":       klCount,
	})
	t.Logf("PPL q8_0=%.4f q4_k=%.4f (ratio %.6f) KL(q8||q4)=%.6f over %d samples",
		q8.Perplexity, q4.Perplexity, q4.Perplexity/q8.Perplexity, meanKL, klCount)
}
