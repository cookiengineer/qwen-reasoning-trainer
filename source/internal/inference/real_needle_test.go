package inference

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

const needleMagic = "7391"

// TestRealNeedle checks long-context retrieval and, at 384k/512k, exercises the
// automatic YaRN factors 1.5 and 2.0.
//
//	QWEN38_REAL_DEVICE=1 QWEN38_REAL_NEEDLE=1 QWEN38_NEEDLE_CONTEXT=384k \
//	  go test ./internal/inference -run TestRealNeedle -v
func TestRealNeedle(t *testing.T) {
	requireEnv(t, "QWEN38_REAL_DEVICE")
	requireEnv(t, "QWEN38_REAL_NEEDLE")

	contextLength := envContext(t, "QWEN38_NEEDLE_CONTEXT", "128k")
	decodeTokens := envInt(t, "QWEN38_NEEDLE_GENERATE", 24)
	depthPercent := envInt(t, "QWEN38_NEEDLE_DEPTH_PERCENT", 50)
	if depthPercent > 100 {
		t.Fatalf("QWEN38_NEEDLE_DEPTH_PERCENT=%d: want 0..100", depthPercent)
	}
	promptTokens := contextLength - decodeTokens
	if promptTokens < 64 {
		t.Fatalf("context %d is too small for a needle prompt", contextLength)
	}

	corpus, err := os.ReadFile(realCorpusPath(t))
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	engine, _ := loadRealEngine(t, contextLength, "q4_k")
	defer engine.Close()

	corpusTokens := engine.Vocab.Encode(string(corpus))
	if len(corpusTokens) == 0 {
		t.Fatal("corpus encoded to zero tokens")
	}
	secret := engine.Vocab.Encode(fmt.Sprintf(" The secret magic number is %s. ", needleMagic))
	query := engine.Vocab.Encode("\nQuestion: What is the secret magic number?\nAnswer:")
	prompt := buildNeedlePrompt(corpusTokens, promptTokens, secret, query, depthPercent)
	if len(prompt) != promptTokens {
		t.Fatalf("needle prompt has %d tokens, want %d", len(prompt), promptTokens)
	}

	state, err := qwen38.NewStateWithOptions(engine.Model.Cfg, qwen38.StateOptions{
		Backend:       engine.Backend,
		KVType:        engine.KVCacheType(),
		Capacity:      contextLength,
		DeviceForward: true,
	})
	if err != nil {
		t.Fatalf("device state: %v", err)
	}
	defer func() {
		_ = compute.Run(engine.Backend, func() error {
			state.Free()
			return nil
		})
	}()

	chunk := engine.PrefillChunk()
	var result *qwen38.ForwardResult
	prefillStart := time.Now()
	for start := 0; start < len(prompt); start += chunk {
		end := start + chunk
		if end > len(prompt) {
			end = len(prompt)
		}
		positions := make([]int32, end-start)
		for index := range positions {
			positions[index] = int32(start + index)
		}
		result, err = engine.Model.Forward(prompt[start:end], qwen38.ForwardOptions{
			State:         state,
			Positions:     positions,
			LastTokenOnly: true,
		})
		if err != nil {
			t.Fatalf("prefill [%d:%d]: %v", start, end, err)
		}
	}
	prefillSeconds := time.Since(prefillStart)

	generated := make([]int32, 0, decodeTokens)
	decodeStart := time.Now()
	for index := 0; index < decodeTokens; index++ {
		next := argmax(result.Logits.F32)
		if engine.Vocab.IsEOS(next) {
			break
		}
		generated = append(generated, next)
		position := len(prompt) + index
		result, err = engine.Model.Forward([]int32{next}, qwen38.ForwardOptions{
			State:         state,
			Positions:     []int32{int32(position)},
			LastTokenOnly: true,
		})
		if err != nil {
			t.Fatalf("decode step %d: %v", index, err)
		}
	}
	decodeSeconds := time.Since(decodeStart)
	answer := engine.Vocab.Decode(generated)
	passed := strings.Contains(answer, needleMagic)
	logReal(t, "needle", map[string]any{
		"context":             contextLength,
		"prompt_tokens":       promptTokens,
		"depth_percent":       depthPercent,
		"prefill_chunk":       chunk,
		"prefill_seconds":     prefillSeconds.Seconds(),
		"decode_seconds":      decodeSeconds.Seconds(),
		"generated_tokens":    len(generated),
		"answer":              answer,
		"pass":                passed,
		"rope_scaling":        engine.Model.Cfg.RopeScaling,
		"rope_factor":         engine.Model.Cfg.RopeScaleFactor,
		"attention_scale":     engine.Model.Cfg.AttnScale,
		"host_rss_high_water": rssHighWaterBytes(t),
	})
	if !passed {
		t.Fatalf("needle at %s (depth %d%%) not recovered; answer %q",
			FormatTokens(contextLength), depthPercent, answer)
	}
}

// buildNeedlePrompt repeats corpus tokens to fill promptTokens, inserts the
// secret at depthPercent, and appends the query.
func buildNeedlePrompt(corpus []int32, promptTokens int, secret, query []int32, depthPercent int) []int32 {
	fillerLen := promptTokens - len(secret) - len(query)
	if fillerLen < 0 {
		fillerLen = 0
	}
	filler := make([]int32, fillerLen)
	for index := range filler {
		filler[index] = corpus[index%len(corpus)]
	}
	split := fillerLen * depthPercent / 100
	prompt := make([]int32, 0, promptTokens)
	prompt = append(prompt, filler[:split]...)
	prompt = append(prompt, secret...)
	prompt = append(prompt, filler[split:]...)
	prompt = append(prompt, query...)
	return prompt
}
