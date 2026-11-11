package inference

import (
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// TestRealDecodeBench isolates steady-state decode from prefill. It reports
// per-step latency and Vulkan dispatch/submit counts for a given KV length.
//
//	QWEN38_REAL_DEVICE=1 QWEN38_REAL_DECODE=1 QWEN38_DECODE_CONTEXT=4k \
//	  go test ./internal/inference -run TestRealDecodeBench -v
func TestRealDecodeBench(t *testing.T) {
	requireEnv(t, "QWEN38_REAL_DEVICE")
	requireEnv(t, "QWEN38_REAL_DECODE")

	contextLength := envContext(t, "QWEN38_DECODE_CONTEXT", "4k")
	decodeTokens := envInt(t, "QWEN38_DECODE_TOKENS", 16)
	prefillTokens := envInt(t, "QWEN38_DECODE_PREFILL", contextLength/2)
	if prefillTokens+decodeTokens > contextLength {
		prefillTokens = contextLength - decodeTokens
	}
	if prefillTokens < 1 {
		t.Fatalf("prefill tokens %d leave no room for %d decode tokens", prefillTokens, decodeTokens)
	}

	engine, backend := loadRealEngine(t, contextLength, "q4_k")
	defer engine.Close()
	backend.ResetStats()

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

	tokens := syntheticTokens(prefillTokens, engine.Model.Cfg.NVocab)
	positions := make([]int32, prefillTokens)
	for i := range positions {
		positions[i] = int32(i)
	}
	chunk := engine.PrefillChunk()
	var result *qwen38.ForwardResult
	prefillStart := time.Now()
	for start := 0; start < prefillTokens; start += chunk {
		end := start + chunk
		if end > prefillTokens {
			end = prefillTokens
		}
		result, err = engine.Model.Forward(tokens[start:end], qwen38.ForwardOptions{
			State:         state,
			Positions:     positions[start:end],
			LastTokenOnly: true,
		})
		if err != nil {
			t.Fatalf("prefill [%d:%d]: %v", start, end, err)
		}
	}
	prefillSeconds := time.Since(prefillStart)

	// Warm one decode step so lazy allocations are not counted.
	next := argmax(result.Logits.F32)
	result, err = engine.Model.Forward([]int32{next}, qwen38.ForwardOptions{
		State:         state,
		Positions:     []int32{int32(prefillTokens)},
		LastTokenOnly: true,
	})
	if err != nil {
		t.Fatalf("warm decode: %v", err)
	}
	backend.ResetStats()

	stepSeconds := make([]float64, decodeTokens)
	decodeStart := time.Now()
	for i := 0; i < decodeTokens; i++ {
		next = argmax(result.Logits.F32)
		stepStart := time.Now()
		result, err = engine.Model.Forward([]int32{next}, qwen38.ForwardOptions{
			State:         state,
			Positions:     []int32{int32(prefillTokens + 1 + i)},
			LastTokenOnly: true,
		})
		if err != nil {
			t.Fatalf("decode step %d: %v", i, err)
		}
		stepSeconds[i] = time.Since(stepStart).Seconds()
	}
	decodeSeconds := time.Since(decodeStart)
	stats := backend.Stats()
	var meanStep, minStep, maxStep float64
	if len(stepSeconds) > 0 {
		minStep, maxStep = stepSeconds[0], stepSeconds[0]
		for _, step := range stepSeconds {
			meanStep += step
			if step < minStep {
				minStep = step
			}
			if step > maxStep {
				maxStep = step
			}
		}
		meanStep /= float64(len(stepSeconds))
	}
	fields := map[string]any{
		"context":               contextLength,
		"prefill_tokens":        prefillTokens,
		"decode_tokens":         decodeTokens,
		"prefill_seconds":       prefillSeconds.Seconds(),
		"decode_seconds":        decodeSeconds.Seconds(),
		"decode_tokens_per_sec": float64(decodeTokens) / decodeSeconds.Seconds(),
		"mean_step_ms":          meanStep * 1000,
		"min_step_ms":           minStep * 1000,
		"max_step_ms":           maxStep * 1000,
		"dispatches":            stats.Dispatches,
		"submits":               stats.Submits,
		"bytes_uploaded":        stats.BytesUploaded,
		"bytes_downloaded":      stats.BytesDownloaded,
		"device_bytes":          stats.DeviceBytes,
		"peak_device_bytes":     stats.PeakDeviceBytes,
		"host_rss_high_water":   rssHighWaterBytes(t),
		"kv_type":               kvTypeName(engine.KVCacheType()),
	}
	logReal(t, "decode_bench", fields)
	t.Logf("context=%d prefill=%d decode=%.2f tok/s step=%.1fms (%.1f..%.1f) dispatches=%d submits=%d",
		contextLength, prefillTokens, fields["decode_tokens_per_sec"].(float64),
		meanStep*1000, minStep*1000, maxStep*1000, stats.Dispatches, stats.Submits)
}
