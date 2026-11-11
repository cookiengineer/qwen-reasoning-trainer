package inference

import (
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// TestRealLongContextBench measures chunked-prefill and cached-decode
// throughput for one context preset. Each preset is intended to run in its own
// process (the Makefile target does this) so host weights are reclaimed between
// runs.
//
//	QWEN38_REAL_DEVICE=1 QWEN38_REAL_BENCH=1 QWEN38_BENCH_CONTEXT=128k \
//	  go test ./internal/inference -run TestRealLongContextBench -v
func TestRealLongContextBench(t *testing.T) {
	requireEnv(t, "QWEN38_REAL_DEVICE")
	requireEnv(t, "QWEN38_REAL_BENCH")

	contextLength := envContext(t, "QWEN38_BENCH_CONTEXT", "128k")
	decodeTokens := envInt(t, "QWEN38_BENCH_DECODE", 32)
	promptTokens := envInt(t, "QWEN38_BENCH_PROMPT", contextLength-decodeTokens)
	if promptTokens > contextLength-decodeTokens {
		promptTokens = contextLength - decodeTokens
	}
	if promptTokens < 1 {
		t.Fatalf("prompt tokens %d leave no room for %d decode tokens in context %d",
			promptTokens, decodeTokens, contextLength)
	}
	kvType := "q4_k"
	if value := envString(t, "QWEN38_BENCH_KV"); value != "" {
		kvType = value
	}

	engine, backend := loadRealEngine(t, contextLength, kvType)
	defer engine.Close()

	chunk := engine.PrefillChunk()
	if override := envInt(t, "QWEN38_BENCH_CHUNK", 0); override > 0 {
		chunk = override
	}
	if chunk > promptTokens {
		chunk = promptTokens
	}

	tokens := syntheticTokens(promptTokens, engine.Model.Cfg.NVocab)
	positions := make([]int32, len(tokens))
	for index := range positions {
		positions[index] = int32(index)
	}

	backend.ResetStats()
	startingDeviceBytes := backend.Stats().DeviceBytes

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

	var result *qwen38.ForwardResult
	prefillStart := time.Now()
	lastProgress := prefillStart
	for start := 0; start < len(tokens); start += chunk {
		end := start + chunk
		if end > len(tokens) {
			end = len(tokens)
		}
		result, err = engine.Model.Forward(tokens[start:end], qwen38.ForwardOptions{
			State:         state,
			Positions:     positions[start:end],
			LastTokenOnly: true,
		})
		if err != nil {
			t.Fatalf("prefill [%d:%d]: %v", start, end, err)
		}
		if time.Since(lastProgress) >= 15*time.Second || end == len(tokens) {
			elapsed := time.Since(prefillStart).Seconds()
			done := end
			rate := float64(done) / elapsed
			eta := 0.0
			if rate > 0 {
				eta = float64(len(tokens)-done) / rate
			}
			t.Logf("prefill progress=%d/%d (%.1f%%) %.2f tok/s eta=%.0fs",
				done, len(tokens), 100*float64(done)/float64(len(tokens)), rate, eta)
			lastProgress = time.Now()
		}
	}
	prefillSeconds := time.Since(prefillStart)

	decodeStart := time.Now()
	for index := 0; index < decodeTokens; index++ {
		if result == nil || len(result.Logits.F32) == 0 {
			t.Fatal("prefill produced no logits")
		}
		next := argmax(result.Logits.F32)
		position := len(tokens) + index
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

	stats := backend.Stats()
	rss := rssHighWaterBytes(t)
	kvEstimate := engine.EstimatedKVBytes(contextLength)
	fields := map[string]any{
		"context":                contextLength,
		"prompt_tokens":          promptTokens,
		"decode_tokens":          decodeTokens,
		"prefill_chunk":          chunk,
		"kv_type":                kvTypeName(engine.KVCacheType()),
		"kv_estimate_bytes":      kvEstimate,
		"starting_device_bytes":  startingDeviceBytes,
		"device_bytes":           stats.DeviceBytes,
		"peak_device_bytes":      stats.PeakDeviceBytes,
		"bytes_uploaded":         stats.BytesUploaded,
		"bytes_downloaded":       stats.BytesDownloaded,
		"prefill_seconds":        prefillSeconds.Seconds(),
		"prefill_tokens_per_sec": float64(promptTokens) / prefillSeconds.Seconds(),
		"decode_seconds":         decodeSeconds.Seconds(),
		"decode_tokens_per_sec":  float64(decodeTokens) / decodeSeconds.Seconds(),
		"host_rss_high_water":    rss,
		"rope_scaling":           engine.Model.Cfg.RopeScaling,
		"rope_factor":            engine.Model.Cfg.RopeScaleFactor,
		"attention_scale":        engine.Model.Cfg.AttnScale,
		"backend":                "vulkan",
		"top_dispatches":         topCounts(stats.ShaderDispatches, 12),
		"top_shader_ns":          topCounts(stats.ShaderTicks, 12),
		"cpu_record_ns":          stats.CPURecordNs,
	}
	logReal(t, "long_context_bench", fields)
	t.Logf("context=%s prompt=%d chunk=%d prefill=%.2f tok/s decode=%.2f tok/s peak=%s rss=%s",
		FormatTokens(contextLength), promptTokens, chunk,
		fields["prefill_tokens_per_sec"].(float64), fields["decode_tokens_per_sec"].(float64),
		memString(stats.PeakDeviceBytes), memString(rss))
}
