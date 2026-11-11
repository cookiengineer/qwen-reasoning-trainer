package inference

import (
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// TestRealDeviceLogitsMatchHost compares the device-resident forward pass with
// the host CPU reference on the real 27B model. The host and device models share
// the same loaded weight bytes; only the execution path and KV storage differ.
//
//	QWEN38_REAL_DEVICE=1 QWEN38_MODEL=... go test ./internal/inference -run TestRealDevice -v
func TestRealDeviceLogitsMatchHost(t *testing.T) {
	requireEnv(t, "QWEN38_REAL_DEVICE")

	const promptTokens = 8
	const decodeTokens = 2
	engine, _ := loadRealEngine(t, 512, "q4_k")
	defer engine.Close()

	cfg := engine.Model.Cfg
	tokens := syntheticTokens(promptTokens+decodeTokens, cfg.NVocab)
	positions := make([]int32, len(tokens))
	for index := range positions {
		positions[index] = int32(index)
	}

	// The CPU reference shares engine.Model.W; NewModel does not copy weights.
	host := qwen38.NewModel(engine.Model.W)
	hostState := qwen38.NewState(cfg)

	started := time.Now()
	wantPrefill, err := host.Forward(tokens[:promptTokens], qwen38.ForwardOptions{
		State:         hostState,
		Positions:     positions[:promptTokens],
		LastTokenOnly: true,
	})
	if err != nil {
		t.Fatalf("host prefill: %v", err)
	}
	wantDecode, err := host.Forward(tokens[promptTokens:], qwen38.ForwardOptions{
		State:         hostState,
		Positions:     positions[promptTokens:],
		LastTokenOnly: true,
	})
	if err != nil {
		t.Fatalf("host decode: %v", err)
	}
	hostElapsed := time.Since(started)
	logReal(t, "parity_host_reference", map[string]any{
		"seconds":       hostElapsed.Seconds(),
		"prompt_tokens": promptTokens,
		"decode_tokens": decodeTokens,
	})

	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ8_0} {
		typ := typ
		t.Run(kvTypeName(typ), func(t *testing.T) {
			state, err := qwen38.NewStateWithOptions(cfg, qwen38.StateOptions{
				Backend:       engine.Backend,
				KVType:        typ,
				Capacity:      len(tokens),
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

			started := time.Now()
			gotPrefill, err := engine.Model.Forward(tokens[:promptTokens], qwen38.ForwardOptions{
				State:         state,
				Positions:     positions[:promptTokens],
				LastTokenOnly: true,
			})
			if err != nil {
				t.Fatalf("device prefill: %v", err)
			}
			gotDecode, err := engine.Model.Forward(tokens[promptTokens:], qwen38.ForwardOptions{
				State:         state,
				Positions:     positions[promptTokens:],
				LastTokenOnly: true,
			})
			if err != nil {
				t.Fatalf("device decode: %v", err)
			}
			deviceElapsed := time.Since(started)

			prefillMetrics := compareLogits(t, "prefill", wantPrefill.Logits.F32, gotPrefill.Logits.F32)
			decodeMetrics := compareLogits(t, "decode", wantDecode.Logits.F32, gotDecode.Logits.F32)
			logReal(t, "parity", map[string]any{
				"kv_type":          kvTypeName(typ),
				"seconds":          deviceElapsed.Seconds(),
				"prefill_cosine":   prefillMetrics.Cosine,
				"prefill_max_abs":  prefillMetrics.MaxAbs,
				"prefill_mean_abs": prefillMetrics.MeanAbs,
				"prefill_top5":     prefillMetrics.Top5Match,
				"decode_cosine":    decodeMetrics.Cosine,
				"decode_max_abs":   decodeMetrics.MaxAbs,
				"decode_mean_abs":  decodeMetrics.MeanAbs,
				"decode_top5":      decodeMetrics.Top5Match,
			})
		})
	}
}

// TestRealDeviceMemoryEstimate checks that the resident weight upload and the
// KV allocations match the sizes derived from the loaded model and estimator.
func TestRealDeviceMemoryEstimate(t *testing.T) {
	requireEnv(t, "QWEN38_REAL_DEVICE")

	const contextLength = 512
	engine, backend := loadRealEngine(t, contextLength, "q4_k")
	defer engine.Close()

	wantWeights := loadedWeightBytes(engine.Model.W)
	afterLoad := backend.Stats()
	logReal(t, "memory_after_load", map[string]any{
		"loaded_weight_bytes": wantWeights,
		"engine_model_bytes":  engine.ModelBytes(),
		"device_bytes":        afterLoad.DeviceBytes,
		"peak_device_bytes":   afterLoad.PeakDeviceBytes,
	})
	if afterLoad.DeviceBytes < wantWeights {
		t.Fatalf("device bytes %d < uploaded weight bytes %d", afterLoad.DeviceBytes, wantWeights)
	}
	// Temporary upload staging buffers are retained in the backend pool.
	if excess := afterLoad.DeviceBytes - wantWeights; excess > 2<<30 {
		t.Fatalf("device bytes exceed uploaded weights by %s", memString(excess))
	}

	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ8_0} {
		typ := typ
		t.Run(kvTypeName(typ), func(t *testing.T) {
			before := backend.Stats().DeviceBytes
			state, err := qwen38.NewStateWithOptions(engine.Model.Cfg, qwen38.StateOptions{
				Backend:       engine.Backend,
				KVType:        typ,
				Capacity:      contextLength,
				DeviceForward: true,
			})
			if err != nil {
				t.Fatalf("device state: %v", err)
			}
			after := backend.Stats()
			wantKV := EstimateKVBytes(engine.Cfg, contextLength, typ)
			recurrent := deviceRecurrentStateBytes(engine.Model.Cfg)
			gotKV := after.DeviceBytes - before
			logReal(t, "memory_kv", map[string]any{
				"kv_type":               kvTypeName(typ),
				"context":               contextLength,
				"estimated_kv":          wantKV,
				"recurrent_state_bytes": recurrent,
				"allocated_delta":       gotKV,
				"peak_device_bytes":     after.PeakDeviceBytes,
			})
			// The first state for a fresh backend also allocates the fixed
			// conv/SSM buffers; later states reuse them from the buffer pool.
			if gotKV != wantKV && gotKV != wantKV+recurrent {
				t.Errorf("device allocation %d bytes != KV estimate %d (or %d with recurrent state)",
					gotKV, wantKV, wantKV+recurrent)
			}
			if after.PeakDeviceBytes < after.DeviceBytes {
				t.Errorf("peak device bytes %d < live %d", after.PeakDeviceBytes, after.DeviceBytes)
			}
			if after.PeakDeviceBytes > after.DeviceBytes+(64<<20) {
				t.Errorf("peak device bytes %d unexpectedly exceed live %d", after.PeakDeviceBytes, after.DeviceBytes)
			}
			_ = compute.Run(engine.Backend, func() error {
				state.Free()
				return nil
			})
		})
	}
}
