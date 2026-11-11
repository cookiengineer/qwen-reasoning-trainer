package evaluate

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// CachedPPLOptions configures PerplexityCached.
type CachedPPLOptions struct {
	// Backend enables device-resident state and KV caching. Nil uses the host
	// reference path with float32 host KV.
	Backend compute.Backend
	// KVType selects the quantized device KV storage type.
	KVType quant.Type
	// DeviceForward enables the fully device-resident forward path.
	DeviceForward bool
	// Capacity is the KV capacity; <= 0 uses len(tokens).
	Capacity int
	// Chunk is the prefill chunk size; <= 0 uses 512.
	Chunk int
	// SampleStride calls OnSample every N scored positions (0 disables it).
	SampleStride int
	// OnSample receives a logits column for the scored position. The slice is
	// only valid during the callback; copy it to retain.
	OnSample func(position int, logits []float32)
}

// PerplexityCached computes token-level perplexity through a real recurrent
// state, so quantized device KV caches are exercised. Predictions that would
// cross a prefill-chunk boundary are skipped, which loses at most one token per
// chunk.
func PerplexityCached(m *qwen38.Model, tokens []int32, opt CachedPPLOptions) (PPLResult, error) {
	if len(tokens) < 2 {
		return PPLResult{}, fmt.Errorf("evaluate: perplexity needs at least 2 tokens, got %d", len(tokens))
	}
	capacity := opt.Capacity
	if capacity <= 0 {
		capacity = len(tokens)
	}
	chunk := opt.Chunk
	if chunk <= 0 {
		chunk = 512
	}
	if chunk > len(tokens) {
		chunk = len(tokens)
	}
	if chunk < 2 {
		chunk = 2
	}

	var state *qwen38.State
	var err error
	if opt.DeviceForward && opt.Backend != nil {
		err = compute.Run(opt.Backend, func() error {
			var stateErr error
			state, stateErr = qwen38.NewStateWithOptions(m.Cfg, qwen38.StateOptions{
				Backend:       opt.Backend,
				KVType:        opt.KVType,
				Capacity:      capacity,
				DeviceForward: true,
			})
			return stateErr
		})
	} else {
		state = qwen38.NewState(m.Cfg)
	}
	if err != nil {
		return PPLResult{}, err
	}
	defer func() {
		if opt.DeviceForward && opt.Backend != nil {
			_ = compute.Run(opt.Backend, func() error {
				state.Free()
				return nil
			})
			return
		}
		state.Free()
	}()

	nVocab := m.Cfg.NVocab
	var nll float64
	counted := 0
	for start := 0; start < len(tokens); start += chunk {
		end := start + chunk
		if end > len(tokens) {
			end = len(tokens)
		}
		positions := make([]int32, end-start)
		for index := range positions {
			positions[index] = int32(start + index)
		}
		result, err := m.Forward(tokens[start:end], qwen38.ForwardOptions{
			State:         state,
			Positions:     positions,
			LastTokenOnly: false,
		})
		if err != nil {
			return PPLResult{}, err
		}
		if got := result.Logits.Ne(0); got != nVocab {
			return PPLResult{}, fmt.Errorf("evaluate: logits vocab %d, want %d", got, nVocab)
		}
		columns := result.Logits.Ne(1)
		for local := 0; local < columns && start+local+1 < end; local++ {
			position := start + local
			target := int(tokens[position+1])
			nll -= logProbAt(result.Logits.F32, nVocab, local, target)
			counted++
			if opt.SampleStride > 0 && opt.OnSample != nil && position%opt.SampleStride == 0 {
				offset := local * nVocab
				opt.OnSample(position, result.Logits.F32[offset:offset+nVocab])
			}
		}
	}
	if counted == 0 {
		return PPLResult{}, fmt.Errorf("evaluate: no tokens scored")
	}
	return PPLResult{
		NLL:        nll,
		Tokens:     counted,
		Perplexity: math.Exp(nll / float64(counted)),
	}, nil
}

// KLFromLogits returns KL(p || q) for two rows of raw next-token logits.
func KLFromLogits(pLogits, qLogits []float32) float64 {
	return klRow(pLogits, qLogits)
}
