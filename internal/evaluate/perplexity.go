package evaluate

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// DefaultPPLWindow is the default number of tokens scored per forward pass. The
// full-vocabulary logits of one window occupy nVocab*window*4 bytes, so the
// window bounds the peak memory of a perplexity run.
const DefaultPPLWindow = 512

// PPLResult is the outcome of a perplexity evaluation.
type PPLResult struct {
	// NLL is the total negative log-likelihood over the counted tokens.
	NLL float64
	// Tokens is the number of next-token predictions that were counted.
	Tokens int
	// Perplexity is exp(NLL/Tokens).
	Perplexity float64
}

// PerplexityWindowed computes the length-normalized perplexity of tokens under
// m, processing the sequence in non-overlapping windows of at most window
// tokens. A window holds one forward pass; context does not cross a window
// boundary, so a smaller window lowers peak memory at the cost of some
// long-range context. A window <= 0 scores the whole token slice at once.
func PerplexityWindowed(m *qwen38.Model, tokens []int32, window int) (PPLResult, error) {
	if len(tokens) < 2 {
		return PPLResult{}, fmt.Errorf("evaluate: perplexity needs at least 2 tokens, got %d", len(tokens))
	}
	if window == 1 {
		return PPLResult{}, fmt.Errorf("evaluate: window must be 0 (whole input) or >= 2")
	}
	if window <= 0 || window > len(tokens) {
		window = len(tokens)
	}
	nVocab := m.Cfg.NVocab
	var nll float64
	counted := 0
	for start := 0; start < len(tokens); start += window {
		end := start + window
		if end > len(tokens) {
			end = len(tokens)
		}
		win := tokens[start:end]
		if len(win) < 2 {
			continue
		}
		res, err := m.Forward(win, qwen38.ForwardOptions{})
		if err != nil {
			return PPLResult{}, err
		}
		if got := res.Logits.Ne(0); got != nVocab {
			return PPLResult{}, fmt.Errorf("evaluate: logits vocab %d, want %d", got, nVocab)
		}
		nT := res.Logits.Ne(1)
		for t := 0; t+1 < nT && t+1 < len(win); t++ {
			nll -= logProbAt(res.Logits.F32, nVocab, t, int(win[t+1]))
			counted++
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

// logProbAt returns log softmax(logits)[target] for the column pos of a logits
// buffer with GGML layout [nVocab, nTokens] (element (v, t) at v + t*nVocab).
func logProbAt(logits []float32, nVocab, pos, target int) float64 {
	off := pos * nVocab
	if off+nVocab > len(logits) || target < 0 || target >= nVocab {
		return math.Inf(-1)
	}
	max := float32(math.Inf(-1))
	for v := 0; v < nVocab; v++ {
		if logits[off+v] > max {
			max = logits[off+v]
		}
	}
	var sum float64
	for v := 0; v < nVocab; v++ {
		sum += math.Exp(float64(logits[off+v] - max))
	}
	return float64(logits[off+target]-max) - math.Log(sum)
}
