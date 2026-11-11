package inference

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

var seedCounter uint64

// Chat renders the request, generates a response, and streams deltas through
// emit (which may be nil). It is safe to call concurrently; complete forward
// calls are serialized by the model/backend execution guard.
func (e *Engine) Chat(ctx context.Context, req ChatRequest, emit func(Delta) error) (*Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if e.speculative {
		return e.chatSpec(ctx, req, emit)
	}
	e.life.RLock()
	if e.closed {
		e.life.RUnlock()
		return nil, fmt.Errorf("inference: engine is closed")
	}
	defer e.life.RUnlock()

	select {
	case e.sem <- struct{}{}:
		defer func() { <-e.sem }()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	prompt, err := e.render(req)
	if err != nil {
		return nil, err
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}
	if e.contextLen > 0 && len(prompt)+maxTokens > e.contextLen {
		return nil, fmt.Errorf("inference: prompt uses %d tokens; max_tokens %d exceeds context limit %d",
			len(prompt), maxTokens, e.contextLen)
	}

	seed := req.Seed
	if seed == 0 {
		seed = time.Now().UnixNano() ^ int64(atomic.AddUint64(&seedCounter, 1))
	}
	rng := rand.New(rand.NewSource(seed))

	capacity := len(prompt) + maxTokens
	if e.contextLen > 0 && capacity > e.contextLen {
		capacity = e.contextLen
	}
	if _, ok := e.Backend.(qwen38.KVCacheBackend); ok {
		need := EstimateKVBytes(e.Cfg, capacity, e.kvType)
		if !e.reserveKV(need) {
			return nil, fmt.Errorf("inference: KV cache request %s exceeds remaining budget %s",
				FormatBytes(need), FormatBytes(e.kvBudget))
		}
		defer e.releaseKV(need)
	}
	var st *qwen38.State
	err = compute.Run(e.Backend, func() error {
		var stateErr error
		st, stateErr = qwen38.NewStateWithOptions(e.Model.Cfg, qwen38.StateOptions{
			Backend:       e.Backend,
			KVType:        e.kvType,
			Capacity:      capacity,
			DeviceForward: e.Model.DeviceForwardEnabled(),
		})
		return stateErr
	})
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = compute.Run(e.Backend, func() error {
			st.Free()
			return nil
		})
	}()
	chunk := e.prefillChunk
	if chunk <= 0 || chunk > len(prompt) {
		chunk = len(prompt)
	}
	var res *qwen38.ForwardResult
	for start := 0; start < len(prompt); start += chunk {
		end := start + chunk
		if end > len(prompt) {
			end = len(prompt)
		}
		positions := make([]int32, end-start)
		for i := range positions {
			positions[i] = int32(start + i)
		}
		// Intermediate prefill chunks only need to append their KV/state; the
		// discarded logits would otherwise be a full-vocabulary download.
		res, err = e.Model.Forward(prompt[start:end], qwen38.ForwardOptions{
			State:         st,
			Positions:     positions,
			LastTokenOnly: true,
		})
		if err != nil {
			return nil, err
		}
	}

	parser := newStreamParser(req.EnableThinking, toolArgTypes(req.Tools), emit)
	filter := newStopFilter(req.Stop, parser.Feed)
	generated := make([]int32, 0, maxTokens)
	finish := "stop"

	for i := 0; i < maxTokens; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if e.contextLen > 0 && len(prompt)+len(generated) >= e.contextLen {
			finish = "length"
			break
		}
		next := sampleToken(res.Logits.F32, rng, req.Temperature, req.TopK, req.TopP)
		if e.Vocab.IsEOS(next) || (e.hasImEnd && next == e.imEnd) {
			break
		}
		generated = append(generated, next)
		if err := filter.Write(e.Vocab.Decode([]int32{next})); err != nil {
			return nil, err
		}
		if filter.Stopped() {
			break
		}
		if i == maxTokens-1 {
			finish = "length"
			break
		}
		pos := len(prompt) + len(generated) - 1
		res, err = e.Model.Forward([]int32{next}, qwen38.ForwardOptions{
			State:         st,
			Positions:     []int32{int32(pos)},
			LastTokenOnly: true,
		})
		if err != nil {
			return nil, err
		}
	}

	if err := filter.Close(); err != nil {
		return nil, err
	}
	if err := parser.Close(); err != nil {
		return nil, err
	}

	result := parser.Result()
	if len(result.ToolCalls) > 0 {
		finish = "tool_calls"
	}
	result.FinishReason = finish
	result.Usage = Usage{
		PromptTokens:     len(prompt),
		CompletionTokens: len(generated),
	}
	result.Tokens = append([]int32(nil), generated...)
	return &result, nil
}

func sampleToken(logits []float32, rng *rand.Rand, temperature float64, topK int, topP float64) int32 {
	n := len(logits)
	if n == 0 {
		return 0
	}
	if temperature <= 0 {
		return argmax(logits)
	}

	maxLogit := float32(math.Inf(-1))
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	probs := make([]float64, n)
	var sum float64
	for i, v := range logits {
		p := math.Exp(float64(v-maxLogit) / temperature)
		probs[i] = p
		sum += p
	}
	if sum == 0 || math.IsNaN(sum) {
		return argmax(logits)
	}
	for i := range probs {
		probs[i] /= sum
	}

	if topK <= 0 || topK > n {
		topK = n
	}
	if topP <= 0 || topP > 1 {
		topP = 1
	}
	if topK == n && topP == 1 {
		return sampleAll(probs, rng)
	}

	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })
	if topK < len(order) {
		order = order[:topK]
	}
	if topP < 1 {
		var cum float64
		cut := len(order)
		for i, idx := range order {
			cum += probs[idx]
			if cum >= topP {
				cut = i + 1
				break
			}
		}
		order = order[:cut]
	}
	if len(order) == 0 {
		return int32(argmax(logits))
	}
	var total float64
	for _, idx := range order {
		total += probs[idx]
	}
	if total == 0 {
		return int32(order[0])
	}
	r := rng.Float64() * total
	var cum float64
	for _, idx := range order {
		cum += probs[idx]
		if r < cum {
			return int32(idx)
		}
	}
	return int32(order[len(order)-1])
}

func sampleAll(probs []float64, rng *rand.Rand) int32 {
	r := rng.Float64()
	var cum float64
	for i, p := range probs {
		cum += p
		if r < cum {
			return int32(i)
		}
	}
	return int32(len(probs) - 1)
}

func argmax(v []float32) int32 {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return int32(best)
}

type stopFilter struct {
	stops   []string
	maxLen  int
	buf     string
	stopped bool
	out     func(string) error
}

func newStopFilter(stops []string, out func(string) error) *stopFilter {
	cleaned := make([]string, 0, len(stops))
	maxLen := 0
	for _, s := range stops {
		if s == "" {
			continue
		}
		cleaned = append(cleaned, s)
		if len(s) > maxLen {
			maxLen = len(s)
		}
	}
	return &stopFilter{stops: cleaned, maxLen: maxLen, out: out}
}

func (f *stopFilter) Stopped() bool { return f.stopped }

func (f *stopFilter) Write(s string) error {
	if f.stopped {
		return nil
	}
	f.buf += s
	if len(f.stops) == 0 {
		return f.flushAll()
	}
	if idx := earliestStop(f.buf, f.stops); idx >= 0 {
		if idx > 0 {
			if err := f.out(f.buf[:idx]); err != nil {
				return err
			}
		}
		f.buf = ""
		f.stopped = true
		return nil
	}
	keep := f.maxLen - 1
	if keep > len(f.buf) {
		keep = len(f.buf)
	}
	emit := len(f.buf) - keep
	for emit > 0 && emit < len(f.buf) && f.buf[emit]&0xC0 == 0x80 {
		emit--
	}
	if emit > 0 {
		if err := f.out(f.buf[:emit]); err != nil {
			return err
		}
		f.buf = f.buf[emit:]
	}
	return nil
}

func (f *stopFilter) Close() error {
	if f.stopped {
		return nil
	}
	return f.flushAll()
}

func (f *stopFilter) flushAll() error {
	if f.buf == "" {
		return nil
	}
	s := f.buf
	f.buf = ""
	return f.out(s)
}

func earliestStop(s string, stops []string) int {
	best := -1
	for _, stop := range stops {
		if i := strings.Index(s, stop); i >= 0 && (best < 0 || i < best) {
			best = i
		}
	}
	return best
}
