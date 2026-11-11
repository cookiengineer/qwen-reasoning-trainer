package inference

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync/atomic"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

func topProbability(logits []float32, temperature float64) float64 {
	if temperature <= 0 || len(logits) == 0 {
		return 1
	}
	maxLogit := float32(math.Inf(-1))
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v-maxLogit) / temperature)
	}
	if sum == 0 || math.IsNaN(sum) {
		return 0
	}
	return 1 / sum
}

func logitRow(t *compute.Tensor, row, vocab int) []float32 {
	return t.F32[row*vocab : (row+1)*vocab]
}

func (e *Engine) resetMTP(st *qwen38.State, n int) {
	if st == nil {
		return
	}
	if st.MTPKV != nil {
		st.MTPKV.SetLength(n)
	}
	if st.DeviceMTPKV != nil {
		st.DeviceMTPKV.Len = n
	}
}

// primeMTP rebuilds the MTP KV for tokens starting at position len(positions0)
// and advances the pending hidden row. It expects the MTP KV length to already
// be the position before the first token.
func (e *Engine) primeMTP(tokens []int32, res *qwen38.ForwardResult, positions []int32, st *qwen38.State) error {
	if len(tokens) == 0 {
		return nil
	}
	n := e.Model.Cfg.NEmbd
	if e.Backend == nil {
		if st.MTPPrevH == nil {
			st.MTPPrevH = make([]float32, n)
		}
		mr, err := e.Model.ForwardMTP(tokens, res.FinalHidden, st.MTPPrevH, qwen38.MTPOptions{
			Positions: positions,
		}, st)
		if err != nil {
			return err
		}
		copy(st.MTPPrevH, mr.Hidden.F32[(len(tokens)-1)*n:len(tokens)*n])
		return nil
	}
	if res.DeviceHidden == nil {
		return fmt.Errorf("inference: missing device hidden for MTP priming")
	}
	opts := qwen38.MTPOptions{
		Positions:        positions,
		DeviceHiddenRows: res.DeviceHidden,
		DevicePrevHidden: st.DeviceMTPPrevH,
	}
	mr, err := e.Model.ForwardMTP(tokens, nil, nil, opts, st)
	if err != nil {
		return err
	}
	if mr.DeviceHidden != nil {
		e.Backend.Free(mr.DeviceHidden)
	}
	last, err := e.Backend.(interface {
		SliceF32(compute.Buffer, int, []int) (compute.Buffer, error)
	}).SliceF32(res.DeviceHidden, (len(tokens)-1)*n*4, []int{n, 1})
	if err != nil {
		return err
	}
	return compute.Run(e.Backend, func() error {
		return e.Backend.Copy(st.DeviceMTPPrevH, last)
	})
}

type draftResult struct {
	drafts      []int32
	processed   int
	lastHidden  compute.Buffer
	lastHiddenH []float32
}

func (e *Engine) draftMTP(st *qwen38.State, current int32, pos int, maxDraft int, req ChatRequest) (draftResult, error) {
	n := e.Model.Cfg.NEmbd
	var res draftResult
	res.drafts = make([]int32, 0, maxDraft)
	tok := current
	hostPrev := st.MTPPrevH
	devPrev := st.DeviceMTPPrevH
	var owned compute.Buffer
	for i := 0; i < maxDraft; i++ {
		opts := qwen38.MTPOptions{
			Positions:     []int32{int32(pos + i)},
			ComputeLogits: true,
			LastTokenOnly: true,
		}
		if e.Backend != nil {
			opts.DevicePrevHidden = devPrev
		}
		mr, err := e.Model.ForwardMTP([]int32{tok}, nil, hostPrev, opts, st)
		if err != nil {
			if owned != nil {
				e.Backend.Free(owned)
			}
			return res, err
		}
		res.processed = i + 1
		if mr.DeviceHidden != nil {
			if owned != nil {
				e.Backend.Free(owned)
			}
			owned = mr.DeviceHidden
			devPrev = owned
			res.lastHidden = owned
			res.lastHiddenH = nil
		} else if mr.Hidden != nil {
			hostPrev = mr.Hidden.F32[:n]
			res.lastHiddenH = hostPrev
		}
		d := argmax(mr.Logits.F32)
		prob := topProbability(mr.Logits.F32, req.Temperature)
		if i > 0 && prob < e.draftMinProb {
			break
		}
		if e.Vocab.IsEOS(d) || (e.hasImEnd && d == e.imEnd) {
			break
		}
		res.drafts = append(res.drafts, d)
		tok = d
	}
	// On return the caller owns res.lastHidden.
	return res, nil
}

func (e *Engine) setMTPPrev(st *qwen38.State, res *qwen38.ForwardResult, idx int) error {
	n := e.Model.Cfg.NEmbd
	if e.Backend == nil {
		if st.MTPPrevH == nil {
			st.MTPPrevH = make([]float32, n)
		}
		copy(st.MTPPrevH, res.FinalHidden.F32[idx*n:(idx+1)*n])
		return nil
	}
	sl, err := e.Backend.(interface {
		SliceF32(compute.Buffer, int, []int) (compute.Buffer, error)
	}).SliceF32(res.DeviceHidden, idx*n*4, []int{n, 1})
	if err != nil {
		return err
	}
	return compute.Run(e.Backend, func() error {
		return e.Backend.Copy(st.DeviceMTPPrevH, sl)
	})
}

func (e *Engine) appendMTPDraft(st *qwen38.State, tok int32, pos int, dr draftResult) error {
	opts := qwen38.MTPOptions{Positions: []int32{int32(pos)}}
	if e.Backend == nil {
		if dr.lastHiddenH == nil {
			return fmt.Errorf("inference: missing MTP draft hidden")
		}
		mr, err := e.Model.ForwardMTP([]int32{tok}, nil, dr.lastHiddenH, opts, st)
		if err != nil {
			return err
		}
		_ = mr
		return nil
	}
	if dr.lastHidden == nil {
		return fmt.Errorf("inference: missing MTP draft hidden")
	}
	opts.DevicePrevHidden = dr.lastHidden
	mr, err := e.Model.ForwardMTP([]int32{tok}, nil, nil, opts, st)
	if err != nil {
		return err
	}
	if mr.DeviceHidden != nil {
		e.Backend.Free(mr.DeviceHidden)
	}
	return nil
}

func (e *Engine) chatSpec(ctx context.Context, req ChatRequest, emit func(Delta) error) (*Result, error) {
	if ctx == nil {
		ctx = context.Background()
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
			MTP:           e.speculative,
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
		res, err = e.Model.Forward(prompt[start:end], qwen38.ForwardOptions{
			State:         st,
			Positions:     positions,
			LastTokenOnly: true,
			CaptureHidden: true,
		})
		if err != nil {
			return nil, err
		}
		if err := e.primeMTP(prompt[start:end], res, positions, st); err != nil {
			if res.DeviceHidden != nil {
				e.Backend.Free(res.DeviceHidden)
			}
			return nil, err
		}
		if res.DeviceHidden != nil {
			e.Backend.Free(res.DeviceHidden)
		}
	}

	parser := newStreamParser(req.EnableThinking, toolArgTypes(req.Tools), emit)
	filter := newStopFilter(req.Stop, parser.Feed)
	generated := make([]int32, 0, maxTokens)
	finish := "stop"
	var tok int32
	haveTok := false
	stopped := false
	var emitErr error
	useSpec := true
	draftMax := e.draftMax
	rounds, acceptedSum := 0, 0

	emitOne := func(t int32) bool {
		if e.Vocab.IsEOS(t) || (e.hasImEnd && t == e.imEnd) {
			stopped = true
			return false
		}
		generated = append(generated, t)
		if err := filter.Write(e.Vocab.Decode([]int32{t})); err != nil {
			emitErr = err
			stopped = true
			return false
		}
		if filter.Stopped() {
			stopped = true
			return false
		}
		return true
	}
	normalStep := func(t int32, curPos int, prime bool) error {
		if !emitOne(t) {
			return nil
		}
		if len(generated) >= maxTokens {
			finish = "length"
			return nil
		}
		res, err = e.Model.Forward([]int32{t}, qwen38.ForwardOptions{
			State:         st,
			Positions:     []int32{int32(curPos)},
			LastTokenOnly: true,
			CaptureHidden: prime,
		})
		if err != nil {
			return err
		}
		if prime {
			e.resetMTP(st, curPos)
			if err := e.primeMTP([]int32{t}, res, []int32{int32(curPos)}, st); err != nil {
				if res.DeviceHidden != nil {
					e.Backend.Free(res.DeviceHidden)
				}
				return err
			}
			if res.DeviceHidden != nil {
				e.Backend.Free(res.DeviceHidden)
			}
		}
		haveTok = false
		return nil
	}

	for len(generated) < maxTokens && !stopped {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if !haveTok {
			if res == nil || res.Logits == nil || res.Logits.NumElements() == 0 {
				return nil, fmt.Errorf("inference: missing logits")
			}
			tok = sampleToken(res.Logits.F32, rng, req.Temperature, req.TopK, req.TopP)
			haveTok = true
			if e.Vocab.IsEOS(tok) || (e.hasImEnd && tok == e.imEnd) {
				break
			}
		}
		curPos := len(prompt) + len(generated)
		if e.contextLen > 0 && curPos >= e.contextLen {
			finish = "length"
			break
		}
		remaining := maxTokens - len(generated)
		if !useSpec {
			if err := normalStep(tok, curPos, false); err != nil {
				return nil, err
			}
			continue
		}
		maxDraft := draftMax
		if maxDraft > remaining-1 {
			maxDraft = remaining - 1
		}
		if e.contextLen > 0 && maxDraft > e.contextLen-curPos-1 {
			maxDraft = e.contextLen - curPos - 1
		}
		if maxDraft <= 0 {
			if err := normalStep(tok, curPos, true); err != nil {
				return nil, err
			}
			continue
		}

		dr, err := e.draftMTP(st, tok, curPos, maxDraft, req)
		if err != nil {
			return nil, err
		}
		drafts := dr.drafts
		if len(drafts) == 0 {
			if dr.lastHidden != nil {
				e.Backend.Free(dr.lastHidden)
			}
			if err := normalStep(tok, curPos, true); err != nil {
				return nil, err
			}
			continue
		}

		verifyTokens := make([]int32, 0, 1+len(drafts))
		verifyTokens = append(verifyTokens, tok)
		verifyTokens = append(verifyTokens, drafts...)
		verifyPos := make([]int32, len(verifyTokens))
		for i := range verifyPos {
			verifyPos[i] = int32(curPos + i)
		}
		if err := compute.Run(e.Backend, func() error { return st.SpecSnapshot() }); err != nil {
			return nil, err
		}
		verifyRes, err := e.Model.Forward(verifyTokens, qwen38.ForwardOptions{
			State:         st,
			Positions:     verifyPos,
			CaptureHidden: true,
		})
		if err != nil {
			return nil, err
		}

		accepted := 0
		mismatch := false
		var corrected int32
		for i := range drafts {
			y := sampleToken(logitRow(verifyRes.Logits, i, e.Model.Cfg.NVocab), rng, req.Temperature, req.TopK, req.TopP)
			if y == drafts[i] {
				accepted++
				continue
			}
			corrected = y
			mismatch = true
			break
		}
		acceptedTokens := verifyTokens[:1+accepted]
		rounds++
		acceptedSum += accepted
		if rounds >= 2 {
			avg := float64(acceptedSum) / float64(rounds)
			if avg < 0.6*float64(draftMax) {
				useSpec = false
			}
			rounds, acceptedSum = 0, 0
		}
		selected := verifyRes
		if mismatch {
			if verifyRes.DeviceHidden != nil {
				e.Backend.Free(verifyRes.DeviceHidden)
			}
			if err := compute.Run(e.Backend, func() error { return st.SpecRestore() }); err != nil {
				return nil, err
			}
			selected, err = e.Model.Forward(acceptedTokens, qwen38.ForwardOptions{
				State:         st,
				Positions:     verifyPos[:1+accepted],
				CaptureHidden: true,
			})
			if err != nil {
				return nil, err
			}
		}

		var nextTok int32
		if !mismatch {
			nextTok = sampleToken(logitRow(verifyRes.Logits, len(drafts), e.Model.Cfg.NVocab), rng, req.Temperature, req.TopK, req.TopP)
		} else {
			nextTok = corrected
		}

		for _, t := range acceptedTokens {
			if !emitOne(t) {
				break
			}
			if len(generated) >= maxTokens {
				finish = "length"
				stopped = true
				break
			}
		}
		if !stopped {
			// Keep the MTP KV built during drafting. If every draft matched and
			// the last drafted token was never fed back, append it with the last
			// MTP hidden; otherwise discard the rejected tail.
			lastAcceptedPos := curPos + accepted
			if accepted == len(drafts) && dr.processed == len(drafts) {
				if err := e.appendMTPDraft(st, drafts[len(drafts)-1], lastAcceptedPos, dr); err != nil {
					if dr.lastHidden != nil {
						e.Backend.Free(dr.lastHidden)
					}
					if selected.DeviceHidden != nil {
						e.Backend.Free(selected.DeviceHidden)
					}
					return nil, err
				}
			} else {
				e.resetMTP(st, lastAcceptedPos+1)
			}
			if err := e.setMTPPrev(st, selected, accepted); err != nil {
				if dr.lastHidden != nil {
					e.Backend.Free(dr.lastHidden)
				}
				if selected.DeviceHidden != nil {
					e.Backend.Free(selected.DeviceHidden)
				}
				return nil, err
			}
		}
		if dr.lastHidden != nil {
			e.Backend.Free(dr.lastHidden)
		}
		if selected.DeviceHidden != nil {
			e.Backend.Free(selected.DeviceHidden)
		}
		if stopped {
			if len(generated) >= maxTokens {
				finish = "length"
			}
			break
		}
		if e.Vocab.IsEOS(nextTok) || (e.hasImEnd && nextTok == e.imEnd) {
			break
		}
		tok = nextTok
		haveTok = true
	}

	if emitErr != nil {
		return nil, emitErr
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
	result.Usage = Usage{PromptTokens: len(prompt), CompletionTokens: len(generated)}
	result.Tokens = append([]int32(nil), generated...)
	return &result, nil
}
