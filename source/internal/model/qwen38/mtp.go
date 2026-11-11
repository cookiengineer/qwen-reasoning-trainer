package qwen38

import (
	"fmt"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// MTPOptions controls a NextN draft-head forward.
type MTPOptions struct {
	// Positions are the absolute positions of the tokens. Defaults to 0..T-1.
	Positions []int32
	// ComputeLogits runs the shared LM head. Priming only needs the KV update.
	ComputeLogits bool
	// LastTokenOnly projects only the last hidden row when ComputeLogits is set.
	LastTokenOnly bool
	// DeviceHiddenRows is the device-resident hidden rows for the tokens.
	DeviceHiddenRows compute.Buffer
	// DevicePrevHidden is the device-resident hidden row immediately before
	// tokens[0]. Either this or prevHidden must be supplied on the device path.
	DevicePrevHidden compute.Buffer
}

// MTPResult is the output of a NextN draft-head forward.
type MTPResult struct {
	// Hidden is the post-shared-norm hidden state [NEmbd, T].
	Hidden *compute.Tensor
	// Logits is [NVocab, T] (or [NVocab, 1] with LastTokenOnly).
	Logits *compute.Tensor
	// DeviceHidden is the device-resident hidden state when the device path is
	// used. The caller owns and must free it.
	DeviceHidden compute.Buffer
}

func (m *Model) mtpAsLayer() *LayerWeights {
	mtp := m.W.MTP
	return &LayerWeights{
		AttnNorm:     mtp.AttnNorm,
		PostAttnNorm: mtp.PostAttnNorm,
		AttnQ:        mtp.AttnQ,
		AttnK:        mtp.AttnK,
		AttnV:        mtp.AttnV,
		AttnOutput:   mtp.AttnOutput,
		AttnQNorm:    mtp.AttnQNorm,
		AttnKNorm:    mtp.AttnKNorm,
		FfnGate:      mtp.FfnGate,
		FfnUp:        mtp.FfnUp,
		FfnDown:      mtp.FfnDown,
	}
}

// ForwardMTP runs the single NextN block. prevHidden is the hidden state of
// the token immediately before tokens[0]; hiddenRows, when non-nil, holds the
// hidden states of tokens themselves and supplies the shifted input for tokens
// 1..T-1. On the host path the result is computed with the CPU reference ops.
func (m *Model) ForwardMTP(tokens []int32, hiddenRows *compute.Tensor, prevHidden []float32, opts MTPOptions, st *State) (*MTPResult, error) {
	if m.W.MTP == nil {
		return nil, fmt.Errorf("qwen38: model has no MTP head")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	var res *MTPResult
	err := compute.Run(m.Backend, func() error {
		if m.device != nil && st != nil && st.deviceForward {
			var err error
			res, err = m.forwardMTPDevice(tokens, hiddenRows, prevHidden, opts, st)
			return err
		}
		var err error
		res, err = m.forwardMTPHost(tokens, hiddenRows, prevHidden, opts, st)
		return err
	})
	return res, err
}

func (m *Model) forwardMTPHost(tokens []int32, hiddenRows *compute.Tensor, prevHidden []float32, opts MTPOptions, st *State) (*MTPResult, error) {
	cfg := m.Cfg
	mtp := m.W.MTP
	T := len(tokens)
	if T == 0 {
		return nil, fmt.Errorf("qwen38: empty MTP input")
	}
	if st == nil {
		st = NewState(cfg)
	}
	if st.MTPKV == nil {
		st.MTPKV = &kvCache{}
	}
	positions := opts.Positions
	if positions == nil {
		positions = make([]int32, T)
		for i := range positions {
			positions[i] = int32(i)
		}
	}
	if len(positions) != T {
		return nil, fmt.Errorf("qwen38: MTP positions length %d != tokens %d", len(positions), T)
	}
	if prevHidden != nil && len(prevHidden) != cfg.NEmbd {
		return nil, fmt.Errorf("qwen38: MTP prev hidden length %d != %d", len(prevHidden), cfg.NEmbd)
	}

	emb, err := m.W.TokenEmbd.GetRows(tokens)
	if err != nil {
		return nil, err
	}
	eNorm := rms(emb, mtp.ENorm, cfg.RmsEps)

	hPrev := compute.NewF32(cfg.NEmbd, T)
	for t := 0; t < T; t++ {
		var src []float32
		if t == 0 {
			src = prevHidden
		} else if hiddenRows != nil {
			start := (t - 1) * cfg.NEmbd
			if start+cfg.NEmbd <= len(hiddenRows.F32) {
				src = hiddenRows.F32[start : start+cfg.NEmbd]
			}
		}
		copy(hPrev.F32[t*cfg.NEmbd:(t+1)*cfg.NEmbd], src)
	}
	hNorm := rms(hPrev, mtp.HNorm, cfg.RmsEps)

	concat := compute.NewF32(2*cfg.NEmbd, T)
	for t := 0; t < T; t++ {
		copy(concat.F32[t*2*cfg.NEmbd:t*2*cfg.NEmbd+cfg.NEmbd], eNorm.F32[t*cfg.NEmbd:(t+1)*cfg.NEmbd])
		copy(concat.F32[t*2*cfg.NEmbd+cfg.NEmbd:(t+1)*2*cfg.NEmbd], hNorm.F32[t*cfg.NEmbd:(t+1)*cfg.NEmbd])
	}

	cur, err := m.mm(mtp.EHProj, concat)
	if err != nil {
		return nil, err
	}
	lw := m.mtpAsLayer()
	h := rms(cur, lw.AttnNorm, cfg.RmsEps)
	attnOut, err := m.fullAttnCache(lw, h, positions, st, st.MTPKV, -1)
	if err != nil {
		return nil, err
	}
	cur, err = compute.Add(cur, attnOut)
	if err != nil {
		return nil, err
	}
	h2 := rms(cur, lw.PostAttnNorm, cfg.RmsEps)
	ffnOut, err := m.ffn(lw, h2)
	if err != nil {
		return nil, err
	}
	cur, err = compute.Add(cur, ffnOut)
	if err != nil {
		return nil, err
	}
	hidden := rms(cur, mtp.SharedNorm, cfg.RmsEps)
	res := &MTPResult{Hidden: hidden}
	if opts.ComputeLogits {
		x := hidden
		if opts.LastTokenOnly && T > 1 {
			x = &compute.Tensor{
				Dims: []int{cfg.NEmbd, 1},
				Type: hidden.Type,
				F32:  hidden.F32[(T-1)*cfg.NEmbd : T*cfg.NEmbd],
			}
		}
		logits, err := m.mm(m.W.Output, x)
		if err != nil {
			return nil, err
		}
		res.Logits = logits
	}
	return res, nil
}

func (m *Model) forwardMTPDevice(tokens []int32, hiddenRows *compute.Tensor, prevHidden []float32, opts MTPOptions, st *State) (*MTPResult, error) {
	cfg := m.Cfg
	be := m.Backend
	d := m.device
	if d == nil || d.mtp == nil {
		return nil, fmt.Errorf("qwen38: device MTP weights missing")
	}
	T := len(tokens)
	if T == 0 {
		return nil, fmt.Errorf("qwen38: empty MTP input")
	}
	if st == nil || st.DeviceMTPKV == nil {
		return nil, fmt.Errorf("qwen38: MTP device state missing")
	}
	positions := opts.Positions
	if positions == nil {
		positions = make([]int32, T)
		for i := range positions {
			positions[i] = int32(i)
		}
	}
	if len(positions) != T {
		return nil, fmt.Errorf("qwen38: MTP positions length %d != tokens %d", len(positions), T)
	}
	prevBuf := opts.DevicePrevHidden
	if prevBuf == nil {
		if len(prevHidden) != cfg.NEmbd {
			return nil, fmt.Errorf("qwen38: MTP device prev hidden missing")
		}
		var err error
		prevBuf, err = be.Upload(&compute.Tensor{Dims: []int{cfg.NEmbd}, F32: prevHidden})
		if err != nil {
			return nil, err
		}
		defer be.Free(prevBuf)
	}
	if T > 1 && opts.DeviceHiddenRows == nil {
		return nil, fmt.Errorf("qwen38: MTP device hidden rows missing")
	}

	be.BeginScope()
	var hidden compute.Buffer
	cleanup := func() {
		if hidden != nil {
			be.Free(hidden)
		}
		be.EndScope()
	}
	emb, err := be.GetRowsWeight(d.embed, tokens)
	if err != nil {
		cleanup()
		return nil, err
	}
	rb := be.(DeviceForwardBackend)
	concat, err := rb.MTPConcat(emb, opts.DeviceHiddenRows, prevBuf, d.mtp.eNorm, d.mtp.hNorm, cfg.NEmbd, T, cfg.RmsEps)
	if err != nil {
		cleanup()
		return nil, err
	}
	cur, err := be.MatMulWeight(d.mtp.ehProj, concat)
	if err != nil {
		cleanup()
		return nil, err
	}
	h, err := be.RMSNorm(cur, d.mtp.attn.norm, cfg.RmsEps)
	if err != nil {
		cleanup()
		return nil, err
	}
	_, hidden, err = m.deviceAttnBlockCache(-1, d.mtp.attn, h, cur, d.mtp.sharedNorm, positions, st, st.DeviceMTPKV)
	if err != nil {
		cleanup()
		return nil, err
	}
	res := &MTPResult{DeviceHidden: hidden}
	if opts.ComputeLogits {
		x := hidden
		if opts.LastTokenOnly && T > 1 {
			x, err = rb.LastToken(hidden)
			if err != nil {
				cleanup()
				return nil, err
			}
		}
		lb, err := be.MatMulWeight(d.lmHead, x)
		if err != nil {
			cleanup()
			return nil, err
		}
		logits, err := be.Download(lb)
		if err != nil {
			cleanup()
			return nil, err
		}
		res.Logits = logits
	}
	be.EndScope(hidden)
	return res, nil
}
