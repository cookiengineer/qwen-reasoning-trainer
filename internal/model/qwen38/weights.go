package qwen38

import (
	"fmt"
	"math/rand"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// LayerWeights holds the tensors of one transformer block.
type LayerWeights struct {
	AttnNorm     *compute.Tensor
	PostAttnNorm *compute.Tensor

	// Full-attention layers.
	AttnQ      *compute.Tensor
	AttnK      *compute.Tensor
	AttnV      *compute.Tensor
	AttnOutput *compute.Tensor
	AttnQNorm  *compute.Tensor
	AttnKNorm  *compute.Tensor

	// Linear-attention (GatedDeltaNet) layers.
	AttnQKV   *compute.Tensor
	AttnGate  *compute.Tensor
	SSMOut    *compute.Tensor
	SSMAlpha  *compute.Tensor
	SSMBeta   *compute.Tensor
	SSMConv1d *compute.Tensor
	SSMNorm   *compute.Tensor
	SSMA      *compute.Tensor
	SSMDt     *compute.Tensor

	// Feed-forward.
	FfnGate *compute.Tensor
	FfnUp   *compute.Tensor
	FfnDown *compute.Tensor
}

// Weights holds all model parameters as float32 host tensors.
//
// Note: dequantizing a 27B model to float32 requires roughly 108 GiB. Load
// options for quantized or float16 storage will be added in a later task; this
// path is intended for small models and for validation of the forward pass.
type Weights struct {
	Cfg        *Config
	TokenEmbd  *compute.Tensor
	OutputNorm *compute.Tensor
	Output     *compute.Tensor
	Layers     []LayerWeights
}

func loadTensor(g *gguf.File, name string) (*compute.Tensor, error) {
	ti, ok := g.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("qwen38: missing tensor %q", name)
	}
	f32, err := g.ReadTensorF32(ti)
	if err != nil {
		return nil, err
	}
	dims := make([]int, len(ti.Dims))
	for i, d := range ti.Dims {
		dims[i] = int(d)
	}
	return &compute.Tensor{Dims: dims, Type: quant.TypeF32, F32: f32}, nil
}

// LoadFromGGUF loads and dequantizes all trunk-layer weights required by the
// forward pass. MTP and vision tensors are ignored.
func LoadFromGGUF(g *gguf.File, mc *modelcfg.Config) (*Weights, error) {
	cfg, err := FromModelcfg(mc)
	if err != nil {
		return nil, err
	}
	w := &Weights{Cfg: cfg}

	if w.TokenEmbd, err = loadTensor(g, "token_embd.weight"); err != nil {
		return nil, err
	}
	if w.OutputNorm, err = loadTensor(g, "output_norm.weight"); err != nil {
		return nil, err
	}
	if ti, ok := g.Tensor("output.weight"); ok {
		if w.Output, err = loadTensor(g, "output.weight"); err != nil {
			return nil, err
		}
		_ = ti
	} else {
		w.Output = w.TokenEmbd
	}

	trunk := cfg.TrunkLayers()
	w.Layers = make([]LayerWeights, trunk)
	for il := 0; il < trunk; il++ {
		lw := LayerWeights{}
		prefix := fmt.Sprintf("blk.%d.", il)
		req := func(dst **compute.Tensor, suffix string) error {
			t, err := loadTensor(g, prefix+suffix)
			if err != nil {
				return err
			}
			*dst = t
			return nil
		}
		if err := req(&lw.AttnNorm, "attn_norm.weight"); err != nil {
			return nil, err
		}
		if err := req(&lw.PostAttnNorm, "post_attention_norm.weight"); err != nil {
			return nil, err
		}
		if cfg.IsRecurrent(il) {
			for _, e := range []struct {
				dst    **compute.Tensor
				suffix string
			}{
				{&lw.AttnQKV, "attn_qkv.weight"},
				{&lw.AttnGate, "attn_gate.weight"},
				{&lw.SSMOut, "ssm_out.weight"},
				{&lw.SSMAlpha, "ssm_alpha.weight"},
				{&lw.SSMBeta, "ssm_beta.weight"},
				{&lw.SSMConv1d, "ssm_conv1d.weight"},
				{&lw.SSMNorm, "ssm_norm.weight"},
				{&lw.SSMA, "ssm_a"},
				{&lw.SSMDt, "ssm_dt.bias"},
			} {
				if err := req(e.dst, e.suffix); err != nil {
					return nil, err
				}
			}
		} else {
			for _, e := range []struct {
				dst    **compute.Tensor
				suffix string
			}{
				{&lw.AttnQ, "attn_q.weight"},
				{&lw.AttnK, "attn_k.weight"},
				{&lw.AttnV, "attn_v.weight"},
				{&lw.AttnOutput, "attn_output.weight"},
				{&lw.AttnQNorm, "attn_q_norm.weight"},
				{&lw.AttnKNorm, "attn_k_norm.weight"},
			} {
				if err := req(e.dst, e.suffix); err != nil {
					return nil, err
				}
			}
		}
		for _, e := range []struct {
			dst    **compute.Tensor
			suffix string
		}{
			{&lw.FfnGate, "ffn_gate.weight"},
			{&lw.FfnUp, "ffn_up.weight"},
			{&lw.FfnDown, "ffn_down.weight"},
		} {
			if err := req(e.dst, e.suffix); err != nil {
				return nil, err
			}
		}
		w.Layers[il] = lw
	}
	return w, nil
}

// NewRandom builds a small randomly-initialized model for tests and
// validation. Weights are drawn from a normal distribution scaled by 0.02.
func NewRandom(cfg *Config, seed int64) *Weights {
	rng := rand.New(rand.NewSource(seed))
	mk := func(dims ...int) *compute.Tensor {
		t := compute.NewF32(dims...)
		for i := range t.F32 {
			t.F32[i] = float32(rng.NormFloat64()) * 0.02
		}
		return t
	}
	ones := func(n int) *compute.Tensor {
		t := compute.NewF32(n)
		for i := range t.F32 {
			t.F32[i] = 1
		}
		return t
	}
	w := &Weights{Cfg: cfg}
	w.TokenEmbd = mk(cfg.NEmbd, cfg.NVocab)
	w.OutputNorm = ones(cfg.NEmbd)
	w.Output = mk(cfg.NEmbd, cfg.NVocab)
	trunk := cfg.TrunkLayers()
	w.Layers = make([]LayerWeights, trunk)
	for il := 0; il < trunk; il++ {
		lw := LayerWeights{}
		lw.AttnNorm = ones(cfg.NEmbd)
		lw.PostAttnNorm = ones(cfg.NEmbd)
		if cfg.IsRecurrent(il) {
			lw.AttnQKV = mk(cfg.NEmbd, cfg.ConvDim())
			lw.AttnGate = mk(cfg.NEmbd, cfg.ValueDim())
			lw.SSMOut = mk(cfg.ValueDim(), cfg.NEmbd)
			lw.SSMAlpha = mk(cfg.NEmbd, cfg.DtRank)
			lw.SSMBeta = mk(cfg.NEmbd, cfg.DtRank)
			lw.SSMConv1d = mk(cfg.DConv, cfg.ConvDim())
			lw.SSMNorm = ones(cfg.HeadVDim())
			lw.SSMA = ones(cfg.DtRank)
			lw.SSMDt = mk(cfg.DtRank)
		} else {
			lw.AttnQ = mk(cfg.NEmbd, cfg.NHead*cfg.HeadDim*2)
			lw.AttnK = mk(cfg.NEmbd, cfg.NHeadKV*cfg.HeadDim)
			lw.AttnV = mk(cfg.NEmbd, cfg.NHeadKV*cfg.HeadDim)
			lw.AttnOutput = mk(cfg.NHead*cfg.HeadDim, cfg.NEmbd)
			lw.AttnQNorm = ones(cfg.HeadDim)
			lw.AttnKNorm = ones(cfg.HeadDim)
		}
		lw.FfnGate = mk(cfg.NEmbd, cfg.NFf)
		lw.FfnUp = mk(cfg.NEmbd, cfg.NFf)
		lw.FfnDown = mk(cfg.NFf, cfg.NEmbd)
		w.Layers[il] = lw
	}
	return w
}
