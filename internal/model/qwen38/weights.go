package qwen38

import (
	"fmt"
	"math/rand"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
)

// LayerWeights holds the parameters of one transformer block. Matrices stay
// quantized in *Weight; norms and biases are small and stored as float32.
type LayerWeights struct {
	AttnNorm     []float32
	PostAttnNorm []float32

	// Full-attention layers.
	AttnQ      *Weight
	AttnK      *Weight
	AttnV      *Weight
	AttnOutput *Weight
	AttnQNorm  []float32
	AttnKNorm  []float32

	// Linear-attention (GatedDeltaNet) layers.
	AttnQKV   *Weight
	AttnGate  *Weight
	SSMOut    *Weight
	SSMAlpha  *Weight
	SSMBeta   *Weight
	SSMConv1d *compute.Tensor
	SSMNorm   []float32
	SSMA      []float32
	SSMDt     []float32

	// Feed-forward.
	FfnGate *Weight
	FfnUp   *Weight
	FfnDown *Weight
}

// Weights holds all model parameters. Large matrices keep their on-disk
// quantization; the resident footprint is close to the GGUF file size.
type Weights struct {
	Cfg        *Config
	TokenEmbd  *Weight
	OutputNorm []float32
	Output     *Weight
	Layers     []LayerWeights
}

func loadVector(g *gguf.File, name string) ([]float32, error) {
	w, err := loadWeightFromGGUF(g, name)
	if err != nil {
		return nil, err
	}
	return w.Vector()
}

// LoadFromGGUF loads the trunk-layer weights needed by the forward pass. MTP
// and vision tensors are ignored.
func LoadFromGGUF(g *gguf.File, mc *modelcfg.Config) (*Weights, error) {
	cfg, err := FromModelcfg(mc)
	if err != nil {
		return nil, err
	}
	w := &Weights{Cfg: cfg}

	if w.TokenEmbd, err = loadWeightFromGGUF(g, "token_embd.weight"); err != nil {
		return nil, err
	}
	if w.OutputNorm, err = loadVector(g, "output_norm.weight"); err != nil {
		return nil, err
	}
	if _, ok := g.Tensor("output.weight"); ok {
		if w.Output, err = loadWeightFromGGUF(g, "output.weight"); err != nil {
			return nil, err
		}
	} else {
		w.Output = w.TokenEmbd
	}

	trunk := cfg.TrunkLayers()
	w.Layers = make([]LayerWeights, trunk)
	for il := 0; il < trunk; il++ {
		lw := LayerWeights{}
		prefix := fmt.Sprintf("blk.%d.", il)
		vecreq := func(dst *[]float32, suffix string) error {
			v, err := loadVector(g, prefix+suffix)
			if err != nil {
				return err
			}
			*dst = v
			return nil
		}
		wreq := func(dst **Weight, suffix string) error {
			v, err := loadWeightFromGGUF(g, prefix+suffix)
			if err != nil {
				return err
			}
			*dst = v
			return nil
		}
		if err := vecreq(&lw.AttnNorm, "attn_norm.weight"); err != nil {
			return nil, err
		}
		if err := vecreq(&lw.PostAttnNorm, "post_attention_norm.weight"); err != nil {
			return nil, err
		}
		if cfg.IsRecurrent(il) {
			for _, e := range []struct {
				dst    **Weight
				suffix string
			}{
				{&lw.AttnQKV, "attn_qkv.weight"},
				{&lw.AttnGate, "attn_gate.weight"},
				{&lw.SSMOut, "ssm_out.weight"},
				{&lw.SSMAlpha, "ssm_alpha.weight"},
				{&lw.SSMBeta, "ssm_beta.weight"},
			} {
				if err := wreq(e.dst, e.suffix); err != nil {
					return nil, err
				}
			}
			conv, err := loadVector(g, prefix+"ssm_conv1d.weight")
			if err != nil {
				return nil, err
			}
			lw.SSMConv1d = &compute.Tensor{Dims: []int{cfg.DConv, cfg.ConvDim()}, F32: conv}
			if err := vecreq(&lw.SSMNorm, "ssm_norm.weight"); err != nil {
				return nil, err
			}
			if err := vecreq(&lw.SSMA, "ssm_a"); err != nil {
				return nil, err
			}
			if err := vecreq(&lw.SSMDt, "ssm_dt.bias"); err != nil {
				return nil, err
			}
		} else {
			for _, e := range []struct {
				dst    **Weight
				suffix string
			}{
				{&lw.AttnQ, "attn_q.weight"},
				{&lw.AttnK, "attn_k.weight"},
				{&lw.AttnV, "attn_v.weight"},
				{&lw.AttnOutput, "attn_output.weight"},
			} {
				if err := wreq(e.dst, e.suffix); err != nil {
					return nil, err
				}
			}
			if err := vecreq(&lw.AttnQNorm, "attn_q_norm.weight"); err != nil {
				return nil, err
			}
			if err := vecreq(&lw.AttnKNorm, "attn_k_norm.weight"); err != nil {
				return nil, err
			}
		}
		for _, e := range []struct {
			dst    **Weight
			suffix string
		}{
			{&lw.FfnGate, "ffn_gate.weight"},
			{&lw.FfnUp, "ffn_up.weight"},
			{&lw.FfnDown, "ffn_down.weight"},
		} {
			if err := wreq(e.dst, e.suffix); err != nil {
				return nil, err
			}
		}
		w.Layers[il] = lw
	}
	return w, nil
}

// NewRandom builds a small randomly-initialized model for tests and
// validation. Weights are normal with standard deviation 0.02.
func NewRandom(cfg *Config, seed int64) *Weights {
	rng := rand.New(rand.NewSource(seed))
	mk := func(dims ...int) *Weight {
		n := 1
		for _, d := range dims {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(rng.NormFloat64()) * 0.02
		}
		return NewWeightF32(dims, data)
	}
	ones := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
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
			lw.SSMConv1d = NewWeightF32([]int{cfg.DConv, cfg.ConvDim()}, mk2(rng, cfg.DConv*cfg.ConvDim())).Tensor()
			lw.SSMNorm = ones(cfg.HeadVDim())
			lw.SSMA = ones(cfg.DtRank)
			lw.SSMDt = mk2(rng, cfg.DtRank)
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

func mk2(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64()) * 0.02
	}
	return v
}
