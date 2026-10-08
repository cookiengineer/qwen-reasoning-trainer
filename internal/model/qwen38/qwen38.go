// Package qwen38 implements the Qwen3.8 (27B) hybrid transformer forward pass.
//
// The model's architecture is published as Qwen3_5ForConditionalGeneration and
// its GGUF arch string is "qwen35"; this project refers to it as Qwen3.8. Each
// block is either a full-attention decoder block or a GatedDeltaNet
// (linear-attention) block; the layer pattern is 3 linear then 1 full.
//
// The forward pass operates on host compute.Tensor values and uses the
// backend-agnostic ops in the compute package. It is the reference used to
// collect abliteration residuals and to validate inference.
package qwen38

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
)

// Config holds the derived dimensions used by the forward pass.
type Config struct {
	NEmbd   int
	NHead   int
	NHeadKV int
	HeadDim int
	NRot    int
	NFf     int
	NLayer  int
	NVocab  int

	RopeTheta float64
	RmsEps    float32

	DConv      int
	DState     int
	DInner     int
	DtRank     int
	GroupCount int

	FullAttnInterval int
	NextNPredict     int
	LayerTypes       []modelcfg.LayerType
}

// KeyDim returns the combined per-token q/k width of a linear layer.
func (c *Config) KeyDim() int { return c.DState * c.GroupCount }

// ValueDim returns the per-token value width of a linear layer.
func (c *Config) ValueDim() int { return c.DInner }

// ConvDim returns the conv channel count of a linear layer.
func (c *Config) ConvDim() int { return 2*c.KeyDim() + c.ValueDim() }

// HeadVDim returns the value head dimension of a linear layer.
func (c *Config) HeadVDim() int { return c.DInner / c.DtRank }

// TrunkLayers returns the number of non-MTP layers.
func (c *Config) TrunkLayers() int {
	if c.NextNPredict > 0 && c.NextNPredict < c.NLayer {
		return c.NLayer - c.NextNPredict
	}
	return c.NLayer
}

// IsRecurrent reports whether layer il is a linear-attention block.
func (c *Config) IsRecurrent(il int) bool {
	if il < 0 || il >= c.TrunkLayers() {
		return false
	}
	if il < len(c.LayerTypes) {
		return c.LayerTypes[il] == modelcfg.LayerLinearAttention
	}
	return (il+1)%c.FullAttnInterval != 0
}

// FromModelcfg derives the forward-pass configuration.
func FromModelcfg(mc *modelcfg.Config) (*Config, error) {
	c := &Config{
		NEmbd:            mc.EmbeddingLength,
		NHead:            mc.HeadCount,
		NHeadKV:          mc.HeadCountKV,
		HeadDim:          mc.KeyLength,
		NRot:             mc.RopeDimCount,
		NFf:              mc.FeedForwardLength,
		NLayer:           mc.BlockCount,
		NVocab:           mc.VocabSize,
		RopeTheta:        mc.RopeTheta,
		RmsEps:           float32(mc.RMSNormEps),
		DConv:            mc.SSMConvKernel,
		DState:           mc.SSMStateSize,
		DInner:           mc.SSMInnerSize,
		DtRank:           mc.SSMTimeStepRank,
		GroupCount:       mc.SSMGroupCount,
		FullAttnInterval: mc.FullAttentionInterval,
		NextNPredict:     mc.NextNPredictLayers,
		LayerTypes:       mc.LayerTypes,
	}
	if c.HeadDim == 0 && c.NHead > 0 {
		c.HeadDim = c.NEmbd / c.NHead
	}
	if c.NRot == 0 {
		c.NRot = c.HeadDim
	}
	if c.NEmbd == 0 || c.NLayer == 0 {
		return nil, fmt.Errorf("qwen38: incomplete config (n_embd=%d n_layer=%d)", c.NEmbd, c.NLayer)
	}
	return c, nil
}

// Model binds configuration and weights for a forward pass.
type Model struct {
	Cfg *Config
	W   *Weights
}

// NewModel creates a model from loaded weights.
func NewModel(w *Weights) *Model { return &Model{Cfg: w.Cfg, W: w} }

// rms applies RMS normalization over the leading dimension.
func rms(x *compute.Tensor, w []float32, eps float32) *compute.Tensor {
	return compute.RMSNorm(x, w, eps)
}

// mm multiplies weight w [K,N] by activation x [K,M], giving [N,M].
func mm(w, x *compute.Tensor) (*compute.Tensor, error) {
	return compute.MatMul(w, x)
}

func add(a, b *compute.Tensor) (*compute.Tensor, error) {
	return compute.Add(a, b)
}

func sigmoidInto(t *compute.Tensor) *compute.Tensor {
	out := t.Clone()
	for i, v := range out.F32 {
		out.F32[i] = compute.Sigmoid(v)
	}
	return out
}

func siluInto(t *compute.Tensor) *compute.Tensor {
	out := t.Clone()
	for i, v := range out.F32 {
		out.F32[i] = compute.Silu(v)
	}
	return out
}

func softplusInto(t *compute.Tensor) *compute.Tensor {
	out := t.Clone()
	for i, v := range out.F32 {
		out.F32[i] = softplus(v)
	}
	return out
}

func softplus(v float32) float32 {
	return float32(math.Log1p(math.Exp(float64(v))))
}
