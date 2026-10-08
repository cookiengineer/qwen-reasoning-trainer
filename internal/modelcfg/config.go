// Package modelcfg parses Qwen3.5 GGUF hyperparameters and produces an inspect
// report. Key names follow references/llama_cpp/src/llama-arch.cpp.
package modelcfg

import (
	"fmt"
	"sort"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// LayerType classifies a transformer block.
type LayerType string

const (
	// LayerFullAttention is a standard causal self-attention block.
	LayerFullAttention LayerType = "full_attention"
	// LayerLinearAttention is a GatedDeltaNet (gated delta rule) block.
	LayerLinearAttention LayerType = "linear_attention"
)

// Config holds the subset of hyperparameters relevant to this project.
type Config struct {
	Arch        string
	Name        string
	TensorCount int

	BlockCount        int
	EmbeddingLength   int
	FeedForwardLength int
	HeadCount         int
	HeadCountKV       int
	KeyLength         int
	ValueLength       int
	ContextLength     int
	VocabSize         int

	RopeTheta             float64
	RopeSections          []int
	RMSNormEps            float64
	FullAttentionInterval int
	NextNPredictLayers    int

	SSMConvKernel   int
	SSMInnerSize    int
	SSMStateSize    int
	SSMTimeStepRank int
	SSMGroupCount   int

	LayerTypes []LayerType
	MTPCount   int
}

// FromGGUF builds a Config from an open GGUF file.
func FromGGUF(g *gguf.File) (*Config, error) {
	arch, _ := g.Str("general.architecture")
	if arch == "" {
		return nil, fmt.Errorf("modelcfg: missing general.architecture")
	}
	c := &Config{Arch: arch}
	c.Name, _ = g.Str("general.name")

	key := func(suffix string) string { return arch + "." + suffix }

	c.BlockCount = int(intMeta(g, key("block_count")))
	c.EmbeddingLength = int(intMeta(g, key("embedding_length")))
	c.FeedForwardLength = int(intMeta(g, key("feed_forward_length")))
	c.HeadCount = int(intMeta(g, key("attention.head_count")))
	c.HeadCountKV = int(intMeta(g, key("attention.head_count_kv")))
	c.KeyLength = int(intMeta(g, key("attention.key_length")))
	c.ValueLength = int(intMeta(g, key("attention.value_length")))
	c.ContextLength = int(intMeta(g, key("context_length")))
	if c.KeyLength == 0 && c.HeadCount > 0 && c.EmbeddingLength > 0 {
		c.KeyLength = c.EmbeddingLength / c.HeadCount
	}
	c.RopeTheta = floatMeta(g, key("rope.freq_base"))
	c.RMSNormEps = floatMeta(g, key("attention.layer_norm_rms_epsilon"))
	c.FullAttentionInterval = int(intMeta(g, key("full_attention_interval")))
	if c.FullAttentionInterval == 0 {
		c.FullAttentionInterval = 4
	}
	c.NextNPredictLayers = int(intMeta(g, key("nextn_predict_layers")))
	c.SSMConvKernel = int(intMeta(g, key("ssm.conv_kernel")))
	c.SSMInnerSize = int(intMeta(g, key("ssm.inner_size")))
	c.SSMStateSize = int(intMeta(g, key("ssm.state_size")))
	c.SSMTimeStepRank = int(intMeta(g, key("ssm.time_step_rank")))
	c.SSMGroupCount = int(intMeta(g, key("ssm.group_count")))

	if secs, ok := g.Get(key("rope.dimension_sections")); ok && secs.Type == gguf.ValueTypeArray {
		for _, v := range secs.Array {
			if n, ok := v.AsInt(); ok {
				c.RopeSections = append(c.RopeSections, int(n))
			}
		}
	}

	if toks, ok := g.Strings("tokenizer.ggml.tokens"); ok {
		c.VocabSize = len(toks)
	}

	c.LayerTypes = classifyLayers(g, c)
	c.MTPCount = c.NextNPredictLayers
	return c, nil
}

// TrunkLayers returns the number of trunk (non-MTP) transformer layers. The
// GGUF block_count includes the MTP layers declared by nextn_predict_layers.
func (c *Config) TrunkLayers() int {
	if c.NextNPredictLayers > 0 && c.NextNPredictLayers < c.BlockCount {
		return c.BlockCount - c.NextNPredictLayers
	}
	return c.BlockCount
}

// IsRecurrent reports whether layer il is a linear-attention block. MTP layers
// are full-attention decoder blocks and are never recurrent.
func (c *Config) IsRecurrent(il int) bool {
	if il < 0 || il >= c.TrunkLayers() {
		return false
	}
	if il < len(c.LayerTypes) {
		return c.LayerTypes[il] == LayerLinearAttention
	}
	return (il+1)%c.FullAttentionInterval != 0
}

func classifyLayers(g *gguf.File, c *Config) []LayerType {
	types := make([]LayerType, c.BlockCount)
	// Prefer an explicit recurrent-layer array when present.
	if v, ok := g.Get(c.Arch + ".attention.recurrent_layers"); ok && v.Type == gguf.ValueTypeArray {
		for i := 0; i < c.BlockCount && i < len(v.Array); i++ {
			if v.Array[i].Bool {
				types[i] = LayerLinearAttention
			} else {
				types[i] = LayerFullAttention
			}
		}
	} else {
		for i := 0; i < c.BlockCount; i++ {
			if (i+1)%c.FullAttentionInterval == 0 {
				types[i] = LayerFullAttention
			} else {
				types[i] = LayerLinearAttention
			}
		}
	}
	// The trailing MTP layers are full-attention decoder blocks.
	for i := c.TrunkLayers(); i < c.BlockCount; i++ {
		types[i] = LayerFullAttention
	}
	return types
}

func intMeta(g *gguf.File, key string) int64 {
	if v, ok := g.Int(key); ok {
		return v
	}
	return 0
}

func floatMeta(g *gguf.File, key string) float64 {
	if v, ok := g.Float(key); ok {
		return v
	}
	return 0
}

// TensorStat aggregates tensor bytes by quant type.
type TensorStat struct {
	Type  quant.Type
	Count int
	Bytes int64
}

// Report summarizes a model file.
type Report struct {
	Config      *Config
	TensorCount int
	TotalBytes  int64
	Types       []TensorStat
	Unsupported []string
}

// NewReport builds an inspect report for an open GGUF file.
func NewReport(g *gguf.File) (*Report, error) {
	c, err := FromGGUF(g)
	if err != nil {
		return nil, err
	}
	r := &Report{Config: c, TensorCount: len(g.Tensors)}
	byType := map[quant.Type]*TensorStat{}
	for _, t := range g.Tensors {
		sz, err := t.ByteSize()
		if err != nil {
			return nil, fmt.Errorf("modelcfg: tensor %s: %w", t.Name, err)
		}
		r.TotalBytes += sz
		st, ok := byType[t.Type]
		if !ok {
			st = &TensorStat{Type: t.Type}
			byType[t.Type] = st
		}
		st.Count++
		st.Bytes += sz
		if !t.Type.DequantSupported() {
			r.Unsupported = append(r.Unsupported, fmt.Sprintf("%s (%s)", t.Name, t.Type))
		}
	}
	for _, st := range byType {
		r.Types = append(r.Types, *st)
	}
	sort.Slice(r.Types, func(i, j int) bool { return r.Types[i].Bytes > r.Types[j].Bytes })
	sort.Strings(r.Unsupported)
	return r, nil
}

func humanBytes(b int64) string {
	const unit = 1024.0
	f := float64(b)
	switch {
	case f >= unit*unit*unit:
		return fmt.Sprintf("%.2f GiB", f/(unit*unit*unit))
	case f >= unit*unit:
		return fmt.Sprintf("%.2f MiB", f/(unit*unit))
	case f >= unit:
		return fmt.Sprintf("%.2f KiB", f/unit)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

// Format writes a human-readable report.
func (r *Report) Format(w *strings.Builder) {
	c := r.Config
	bp := func(k, v string) { fmt.Fprintf(w, "  %-26s %s\n", k+":", v) }

	fmt.Fprintf(w, "Model\n")
	bp("architecture", c.Arch)
	if c.Name != "" {
		bp("name", c.Name)
	}
	bp("block_count", fmt.Sprintf("%d", c.BlockCount))
	bp("embedding_length", fmt.Sprintf("%d", c.EmbeddingLength))
	bp("feed_forward_length", fmt.Sprintf("%d", c.FeedForwardLength))
	bp("head_count", fmt.Sprintf("%d", c.HeadCount))
	bp("head_count_kv", fmt.Sprintf("%d", c.HeadCountKV))
	bp("key_length", fmt.Sprintf("%d", c.KeyLength))
	bp("context_length", fmt.Sprintf("%d", c.ContextLength))
	bp("vocab_size", fmt.Sprintf("%d", c.VocabSize))
	bp("rope_theta", fmt.Sprintf("%g", c.RopeTheta))
	bp("rms_norm_eps", fmt.Sprintf("%g", c.RMSNormEps))
	bp("full_attention_interval", fmt.Sprintf("%d", c.FullAttentionInterval))
	bp("ssm_conv_kernel", fmt.Sprintf("%d", c.SSMConvKernel))
	bp("ssm_inner_size", fmt.Sprintf("%d", c.SSMInnerSize))
	bp("ssm_state_size", fmt.Sprintf("%d", c.SSMStateSize))
	bp("ssm_time_step_rank", fmt.Sprintf("%d", c.SSMTimeStepRank))
	bp("ssm_group_count", fmt.Sprintf("%d", c.SSMGroupCount))
	if c.MTPCount > 0 {
		bp("mtp_layers", fmt.Sprintf("%d", c.MTPCount))
	}

	full, linear := 0, 0
	for i := 0; i < c.TrunkLayers(); i++ {
		switch c.LayerTypes[i] {
		case LayerFullAttention:
			full++
		case LayerLinearAttention:
			linear++
		}
	}
	bp("layer_types", fmt.Sprintf("%d full_attention, %d linear_attention (trunk)", full, linear))

	fmt.Fprintf(w, "\nTensors\n")
	bp("count", fmt.Sprintf("%d", r.TensorCount))
	bp("total_bytes", fmt.Sprintf("%s (%d)", humanBytes(r.TotalBytes), r.TotalBytes))
	for _, st := range r.Types {
		fmt.Fprintf(w, "    %-10s %5d tensors  %s\n", st.Type, st.Count, humanBytes(st.Bytes))
	}
	if len(r.Unsupported) > 0 {
		fmt.Fprintf(w, "\nUnsupported tensor types (dequant not implemented):\n")
		for _, u := range r.Unsupported {
			fmt.Fprintf(w, "    %s\n", u)
		}
	}
}

// String returns the formatted report.
func (r *Report) String() string {
	var b strings.Builder
	r.Format(&b)
	return b.String()
}
