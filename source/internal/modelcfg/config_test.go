package modelcfg

import (
	"bytes"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func synthGGUF(t *testing.T, recurrent []bool) []byte {
	t.Helper()
	const blocks = 8
	kvs := []gguf.KV{
		{Key: "general.architecture", Value: gguf.StringValue("qwen35")},
		{Key: "general.name", Value: gguf.StringValue("Qwen3.8-27B")},
		{Key: "general.alignment", Value: gguf.Uint32Value(32)},
		{Key: "qwen35.block_count", Value: gguf.Uint32Value(blocks)},
		{Key: "qwen35.embedding_length", Value: gguf.Uint32Value(5120)},
		{Key: "qwen35.feed_forward_length", Value: gguf.Uint32Value(17408)},
		{Key: "qwen35.attention.head_count", Value: gguf.Uint32Value(24)},
		{Key: "qwen35.attention.head_count_kv", Value: gguf.Uint32Value(4)},
		{Key: "qwen35.attention.key_length", Value: gguf.Uint32Value(256)},
		{Key: "qwen35.context_length", Value: gguf.Uint32Value(262144)},
		{Key: "qwen35.rope.freq_base", Value: gguf.Float32Value(10000000)},
		{Key: "qwen35.attention.layer_norm_rms_epsilon", Value: gguf.Float32Value(1e-6)},
		{Key: "qwen35.full_attention_interval", Value: gguf.Uint32Value(4)},
		{Key: "qwen35.ssm.conv_kernel", Value: gguf.Uint32Value(4)},
		{Key: "qwen35.ssm.inner_size", Value: gguf.Uint32Value(6144)},
		{Key: "qwen35.ssm.state_size", Value: gguf.Uint32Value(128)},
		{Key: "qwen35.ssm.time_step_rank", Value: gguf.Uint32Value(48)},
		{Key: "qwen35.ssm.group_count", Value: gguf.Uint32Value(16)},
		{Key: "tokenizer.ggml.tokens", Value: gguf.StringArray([]string{"a", "b", "c", "d"})},
	}
	if recurrent != nil {
		arr := make([]gguf.Value, len(recurrent))
		for i, r := range recurrent {
			arr[i] = gguf.Value{Type: gguf.ValueTypeBool, Bool: r}
		}
		kvs = append(kvs, gguf.KV{Key: "qwen35.attention.recurrent_layers", Value: gguf.Value{Type: gguf.ValueTypeArray, Array: arr}})
	}
	tensors := []gguf.WriteTensor{
		{Name: "token_embd.weight", Dims: []uint64{256}, Type: quant.TypeF32, Data: make([]byte, 1024)},
		{Name: "blk.0.ffn_down.weight", Dims: []uint64{256}, Type: quant.TypeQ4_K, Data: make([]byte, 144)},
		{Name: "blk.3.attn_output.weight", Dims: []uint64{32}, Type: quant.TypeQ4_0, Data: make([]byte, 18)},
		{Name: "blk.1.ffn_down.weight", Dims: []uint64{256}, Type: quant.TypeQ2_K, Data: make([]byte, 84)},
	}
	buf, err := gguf.Encode(kvs, tensors)
	if err != nil {
		t.Fatal(err)
	}
	return buf
}

func TestFromGGUF(t *testing.T) {
	buf := synthGGUF(t, nil)
	g, err := gguf.ReadFrom(bytes.NewReader(buf), int64(len(buf)))
	if err != nil {
		t.Fatal(err)
	}
	c, err := FromGGUF(g)
	if err != nil {
		t.Fatal(err)
	}
	if c.Arch != "qwen35" || c.BlockCount != 8 || c.EmbeddingLength != 5120 {
		t.Errorf("config = %+v", c)
	}
	if c.HeadCount != 24 || c.HeadCountKV != 4 || c.KeyLength != 256 {
		t.Errorf("heads = %+v", c)
	}
	if c.VocabSize != 4 {
		t.Errorf("vocab = %d, want 4", c.VocabSize)
	}
	if c.FullAttentionInterval != 4 {
		t.Errorf("interval = %d", c.FullAttentionInterval)
	}
	full, linear := 0, 0
	for i := 0; i < c.BlockCount; i++ {
		if c.IsRecurrent(i) {
			linear++
		} else {
			full++
		}
	}
	// Pattern 3 linear + 1 full per group of 4 with interval 4 -> full at 3,7.
	if full != 2 || linear != 6 {
		t.Fatalf("full=%d linear=%d, want 2/6", full, linear)
	}
	if c.IsRecurrent(3) || c.IsRecurrent(7) {
		t.Error("layers 3 and 7 must be full attention")
	}
	if !c.IsRecurrent(0) || !c.IsRecurrent(2) {
		t.Error("layers 0 and 2 must be linear attention")
	}
}

func TestExplicitRecurrentLayers(t *testing.T) {
	// Explicit array takes precedence: mark every layer full attention.
	rec := make([]bool, 8)
	buf := synthGGUF(t, rec)
	g, err := gguf.ReadFrom(bytes.NewReader(buf), int64(len(buf)))
	if err != nil {
		t.Fatal(err)
	}
	c, err := FromGGUF(g)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < c.BlockCount; i++ {
		if c.IsRecurrent(i) {
			t.Fatalf("layer %d should not be recurrent", i)
		}
	}
}

func TestMTPClassification(t *testing.T) {
	// block_count includes the MTP layers. With block_count 5, interval 4 and
	// nextn 1, layers 0..3 are trunk (full at 3) and layer 4 is an MTP block
	// that must be full attention (not linear).
	kvs := []gguf.KV{
		{Key: "general.architecture", Value: gguf.StringValue("qwen35")},
		{Key: "qwen35.block_count", Value: gguf.Uint32Value(5)},
		{Key: "qwen35.full_attention_interval", Value: gguf.Uint32Value(4)},
		{Key: "qwen35.nextn_predict_layers", Value: gguf.Uint32Value(1)},
	}
	tensors := []gguf.WriteTensor{
		{Name: "blk.4.nextn.eh_proj.weight", Dims: []uint64{256}, Type: quant.TypeQ6_K, Data: make([]byte, 210)},
	}
	buf, err := gguf.Encode(kvs, tensors)
	if err != nil {
		t.Fatal(err)
	}
	g, err := gguf.ReadFrom(bytes.NewReader(buf), int64(len(buf)))
	if err != nil {
		t.Fatal(err)
	}
	c, err := FromGGUF(g)
	if err != nil {
		t.Fatal(err)
	}
	if c.TrunkLayers() != 4 {
		t.Fatalf("trunk layers = %d, want 4", c.TrunkLayers())
	}
	if c.MTPCount != 1 {
		t.Fatalf("mtp count = %d, want 1", c.MTPCount)
	}
	if c.IsRecurrent(4) {
		t.Error("MTP layer 4 must not be recurrent")
	}
	if c.LayerTypes[4] != LayerFullAttention {
		t.Errorf("layer 4 type = %s, want full_attention", c.LayerTypes[4])
	}
}

func TestReport(t *testing.T) {
	buf := synthGGUF(t, nil)
	g, err := gguf.ReadFrom(bytes.NewReader(buf), int64(len(buf)))
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewReport(g)
	if err != nil {
		t.Fatal(err)
	}
	if r.TensorCount != 4 {
		t.Errorf("tensor count = %d, want 4", r.TensorCount)
	}
	if len(r.Types) != 4 {
		t.Errorf("distinct types = %d, want 4", len(r.Types))
	}
	// Q2_K has no decoder, so it must be listed as unsupported.
	if len(r.Unsupported) != 1 {
		t.Fatalf("unsupported = %v, want 1 entry", r.Unsupported)
	}
	if got := r.String(); len(got) == 0 {
		t.Error("empty report")
	}
}
