package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func buildTestFile(t *testing.T) []byte {
	t.Helper()
	kvs := []KV{
		{Key: "general.architecture", Value: StringValue("qwen35")},
		{Key: "general.name", Value: StringValue("test-model")},
		{Key: "general.alignment", Value: Uint32Value(32)},
		{Key: "qwen35.block_count", Value: Uint32Value(8)},
		{Key: "qwen35.attention.head_count", Value: Uint32Value(4)},
		{Key: "qwen35.full_attention_interval", Value: Uint32Value(4)},
		{Key: "qwen35.rope.freq_base", Value: Float32Value(10000000)},
		{Key: "test.flag", Value: BoolValue(true)},
		{Key: "tokenizer.ggml.tokens", Value: StringArray([]string{"a", "b", "c"})},
	}
	data := make([]byte, 8*4)
	for i := range 8 {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i+1)))
	}
	tensors := []WriteTensor{
		{Name: "token_embd.weight", Dims: []uint64{4, 2}, Type: quant.TypeF32, Data: data},
	}
	buf, err := Encode(kvs, tensors)
	if err != nil {
		t.Fatal(err)
	}
	return buf
}

func TestRoundTrip(t *testing.T) {
	buf := buildTestFile(t)
	g, err := ReadFrom(bytes.NewReader(buf), int64(len(buf)))
	if err != nil {
		t.Fatal(err)
	}
	if g.Version != Version {
		t.Errorf("version = %d, want %d", g.Version, Version)
	}
	if g.Alignment != 32 {
		t.Errorf("alignment = %d, want 32", g.Alignment)
	}
	if arch, _ := g.Str("general.architecture"); arch != "qwen35" {
		t.Errorf("arch = %q", arch)
	}
	if n, _ := g.Uint("qwen35.block_count"); n != 8 {
		t.Errorf("block_count = %d, want 8", n)
	}
	if f, _ := g.Float("qwen35.rope.freq_base"); f != 10000000 {
		t.Errorf("rope base = %v", f)
	}
	v, ok := g.Get("test.flag")
	if !ok || !v.Bool {
		t.Errorf("test.flag = %+v", v)
	}
	toks, ok := g.Strings("tokenizer.ggml.tokens")
	if !ok || len(toks) != 3 || toks[2] != "c" {
		t.Errorf("tokens = %v", toks)
	}

	tensor, ok := g.Tensor("token_embd.weight")
	if !ok {
		t.Fatal("tensor missing")
	}
	if tensor.Type != quant.TypeF32 || tensor.NumElements() != 8 {
		t.Errorf("tensor = %v", tensor)
	}
	if g.DataStart()%32 != 0 {
		t.Errorf("data start %d not aligned", g.DataStart())
	}
	got, err := g.ReadTensorF32(tensor)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 8 {
		if got[i] != float32(i+1) {
			t.Errorf("tensor[%d] = %v, want %d", i, got[i], i+1)
		}
	}
}

func TestBadMagic(t *testing.T) {
	buf := buildTestFile(t)
	buf[0] = 0
	_, err := ReadFrom(bytes.NewReader(buf), int64(len(buf)))
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid, got %v", err)
	}
}

func TestTruncated(t *testing.T) {
	buf := buildTestFile(t)
	_, err := ReadFrom(bytes.NewReader(buf[:16]), 16)
	if err == nil {
		t.Fatal("expected error for truncated header")
	}
}

func TestTensorOutOfBounds(t *testing.T) {
	kvs := []KV{{Key: "general.architecture", Value: StringValue("qwen35")}}
	tensors := []WriteTensor{
		{Name: "t", Dims: []uint64{4}, Type: quant.TypeF32, Data: make([]byte, 16)},
	}
	buf, err := Encode(kvs, tensors)
	if err != nil {
		t.Fatal(err)
	}
	// Corrupt the file size to simulate truncation past the tensor data.
	_, err = ReadFrom(bytes.NewReader(buf[:len(buf)-4]), int64(len(buf)-4))
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("expected ErrInvalid, got %v", err)
	}
}

func TestOpenMissing(t *testing.T) {
	if _, err := Open("/nonexistent/model.gguf"); err == nil {
		t.Fatal("expected error")
	}
}
