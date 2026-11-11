package inference

import (
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
)

func real27BConfig() *modelcfg.Config {
	return &modelcfg.Config{
		BlockCount:            65,
		EmbeddingLength:       5120,
		HeadCount:             24,
		HeadCountKV:           4,
		KeyLength:             256,
		FullAttentionInterval: 4,
		NextNPredictLayers:    1,
	}
}

func TestParseContextLength(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"128k", 128 * 1024},
		{"256K", 256 * 1024},
		{"384k", 384 * 1024},
		{"512k", 512 * 1024},
		{"524288", 524288},
		{" 8k ", 8192},
	}
	for _, tc := range cases {
		got, err := ParseContextLength(tc.in)
		if err != nil {
			t.Fatalf("%q: %v", tc.in, err)
		}
		if got != tc.want {
			t.Fatalf("%q = %d, want %d", tc.in, got, tc.want)
		}
	}
	for _, in := range []string{"", "0", "-1", "k", "12x", "1.5k"} {
		if _, err := ParseContextLength(in); err == nil {
			t.Fatalf("%q: expected error", in)
		}
	}
}

func TestEstimateKVBytes27B(t *testing.T) {
	cfg := real27BConfig()
	got4 := EstimateKVBytes(cfg, 512*1024, DefaultKVCacheType)
	if got4 != 9<<30 {
		t.Fatalf("q4_k 512k = %d, want %d", got4, uint64(9)<<30)
	}
	got8 := EstimateKVBytes(cfg, 512*1024, AlternativeKVCacheType)
	if got8 != 17<<30 {
		t.Fatalf("q8_0 512k = %d, want %d", got8, uint64(17)<<30)
	}
	if got := EstimateKVBytes(cfg, 128*1024, DefaultKVCacheType); got != (9<<30)/4 {
		t.Fatalf("q4_k 128k = %d, want %d", got, uint64(9<<30)/4)
	}
}

func TestAutoPrefillChunk(t *testing.T) {
	if got := AutoPrefillChunk(32*1024, compute.Capabilities{}); got != 2048 {
		t.Fatalf("32k chunk = %d", got)
	}
	if got := AutoPrefillChunk(128*1024, compute.Capabilities{}); got != 1024 {
		t.Fatalf("128k chunk = %d", got)
	}
	if got := AutoPrefillChunk(256*1024, compute.Capabilities{}); got != 512 {
		t.Fatalf("256k chunk = %d", got)
	}
	if got := AutoPrefillChunk(512*1024, compute.Capabilities{}); got != 256 {
		t.Fatalf("512k chunk = %d", got)
	}
	if got := AutoPrefillChunk(512*1024, compute.Capabilities{MemoryBytes: 8 << 30}); got != 64 {
		t.Fatalf("512k low-memory chunk = %d", got)
	}
	if got := AutoPrefillChunk(512*1024, compute.Capabilities{MemoryBytes: 80 << 30}); got != 512 {
		t.Fatalf("512k high-memory chunk = %d", got)
	}
}

func TestFormatTokens(t *testing.T) {
	if got := FormatTokens(131072); got != "128k" {
		t.Fatalf("FormatTokens = %q", got)
	}
	if got := FormatTokens(1234); got != "1234" {
		t.Fatalf("FormatTokens = %q", got)
	}
}

func TestKVBudgetReservation(t *testing.T) {
	e := &Engine{kvLimit: true, kvBudget: 10}
	if !e.reserveKV(6) {
		t.Fatal("first reservation failed")
	}
	if e.reserveKV(5) {
		t.Fatal("over-budget reservation succeeded")
	}
	e.releaseKV(4)
	if !e.reserveKV(4) {
		t.Fatal("released budget was not reusable")
	}
	e.releaseKV(6)
	if e.kvUsed != 0 {
		t.Fatalf("kvUsed = %d, want 0", e.kvUsed)
	}
}
