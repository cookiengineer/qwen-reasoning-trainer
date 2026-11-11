package qwen38

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func cachedAttentionConfig() *Config {
	return &Config{
		NEmbd: 512, NHead: 2, NHeadKV: 1, HeadDim: 256, NRot: 256,
		NFf: 256, NLayer: 1, NVocab: 32,
		RopeTheta: 10000, RmsEps: 1e-6,
		DConv: 4, DState: 4, DInner: 16, DtRank: 4, GroupCount: 2,
		FullAttnInterval: 1,
		LayerTypes:       []modelcfg.LayerType{modelcfg.LayerFullAttention},
	}
}

func TestCachedAttentionStateMatchesHost(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()

	cfg := cachedAttentionConfig()
	w := NewRandom(cfg, 123)
	host := NewModel(w)
	device := NewModel(w)
	device.Backend = be
	defer device.FreeDevice()

	tokens := []int32{1, 2, 3, 4, 5}
	hostState := NewState(cfg)
	devState, err := NewStateWithOptions(cfg, StateOptions{
		Backend: be, KVType: quant.TypeQ4_K, Capacity: len(tokens) + 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer devState.Free()

	for i, tok := range tokens {
		pos := []int32{int32(i)}
		want, err := host.Forward([]int32{tok}, ForwardOptions{State: hostState, Positions: pos, LastTokenOnly: true})
		if err != nil {
			t.Fatal(err)
		}
		got, err := device.Forward([]int32{tok}, ForwardOptions{State: devState, Positions: pos, LastTokenOnly: true})
		if err != nil {
			t.Fatal(err)
		}
		for j := range want.Logits.F32 {
			tol := float32(2.5e-1 + 2.5e-1*math.Abs(float64(want.Logits.F32[j])))
			if d := float32(math.Abs(float64(got.Logits.F32[j] - want.Logits.F32[j]))); d > tol {
				t.Fatalf("token %d logit %d: device %v host %v (diff %v)", i, j, got.Logits.F32[j], want.Logits.F32[j], d)
			}
		}
	}
}

func TestCachedAttentionChunkedPrefillMatchesHost(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()

	cfg := cachedAttentionConfig()
	w := NewRandom(cfg, 321)
	host := NewModel(w)
	device := NewModel(w)
	device.Backend = be
	defer device.FreeDevice()

	tokens := []int32{3, 1, 4, 1, 5, 9}
	want, err := host.Forward(tokens, ForwardOptions{State: NewState(cfg), LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	devState, err := NewStateWithOptions(cfg, StateOptions{
		Backend: be, KVType: quant.TypeQ4_K, Capacity: len(tokens),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer devState.Free()
	got, err := device.Forward(tokens, ForwardOptions{State: devState, LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	for j := range want.Logits.F32 {
		tol := float32(2.5e-1 + 2.5e-1*math.Abs(float64(want.Logits.F32[j])))
		if d := float32(math.Abs(float64(got.Logits.F32[j] - want.Logits.F32[j]))); d > tol {
			t.Fatalf("logit %d: device %v host %v (diff %v)", j, got.Logits.F32[j], want.Logits.F32[j], d)
		}
	}
}
