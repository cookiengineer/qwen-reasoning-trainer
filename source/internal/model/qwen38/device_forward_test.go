package qwen38

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func deviceForwardConfig() *Config {
	return &Config{
		NEmbd: 512, NHead: 2, NHeadKV: 1, HeadDim: 256, NRot: 256,
		NFf: 256, NLayer: 4, NVocab: 32,
		RopeTheta: 10000, RmsEps: 1e-6,
		DConv: 4, DState: 4, DInner: 16, DtRank: 4, GroupCount: 2,
		FullAttnInterval: 4,
		LayerTypes: []modelcfg.LayerType{
			modelcfg.LayerLinearAttention,
			modelcfg.LayerLinearAttention,
			modelcfg.LayerLinearAttention,
			modelcfg.LayerFullAttention,
		},
	}
}

func TestDeviceForwardMatchesHost(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()

	cfg := deviceForwardConfig()
	w := NewRandom(cfg, 777)
	host := NewModel(w)
	device := NewModel(w)
	device.Backend = be
	if err := device.EnableDeviceForward(); err != nil {
		t.Fatal(err)
	}
	defer device.FreeDevice()

	tokens := []int32{1, 2, 3, 4, 5, 6}
	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ8_0} {
		t.Run(typ.String(), func(t *testing.T) {
			devState, err := NewStateWithOptions(cfg, StateOptions{
				Backend: be, KVType: typ, Capacity: len(tokens), DeviceForward: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			defer devState.Free()
			hs := NewState(cfg)
			for i, tok := range tokens {
				pos := []int32{int32(i)}
				want, err := host.Forward([]int32{tok}, ForwardOptions{
					State: hs, Positions: pos, LastTokenOnly: true,
				})
				if err != nil {
					t.Fatal(err)
				}
				got, err := device.Forward([]int32{tok}, ForwardOptions{
					State: devState, Positions: pos, LastTokenOnly: true,
				})
				if err != nil {
					t.Fatal(err)
				}
				for j := range want.Logits.F32 {
					tol := float32(3e-1 + 3e-1*math.Abs(float64(want.Logits.F32[j])))
					if d := float32(math.Abs(float64(got.Logits.F32[j] - want.Logits.F32[j]))); d > tol {
						t.Fatalf("%s token %d logit %d: device %v host %v (diff %v)",
							typ, i, j, got.Logits.F32[j], want.Logits.F32[j], d)
					}
				}
			}
		})
	}
}

func TestDeviceForwardChunkedPrefillMatchesHost(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()

	cfg := deviceForwardConfig()
	w := NewRandom(cfg, 888)
	host := NewModel(w)
	device := NewModel(w)
	device.Backend = be
	if err := device.EnableDeviceForward(); err != nil {
		t.Fatal(err)
	}
	defer device.FreeDevice()

	tokens := []int32{3, 1, 4, 1, 5, 9}
	want, err := host.Forward(tokens, ForwardOptions{State: NewState(cfg), LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	devState, err := NewStateWithOptions(cfg, StateOptions{
		Backend: be, KVType: quant.TypeQ8_0, Capacity: len(tokens), DeviceForward: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer devState.Free()

	var got *ForwardResult
	for start := 0; start < len(tokens); start += 2 {
		end := start + 2
		if end > len(tokens) {
			end = len(tokens)
		}
		pos := make([]int32, end-start)
		for i := range pos {
			pos[i] = int32(start + i)
		}
		got, err = device.Forward(tokens[start:end], ForwardOptions{
			State: devState, Positions: pos, LastTokenOnly: end == len(tokens),
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	for j := range want.Logits.F32 {
		tol := float32(3e-1 + 3e-1*math.Abs(float64(want.Logits.F32[j])))
		if d := float32(math.Abs(float64(got.Logits.F32[j] - want.Logits.F32[j]))); d > tol {
			t.Fatalf("logit %d: device %v host %v (diff %v)", j, got.Logits.F32[j], want.Logits.F32[j], d)
		}
	}
}

func TestDeviceForwardChunkedPrefillAttentionOnly(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()
	cfg := cachedAttentionConfig()
	w := NewRandom(cfg, 889)
	host := NewModel(w)
	device := NewModel(w)
	device.Backend = be
	if err := device.EnableDeviceForward(); err != nil {
		t.Fatal(err)
	}
	defer device.FreeDevice()
	tokens := []int32{3, 1, 4, 1, 5, 9}
	want, err := host.Forward(tokens, ForwardOptions{State: NewState(cfg), LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	devState, err := NewStateWithOptions(cfg, StateOptions{
		Backend: be, KVType: quant.TypeQ8_0, Capacity: len(tokens), DeviceForward: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer devState.Free()
	var got *ForwardResult
	for start := 0; start < len(tokens); start += 2 {
		end := start + 2
		if end > len(tokens) {
			end = len(tokens)
		}
		pos := make([]int32, end-start)
		for i := range pos {
			pos[i] = int32(start + i)
		}
		got, err = device.Forward(tokens[start:end], ForwardOptions{
			State: devState, Positions: pos, LastTokenOnly: end == len(tokens),
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	for j := range want.Logits.F32 {
		tol := float32(3e-1 + 3e-1*math.Abs(float64(want.Logits.F32[j])))
		if d := float32(math.Abs(float64(got.Logits.F32[j] - want.Logits.F32[j]))); d > tol {
			t.Fatalf("logit %d: device %v host %v (diff %v)", j, got.Logits.F32[j], want.Logits.F32[j], d)
		}
	}
}

func TestDeviceForwardChunkedGDNOnly(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()
	cfg := deviceForwardConfig()
	cfg.NLayer = 4
	cfg.FullAttnInterval = 100
	cfg.LayerTypes = []modelcfg.LayerType{
		modelcfg.LayerLinearAttention, modelcfg.LayerLinearAttention,
		modelcfg.LayerLinearAttention, modelcfg.LayerLinearAttention,
	}
	w := NewRandom(cfg, 890)
	host := NewModel(w)
	host.Backend = be
	defer host.FreeDevice()
	device := NewModel(w)
	device.Backend = be
	if err := device.EnableDeviceForward(); err != nil {
		t.Fatal(err)
	}
	defer device.FreeDevice()
	tokens := []int32{3, 1, 4, 1, 5, 9}
	want, err := host.Forward(tokens, ForwardOptions{State: NewState(cfg), LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	for chunk := 1; chunk <= 3; chunk++ {
		devState, err := NewStateWithOptions(cfg, StateOptions{Backend: be, DeviceForward: true})
		if err != nil {
			t.Fatal(err)
		}
		var got *ForwardResult
		for start := 0; start < len(tokens); start += chunk {
			end := start + chunk
			if end > len(tokens) {
				end = len(tokens)
			}
			pos := make([]int32, end-start)
			for i := range pos {
				pos[i] = int32(start + i)
			}
			got, err = device.Forward(tokens[start:end], ForwardOptions{
				State: devState, Positions: pos, LastTokenOnly: end == len(tokens),
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		for j := range want.Logits.F32 {
			tol := float32(3e-1 + 3e-1*math.Abs(float64(want.Logits.F32[j])))
			if d := float32(math.Abs(float64(got.Logits.F32[j] - want.Logits.F32[j]))); d > tol {
				devState.Free()
				t.Fatalf("chunk %d logit %d: device %v host %v (diff %v)", chunk, j, got.Logits.F32[j], want.Logits.F32[j], d)
			}
		}
		devState.Free()
	}
}

func TestDeviceForwardTwoTokenGDNParity(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()
	cfg := deviceForwardConfig()
	cfg.NLayer = 4
	cfg.FullAttnInterval = 100
	cfg.LayerTypes = []modelcfg.LayerType{
		modelcfg.LayerLinearAttention, modelcfg.LayerLinearAttention,
		modelcfg.LayerLinearAttention, modelcfg.LayerLinearAttention,
	}
	w := NewRandom(cfg, 891)
	host := NewModel(w)
	host.Backend = be
	defer host.FreeDevice()
	device := NewModel(w)
	device.Backend = be
	if err := device.EnableDeviceForward(); err != nil {
		t.Fatal(err)
	}
	defer device.FreeDevice()
	tokens := []int32{3, 1}
	want, err := host.Forward(tokens, ForwardOptions{State: NewState(cfg), LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	devState, err := NewStateWithOptions(cfg, StateOptions{Backend: be, DeviceForward: true})
	if err != nil {
		t.Fatal(err)
	}
	defer devState.Free()
	got, err := device.Forward(tokens, ForwardOptions{State: devState, LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	for j := range want.Logits.F32 {
		tol := float32(3e-1 + 3e-1*math.Abs(float64(want.Logits.F32[j])))
		if d := float32(math.Abs(float64(got.Logits.F32[j] - want.Logits.F32[j]))); d > tol {
			t.Fatalf("logit %d: device %v host %v (diff %v)", j, got.Logits.F32[j], want.Logits.F32[j], d)
		}
	}
}

func TestDeviceForwardGDNTwoTokenSelfConsistency(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()
	cfg := deviceForwardConfig()
	cfg.NLayer = 4
	cfg.FullAttnInterval = 100
	cfg.LayerTypes = []modelcfg.LayerType{
		modelcfg.LayerLinearAttention, modelcfg.LayerLinearAttention,
		modelcfg.LayerLinearAttention, modelcfg.LayerLinearAttention,
	}
	w := NewRandom(cfg, 892)
	device := NewModel(w)
	device.Backend = be
	if err := device.EnableDeviceForward(); err != nil {
		t.Fatal(err)
	}
	defer device.FreeDevice()
	tokens := []int32{3, 1}

	s1, _ := NewStateWithOptions(cfg, StateOptions{Backend: be, DeviceForward: true})
	defer s1.Free()
	two, err := device.Forward(tokens, ForwardOptions{State: s1, LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}

	s2, _ := NewStateWithOptions(cfg, StateOptions{Backend: be, DeviceForward: true})
	defer s2.Free()
	if _, err := device.Forward(tokens[:1], ForwardOptions{State: s2, Positions: []int32{0}, LastTokenOnly: true}); err != nil {
		t.Fatal(err)
	}
	one, err := device.Forward(tokens[1:], ForwardOptions{State: s2, Positions: []int32{1}, LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	for j := range two.Logits.F32 {
		if d := math.Abs(float64(two.Logits.F32[j] - one.Logits.F32[j])); d > 1e-3 {
			t.Fatalf("logit %d: two-token %v one-token %v (diff %v)", j, two.Logits.F32[j], one.Logits.F32[j], d)
		}
	}
}
