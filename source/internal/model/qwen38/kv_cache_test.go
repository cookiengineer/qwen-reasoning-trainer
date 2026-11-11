package qwen38

import (
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func TestKVCacheGrowthPreservesValues(t *testing.T) {
	cfg := tinyConfig()
	st := NewStateWithCapacity(cfg, 4)
	cache := st.KV[3]
	if cap(cache.k) == 0 {
		t.Fatal("capacity hint did not reserve KV storage")
	}
	stride := cfg.HeadDim * cfg.NHeadKV
	base := &cache.k[:cap(cache.k)][0]

	for i := 0; i < 4; i++ {
		k := compute.NewF32(cfg.HeadDim, cfg.NHeadKV, 1)
		v := compute.NewF32(cfg.HeadDim, cfg.NHeadKV, 1)
		for j := 0; j < stride; j++ {
			k.F32[j] = float32(i*100 + j)
			v.F32[j] = float32(i*100 + j + 5000)
		}
		nk, _, total := appendKVCache(cache, k, v, cfg.HeadDim, cfg.NHeadKV, 1)
		if total != i+1 {
			t.Fatalf("total = %d, want %d", total, i+1)
		}
		if &nk.F32[0] != base {
			t.Fatalf("cache reallocated within reserved capacity at token %d", i)
		}
	}

	k := compute.NewF32(cfg.HeadDim, cfg.NHeadKV, 1)
	v := compute.NewF32(cfg.HeadDim, cfg.NHeadKV, 1)
	for j := 0; j < stride; j++ {
		k.F32[j] = 999
		v.F32[j] = 888
	}
	nk, nv, total := appendKVCache(cache, k, v, cfg.HeadDim, cfg.NHeadKV, 1)
	if total != 5 {
		t.Fatalf("total = %d, want 5", total)
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < stride; j++ {
			if got, want := nk.F32[i*stride+j], float32(i*100+j); got != want {
				t.Fatalf("k[%d,%d] = %v, want %v", i, j, got, want)
			}
			if got, want := nv.F32[i*stride+j], float32(i*100+j+5000); got != want {
				t.Fatalf("v[%d,%d] = %v, want %v", i, j, got, want)
			}
		}
	}
	if nk.F32[4*stride] != 999 || nv.F32[4*stride] != 888 {
		t.Fatalf("appended token not stored: k=%v v=%v", nk.F32[4*stride], nv.F32[4*stride])
	}
}

func TestNewStateLazyKVCache(t *testing.T) {
	cfg := tinyConfig()
	st := NewState(cfg)
	for il := 0; il < cfg.TrunkLayers(); il++ {
		if cfg.IsRecurrent(il) {
			continue
		}
		if cap(st.KV[il].k) != 0 || cap(st.KV[il].v) != 0 {
			t.Fatalf("layer %d KV cache is not lazy", il)
		}
	}
}
