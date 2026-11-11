package qwen38

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// TestDeviceBatchMatchesSequential checks that a small verification batch
// produces the same per-position logits as one-token-at-a-time decoding.
func TestDeviceBatchMatchesSequential(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()
	cfg := deviceForwardConfig()
	w := NewRandom(cfg, 404)
	m := NewModel(w)
	m.Backend = be
	if err := m.EnableDeviceForward(); err != nil {
		t.Fatal(err)
	}
	defer m.FreeDevice()
	prompt := []int32{1, 2, 3}
	tokens := []int32{5, 6, 7, 8, 9}
	pos := make([]int32, len(tokens))
	for i := range pos {
		pos[i] = int32(len(prompt) + i)
	}
	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ8_0} {
		a, err := NewStateWithOptions(cfg, StateOptions{Backend: be, KVType: typ, Capacity: 16, DeviceForward: true})
		if err != nil {
			t.Fatal(err)
		}
		b, err := NewStateWithOptions(cfg, StateOptions{Backend: be, KVType: typ, Capacity: 16, DeviceForward: true})
		if err != nil {
			a.Free()
			t.Fatal(err)
		}
		ppos := []int32{0, 1, 2}
		if _, err := m.Forward(prompt, ForwardOptions{State: a, Positions: ppos, LastTokenOnly: true}); err != nil {
			t.Fatal(err)
		}
		if _, err := m.Forward(prompt, ForwardOptions{State: b, Positions: ppos, LastTokenOnly: true}); err != nil {
			t.Fatal(err)
		}
		batch, err := m.Forward(tokens, ForwardOptions{State: a, Positions: pos, CaptureHidden: true})
		if err != nil {
			t.Fatal(err)
		}
		for i, tok := range tokens {
			seq, err := m.Forward([]int32{tok}, ForwardOptions{State: b, Positions: pos[i : i+1], LastTokenOnly: true})
			if err != nil {
				t.Fatal(err)
			}
			want := seq.Logits.F32
			got := batch.Logits.F32[i*cfg.NVocab : (i+1)*cfg.NVocab]
			for j := range want {
				tol := float32(2e-2 + 2e-2*math.Abs(float64(want[j])))
				if d := float32(math.Abs(float64(got[j] - want[j]))); d > tol {
					t.Fatalf("%s token %d logit %d: batch %v seq %v (diff %v)", typ, i, j, got[j], want[j], d)
				}
			}
		}
		if batch.DeviceHidden != nil {
			be.Free(batch.DeviceHidden)
		}
		a.Free()
		b.Free()
	}
}

// TestDeviceSpecRestore checks the speculative snapshot/restore of the GDN and
// conv state on the device.
func TestDeviceSpecRestore(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()
	cfg := deviceForwardConfig()
	w := NewRandom(cfg, 505)
	m := NewModel(w)
	m.Backend = be
	if err := m.EnableDeviceForward(); err != nil {
		t.Fatal(err)
	}
	defer m.FreeDevice()
	prompt := []int32{1, 2, 3}
	drafts := []int32{4, 5, 6, 7}
	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ8_0} {
		a, err := NewStateWithOptions(cfg, StateOptions{Backend: be, KVType: typ, Capacity: 16, DeviceForward: true})
		if err != nil {
			t.Fatal(err)
		}
		ref, err := NewStateWithOptions(cfg, StateOptions{Backend: be, KVType: typ, Capacity: 16, DeviceForward: true})
		if err != nil {
			a.Free()
			t.Fatal(err)
		}
		ppos := []int32{0, 1, 2}
		if _, err := m.Forward(prompt, ForwardOptions{State: a, Positions: ppos, LastTokenOnly: true}); err != nil {
			t.Fatal(err)
		}
		if _, err := m.Forward(prompt, ForwardOptions{State: ref, Positions: ppos, LastTokenOnly: true}); err != nil {
			t.Fatal(err)
		}
		if err := a.SpecSnapshot(); err != nil {
			t.Fatal(err)
		}
		batch := append([]int32{9}, drafts...)
		bpos := []int32{3, 4, 5, 6, 7}
		if _, err := m.Forward(batch, ForwardOptions{State: a, Positions: bpos, LastTokenOnly: false}); err != nil {
			t.Fatal(err)
		}
		if err := a.SpecRestore(); err != nil {
			t.Fatal(err)
		}
		got, err := m.Forward([]int32{9}, ForwardOptions{State: a, Positions: []int32{3}, LastTokenOnly: true})
		if err != nil {
			t.Fatal(err)
		}
		want, err := m.Forward([]int32{9}, ForwardOptions{State: ref, Positions: []int32{3}, LastTokenOnly: true})
		if err != nil {
			t.Fatal(err)
		}
		for i := range want.Logits.F32 {
			if got.Logits.F32[i] != want.Logits.F32[i] {
				t.Fatalf("%s logit %d: restored %v reference %v", typ, i, got.Logits.F32[i], want.Logits.F32[i])
			}
		}
		a.Free()
		ref.Free()
	}
}
