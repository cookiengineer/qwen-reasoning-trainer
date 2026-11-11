package qwen38

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func tinyMTPConfig() *Config {
	cfg := tinyConfig()
	cfg.NextNPredict = 1
	cfg.LayerTypes = append(cfg.LayerTypes, modelcfg.LayerFullAttention)
	return cfg
}

func TestMTPHostForward(t *testing.T) {
	cfg := tinyMTPConfig()
	m := NewModel(NewRandom(cfg, 11))
	st := NewState(cfg)
	tokens := []int32{1, 2, 3, 4}
	base, err := m.Forward(tokens, ForwardOptions{State: st, CaptureHidden: true})
	if err != nil {
		t.Fatal(err)
	}
	if base.FinalHidden == nil || base.FinalHidden.Ne(0) != cfg.NEmbd || base.FinalHidden.Ne(1) != len(tokens) {
		t.Fatalf("final hidden = %+v", base.FinalHidden)
	}
	pos := []int32{0, 1, 2, 3}
	res, err := m.ForwardMTP(tokens, base.FinalHidden, make([]float32, cfg.NEmbd), MTPOptions{
		Positions: pos, ComputeLogits: true, LastTokenOnly: true,
	}, st)
	if err != nil {
		t.Fatal(err)
	}
	if res.Hidden == nil || res.Hidden.Ne(1) != len(tokens) {
		t.Fatalf("mtp hidden dims = %v", res.Hidden)
	}
	if res.Logits == nil || res.Logits.Ne(0) != cfg.NVocab || res.Logits.Ne(1) != 1 {
		t.Fatalf("mtp logits dims = %v", res.Logits)
	}
	for _, v := range res.Logits.F32 {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("mtp logit = %v", v)
		}
	}
}

func TestDeviceMTPMatchesHost(t *testing.T) {
	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()

	cfg := deviceForwardConfig()
	cfg.NLayer = 5
	cfg.NextNPredict = 1
	cfg.LayerTypes = append(cfg.LayerTypes, modelcfg.LayerFullAttention)
	w := NewRandom(cfg, 22)
	host := NewModel(w)
	device := NewModel(w)
	device.Backend = be
	if err := device.EnableDeviceForward(); err != nil {
		t.Fatal(err)
	}
	defer device.FreeDevice()

	tokens := []int32{1, 2, 3, 4}
	positions := []int32{0, 1, 2, 3}
	hs := NewState(cfg)
	hb, err := host.Forward(tokens, ForwardOptions{State: hs, CaptureHidden: true})
	if err != nil {
		t.Fatal(err)
	}
	hw, err := host.ForwardMTP(tokens, hb.FinalHidden, make([]float32, cfg.NEmbd), MTPOptions{
		Positions: positions, ComputeLogits: true, LastTokenOnly: true,
	}, hs)
	if err != nil {
		t.Fatal(err)
	}

	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ8_0} {
		ds, err := NewStateWithOptions(cfg, StateOptions{
			Backend: be, KVType: typ, Capacity: 8, DeviceForward: true, MTP: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		db, err := device.Forward(tokens, ForwardOptions{State: ds, CaptureHidden: true})
		if err != nil {
			ds.Free()
			t.Fatal(err)
		}
		prev, err := be.Upload(&compute.Tensor{Dims: []int{cfg.NEmbd}, F32: make([]float32, cfg.NEmbd)})
		if err != nil {
			ds.Free()
			t.Fatal(err)
		}
		dw, err := device.ForwardMTP(tokens, nil, nil, MTPOptions{
			Positions: positions, ComputeLogits: true, LastTokenOnly: true,
			DeviceHiddenRows: db.DeviceHidden, DevicePrevHidden: prev,
		}, ds)
		be.Free(prev)
		be.Free(db.DeviceHidden)
		if err != nil {
			ds.Free()
			t.Fatal(err)
		}
		if dw.Logits == nil {
			be.Free(dw.DeviceHidden)
			ds.Free()
			t.Fatal("device mtp logits missing")
		}
		for i := range hw.Logits.F32 {
			tol := float32(3e-1 + 3e-1*math.Abs(float64(hw.Logits.F32[i])))
			if d := float32(math.Abs(float64(dw.Logits.F32[i] - hw.Logits.F32[i]))); d > tol {
				be.Free(dw.DeviceHidden)
				ds.Free()
				t.Fatalf("%s mtp logit %d: device %v host %v (diff %v)", typ, i, dw.Logits.F32[i], hw.Logits.F32[i], d)
			}
		}
		be.Free(dw.DeviceHidden)
		ds.Free()
	}
}

func testArgmax(v []float32) int32 {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return int32(best)
}

// TestSpeculativeRollback checks that a partial speculative round rewinds the
// recurrent state to exactly the same point as a cold forward of the accepted
// prefix.
func TestSpeculativeRollback(t *testing.T) {
	cfg := tinyMTPConfig()
	m := NewModel(NewRandom(cfg, 33))
	n := cfg.NEmbd
	prompt := []int32{1, 2, 3}
	st := NewState(cfg)
	res, err := m.Forward(prompt, ForwardOptions{State: st, CaptureHidden: true})
	if err != nil {
		t.Fatal(err)
	}
	pos := []int32{0, 1, 2}
	mtpRes, err := m.ForwardMTP(prompt, res.FinalHidden, make([]float32, n), MTPOptions{Positions: pos}, st)
	if err != nil {
		t.Fatal(err)
	}
	st.MTPPrevH = append([]float32(nil), mtpRes.Hidden.F32[(len(prompt)-1)*n:]...)
	current := testArgmax(logitRowTest(res.Logits, len(prompt)-1, cfg.NVocab))

	// Greedy MTP drafts.
	drafts := make([]int32, 0, 3)
	tok := current
	prev := append([]float32(nil), st.MTPPrevH...)
	for i := 0; i < 3; i++ {
		mr, err := m.ForwardMTP([]int32{tok}, nil, prev, MTPOptions{
			Positions: []int32{int32(len(prompt) + i)}, ComputeLogits: true, LastTokenOnly: true,
		}, st)
		if err != nil {
			t.Fatal(err)
		}
		d := testArgmax(mr.Logits.F32)
		drafts = append(drafts, d)
		prev = mr.Hidden.F32[:n]
		tok = d
	}

	if err := st.SpecSnapshot(); err != nil {
		t.Fatal(err)
	}
	verifyTokens := append([]int32{current}, drafts...)
	verifyPos := make([]int32, len(verifyTokens))
	for i := range verifyPos {
		verifyPos[i] = int32(len(prompt) + i)
	}
	verify, err := m.Forward(verifyTokens, ForwardOptions{State: st, Positions: verifyPos, CaptureHidden: true})
	if err != nil {
		t.Fatal(err)
	}
	accepted := 0
	for i := range drafts {
		if testArgmax(logitRowTest(verify.Logits, i, cfg.NVocab)) == drafts[i] {
			accepted++
			continue
		}
		break
	}
	acceptedTokens := verifyTokens[:1+accepted]
	var selected *ForwardResult
	if accepted == len(drafts) {
		selected = verify
	} else {
		if err := st.SpecRestore(); err != nil {
			t.Fatal(err)
		}
		selected, err = m.Forward(acceptedTokens, ForwardOptions{
			State: st, Positions: verifyPos[:1+accepted], CaptureHidden: true,
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	// Cold reference from the same weights.
	ref := NewModel(NewRandom(cfg, 33))
	refState := NewState(cfg)
	all := append(append([]int32(nil), prompt...), acceptedTokens...)
	refPos := make([]int32, len(all))
	for i := range refPos {
		refPos[i] = int32(i)
	}
	refRes, err := ref.Forward(all, ForwardOptions{State: refState, Positions: refPos, CaptureHidden: true})
	if err != nil {
		t.Fatal(err)
	}
	if accepted == len(drafts) {
		// selected has all accepted tokens; reference includes them.
	} else {
		// same as above; acceptedTokens is the prefix.
	}
	got := selected.FinalHidden.F32[(len(acceptedTokens)-1)*n:]
	want := refRes.FinalHidden.F32[(len(all)-1)*n:]
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > 1e-5 {
			t.Fatalf("hidden %d: speculative %v cold %v (diff %v)", i, got[i], want[i], d)
		}
	}
	// The next-token logits must also agree.
	gotLogit := logitRowTest(selected.Logits, len(acceptedTokens)-1, cfg.NVocab)
	wantLogit := logitRowTest(refRes.Logits, len(all)-1, cfg.NVocab)
	for i := range wantLogit {
		if d := math.Abs(float64(gotLogit[i] - wantLogit[i])); d > 1e-5 {
			t.Fatalf("logit %d: speculative %v cold %v (diff %v)", i, gotLogit[i], wantLogit[i], d)
		}
	}
}

func logitRowTest(t *compute.Tensor, row, vocab int) []float32 {
	return t.F32[row*vocab : (row+1)*vocab]
}
