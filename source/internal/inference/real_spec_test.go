package inference

import (
	"context"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// TestRealSpeculativeRuns checks that the MTP speculative path runs end to end
// on the real model. Kernel ordering can change greedy tie-breaks, so this
// test asserts the first token (identical prefill path) and accounting rather
// than full-sequence token equality.
func TestRealSpeculativeRuns(t *testing.T) {
	requireEnv(t, "QWEN38_REAL_DEVICE")
	engine, err := Load(LoadOptions{
		Path:          realModelPath(t),
		Device:        "vulkan",
		ContextLength: 4096,
		MaxConcurrent: 1,
		KVCacheType:   "q4_k",
		Speculative:   true,
		DraftMax:      4,
		DraftMinProb:  0.6,
		LogLevel:      "quiet",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	if !engine.Speculative() {
		t.Fatal("MTP speculative decoding did not activate")
	}
	req := ChatRequest{
		Messages:       []Message{{Role: "user", Content: "Hi"}},
		MaxTokens:      24,
		Temperature:    0,
		EnableThinking: true,
	}
	spec, err := engine.Chat(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	engine.speculative = false
	plain, err := engine.Chat(context.Background(), req, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.Tokens) == 0 {
		t.Fatal("speculative produced no tokens")
	}
	if len(plain.Tokens) == 0 {
		t.Fatal("plain run produced no tokens")
	}
	if spec.Tokens[0] != plain.Tokens[0] {
		t.Fatalf("first token differs: spec %d plain %d", spec.Tokens[0], plain.Tokens[0])
	}
	if spec.Usage.CompletionTokens != len(spec.Tokens) {
		t.Fatalf("spec usage %d != tokens %d", spec.Usage.CompletionTokens, len(spec.Tokens))
	}
	if spec.FinishReason == "" {
		t.Fatal("speculative finish reason is empty")
	}
	t.Logf("spec tokens=%v plain tokens=%v", spec.Tokens, plain.Tokens)
}

func TestRealBatchMatchesSequential(t *testing.T) {
	requireEnv(t, "QWEN38_REAL_DEVICE")
	engine, err := Load(LoadOptions{
		Path:          realModelPath(t),
		Device:        "vulkan",
		ContextLength: 4096,
		MaxConcurrent: 1,
		KVCacheType:   "q4_k",
		LogLevel:      "quiet",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	prompt, err := engine.render(ChatRequest{Messages: []Message{{Role: "user", Content: "Hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	cfg := engine.Model.Cfg
	mk := func() *qwen38.State {
		st, err := qwen38.NewStateWithOptions(cfg, qwen38.StateOptions{
			Backend: engine.Backend, KVType: engine.kvType, Capacity: 4096, DeviceForward: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		return st
	}
	a, b := mk(), mk()
	defer a.Free()
	defer b.Free()
	ppos := make([]int32, len(prompt))
	for i := range ppos {
		ppos[i] = int32(i)
	}
	pa, err := engine.Model.Forward(prompt, qwen38.ForwardOptions{State: a, Positions: ppos, LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := engine.Model.Forward(prompt, qwen38.ForwardOptions{State: b, Positions: ppos, LastTokenOnly: true}); err != nil {
		t.Fatal(err)
	}
	cur := argmax(pa.Logits.F32)
	tokens := []int32{cur, 1144, 310, 5707, 310}
	pos := make([]int32, len(tokens))
	for i := range pos {
		pos[i] = int32(len(prompt) + i)
	}
	batch, err := engine.Model.Forward(tokens, qwen38.ForwardOptions{State: a, Positions: pos})
	if err != nil {
		t.Fatal(err)
	}
	for i, tok := range tokens {
		seq, err := engine.Model.Forward([]int32{tok}, qwen38.ForwardOptions{State: b, Positions: pos[i : i+1], LastTokenOnly: true})
		if err != nil {
			t.Fatal(err)
		}
		got := batch.Logits.F32[i*cfg.NVocab : (i+1)*cfg.NVocab]
		want := seq.Logits.F32
		maxd := float32(0)
		argG, argW := int32(0), int32(0)
		for j := range want {
			if d := float32(absf(float64(got[j] - want[j]))); d > maxd {
				maxd = d
			}
			if got[j] > got[argG] {
				argG = int32(j)
			}
			if want[j] > want[argW] {
				argW = int32(j)
			}
		}
		t.Logf("row %d maxdiff=%v batch_arg=%d seq_arg=%d", i, maxd, argG, argW)
		if argG != argW {
			t.Fatalf("row %d argmax differs: batch %d seq %d (maxdiff %v)", i, argG, argW, maxd)
		}
	}
}

func absf(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func TestRealSpecRestoreAfterRound(t *testing.T) {
	requireEnv(t, "QWEN38_REAL_DEVICE")
	engine, err := Load(LoadOptions{
		Path: realModelPath(t), Device: "vulkan", ContextLength: 4096,
		MaxConcurrent: 1, KVCacheType: "q4_k", LogLevel: "quiet",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()
	prompt, err := engine.render(ChatRequest{Messages: []Message{{Role: "user", Content: "Hi"}}})
	if err != nil {
		t.Fatal(err)
	}
	cfg := engine.Model.Cfg
	mk := func() *qwen38.State {
		st, err := qwen38.NewStateWithOptions(cfg, qwen38.StateOptions{
			Backend: engine.Backend, KVType: engine.kvType, Capacity: 4096, DeviceForward: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		return st
	}
	a, b := mk(), mk()
	defer a.Free()
	defer b.Free()
	ppos := make([]int32, len(prompt))
	for i := range ppos {
		ppos[i] = int32(i)
	}
	if _, err := engine.Model.Forward(prompt, qwen38.ForwardOptions{State: a, Positions: ppos, LastTokenOnly: true}); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.Model.Forward(prompt, qwen38.ForwardOptions{State: b, Positions: ppos, LastTokenOnly: true}); err != nil {
		t.Fatal(err)
	}
	round1 := []int32{1596, 1144, 310, 5707, 310}
	r1pos := make([]int32, len(round1))
	for i := range r1pos {
		r1pos[i] = int32(len(prompt) + i)
	}
	if _, err := engine.Model.Forward(round1, qwen38.ForwardOptions{State: a, Positions: r1pos}); err != nil {
		t.Fatal(err)
	}
	// Reference uses the same batched round-1 forward so this test isolates
	// snapshot/restore rather than batch-vs-sequential numerics.
	if _, err := engine.Model.Forward(round1, qwen38.ForwardOptions{State: b, Positions: r1pos}); err != nil {
		t.Fatal(err)
	}
	// Round 2: verify a batch, then restore and replay only the accepted token.
	verify := []int32{1156, 25, 328, 12675, 1}
	vpos := make([]int32, len(verify))
	for i := range vpos {
		vpos[i] = int32(len(prompt) + len(round1) + i)
	}
	if err := a.SpecSnapshot(); err != nil {
		t.Fatal(err)
	}
	if _, err := engine.Model.Forward(verify, qwen38.ForwardOptions{State: a, Positions: vpos}); err != nil {
		t.Fatal(err)
	}
	if err := a.SpecRestore(); err != nil {
		t.Fatal(err)
	}
	accepted := []int32{1156, 25}
	for i, tok := range accepted {
		if _, err := engine.Model.Forward([]int32{tok}, qwen38.ForwardOptions{State: a, Positions: vpos[i : i+1], LastTokenOnly: true}); err != nil {
			t.Fatal(err)
		}
		if _, err := engine.Model.Forward([]int32{tok}, qwen38.ForwardOptions{State: b, Positions: vpos[i : i+1], LastTokenOnly: true}); err != nil {
			t.Fatal(err)
		}
	}
	// Compare the next token logits.
	nextPos := int32(len(prompt) + len(round1) + len(accepted))
	ga, err := engine.Model.Forward([]int32{328}, qwen38.ForwardOptions{State: a, Positions: []int32{nextPos}, LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	gb, err := engine.Model.Forward([]int32{328}, qwen38.ForwardOptions{State: b, Positions: []int32{nextPos}, LastTokenOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := range ga.Logits.F32 {
		if ga.Logits.F32[i] != gb.Logits.F32[i] {
			t.Fatalf("logit %d: restored %v reference %v", i, ga.Logits.F32[i], gb.Logits.F32[i])
		}
	}
}
