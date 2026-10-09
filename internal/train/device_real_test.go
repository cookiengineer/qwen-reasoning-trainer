package train

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
)

// TestRealDeviceStep runs a real Qwen3.8-27B resident forward + loss + backward
// step at a long sequence length on the Vulkan device. It is opt-in and heavy
// (loads ~16 GiB and runs many dispatches), so it is excluded from make check:
//
//	QWEN38_REAL_DEVICE=1 QWEN38_REAL_T=512 go test ./internal/train -run TestRealDeviceStep -v
//
// QWEN38_MODEL overrides the GGUF path; QWEN38_REAL_T overrides the sequence
// length (default 512).
func TestRealDeviceStep(t *testing.T) {
	if os.Getenv("QWEN38_REAL_DEVICE") != "1" {
		t.Skip("set QWEN38_REAL_DEVICE=1 to run the real-model device step")
	}
	path := os.Getenv("QWEN38_MODEL")
	if path == "" {
		// `go test` runs with CWD at the package directory, so look from the
		// repo root (../../) as well as the conventional models/ dir.
		for _, cand := range []string{
			"models/Qwen3.8-27B-UD-Q4_K_M.gguf",
			"../../models/Qwen3.8-27B-UD-Q4_K_M.gguf",
		} {
			if _, err := os.Stat(cand); err == nil {
				path = cand
				break
			}
		}
	}
	if path == "" {
		path = "models/Qwen3.8-27B-UD-Q4_K_M.gguf"
	}
	T := 512
	if s := os.Getenv("QWEN38_REAL_T"); s != "" {
		if _, err := fmt.Sscanf(s, "%d", &T); err != nil {
			t.Fatalf("bad QWEN38_REAL_T=%q", s)
		}
	}

	g, err := gguf.Open(path)
	if err != nil {
		t.Skipf("model not found (%v)", err)
	}
	defer g.Close()
	mc, err := modelcfg.FromGGUF(g)
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	w, err := qwen38.LoadFromGGUF(g, mc)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loaded weights in %s", time.Since(t0))

	be, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer be.Close()
	caps := be.Capabilities()
	t.Logf("device: %s (%d bytes)", caps.Name, caps.MemoryBytes)

	loCfg := DefaultLoRA()
	t0 = time.Now()
	m, err := NewDeviceModel(be, w.Cfg, w, loCfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := be.Sync(); err != nil {
		t.Fatal(err)
	}
	t.Logf("built %d-layer device model with %d adapters in %s", w.Cfg.TrunkLayers(), len(m.AdapterLins()), time.Since(t0))

	m.Checkpoint = true
	tokens := make([]int32, T)
	targets := make([]int32, T)
	for i := range tokens {
		tokens[i] = int32(1 + (i*7919+13)%(w.Cfg.NVocab-1))
		targets[i] = int32(1 + ((i+1)*7919+13)%(w.Cfg.NVocab-1))
	}

	tF := time.Now()
	logits, ctx, err := m.Forward(tokens)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	loss, dlogits, err := m.Loss(logits, targets, -100)
	if err != nil {
		t.Fatalf("loss: %v", err)
	}
	be.Free(logits)
	tFL := time.Since(tF)

	tB := time.Now()
	grads, err := m.Backward(ctx, dlogits)
	if err != nil {
		t.Fatalf("backward: %v", err)
	}
	be.Free(dlogits)
	if err := be.Sync(); err != nil {
		t.Fatal(err)
	}
	tBL := time.Since(tB)

	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("non-finite loss %g", loss)
	}
	// Gradients must be finite and not all zero. dA is zero on the first step
	// (B is zero-initialized), so the non-zero signal is dB.
	var maxAbs float32
	for _, gr := range grads {
		for _, buf := range []struct {
			name string
			b    compute.Buffer
		}{{"dA", gr.dA}, {"dB", gr.dB}} {
			gt, err := be.Download(buf.b)
			if err != nil {
				t.Fatal(err)
			}
			for _, f := range gt.F32 {
				if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
					t.Fatalf("non-finite %s gradient", buf.name)
				}
				if a := float32(math.Abs(float64(f))); a > maxAbs {
					maxAbs = a
				}
			}
		}
	}
	t.Logf("T=%d forward+loss %s, backward %s (total %s), loss %.4f, %d adapter grads, max|grad| %.3g",
		T, tFL, tBL, tFL+tBL, loss, len(grads), maxAbs)
	if maxAbs == 0 {
		t.Fatal("all-zero adapter gradient")
	}
}
