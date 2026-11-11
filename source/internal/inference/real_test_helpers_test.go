package inference

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// requireEnv skips a test unless the named environment variable is "1".
func requireEnv(t *testing.T, key string) {
	t.Helper()
	if os.Getenv(key) != "1" {
		t.Skipf("set %s=1 to run this opt-in test", key)
	}
}

// realModelPath resolves the real GGUF from QWEN38_MODEL or the usual local
// repository layout. It is an error (not a skip) once a test has opted in.
func realModelPath(t *testing.T) string {
	t.Helper()
	if path := os.Getenv("QWEN38_MODEL"); path != "" {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("QWEN38_MODEL=%q: %v", path, err)
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			t.Fatal(err)
		}
		return abs
	}
	candidates := []string{
		"../../../models/Qwen3.8-27B-UD-Q4_K_M.gguf",
		"models/Qwen3.8-27B-UD-Q4_K_M.gguf",
	}
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			abs, err := filepath.Abs(candidate)
			if err != nil {
				t.Fatal(err)
			}
			return abs
		}
	}
	t.Fatalf("real model not found; set QWEN38_MODEL=... (tried %v)", candidates)
	return ""
}

// realCorpusPath resolves the WikiText-2 evaluation corpus.
func realCorpusPath(t *testing.T) string {
	t.Helper()
	if path := os.Getenv("QWEN38_PPL_TEXT"); path != "" {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("QWEN38_PPL_TEXT=%q: %v", path, err)
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			t.Fatal(err)
		}
		return abs
	}
	candidates := []string{
		"../../../datasets/wikitext2-test.txt",
		"datasets/wikitext2-test.txt",
	}
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			abs, err := filepath.Abs(candidate)
			if err != nil {
				t.Fatal(err)
			}
			return abs
		}
	}
	t.Fatalf("evaluation corpus not found; set QWEN38_PPL_TEXT=... (tried %v)", candidates)
	return ""
}

// loadRealEngine loads the real model on Vulkan and returns its concrete
// backend so tests can inspect device memory statistics.
func loadRealEngine(t *testing.T, contextLen int, kvType string) (*Engine, *vulkan.Backend) {
	t.Helper()
	engine, err := Load(LoadOptions{
		Path:          realModelPath(t),
		Device:        "vulkan",
		ContextLength: contextLen,
		MaxConcurrent: 1,
		KVCacheType:   kvType,
		LogLevel:      "quiet",
	})
	if err != nil {
		t.Fatalf("load real model: %v", err)
	}
	backend, ok := engine.Backend.(*vulkan.Backend)
	if !ok {
		engine.Close()
		t.Fatalf("backend is %T, want *vulkan.Backend", engine.Backend)
	}
	return engine, backend
}

// loadedWeightBytes is the device upload size implied by the loaded trunk
// weights. It matches EnableDeviceForward: quantized matrices keep their raw
// byte size and float vectors occupy 4 bytes per element.
func loadedWeightBytes(w *qwen38.Weights) uint64 {
	var n uint64
	addWeight := func(w *qwen38.Weight) {
		if w != nil {
			n += uint64(len(w.Raw))
		}
	}
	addVec := func(v []float32) { n += uint64(len(v)) * 4 }
	addTensor := func(t *compute.Tensor) {
		if t != nil {
			n += uint64(len(t.F32)) * 4
		}
	}

	addWeight(w.TokenEmbd)
	if w.Output != w.TokenEmbd {
		addWeight(w.Output)
	}
	addVec(w.OutputNorm)
	for index := range w.Layers {
		lw := &w.Layers[index]
		addVec(lw.AttnNorm)
		addVec(lw.PostAttnNorm)
		if w.Cfg.IsRecurrent(index) {
			addVec(lw.SSMNorm)
			addTensor(lw.SSMConv1d)
			addWeight(lw.SSMAlpha)
			addWeight(lw.SSMBeta)
			addWeight(lw.AttnQKV)
			addWeight(lw.AttnGate)
			addWeight(lw.SSMOut)
		} else {
			addVec(lw.AttnQNorm)
			addVec(lw.AttnKNorm)
			addWeight(lw.AttnQ)
			addWeight(lw.AttnK)
			addWeight(lw.AttnV)
			addWeight(lw.AttnOutput)
		}
		addWeight(lw.FfnGate)
		addWeight(lw.FfnUp)
		addWeight(lw.FfnDown)
	}
	return n
}

// deviceRecurrentStateBytes is the fixed per-state conv/SSM allocation implied
// by the config. It is separate from the KV cache and is allocated for every
// device-forward state.
func deviceRecurrentStateBytes(cfg *qwen38.Config) uint64 {
	var n uint64
	for layer := 0; layer < cfg.TrunkLayers(); layer++ {
		if !cfg.IsRecurrent(layer) {
			continue
		}
		n += uint64(cfg.DConv-1) * uint64(cfg.ConvDim()) * 4
		n += uint64(cfg.HeadVDim()) * uint64(cfg.HeadVDim()) * uint64(cfg.DtRank) * 4
	}
	return n
}

// rssHighWaterBytes returns VmHWM from /proc/self/status, or 0 when the
// platform does not expose it.
func rssHighWaterBytes(t *testing.T) uint64 {
	t.Helper()
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "VmHWM:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0
		}
		kb, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0
		}
		return kb * 1024
	}
	return 0
}

// syntheticTokens returns n deterministic token ids in [2, vocab).
func syntheticTokens(n, vocab int) []int32 {
	if vocab <= 4 {
		vocab = 4
	}
	out := make([]int32, n)
	state := uint64(0x9e3779b97f4a7c15)
	for index := range out {
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		out[index] = int32(2 + int(state%uint64(vocab-2)))
	}
	return out
}

// envString reads an environment override, returning "" when unset.
func envString(t *testing.T, key string) string {
	t.Helper()
	return os.Getenv(key)
}

// envInt reads a positive integer environment override.
func envInt(t *testing.T, key string, fallback int) int {
	t.Helper()
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	n, err := strconv.Atoi(value)
	if err != nil || n <= 0 {
		t.Fatalf("%s=%q: want a positive integer", key, value)
	}
	return n
}

// envContext reads a context preset such as "128k" or a raw token count.
func envContext(t *testing.T, key, fallback string) int {
	t.Helper()
	value := os.Getenv(key)
	if value == "" {
		value = fallback
	}
	n, err := ParseContextLength(value)
	if err != nil {
		t.Fatalf("%s=%q: %v", key, value, err)
	}
	return n
}

// logReal emits one structured, machine-readable result line.
func logReal(t *testing.T, event string, fields map[string]any) {
	t.Helper()
	if fields == nil {
		fields = map[string]any{}
	}
	fields["event"] = event
	data, err := json.Marshal(fields)
	if err != nil {
		t.Fatalf("marshal %s: %v", event, err)
	}
	t.Logf("REAL %s", data)
}

// logitsMetrics summarizes the difference between a reference and candidate
// logits row.
type logitsMetrics struct {
	Cosine    float64
	MaxAbs    float64
	MeanAbs   float64
	RefMaxAbs float64
	ArgmaxRef int
	ArgmaxGot int
	Top5Match int
}

func compareLogits(t *testing.T, label string, want, got []float32) logitsMetrics {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: logits length %d != %d", label, len(got), len(want))
	}
	if len(want) == 0 {
		t.Fatal(label + ": empty logits")
	}
	var dot, normWant, normGot, maxAbs, sumAbs, refMaxAbs float64
	argmaxWant, argmaxGot := 0, 0
	for index := range want {
		w := float64(want[index])
		g := float64(got[index])
		dot += w * g
		normWant += w * w
		normGot += g * g
		diff := absFloat(g - w)
		sumAbs += diff
		if diff > maxAbs {
			maxAbs = diff
		}
		if absFloat(w) > refMaxAbs {
			refMaxAbs = absFloat(w)
		}
		if want[index] > want[argmaxWant] {
			argmaxWant = index
		}
		if got[index] > got[argmaxGot] {
			argmaxGot = index
		}
	}
	cosine := 0.0
	if normWant > 0 && normGot > 0 {
		cosine = dot / (sqrtFloat(normWant) * sqrtFloat(normGot))
	}
	m := logitsMetrics{
		Cosine:    cosine,
		MaxAbs:    maxAbs,
		MeanAbs:   sumAbs / float64(len(want)),
		RefMaxAbs: refMaxAbs,
		ArgmaxRef: argmaxWant,
		ArgmaxGot: argmaxGot,
		Top5Match: topKMatch(want, got, 5),
	}
	if m.ArgmaxRef != m.ArgmaxGot {
		t.Errorf("%s: argmax %d != %d", label, m.ArgmaxGot, m.ArgmaxRef)
	}
	if m.Cosine < 0.995 {
		t.Errorf("%s: cosine %.6f < 0.995", label, m.Cosine)
	}
	if m.Top5Match < 4 {
		t.Errorf("%s: top-5 overlap %d < 4", label, m.Top5Match)
	}
	return m
}

func topKMatch(want, got []float32, k int) int {
	wantTop := topKIndices(want, k)
	gotTop := topKIndices(got, k)
	set := make(map[int]bool, len(wantTop))
	for _, index := range wantTop {
		set[index] = true
	}
	match := 0
	for _, index := range gotTop {
		if set[index] {
			match++
		}
	}
	return match
}

func topKIndices(values []float32, k int) []int {
	indices := make([]int, len(values))
	for index := range indices {
		indices[index] = index
	}
	sort.Slice(indices, func(a, b int) bool {
		return values[indices[a]] > values[indices[b]]
	})
	if k > len(indices) {
		k = len(indices)
	}
	return indices[:k]
}

func absFloat(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func sqrtFloat(v float64) float64 {
	if v <= 0 {
		return 0
	}
	guess := v
	for i := 0; i < 32; i++ {
		guess = 0.5 * (guess + v/guess)
	}
	return guess
}

func kvTypeName(typ quant.Type) string {
	return typ.String()
}

func formatDuration(d time.Duration) string {
	return d.Round(time.Millisecond).String()
}

func memString(bytes uint64) string {
	const unit = 1024
	if bytes >= unit*unit*unit {
		return fmt.Sprintf("%.2f GiB", float64(bytes)/float64(unit*unit*unit))
	}
	if bytes >= unit*unit {
		return fmt.Sprintf("%.2f MiB", float64(bytes)/float64(unit*unit))
	}
	return fmt.Sprintf("%d B", bytes)
}

func runtimeVersion() string { return runtime.Version() }

func totalCounts(m map[string]uint64) uint64 {
	var total uint64
	for _, v := range m {
		total += v
	}
	return total
}

// topCounts formats the largest map entries as "name=value,...".
func topCounts(m map[string]uint64, n int) string {
	if len(m) == 0 {
		return ""
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if m[keys[i]] != m[keys[j]] {
			return m[keys[i]] > m[keys[j]]
		}
		return keys[i] < keys[j]
	})
	if n > len(keys) {
		n = len(keys)
	}
	var b strings.Builder
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%s=%d", keys[i], m[keys[i]])
	}
	return b.String()
}
