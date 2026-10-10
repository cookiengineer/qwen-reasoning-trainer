package verifyquants

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// TestReferenceQuants diffs internal/quant's block decoders against the *actual*
// llama.cpp routines in ggml-quants.c by compiling the vendored reference
// (third_party/ggml) with a small harness (harness/ref_quants.c) and running
// both on identical raw blocks. It is opt-in:
//
//	QWEN38_VERIFY_QUANTS=1 go test ./e2e-tests/verify-quants -run TestReferenceQuants -v
//
// It is self-contained (no references/ dependency). CC overrides the compiler;
// QWEN38_LLAMA_SRC optionally points at an external llama.cpp ggml checkout to
// compare against a different revision.
func TestReferenceQuants(t *testing.T) {
	if os.Getenv("QWEN38_VERIFY_QUANTS") == "" {
		t.Skip("set QWEN38_VERIFY_QUANTS=1 to run the llama.cpp quant reference check")
	}

	// Default to the vendored subset (relative to this package's dir).
	src := os.Getenv("QWEN38_LLAMA_SRC")
	if src == "" {
		src = filepath.Join("third_party", "ggml")
	}
	for _, d := range []string{"src/ggml-quants.c", "src/ggml-quants.h", "src/ggml-common.h", "include/ggml.h"} {
		if _, err := os.Stat(filepath.Join(src, d)); err != nil {
			t.Fatalf("ggml source not found under %s (%v)", src, err)
		}
	}
	harness := filepath.Join("harness", "ref_quants.c")
	if _, err := os.Stat(harness); err != nil {
		t.Fatalf("harness not found (%v)", err)
	}

	cc := os.Getenv("CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		t.Fatalf("C compiler %q not found: %v", cc, err)
	}

	bin := filepath.Join(t.TempDir(), "ref_quants")
	build := exec.Command(cc, "-O1", "-w",
		"-I", filepath.Join(src, "include"),
		"-I", filepath.Join(src, "src"),
		"-I", filepath.Join(src, "src", "ggml-cpu"),
		"-o", bin,
		harness,
		filepath.Join(src, "src", "ggml-quants.c"),
		"-lm",
	)
	build.Stderr = os.Stderr
	if out, err := build.Output(); err != nil {
		t.Fatalf("build harness: %v\n%s", err, out)
	}

	cases := []struct {
		typ   quant.Type
		id    int
		block int
		size  int
	}{
		{quant.TypeQ8_0, 0, 32, 34},
		{quant.TypeQ4_K, 1, 256, 144},
		{quant.TypeQ5_K, 2, 256, 176},
		{quant.TypeQ6_K, 3, 256, 210},
		{quant.TypeQ3_K, 4, 256, 110},
		{quant.TypeIQ4_NL, 5, 32, 18},
		{quant.TypeIQ4_XS, 6, 256, 136},
		{quant.TypeIQ3_S, 7, 256, 110},
	}

	rng := rand.New(rand.NewSource(20261010))
	const blocksPerFixture = 6
	for _, c := range cases {
		if got := c.typ.BlockSize(); got != c.block {
			t.Fatalf("%s: block %d != %d", c.typ, got, c.block)
		}
		if got := c.typ.TypeSize(); got != c.size {
			t.Fatalf("%s: typeSize %d != %d", c.typ, got, c.size)
		}
		n := c.block * blocksPerFixture
		nbytes := c.size * blocksPerFixture

		// 4 fixtures produced by our encoder (valid scales; also verifies that
		// llama.cpp decodes our blocks), 4 random-byte fixtures for coverage.
		trials := make([][]byte, 0, 8)
		for i := 0; i < 4; i++ {
			f := make([]float32, n)
			for j := range f {
				f[j] = float32(rng.NormFloat64()) * 1.5
			}
			raw, err := quant.Quantize(c.typ, f)
			if err != nil {
				t.Fatalf("%s: quantize: %v", c.typ, err)
			}
			trials = append(trials, raw)
		}
		for i := 0; i < 4; i++ {
			raw := make([]byte, nbytes)
			if _, err := rng.Read(raw); err != nil {
				t.Fatal(err)
			}
			trials = append(trials, raw)
		}

		checked := 0
		for ti, raw := range trials {
			got, err := quant.Dequant(c.typ, raw, int64(n))
			if err != nil {
				t.Fatalf("%s: Go dequant: %v", c.typ, err)
			}
			want, err := runRefQuants(bin, "dequant", c.id, n, nbytes, raw)
			if err != nil {
				t.Fatalf("%s: ref harness: %v", c.typ, err)
			}
			if !allFinite(got) || !allFinite(want) {
				// Random blocks can contain non-finite fp16 scales; both
				// decoders agree (NaN) but comparing is meaningless.
				continue
			}
			checked++
			var worst float64
			worstIdx := -1
			for j := range want {
				d := math.Abs(float64(got[j] - want[j]))
				if d > worst {
					worst, worstIdx = d, j
				}
			}
			if worst > 1e-6 {
				t.Errorf("%s trial %d: max |go-ref| = %g at %d (go=%v ref=%v)",
					c.typ, ti, worst, worstIdx, got[worstIdx], want[worstIdx])
			}
		}
		if checked == 0 {
			t.Errorf("%s: no finite fixtures compared", c.typ)
		}
		t.Logf("%s: %d fixtures matched llama.cpp dequant", c.typ, checked)
	}
}

func runRefQuants(bin, mode string, id, n, nbytes int, raw []byte) ([]float32, error) {
	cmd := exec.Command(bin, mode, strconv.Itoa(id), strconv.Itoa(n), strconv.Itoa(nbytes))
	cmd.Stdin = bytes.NewReader(raw)
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%v: %s", err, errb.String())
	}
	wantBytes := out.Bytes()
	if len(wantBytes) != n*4 {
		return nil, fmt.Errorf("harness returned %d bytes, want %d", len(wantBytes), n*4)
	}
	want := make([]float32, n)
	for i := range want {
		want[i] = math.Float32frombits(binary.LittleEndian.Uint32(wantBytes[i*4:]))
	}
	return want, nil
}

// runRefQuantize sends n float32 values to the harness quantizer and returns the
// raw quantized blocks it produced.
func runRefQuantize(bin string, id, n, nbytes int, f []float32) ([]byte, error) {
	in := make([]byte, n*4)
	for i, v := range f {
		binary.LittleEndian.PutUint32(in[i*4:], math.Float32bits(v))
	}
	cmd := exec.Command(bin, "quantize", strconv.Itoa(id), strconv.Itoa(n), strconv.Itoa(nbytes))
	cmd.Stdin = bytes.NewReader(in)
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%v: %s", err, errb.String())
	}
	if out.Len() != nbytes {
		return nil, fmt.Errorf("harness returned %d bytes, want %d", out.Len(), nbytes)
	}
	return out.Bytes(), nil
}

// byteExactEncoders lists the types whose Go encoder must be byte-identical to
// llama.cpp's reference quantizer. It grows as each encoder is ported; types not
// listed are only reported (so `make verify-quants` stays green during the port).
var byteExactEncoders = map[quant.Type]bool{
	quant.TypeQ8_0:   true,
	quant.TypeQ4_K:   true,
	quant.TypeQ5_K:   true,
	quant.TypeQ6_K:   true,
	quant.TypeQ3_K:   true,
	quant.TypeIQ4_NL: true,
	quant.TypeIQ4_XS: true,
	quant.TypeIQ3_S:  true,
}

// TestReferenceQuantize diffs internal/quant's block *encoders* against
// llama.cpp's reference quantizers (quantize_row_*_ref) on identical float input.
// It is opt-in and reuses the harness built by TestReferenceQuants. Types in
// byteExactEncoders must match exactly; the rest are logged.
//
//	QWEN38_VERIFY_QUANTS=1 go test ./e2e-tests/verify-quants -run TestReferenceQuantize -v
func TestReferenceQuantize(t *testing.T) {
	if os.Getenv("QWEN38_VERIFY_QUANTS") == "" {
		t.Skip("set QWEN38_VERIFY_QUANTS=1 to run the llama.cpp quant reference check")
	}
	src := os.Getenv("QWEN38_LLAMA_SRC")
	if src == "" {
		src = filepath.Join("third_party", "ggml")
	}
	cc := os.Getenv("CC")
	if cc == "" {
		cc = "cc"
	}
	bin := filepath.Join(t.TempDir(), "ref_quants")
	build := exec.Command(cc, "-O1", "-w",
		"-I", filepath.Join(src, "include"),
		"-I", filepath.Join(src, "src"),
		"-I", filepath.Join(src, "src", "ggml-cpu"),
		"-o", bin,
		filepath.Join("harness", "ref_quants.c"),
		filepath.Join(src, "src", "ggml-quants.c"),
		"-lm",
	)
	build.Stderr = os.Stderr
	if out, err := build.Output(); err != nil {
		t.Fatalf("build harness: %v\n%s", err, out)
	}

	cases := []struct {
		typ   quant.Type
		id    int
		block int
		size  int
	}{
		{quant.TypeQ8_0, 0, 32, 34},
		{quant.TypeQ4_K, 1, 256, 144},
		{quant.TypeQ5_K, 2, 256, 176},
		{quant.TypeQ6_K, 3, 256, 210},
		{quant.TypeQ3_K, 4, 256, 110},
		{quant.TypeIQ4_NL, 5, 32, 18},
		{quant.TypeIQ4_XS, 6, 256, 136},
		{quant.TypeIQ3_S, 7, 256, 110},
	}
	rng := rand.New(rand.NewSource(0x5eed))
	const blocksPerFixture = 8
	for _, c := range cases {
		n := c.block * blocksPerFixture
		nbytes := c.size * blocksPerFixture
		var totalBytes, differBytes, trialsDiffer, trials int
		for trial := 0; trial < 8; trial++ {
			f := make([]float32, n)
			scale := []float64{1.5, 0.05, 12, 1e-3, 3, 0.7, 40, 0.2}[trial]
			for j := range f {
				f[j] = float32(rng.NormFloat64() * scale)
			}
			got, err := quant.Quantize(c.typ, f)
			if err != nil {
				t.Fatalf("%s: quantize: %v", c.typ, err)
			}
			want, err := runRefQuantize(bin, c.id, n, nbytes, f)
			if err != nil {
				t.Fatalf("%s: ref quantize: %v", c.typ, err)
			}
			trials++
			totalBytes += nbytes
			trialDiff := 0
			for i := range want {
				if got[i] != want[i] {
					differBytes++
					trialDiff++
				}
			}
			if trialDiff > 0 {
				trialsDiffer++
			}
		}
		pct := 100 * float64(totalBytes-differBytes) / float64(totalBytes)
		t.Logf("%s: encoder byte-match %.2f%% (%d/%d bytes differ across %d trials)",
			c.typ, pct, differBytes, totalBytes, trials)
		if byteExactEncoders[c.typ] && differBytes > 0 {
			t.Errorf("%s: encoder is not byte-exact: %d/%d bytes differ", c.typ, differBytes, totalBytes)
		}
	}
}

func allFinite(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}
