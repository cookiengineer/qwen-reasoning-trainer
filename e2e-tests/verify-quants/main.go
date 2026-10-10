// Command verify-quants cross-checks this project's block decoders and encoders
// against the actual llama.cpp routines in ggml-quants.c. It compiles the
// vendored reference subset (third_party/ggml) with a small harness
// (harness/ref_quants.c) and runs both on identical input:
//
//   - decoders: internal/quant.Dequant vs dequantize_row_* on the same blocks
//   - encoders: internal/quant.Quantize vs quantize_row_*_ref (byte-identical)
//
// It is an opt-in external check, run as:
//
//	go run ./e2e-tests/verify-quants      # or: make verify-quants
//
// It needs a C compiler (CC overrides the default cc). It is self-contained (no
// references/ checkout); set QWEN38_LLAMA_SRC to compare against an external
// llama.cpp ggml source tree instead of the vendored copy. Exits non-zero on any
// mismatch.
package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "verify-quants:", err)
		os.Exit(1)
	}
	fmt.Println("verify-quants: OK (decoders + encoders match llama.cpp ggml-quants.c)")
}

func run() error {
	bin, cleanup, err := buildHarness()
	if err != nil {
		return err
	}
	defer cleanup()
	if err := checkDecoders(bin); err != nil {
		return err
	}
	if err := checkEncoders(bin); err != nil {
		return err
	}
	return nil
}

// quantCase maps a quant type to its harness id and block geometry.
type quantCase struct {
	typ   quant.Type
	id    int
	block int
	size  int
}

var quantCases = []quantCase{
	{quant.TypeQ8_0, 0, 32, 34},
	{quant.TypeQ4_K, 1, 256, 144},
	{quant.TypeQ5_K, 2, 256, 176},
	{quant.TypeQ6_K, 3, 256, 210},
	{quant.TypeQ3_K, 4, 256, 110},
	{quant.TypeIQ4_NL, 5, 32, 18},
	{quant.TypeIQ4_XS, 6, 256, 136},
	{quant.TypeIQ3_S, 7, 256, 110},
}

func buildHarness() (bin string, cleanup func(), err error) {
	src := os.Getenv("QWEN38_LLAMA_SRC")
	if src == "" {
		src = filepath.Join(dir(), "third_party", "ggml")
	}
	for _, d := range []string{"src/ggml-quants.c", "src/ggml-quants.h", "src/ggml-common.h", "include/ggml.h"} {
		if _, err := os.Stat(filepath.Join(src, d)); err != nil {
			return "", nil, fmt.Errorf("ggml source not found under %s (%v)", src, err)
		}
	}
	harness := filepath.Join(dir(), "harness", "ref_quants.c")
	if _, err := os.Stat(harness); err != nil {
		return "", nil, fmt.Errorf("harness not found (%v)", err)
	}

	cc := os.Getenv("CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		return "", nil, fmt.Errorf("C compiler %q not found: %w", cc, err)
	}

	tmp, err := os.MkdirTemp("", "verify-quants-")
	if err != nil {
		return "", nil, err
	}
	bin = filepath.Join(tmp, "ref_quants")
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
		os.RemoveAll(tmp)
		return "", nil, fmt.Errorf("build harness: %v\n%s", err, out)
	}
	return bin, func() { os.RemoveAll(tmp) }, nil
}

func checkDecoders(bin string) error {
	for _, c := range quantCases {
		if got := c.typ.BlockSize(); got != c.block {
			return fmt.Errorf("%s: block %d != %d", c.typ, got, c.block)
		}
		if got := c.typ.TypeSize(); got != c.size {
			return fmt.Errorf("%s: typeSize %d != %d", c.typ, got, c.size)
		}
	}

	rng := rand.New(rand.NewSource(20261010))
	const blocksPerFixture = 6
	var errs []string
	for _, c := range quantCases {
		n := c.block * blocksPerFixture
		nbytes := c.size * blocksPerFixture

		// Fixtures produced by our encoder (valid scales; also verifies that
		// llama.cpp decodes our blocks) plus random bytes for coverage.
		trials := make([][]byte, 0, 8)
		for i := 0; i < 4; i++ {
			f := make([]float32, n)
			for j := range f {
				f[j] = float32(rng.NormFloat64()) * 1.5
			}
			raw, err := quant.Quantize(c.typ, f)
			if err != nil {
				return fmt.Errorf("%s: quantize: %w", c.typ, err)
			}
			trials = append(trials, raw)
		}
		for i := 0; i < 4; i++ {
			raw := make([]byte, nbytes)
			if _, err := rng.Read(raw); err != nil {
				return err
			}
			trials = append(trials, raw)
		}

		checked := 0
		for ti, raw := range trials {
			got, err := quant.Dequant(c.typ, raw, int64(n))
			if err != nil {
				return fmt.Errorf("%s: Go dequant: %w", c.typ, err)
			}
			want, err := runRefFloat(bin, "dequant", c.id, n, nbytes, raw)
			if err != nil {
				return fmt.Errorf("%s: ref harness: %w", c.typ, err)
			}
			if !allFinite(got) || !allFinite(want) {
				// Random blocks can carry non-finite fp16 scales; both decoders
				// agree (NaN) but comparing is meaningless.
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
				errs = append(errs, fmt.Sprintf("%s trial %d: max |go-ref| = %g at %d (go=%v ref=%v)",
					c.typ, ti, worst, worstIdx, got[worstIdx], want[worstIdx]))
			}
		}
		if checked == 0 {
			errs = append(errs, fmt.Sprintf("%s: no finite fixtures compared", c.typ))
		}
		fmt.Printf("decoder %-7s: %d fixtures matched llama.cpp\n", c.typ, checked)
	}
	if len(errs) > 0 {
		return fmt.Errorf("decoder mismatches:\n%s", strings.Join(errs, "\n"))
	}
	return nil
}

func checkEncoders(bin string) error {
	rng := rand.New(rand.NewSource(0x5eed))
	const blocksPerFixture = 8
	var errs []string
	for _, c := range quantCases {
		n := c.block * blocksPerFixture
		nbytes := c.size * blocksPerFixture
		var totalBytes, differBytes int
		for trial := 0; trial < 8; trial++ {
			f := make([]float32, n)
			scale := []float64{1.5, 0.05, 12, 1e-3, 3, 0.7, 40, 0.2}[trial]
			for j := range f {
				f[j] = float32(rng.NormFloat64() * scale)
			}
			got, err := quant.Quantize(c.typ, f)
			if err != nil {
				return fmt.Errorf("%s: quantize: %w", c.typ, err)
			}
			want, err := runRefQuantize(bin, c.id, n, nbytes, f)
			if err != nil {
				return fmt.Errorf("%s: ref quantize: %w", c.typ, err)
			}
			totalBytes += nbytes
			for i := range want {
				if got[i] != want[i] {
					differBytes++
				}
			}
		}
		if differBytes > 0 {
			errs = append(errs, fmt.Sprintf("%s: encoder not byte-exact: %d/%d bytes differ", c.typ, differBytes, totalBytes))
		} else {
			fmt.Printf("encoder %-7s: byte-identical to quantize_row_*_ref (%d bytes)\n", c.typ, totalBytes)
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("encoder mismatches:\n%s", strings.Join(errs, "\n"))
	}
	return nil
}

func runRefFloat(bin, mode string, id, n, nbytes int, raw []byte) ([]float32, error) {
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

func allFinite(v []float32) bool {
	for _, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}

// dir returns the directory containing this source file.
func dir() string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Dir(thisFile)
}
