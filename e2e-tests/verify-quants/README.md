# verify-quants

Opt-in cross-check that diffs this project's block decoders
(`internal/quant.Dequant`) and encoders (`internal/quant.Quantize`) against the
**actual** llama.cpp routines (`dequantize_row_*` / `quantize_row_*_ref`). It is
the only check that executes a reference implementation.

Because the `dequant_*`, `gemv_*` and fused `matmul_*_wq` shaders are validated
against `quant.Dequant` (`TestVulkanDequant`, `TestVulkanGemvFused`,
`TestVulkanMatMulWeightFusedParity`), this transitively validates the quantized
decode math in the Vulkan shaders against llama.cpp.

## Layout

- `main.go` — the check program: compiles the harness + `third_party/ggml` with
  `cc`, compares decoders against `quant.Dequant` on identical raw blocks, and
  asserts encoders are byte-identical to `quantize_row_*_ref`.
- `harness/ref_quants.c` — tiny C program: in `dequant` mode reads
  `(type, nelem, nbytes)` + raw block bytes and writes f32; in `quantize` mode
  reads f32 and writes the quantized blocks.
- `third_party/ggml/` — a **vendored subset** of llama.cpp's ggml, the minimal
  include closure needed to compile `ggml-quants.c` (see below). Vendored so the
  check is self-contained and does not depend on the (gitignored) `references/`
  checkout.

## Run

```sh
make verify-quants
# or
go run ./e2e-tests/verify-quants
```

Requirements: a C compiler (`cc`, or `CC=...`). `QWEN38_LLAMA_SRC=/path/to/ggml`
compares against an external llama.cpp checkout instead of the vendored copy.

## Provenance

- Source: `llama.cpp`, revision `08246a28f6000100433d297c4e037c02e9d2d464`
  (from the `references/llama_cpp` checkout at the time of vendoring).
- Files: `ggml/src/ggml-quants.{c,h}`, `ggml/src/ggml-common.h`,
  `ggml/src/ggml-impl.h`, `ggml/src/ggml-cpu/ggml-cpu-impl.h`,
  `ggml/include/{ggml.h,ggml-cpu.h,ggml-backend.h,ggml-alloc.h,gguf.h}`.
- License: MIT (`third_party/ggml/LICENSE`).
- To refresh: re-copy those files from a newer llama.cpp and update the
  revision above. The include closure is stable: those headers only
  `#include` each other and system headers.

`main.go` feeds each decoder both blocks produced by our encoder (proving
llama.cpp can decode our blocks) and random raw bytes (coverage, skipping
non-finite fp16 scales); all match to `<1e-6`. It also feeds float input to both
quantizers and requires byte-identical output.
