# verify-quants

Opt-in cross-check that diffs this project's block decoders
(`internal/quant.Dequant`) against the **actual** llama.cpp dequantization
routines. It is the only check that executes a reference implementation.

Because the `dequant_*`, `gemv_*` and fused `matmul_*_wq` shaders are validated
against `quant.Dequant` (`TestVulkanDequant`, `TestVulkanGemvFused`,
`TestVulkanMatMulWeightFusedParity`), this transitively validates the quantized
decode math in the Vulkan shaders against llama.cpp.

## Layout

- `harness/ref_quants.c` — tiny program: reads `(type, nelem, nbytes)` + raw
  block bytes on stdin, calls the real `dequantize_row_*`, writes f32 to stdout.
- `third_party/ggml/` — a **vendored subset** of llama.cpp's ggml, the minimal
  include closure needed to compile `ggml-quants.c` (see below). Vendored so the
  check is self-contained and does not depend on the (gitignored) `references/`
  checkout.
- `verify_quants_test.go` — compiles the harness + `third_party/ggml` with `cc`
  and compares against `quant.Dequant` on identical raw blocks.

## Run

```sh
make verify-quants
# or
QWEN38_VERIFY_QUANTS=1 go test ./e2e-tests/verify-quants -run TestReferenceQuants -v
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

`verify_quants_test.go` feeds each decoder both blocks produced by our encoder
(proving llama.cpp can decode our blocks) and random raw bytes (coverage,
skipping non-finite fp16 scales); all match to `<1e-6`.
