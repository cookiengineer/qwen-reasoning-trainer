// Package verifyquants holds the opt-in cross-check that diffs
// internal/quant's block decoders against llama.cpp's own ggml-quants.c.
//
// The reference ggml subset is vendored under third_party/ggml (see README.md),
// so the check is self-contained and does not need the references/ checkout.
package verifyquants
