// Reference quant/dequant harness.
//
// Usage: ref_quants <mode> <type> <nelem> <nbytes>
//   mode = dequant : read <nbytes> raw block bytes on stdin, write <nelem>
//                    little-endian float32 values to stdout.
//   mode = quantize: read <nelem> little-endian float32 values on stdin, write
//                    the quantized block bytes to stdout.
//
// It calls the *actual* llama.cpp reference routines from ggml-quants.c so the
// Go port (and, through the gemv/dequant/matmul parity tests, the Vulkan
// shaders) can be diffed against them.
//
// Build (see verify_quants_test.go / `make verify-quants`):
//   cc -O1 -w -I <ggml/src> -I <ggml/src/ggml-cpu> -I <ggml/include> \
//      -o ref_quants ref_quants.c <ggml/src>/ggml-quants.c -lm

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-quants.h"

// The IQ3_S quantizer needs its grid tables initialized (normally done by
// ggml_quantize_init in ggml.c, which the vendored subset does not include).
void iq3xs_init_impl(int grid_size);

// ggml-quants.c references these ggml.c symbols from paths we never call
// (row-size validation and the quantizer asserts); provide tiny stubs so the
// harness links.
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void) type; (void) ne; return 0; }
size_t ggml_type_size(enum ggml_type type) { (void) type; return 0; }
const char * ggml_type_name(enum ggml_type type) { (void) type; return "stub"; }
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap;
    fprintf(stderr, "ggml_abort at %s:%d: ", file, line);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    abort();
}

enum {
    T_Q8_0 = 0,
    T_Q4_K = 1,
    T_Q5_K = 2,
    T_Q6_K = 3,
    T_Q3_K = 4,
    T_IQ4_NL = 5,
    T_IQ4_XS = 6,
    T_IQ3_S = 7,
};

// Elements per block and bytes per block for each supported type (must match
// internal/quant/types.go).
static int block_elems(int type) {
    switch (type) {
        case T_Q8_0:  return 32;
        case T_IQ4_NL: return 32;
        case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_Q3_K: case T_IQ4_XS: case T_IQ3_S:
            return 256;
        default: return 0;
    }
}
static int block_bytes(int type) {
    switch (type) {
        case T_Q8_0:  return 34;
        case T_Q4_K:  return 144;
        case T_Q5_K:  return 176;
        case T_Q6_K:  return 210;
        case T_Q3_K:  return 110;
        case T_IQ4_NL: return 18;
        case T_IQ4_XS: return 136;
        case T_IQ3_S:  return 110;
        default: return 0;
    }
}

static int do_dequant(int type, int64_t n, int64_t nbytes) {
    uint8_t * raw = (uint8_t *) malloc((size_t) nbytes);
    float   * y   = (float *)   malloc((size_t) n * sizeof(float));
    if (!raw || !y) { return 3; }
    if (fread(raw, 1, (size_t) nbytes, stdin) != (size_t) nbytes) {
        fprintf(stderr, "short read\n"); return 4;
    }
    switch (type) {
        case T_Q8_0:  dequantize_row_q8_0 ((const block_q8_0  *) raw, y, n); break;
        case T_Q4_K:  dequantize_row_q4_K ((const block_q4_K  *) raw, y, n); break;
        case T_Q5_K:  dequantize_row_q5_K ((const block_q5_K  *) raw, y, n); break;
        case T_Q6_K:  dequantize_row_q6_K ((const block_q6_K  *) raw, y, n); break;
        case T_Q3_K:  dequantize_row_q3_K ((const block_q3_K  *) raw, y, n); break;
        case T_IQ4_NL: dequantize_row_iq4_nl((const block_iq4_nl *) raw, y, n); break;
        case T_IQ4_XS: dequantize_row_iq4_xs((const block_iq4_xs *) raw, y, n); break;
        case T_IQ3_S: dequantize_row_iq3_s ((const block_iq3_s  *) raw, y, n); break;
        default: fprintf(stderr, "unknown type %d\n", type); return 5;
    }
    if (fwrite(y, sizeof(float), (size_t) n, stdout) != (size_t) n) { return 6; }
    free(raw); free(y);
    return 0;
}

static int do_quantize(int type, int64_t n) {
    int be = block_elems(type), bb = block_bytes(type);
    if (be == 0 || n % be != 0) {
        fprintf(stderr, "bad element count %lld for type %d\n", (long long) n, type);
        return 2;
    }
    int64_t nbytes = (n / be) * bb;
    float * x   = (float *) malloc((size_t) n * sizeof(float));
    uint8_t * y = (uint8_t *) malloc((size_t) nbytes);
    if (!x || !y) { return 3; }
    if (fread(x, sizeof(float), (size_t) n, stdin) != (size_t) n) {
        fprintf(stderr, "short read\n"); return 4;
    }
    memset(y, 0, (size_t) nbytes);
    switch (type) {
        case T_Q8_0:  quantize_row_q8_0_ref (x, (block_q8_0  *) y, n); break;
        case T_Q4_K:  quantize_row_q4_K_ref (x, (block_q4_K  *) y, n); break;
        case T_Q5_K:  quantize_row_q5_K_ref (x, (block_q5_K  *) y, n); break;
        case T_Q6_K:  quantize_row_q6_K_ref (x, (block_q6_K  *) y, n); break;
        case T_Q3_K:  quantize_row_q3_K_ref (x, (block_q3_K  *) y, n); break;
        case T_IQ4_NL: quantize_row_iq4_nl_ref(x, (block_iq4_nl *) y, n); break;
        case T_IQ4_XS: quantize_row_iq4_xs_ref(x, (block_iq4_xs *) y, n); break;
        case T_IQ3_S: iq3xs_init_impl(512); quantize_row_iq3_s_ref (x, (block_iq3_s  *) y, n); break;
        default: fprintf(stderr, "unknown type %d\n", type); return 5;
    }
    if (fwrite(y, 1, (size_t) nbytes, stdout) != (size_t) nbytes) { return 6; }
    free(x); free(y);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <dequant|quantize> <type> <nelem> <nbytes>\n", argv[0]);
        return 2;
    }
    const char * mode  = argv[1];
    const int    type  = atoi(argv[2]);
    const int64_t n    = atoll(argv[3]);
    const int64_t nb   = atoll(argv[4]);
    if (strcmp(mode, "dequant") == 0)  return do_dequant(type, n, nb);
    if (strcmp(mode, "quantize") == 0) return do_quantize(type, n);
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
}
