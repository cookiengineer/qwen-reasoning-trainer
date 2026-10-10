// Reference dequantizer harness.
//
// Reads: argv[1] = type id, argv[2] = element count, argv[3] = raw byte count;
// raw block bytes on stdin. Writes element_count little-endian float32 values to
// stdout. It calls the *actual* llama.cpp dequantization routines from
// references/llama_cpp/ggml/src/ggml-quants.c so the Go port (and, through the
// gemv/dequant parity tests, the Vulkan shaders) can be diffed against them.
//
// Build (see verify_quants_test.go / `make verify-quants`):
//   cc -O1 -w -I <ggml/src> -I <ggml/src/ggml-cpu> -I <ggml/include> \
//      -o ref_quants ref_quants.c <ggml/src>/ggml-quants.c -lm

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml-quants.h"

// ggml-quants.c references these ggml.c symbols from paths we never call
// (row-size validation and the quantizer asserts); provide tiny stubs so the
// harness links. ggml_abort is only reachable from the *quantize* routines,
// which this harness does not exercise.
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

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <type> <nelem> <nbytes>\n", argv[0]);
        return 2;
    }
    const int      type  = atoi(argv[1]);
    const int64_t  n     = atoll(argv[2]);
    const int64_t  nbytes = atoll(argv[3]);

    uint8_t * raw = (uint8_t *) malloc((size_t) nbytes);
    float   * y   = (float *)   malloc((size_t) n * sizeof(float));
    if (!raw || !y) {
        return 3;
    }
    if (fread(raw, 1, (size_t) nbytes, stdin) != (size_t) nbytes) {
        fprintf(stderr, "short read\n");
        return 4;
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
        default:
            fprintf(stderr, "unknown type %d\n", type);
            return 5;
    }

    if (fwrite(y, sizeof(float), (size_t) n, stdout) != (size_t) n) {
        return 6;
    }
    free(raw);
    free(y);
    return 0;
}
