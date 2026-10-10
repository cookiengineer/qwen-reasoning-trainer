// Fused quantized-weight GEMM body: C[n,m] = sum_k dequant(W[n,k]) * B[m,k].
//
// W is a quantized weight [K, N] (rows of K elements, k-contiguous); B is the
// float32 activation [K, M] in GGML layout. The weight is decoded in registers
// into the shared A-tile, so it is never materialized as float32.
//
// The including shader must supply #version, define QDECODE(n, kk), and include
// this file. Same 64x64 / 16x16 / 4x4 register-tile structure as matmul.comp.

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) readonly  buffer W { uint data[]; };
layout(binding = 1) readonly  buffer B { float b[]; };
layout(binding = 2) writeonly buffer C { float c[]; };

layout(push_constant) uniform P {
    uint K;
    uint N;
    uint M;
    uint pad;
} p;

#include "quant_decode.glsl"

#define TM 64
#define TN 64
#define TK 16
#define TMICRO 4
#define TNMICRO 4

shared float As[TK][TM + 4];
shared float Bs[TK][TN + 4];

void main() {
    uint tx = gl_LocalInvocationID.x; // 0..15
    uint ty = gl_LocalInvocationID.y; // 0..15
    uint tid = ty * 16u + tx;
    uint n0 = gl_WorkGroupID.x * TM;
    uint m0 = gl_WorkGroupID.y * TN;

    float acc[TMICRO][TNMICRO];
    for (uint i = 0u; i < TMICRO; i++) {
        for (uint j = 0u; j < TNMICRO; j++) {
            acc[i][j] = 0.0;
        }
    }

    for (uint k0 = 0u; k0 < p.K; k0 += TK) {
        // A slice: As[k][r] = dequant(W[n0+r, k0+k])
        for (uint idx = tid; idx < TM * TK; idx += 256u) {
            uint r = idx / TK;
            uint k = idx - r * TK;
            uint n = n0 + r;
            uint kk = k0 + k;
            As[k][r] = (n < p.N && kk < p.K) ? QDECODE(n, kk) : 0.0;
        }
        // B slice: Bs[k][r] = B[(m0+r), k0+k]
        for (uint idx = tid; idx < TN * TK; idx += 256u) {
            uint r = idx / TK;
            uint k = idx - r * TK;
            uint m = m0 + r;
            uint kk = k0 + k;
            Bs[k][r] = (m < p.M && kk < p.K) ? b[m * p.K + kk] : 0.0;
        }
        barrier();

        for (uint k = 0u; k < TK; k++) {
            float av[TMICRO];
            float bv[TNMICRO];
            for (uint i = 0u; i < TMICRO; i++) {
                av[i] = As[k][ty * TMICRO + i];
            }
            for (uint j = 0u; j < TNMICRO; j++) {
                bv[j] = Bs[k][tx * TNMICRO + j];
            }
            for (uint i = 0u; i < TMICRO; i++) {
                for (uint j = 0u; j < TNMICRO; j++) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }
        barrier();
    }

    for (uint i = 0u; i < TMICRO; i++) {
        uint n = n0 + ty * TMICRO + i;
        for (uint j = 0u; j < TNMICRO; j++) {
            uint m = m0 + tx * TNMICRO + j;
            if (n < p.N && m < p.M) {
                c[m * p.N + n] = acc[i][j];
            }
        }
    }
}
