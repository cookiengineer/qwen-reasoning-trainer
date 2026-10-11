// Shared per-element quantized-weight decoders for the fused GEMM kernels.
//
// Each function returns dequant(W[n, kk]) for a weight [K, N] whose rows are K
// elements long (GGML layout, row-contiguous; byte (n*K + kk)). It requires the
// including shader to declare, before this include:
//
//   layout(binding = 0) readonly buffer W { uint data[]; };
//   layout(push_constant) uniform P { uint K; ... } p;
//
// and that K is a multiple of the type's block size. Ported from the matching
// gemv_*.comp / internal/quant/dequant.go so the fused decode is element-exact.

#include "quant_iq3s_grid.glsl"

uint gb(uint o) { return (data[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu; }
float hf(uint o) { return unpackHalf2x16(gb(o) | (gb(o + 1u) << 8u)).x; }
int si8(uint o) { uint v = gb(o); return (v >= 128u) ? int(v) - 256 : int(v); }
uint u32at(uint o) {
    return gb(o) | (gb(o + 1u) << 8u) | (gb(o + 2u) << 16u) | (gb(o + 3u) << 24u);
}

const float KVNL[16] = float[16](
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);

// get_scale_min_k4: the scale part.
uint kmGetSc(uint j, uint s) {
    if (j < 4u) {
        return gb(s + j) & 63u;
    }
    return (gb(s + j + 4u) & 0xFu) | ((gb(s + j - 4u) >> 6) << 4);
}

// get_scale_min_k4: the min part.
uint kmGetMn(uint j, uint s) {
    if (j < 4u) {
        return gb(s + j + 4u) & 63u;
    }
    return (gb(s + j + 4u) >> 4) | ((gb(s + j) >> 6) << 4);
}

float deq_f16(uint n, uint kk) {
    return hf((n * p.K + kk) * 2u);
}

float deq_q8_0(uint n, uint kk) {
    uint base = (n * (p.K / 32u) + kk / 32u) * 34u;
    return hf(base) * float(si8(base + 2u + (kk % 32u)));
}

float deq_q4_k(uint n, uint kk) {
    uint base = (n * (p.K / 256u) + kk / 256u) * 144u;
    uint e = kk % 256u;
    uint group = e / 64u;
    uint within = e % 64u;
    uint sub = within / 32u;
    uint l = within % 32u;
    uint is = group * 2u + sub;
    float d = hf(base);
    float mn = hf(base + 2u);
    float sc = float(kmGetSc(is, base + 4u));
    float m = float(kmGetMn(is, base + 4u));
    uint q = gb(base + 16u + group * 32u + l);
    float nib = (sub == 0u) ? float(q & 0xFu) : float(q >> 4u);
    return d * sc * nib - mn * m;
}

float deq_q5_k(uint n, uint kk) {
    uint base = (n * (p.K / 256u) + kk / 256u) * 176u;
    uint e = kk % 256u;
    uint group = e / 64u;
    uint within = e % 64u;
    uint sub = within / 32u;
    uint l = within % 32u;
    uint is = group * 2u + sub;
    float d = hf(base);
    float mn = hf(base + 2u);
    float sc = float(kmGetSc(is, base + 4u));
    float m = float(kmGetMn(is, base + 4u));
    uint q = gb(base + 48u + group * 32u + l);
    uint qh = gb(base + 16u + l);
    uint bit = (sub == 0u) ? (2u * group) : (2u * group + 1u);
    uint hi = ((qh >> bit) & 1u) * 16u;
    uint nib = (sub == 0u) ? (q & 0xFu) : (q >> 4u);
    return d * sc * float(nib + hi) - mn * m;
}

float deq_q6_k(uint n, uint kk) {
    uint base = (n * (p.K / 256u) + kk / 256u) * 210u;
    uint e = kk % 256u;
    uint mblk = e / 128u;
    uint within = e % 128u;
    uint l = within % 32u;
    uint quarter = within / 32u;
    uint is = l / 16u;
    uint ql = base + 64u * mblk;
    uint qh = base + 128u + 32u * mblk;
    uint sc = base + 192u + 8u * mblk;
    int qv;
    if (quarter == 0u) {
        qv = int((gb(ql + l) & 0xFu) | (((gb(qh + l) >> 0u) & 3u) << 4u)) - 32;
    } else if (quarter == 1u) {
        qv = int((gb(ql + l + 32u) & 0xFu) | (((gb(qh + l) >> 2u) & 3u) << 4u)) - 32;
    } else if (quarter == 2u) {
        qv = int((gb(ql + l) >> 4u) | (((gb(qh + l) >> 4u) & 3u) << 4u)) - 32;
    } else {
        qv = int((gb(ql + l + 32u) >> 4u) | (((gb(qh + l) >> 6u) & 3u) << 4u)) - 32;
    }
    return hf(base + 208u) * float(si8(sc + is + 2u * quarter)) * float(qv);
}

float deq_q3_k(uint n, uint kk) {
    const uint KMASK1 = 0x03030303u;
    const uint KMASK2 = 0x0f0f0f0fu;
    uint base = (n * (p.K / 256u) + kk / 256u) * 110u;
    uint e = kk % 256u;
    uint nn = e / 128u;
    uint within = e % 128u;
    uint j = within / 32u;
    uint p32 = within % 32u;
    uint half_ = (p32 < 16u) ? 0u : 1u;
    uint l = (half_ == 0u) ? p32 : p32 - 16u;
    uint a0 = u32at(base + 96u);
    uint a1 = u32at(base + 100u);
    uint tmp = u32at(base + 104u);
    uint aux0 = (a0 & KMASK2) | (((tmp >> 0u) & KMASK1) << 4u);
    uint aux1 = (a1 & KMASK2) | (((tmp >> 2u) & KMASK1) << 4u);
    uint aux2 = ((a0 >> 4u) & KMASK2) | (((tmp >> 4u) & KMASK1) << 4u);
    uint aux3 = ((a1 >> 4u) & KMASK2) | (((tmp >> 6u) & KMASK1) << 4u);
    uint is = 8u * nn + 2u * j + half_;
    uint gi = is >> 2u;
    uint word = (gi == 0u) ? aux0 : (gi == 1u) ? aux1 : (gi == 2u) ? aux2 : aux3;
    int sc = int((word >> ((is & 3u) * 8u)) & 0xFFu);
    sc = (sc >= 128) ? sc - 256 : sc;
    uint qo = base + 32u + 32u * nn + ((half_ == 0u) ? l : l + 16u);
    uint hmbit = 1u << (nn * 4u + j);
    int v = int((gb(qo) >> (2u * j)) & 3u);
    if ((gb(base + ((half_ == 0u) ? l : l + 16u)) & hmbit) == 0u) {
        v -= 4;
    }
    return hf(base + 108u) * float(sc - 32) * float(v);
}

float deq_iq4_nl(uint n, uint kk) {
    uint base = (n * (p.K / 32u) + kk / 32u) * 18u;
    uint e = kk % 32u;
    uint b = gb(base + 2u + (e % 16u));
    uint code = (e < 16u) ? (b & 0xFu) : (b >> 4u);
    return hf(base) * KVNL[code];
}

float deq_iq4_xs(uint n, uint kk) {
    uint base = (n * (p.K / 256u) + kk / 256u) * 136u;
    uint e = kk % 256u;
    uint ib = e / 32u;
    uint within = e % 32u;
    uint sh = gb(base + 2u) | (gb(base + 3u) << 8u);
    uint ls = ((gb(base + 4u + (ib >> 1u)) >> (4u * (ib & 1u))) & 0xFu)
            | (((sh >> (2u * ib)) & 3u) << 4u);
    float dl = hf(base) * float(int(ls) - 32);
    uint qb = gb(base + 8u + ib * 16u + (within % 16u));
    uint code = (within < 16u) ? (qb & 0xFu) : (qb >> 4u);
    return dl * KVNL[code];
}

float deq_iq3_s(uint n, uint kk) {
    uint base = (n * (p.K / 256u) + kk / 256u) * 110u;
    uint e = kk % 256u;
    uint it = e / 64u;
    uint within = e % 64u;
    uint firstHalf = (within < 32u) ? 1u : 0u;
    uint pos = within % 32u;
    uint l = pos / 8u;
    uint j8 = pos % 8u;
    float d = hf(base);
    uint scByte = gb(base + 106u + it);
    // The first (db1) and second (db2) half of each 64-element group consume
    // consecutive qs/signs windows; the second half's pointers are advanced.
    uint sh2 = (firstHalf == 1u) ? 0u : 1u;
    float db = (firstHalf == 1u) ? d * float(1 + 2 * int(scByte & 0xFu))
                                 : d * float(1 + 2 * int(scByte >> 4u));
    uint qsBase = base + 2u + 16u * it + 8u * sh2;
    uint qhByte = gb(base + 66u + 2u * it + sh2);
    uint v1 = gb(qsBase + 2u * l);
    uint v2 = gb(qsBase + 2u * l + 1u);
    uint idx1 = v1 | ((qhByte << (8u - 2u * l)) & 256u);
    uint idx2 = v2 | ((qhByte << (7u - 2u * l)) & 256u);
    uint grid = (j8 < 4u) ? GRID[idx1] : GRID[idx2];
    uint sByte = gb(base + 74u + 8u * it + 4u * sh2 + l);
    float sign = ((sByte >> j8) & 1u) != 0u ? -1.0 : 1.0;
    return db * float((grid >> (8u * (j8 % 4u))) & 0xFFu) * sign;
}
