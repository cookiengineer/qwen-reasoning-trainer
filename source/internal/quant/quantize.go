package quant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Quantize encodes src (float32) into a newly allocated buffer of type t.
// len(src) must be a multiple of t.BlockSize(). It returns ErrUnsupported for
// types whose encoder is not implemented.
func Quantize(t Type, src []float32) ([]byte, error) {
	size, err := t.Size(int64(len(src)))
	if err != nil {
		return nil, err
	}
	dst := make([]byte, size)
	if err := QuantizeTo(t, src, dst); err != nil {
		return nil, err
	}
	return dst, nil
}

// QuantizeTo encodes src into dst. dst must be exactly t.Size(len(src)) bytes.
func QuantizeTo(t Type, src []float32, dst []byte) error {
	switch t {
	case TypeF32:
		quantizeF32(src, dst)
	case TypeF16:
		quantizeF16(src, dst)
	case TypeBF16:
		quantizeBF16(src, dst)
	case TypeQ4_0:
		quantizeQ4_0(src, dst)
	case TypeQ8_0:
		quantizeQ8_0(src, dst)
	case TypeQ4_K:
		quantizeQ4K(src, dst)
	case TypeQ5_K:
		quantizeQ5K(src, dst)
	case TypeQ6_K:
		quantizeQ6K(src, dst)
	case TypeQ3_K:
		quantizeQ3K(src, dst)
	case TypeIQ4_NL:
		quantizeIQ4NL(src, dst)
	case TypeIQ4_XS:
		quantizeIQ4XS(src, dst)
	case TypeIQ3_S:
		quantizeIQ3S(src, dst)
	default:
		return errNoEncoder(t)
	}
	return nil
}

func errNoEncoder(t Type) error {
	return &UnsupportedError{Type: t, Op: "quantize"}
}

// UnsupportedError reports an unimplemented codec operation.
type UnsupportedError struct {
	Type Type
	Op   string
}

func (e *UnsupportedError) Error() string {
	return "quant: " + e.Op + " not implemented for type " + e.Type.String()
}

func (e *UnsupportedError) Is(target error) bool { return target == ErrUnsupported }

func putHalf(dst []byte, f float32) { binary.LittleEndian.PutUint16(dst, F32ToF16(f)) }

func quantizeF32(src []float32, dst []byte) {
	for i, v := range src {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(v))
	}
}

func quantizeF16(src []float32, dst []byte) {
	for i, v := range src {
		putHalf(dst[i*2:], v)
	}
}

func quantizeBF16(src []float32, dst []byte) {
	for i, v := range src {
		binary.LittleEndian.PutUint16(dst[i*2:], F32ToBF16(v))
	}
}

func quantizeQ4_0(src []float32, dst []byte) {
	const qk = 32
	for i := 0; i+qk <= len(src); i += qk {
		block := src[i : i+qk]
		var amax float32
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d := amax / 8
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		out := dst[i/qk*18:]
		putHalf(out[0:2], d)
		for j := 0; j < qk/2; j++ {
			x0 := int(math.Round(float64(block[j]*id))) + 8
			x1 := int(math.Round(float64(block[j+qk/2]*id))) + 8
			x0 = clampInt(x0, 0, 15)
			x1 = clampInt(x1, 0, 15)
			out[2+j] = byte(x0 | (x1 << 4))
		}
	}
}

func quantizeQ8_0(src []float32, dst []byte) {
	const qk = 32
	for i := 0; i+qk <= len(src); i += qk {
		block := src[i : i+qk]
		var amax float32
		for _, v := range block {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		d := amax / 127
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		out := dst[i/qk*34:]
		putHalf(out[0:2], d)
		for j := 0; j < qk; j++ {
			q := clampInt(int(math.Round(float64(block[j]*id))), -128, 127)
			out[2+j] = byte(int8(q))
		}
	}
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// quantizeQ4K encodes src (a multiple of QK_K elements) into Q4_K blocks. It is
// a float32-faithful port of llama.cpp's quantize_row_q4_K_ref (ggml-quants.c):
// each 32-element sub-block is quantized with the weighted make_qkx2_quants
// search (weights av_x + |x|), giving a 6-bit scale and min plus a shared fp16
// super-scale and super-min per 256-element super-block.
func quantizeQ4K(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: q4_K requires a multiple of %d elements", QK_K)
	}
	nb := len(src) / QK_K
	var L [QK_K]uint8
	var Laux [32]uint8
	var weights [32]float32
	var mins, scales [QK_K / 32]float32
	for i := 0; i < nb; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*144:]
		var maxScale, maxMin float32
		for j := 0; j < QK_K/32; j++ {
			var sumX2 float32
			for l := 0; l < 32; l++ {
				sumX2 += block[32*j+l] * block[32*j+l]
			}
			avX := sqrtf(sumX2 / 32)
			for l := 0; l < 32; l++ {
				weights[l] = avX + absf(block[32*j+l])
			}
			scales[j] = makeQKX2Quants(32, 15, block[32*j:32*j+32], weights[:], L[32*j:32*j+32], &mins[j], Laux[:], -1, 0.1, 20, false)
			if scales[j] > maxScale {
				maxScale = scales[j]
			}
			if mins[j] > maxMin {
				maxMin = mins[j]
			}
		}
		invScale := float32(0)
		if maxScale > 0 {
			invScale = 63 / maxScale
		}
		invMin := float32(0)
		if maxMin > 0 {
			invMin = 63 / maxMin
		}
		var ls, lm [QK_K / 32]uint8
		for j := 0; j < QK_K/32; j++ {
			ls[j] = uint8(minInt(63, nearestInt(invScale*scales[j])))
			lm[j] = uint8(minInt(63, nearestInt(invMin*mins[j])))
		}
		packScalesK4(b[4:16], ls, lm)
		putHalf(b[0:2], maxScale/63)
		putHalf(b[2:4], maxMin/63)
		dhf := F16ToF32(F32ToF16(maxScale / 63))
		dmhf := F16ToF32(F32ToF16(maxMin / 63))
		for j := 0; j < QK_K/32; j++ {
			sc, m := getScaleMinK4(j, b[4:16])
			d := dhf * float32(sc)
			if d == 0 {
				continue
			}
			dm := dmhf * float32(m)
			for ii := 0; ii < 32; ii++ {
				l := nearestInt((block[32*j+ii] + dm) / d)
				L[32*j+ii] = uint8(clampInt(l, 0, 15))
			}
		}
		q := b[16:144]
		for g := 0; g < 4; g++ {
			for l := 0; l < 32; l++ {
				q[g*32+l] = L[g*64+l] | (L[g*64+32+l] << 4)
			}
		}
	}
	return nil
}

func minMax(v []float32) (float64, float64) {
	lo, hi := float64(v[0]), float64(v[0])
	for _, x := range v {
		f := float64(x)
		if f < lo {
			lo = f
		}
		if f > hi {
			hi = f
		}
	}
	return lo, hi
}

func scaleOf(v, d float64) int {
	return roundHalf(v / d)
}

func roundHalf(f float64) int {
	if f < 0 {
		return -int(-f + 0.5)
	}
	return int(f + 0.5)
}

// CanQuantize reports whether an encoder exists for type t.
func CanQuantize(t Type) bool {
	switch t {
	case TypeF32, TypeF16, TypeBF16, TypeQ4_0, TypeQ8_0, TypeQ4_K, TypeQ5_K,
		TypeQ6_K, TypeQ3_K, TypeIQ4_NL, TypeIQ4_XS, TypeIQ3_S:
		return true
	default:
		return false
	}
}

func minMaxAbs(v []float32) float64 {
	var m float64
	for _, x := range v {
		a := math.Abs(float64(x))
		if a > m {
			m = a
		}
	}
	return m
}

// packScalesK4 packs 6-bit scale/min pairs the way Q4_K and Q5_K expect.
func packScalesK4(dst []byte, ls, lm [8]uint8) {
	var packed [12]byte
	for j := 0; j < 8; j++ {
		l, m := ls[j], lm[j]
		if j < 4 {
			packed[j] = l
			packed[j+4] = m
		} else {
			packed[j+4] = (l & 0xF) | ((m & 0xF) << 4)
			packed[j-4] |= (l >> 4) << 6
			packed[j] |= (m >> 4) << 6
		}
	}
	copy(dst, packed[:])
}

// quantizeQ5K encodes src into Q5_K blocks (176 bytes / 256 elements).
func quantizeQ5K(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: q5_K requires a multiple of %d elements", QK_K)
	}
	nb := len(src) / QK_K
	var L [QK_K]uint8
	var Laux [32]uint8
	var weights [32]float32
	var mins, scales [QK_K / 32]float32
	for i := 0; i < nb; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*176:]
		var maxScale, maxMin float32
		for j := 0; j < QK_K/32; j++ {
			var sumX2 float32
			for l := 0; l < 32; l++ {
				sumX2 += block[32*j+l] * block[32*j+l]
			}
			avX := sqrtf(sumX2 / 32)
			for l := 0; l < 32; l++ {
				weights[l] = avX + absf(block[32*j+l])
			}
			scales[j] = makeQKX2Quants(32, 31, block[32*j:32*j+32], weights[:], L[32*j:32*j+32], &mins[j], Laux[:], -0.5, 0.1, 15, false)
			if scales[j] > maxScale {
				maxScale = scales[j]
			}
			if mins[j] > maxMin {
				maxMin = mins[j]
			}
		}
		invScale := float32(0)
		if maxScale > 0 {
			invScale = 63 / maxScale
		}
		invMin := float32(0)
		if maxMin > 0 {
			invMin = 63 / maxMin
		}
		var ls, lm [QK_K / 32]uint8
		for j := 0; j < QK_K/32; j++ {
			ls[j] = uint8(minInt(63, nearestInt(invScale*scales[j])))
			lm[j] = uint8(minInt(63, nearestInt(invMin*mins[j])))
		}
		packScalesK4(b[4:16], ls, lm)
		putHalf(b[0:2], maxScale/63)
		putHalf(b[2:4], maxMin/63)
		dhf := F16ToF32(F32ToF16(maxScale / 63))
		dmhf := F16ToF32(F32ToF16(maxMin / 63))
		for j := 0; j < QK_K/32; j++ {
			sc, m := getScaleMinK4(j, b[4:16])
			d := dhf * float32(sc)
			if d == 0 {
				continue
			}
			dm := dmhf * float32(m)
			for ii := 0; ii < 32; ii++ {
				l := nearestInt((block[32*j+ii] + dm) / d)
				L[32*j+ii] = uint8(clampInt(l, 0, 31))
			}
		}
		qh := b[16:48]
		ql := b[48:176]
		for l := range qh {
			qh[l] = 0
		}
		var m1, m2 uint8 = 1, 2
		for n := 0; n < QK_K; n += 64 {
			for j := 0; j < 32; j++ {
				l1 := int(L[n+j])
				if l1 > 15 {
					l1 -= 16
					qh[j] |= m1
				}
				l2 := int(L[n+j+32])
				if l2 > 15 {
					l2 -= 16
					qh[j] |= m2
				}
				ql[j] = uint8(l1 | (l2 << 4))
			}
			m1 <<= 2
			m2 <<= 2
			ql = ql[32:]
		}
	}
	return nil
}

// quantizeQ6K encodes src into Q6_K blocks (210 bytes / 256 elements). It is a
// float32-faithful port of quantize_row_q6_K_ref: each 16-element group uses
// make_qx_quants (rmse_type 1), and the block stores ql/qh/scales/d in that
// order (see block_q6_K in ggml-common.h).
func quantizeQ6K(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: q6_K requires a multiple of %d elements", QK_K)
	}
	nb := len(src) / QK_K
	var L [QK_K]int8
	var scales [QK_K / 16]float32
	for i := 0; i < nb; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*210:]
		var maxScale, maxAbsScale float32
		for ib := 0; ib < QK_K/16; ib++ {
			scale := makeQXQuants(16, 32, block[16*ib:16*ib+16], L[16*ib:16*ib+16], 1, nil)
			scales[ib] = scale
			absScale := absf(scale)
			if absScale > maxAbsScale {
				maxAbsScale = absScale
				maxScale = scale
			}
		}
		if maxAbsScale < groupMaxEps {
			for j := range b {
				b[j] = 0
			}
			putHalf(b[208:210], 0)
			continue
		}
		iscale := -128 / maxScale
		putHalf(b[208:210], 1/iscale)
		sc := b[192:208]
		for ib := 0; ib < QK_K/16; ib++ {
			sc[ib] = byte(int8(minInt(127, nearestInt(iscale*scales[ib]))))
		}
		d := F16ToF32(F32ToF16(1 / iscale))
		for j := 0; j < QK_K/16; j++ {
			dv := d * float32(int8(sc[j]))
			if dv == 0 {
				continue
			}
			for ii := 0; ii < 16; ii++ {
				l := nearestInt(block[16*j+ii] / dv)
				L[16*j+ii] = int8(clampInt(l, -32, 31) + 32)
			}
		}
		ql := b[0:128]
		qh := b[128:192]
		for j := 0; j < QK_K; j += 128 {
			for l := 0; l < 32; l++ {
				q1 := L[j+l] & 0xF
				q2 := L[j+l+32] & 0xF
				q3 := L[j+l+64] & 0xF
				q4 := L[j+l+96] & 0xF
				ql[l] = uint8(q1 | (q3 << 4))
				ql[l+32] = uint8(q2 | (q4 << 4))
				qh[l] = uint8((L[j+l] >> 4) | ((L[j+l+32] >> 4) << 2) | ((L[j+l+64] >> 4) << 4) | ((L[j+l+96] >> 4) << 6))
			}
			ql = ql[64:]
			qh = qh[32:]
		}
	}
	return nil
}

// packQ3K packs 6-bit scales (centered at 32) the way Q3_K expects.
func packQ3K(dst []byte, scv [16]int) {
	for b := 0; b < 4; b++ {
		a0 := (scv[b] & 0xF) | ((scv[8+b] & 0xF) << 4)
		a1 := (scv[4+b] & 0xF) | ((scv[12+b] & 0xF) << 4)
		tmp := ((scv[b] >> 4) & 3) | (((scv[4+b] >> 4) & 3) << 2) |
			(((scv[8+b] >> 4) & 3) << 4) | (((scv[12+b] >> 4) & 3) << 6)
		dst[b] = byte(a0)
		dst[4+b] = byte(a1)
		dst[8+b] = byte(tmp)
	}
}

// quantizeQ3K encodes src into Q3_K blocks (110 bytes / 256 elements).
// quantizeQ3K encodes src into Q3_K blocks (110 bytes / 256 elements). It is a
// float32-faithful port of quantize_row_q3_K_ref: each 16-element group uses
// make_q3_quants (rmse), the block stores hmask/qs/scales/d in that order, and
// the high bit of each 2-bit quant is packed into hmask (one bit per group of 8).
func quantizeQ3K(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: q3_K requires a multiple of %d elements", QK_K)
	}
	nb := len(src) / QK_K
	var L [QK_K]int8
	var scales [QK_K / 16]float32
	for i := 0; i < nb; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*110:]
		var maxScale, amax float32
		for j := 0; j < QK_K/16; j++ {
			scales[j] = makeQ3Quants(16, 4, block[16*j:16*j+16], L[16*j:16*j+16], true)
			s := absf(scales[j])
			if s > amax {
				amax = s
				maxScale = scales[j]
			}
		}
		hmask := b[0:32]
		qs := b[32:96]
		sc := b[96:108]
		for j := range sc {
			sc[j] = 0
		}
		var dhf float32
		if maxScale != 0 {
			iscale := -32 / maxScale
			for j := 0; j < QK_K/16; j++ {
				l := clampInt(nearestInt(iscale*scales[j]), -32, 31) + 32
				if j < 8 {
					sc[j] = byte(l & 0xF)
				} else {
					sc[j-8] |= byte((l & 0xF) << 4)
				}
				l >>= 4
				sc[j%4+8] |= byte(l << (2 * (j / 4)))
			}
			putHalf(b[108:110], 1/iscale)
			dhf = F16ToF32(F32ToF16(1 / iscale))
		} else {
			putHalf(b[108:110], 0)
			dhf = 0
		}
		for j := 0; j < QK_K/16; j++ {
			var base int
			if j < 8 {
				base = int(sc[j] & 0xF)
			} else {
				base = int(sc[j-8] >> 4)
			}
			s := int8((base | (((int(sc[8+j%4]) >> (2 * (j / 4))) & 3) << 4)) - 32)
			d := dhf * float32(s)
			if d == 0 {
				continue
			}
			for ii := 0; ii < 16; ii++ {
				l := nearestInt(block[16*j+ii] / d)
				L[16*j+ii] = int8(clampInt(l, -4, 3) + 4)
			}
		}
		for j := range hmask {
			hmask[j] = 0
		}
		m := 0
		var hm uint8 = 1
		for j := 0; j < QK_K; j++ {
			if L[j] > 3 {
				hmask[m] |= hm
				L[j] -= 4
			}
			m++
			if m == QK_K/8 {
				m = 0
				hm <<= 1
			}
		}
		for j := 0; j < QK_K; j += 128 {
			for l := 0; l < 32; l++ {
				qs[j/4+l] = uint8(int(L[j+l]) | (int(L[j+l+32]) << 2) | (int(L[j+l+64]) << 4) | (int(L[j+l+96]) << 6))
			}
		}
	}
	return nil
}

// nearestKV returns the index into the IQ4 codebook closest to t.
func nearestKV(t float64) uint8 {
	best := 0
	bestD := math.Inf(1)
	for i, v := range iq4nlValues {
		d := math.Abs(float64(v) - t)
		if d < bestD {
			bestD = d
			best = i
		}
	}
	return uint8(best)
}

func divOr0(x float32, d float64) float64 {
	if d == 0 {
		return 0
	}
	return float64(x) / d
}

// bestIndexIQ4 ports best_index_int8 over the 16-value IQ4 codebook.
func bestIndexIQ4(x float32) int {
	const n = 16
	if x <= iq4nlValues[0] {
		return 0
	}
	if x >= iq4nlValues[n-1] {
		return n - 1
	}
	ml, mu := 0, n-1
	for mu-ml > 1 {
		mav := (ml + mu) / 2
		if x < iq4nlValues[mav] {
			mu = mav
		} else {
			ml = mav
		}
	}
	if x-iq4nlValues[mu-1] < iq4nlValues[mu]-x {
		return mu - 1
	}
	return mu
}

// iq4NLImpl is a float32-faithful port of quantize_row_iq4_nl_impl, shared by
// IQ4_NL (super=block=32, ntry=-1) and IQ4_XS (super=256, block=32, ntry=7).
// qw is the optional importance matrix (nil for the reference encoders; B3
// will pass one).
func iq4NLImpl(super, block int, x []float32, dh *uint16, q4 []byte, scalesH []uint16, scalesL []byte, scales, weight []float32, L []int8, qw []float32, ntry int) {
	var sigma2 float32
	for j := 0; j < super; j++ {
		sigma2 += x[j] * x[j]
	}
	sigma2 *= 2 / float32(super)
	for i := range q4 {
		q4[i] = 0
	}
	*dh = F32ToF16(0)
	var maxScale, amaxScale float32
	for ib := 0; ib < super/block; ib++ {
		xb := x[ib*block:]
		Lb := L[ib*block:]
		if qw != nil {
			qwb := qw[ib*block:]
			for j := 0; j < block; j++ {
				weight[j] = qwb[j] * sqrtf(sigma2+xb[j]*xb[j])
			}
		} else {
			for j := 0; j < block; j++ {
				weight[j] = xb[j] * xb[j]
			}
		}
		var amax, maxV float32
		for j := 0; j < block; j++ {
			ax := absf(xb[j])
			if ax > amax {
				amax = ax
				maxV = xb[j]
			}
		}
		if amax < groupMaxEps {
			scales[ib] = 0
			continue
		}
		var d float32
		if ntry > 0 {
			d = -maxV / iq4nlValues[0]
		} else {
			d = maxV / iq4nlValues[0]
		}
		id := 1 / d
		var sumqx, sumq2 float32
		for j := 0; j < block; j++ {
			l := bestIndexIQ4(id * xb[j])
			Lb[j] = int8(l)
			q := iq4nlValues[l]
			w := weight[j]
			sumqx += w * q * xb[j]
			sumq2 += w * q * q
		}
		if sumq2 > 0 {
			d = sumqx / sumq2
		} else {
			d = 0
		}
		best := d * sumqx
		for itry := -ntry; itry <= ntry; itry++ {
			id = (float32(itry) + iq4nlValues[0]) / maxV
			sumqx, sumq2 = 0, 0
			for j := 0; j < block; j++ {
				l := bestIndexIQ4(id * xb[j])
				q := iq4nlValues[l]
				w := weight[j]
				sumqx += w * q * xb[j]
				sumq2 += w * q * q
			}
			if sumq2 > 0 && sumqx*sumqx > best*sumq2 {
				d = sumqx / sumq2
				best = d * sumqx
			}
		}
		scales[ib] = d
		absD := absf(d)
		if absD > amaxScale {
			amaxScale = absD
			maxScale = d
		}
	}
	if super/block > 1 {
		d := -maxScale / 32
		*dh = F32ToF16(d)
		var id float32
		if d != 0 {
			id = 1 / d
		}
		for ib := 0; ib < super/block; ib++ {
			l := clampInt(nearestInt(id*scales[ib]), -32, 31)
			dl := d * float32(l)
			var idl float32
			if dl != 0 {
				idl = 1 / dl
			}
			Lb := L[ib*block:]
			xb := x[ib*block:]
			for j := 0; j < block; j++ {
				Lb[j] = int8(bestIndexIQ4(idl * xb[j]))
			}
			l += 32
			ll := uint8(l & 0xF)
			lh := uint8(l >> 4)
			if ib%2 == 0 {
				scalesL[ib/2] = ll
			} else {
				scalesL[ib/2] |= ll << 4
			}
			scalesH[ib/8] |= uint16(lh) << (2 * uint(ib%8))
		}
	} else {
		*dh = F32ToF16(scales[0])
		if ntry > 0 {
			var id float32
			if scales[0] != 0 {
				id = 1 / scales[0]
			}
			for j := 0; j < super; j++ {
				L[j] = int8(bestIndexIQ4(id * x[j]))
			}
		}
	}
	for i := 0; i < super/32; i++ {
		for j := 0; j < 16; j++ {
			q4[16*i+j] = byte(int(L[32*i+j]) | (int(L[32*i+16+j]) << 4))
		}
	}
}

// quantizeIQ4NL encodes src into IQ4_NL blocks (18 bytes / 32 elements).
func quantizeIQ4NL(src []float32, dst []byte) error {
	if len(src)%32 != 0 {
		return fmt.Errorf("quant: iq4_nl requires a multiple of 32 elements")
	}
	nblock := len(src) / 32
	var L [32]int8
	var weight [32]float32
	var scale [1]float32
	var dh uint16
	for ibl := 0; ibl < nblock; ibl++ {
		b := dst[ibl*18:]
		scale[0] = 0
		iq4NLImpl(32, 32, src[ibl*32:], &dh, b[2:18], nil, nil, scale[:], weight[:], L[:], nil, -1)
		binary.LittleEndian.PutUint16(b[0:2], dh)
	}
	return nil
}

// quantizeIQ4XS encodes src into IQ4_XS blocks (136 bytes / 256 elements).
func quantizeIQ4XS(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: iq4_xs requires a multiple of %d elements", QK_K)
	}
	nblock := len(src) / QK_K
	var L [QK_K]int8
	var weight [32]float32
	var scales [QK_K / 32]float32
	for ibl := 0; ibl < nblock; ibl++ {
		b := dst[ibl*136:]
		var dh uint16
		var scalesH [1]uint16
		var scalesL [4]byte
		iq4NLImpl(QK_K, 32, src[ibl*QK_K:], &dh, b[8:136], scalesH[:], scalesL[:], scales[:], weight[:], L[:], nil, 7)
		binary.LittleEndian.PutUint16(b[0:2], dh)
		binary.LittleEndian.PutUint16(b[2:4], scalesH[0])
		copy(b[4:8], scalesL[:])
	}
	return nil
}

// The IQ3_S encoder lives in iq3s_quant.go (shared grid with dequant).
