package quant

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// ErrUnsupported is returned when a codec is not implemented for a type.
var ErrUnsupported = errors.New("quant: unsupported type")

// DequantSupported reports whether DequantTo can decode type t.
func (t Type) DequantSupported() bool {
	switch t {
	case TypeF32, TypeF16, TypeBF16, TypeQ4_0, TypeQ5_0, TypeQ8_0,
		TypeQ4_K, TypeQ5_K, TypeQ6_K, TypeQ3_K, TypeIQ4_NL, TypeIQ4_XS, TypeIQ3_S:
		return true
	default:
		return false
	}
}

// iq4nlValues is the non-linear 4-bit codebook, ported from kvalues_iq4nl in
// references/llama_cpp/ggml/src/ggml-common.h.
var iq4nlValues = [16]float32{
	-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
}

// kmaskIq2xs holds the sign-bit masks for IQ grids, ported from kmask_iq2xs in
// references/llama_cpp/ggml/src/ggml-common.h.
var kmaskIq2xs = [8]uint8{1, 2, 4, 8, 16, 32, 64, 128}

func halfAt(b []byte) float32 { return F16ToF32(binary.LittleEndian.Uint16(b)) }

// Dequant decodes n elements of type t from src into a newly allocated float32
// slice. n must be a multiple of t.BlockSize().
func Dequant(t Type, src []byte, n int64) ([]float32, error) {
	dst := make([]float32, n)
	if err := DequantTo(t, src, dst); err != nil {
		return nil, err
	}
	return dst, nil
}

// DequantTo decodes len(dst) elements of type t from src into dst. The number
// of elements must be a multiple of the block size and src must be large
// enough. It reuses the caller-provided dst to avoid allocations.
func DequantTo(t Type, src []byte, dst []float32) error {
	n := int64(len(dst))
	need, err := t.Size(n)
	if err != nil {
		return err
	}
	if int64(len(src)) < need {
		return fmt.Errorf("quant: %s: source too small: have %d, need %d", t, len(src), need)
	}
	switch t {
	case TypeF32:
		dequantF32(src, dst)
	case TypeF16:
		dequantF16(src, dst)
	case TypeBF16:
		dequantBF16(src, dst)
	case TypeQ4_0:
		dequantQ4_0(src, dst)
	case TypeQ5_0:
		dequantQ5_0(src, dst)
	case TypeQ8_0:
		dequantQ8_0(src, dst)
	case TypeQ4_K:
		dequantQ4K(src, dst)
	case TypeQ5_K:
		dequantQ5K(src, dst)
	case TypeQ6_K:
		dequantQ6K(src, dst)
	case TypeQ3_K:
		dequantQ3K(src, dst)
	case TypeIQ4_NL:
		dequantIQ4NL(src, dst)
	case TypeIQ4_XS:
		dequantIQ4XS(src, dst)
	case TypeIQ3_S:
		dequantIQ3S(src, dst)
	default:
		return fmt.Errorf("%w: dequantize %s", ErrUnsupported, t)
	}
	return nil
}

func dequantF32(src []byte, dst []float32) {
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
}

func dequantF16(src []byte, dst []float32) {
	for i := range dst {
		dst[i] = F16ToF32(binary.LittleEndian.Uint16(src[i*2:]))
	}
}

func dequantBF16(src []byte, dst []float32) {
	for i := range dst {
		dst[i] = BF16ToF32(binary.LittleEndian.Uint16(src[i*2:]))
	}
}

func dequantQ4_0(src []byte, dst []float32) {
	const qk = 32
	for i := 0; i+qk <= len(dst); i += qk {
		b := src[i/qk*18:]
		d := halfAt(b)
		for j := 0; j < qk/2; j++ {
			x0 := int(b[2+j]&0x0F) - 8
			x1 := int(b[2+j]>>4) - 8
			dst[i+j] = float32(x0) * d
			dst[i+j+qk/2] = float32(x1) * d
		}
	}
}

func dequantQ5_0(src []byte, dst []float32) {
	const qk = 32
	for i := 0; i+qk <= len(dst); i += qk {
		b := src[i/qk*22:]
		d := halfAt(b)
		qh := binary.LittleEndian.Uint32(b[2:6])
		for j := 0; j < qk/2; j++ {
			xh0 := uint8(((qh >> j) << 4) & 0x10)
			xh1 := uint8((qh >> (j + 12)) & 0x10)
			x0 := int(int32(b[6+j]&0x0F)|int32(xh0)) - 16
			x1 := int(int32(b[6+j]>>4)|int32(xh1)) - 16
			dst[i+j] = float32(x0) * d
			dst[i+j+qk/2] = float32(x1) * d
		}
	}
}

func dequantQ8_0(src []byte, dst []float32) {
	const qk = 32
	for i := 0; i+qk <= len(dst); i += qk {
		b := src[i/qk*34:]
		d := halfAt(b)
		for j := 0; j < qk; j++ {
			dst[i+j] = float32(int8(b[2+j])) * d
		}
	}
}

// getScaleMinK4 ports get_scale_min_k4 from ggml-quants.c.
func getScaleMinK4(j int, q []byte) (uint8, uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	d := (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
	m := (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	return d, m
}

func dequantQ4K(src []byte, dst []float32) {
	for i := 0; i+QK_K <= len(dst); i += QK_K {
		b := src[i/QK_K*144:]
		d := halfAt(b[0:2])
		min := halfAt(b[2:4])
		scales := b[4:16]
		q := b[16:144]
		is := 0
		out := i
		for j := 0; j < QK_K; j += 64 {
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1, mn1 := d*float32(sc1), min*float32(m1)
			d2, mn2 := d*float32(sc2), min*float32(m2)
			for l := 0; l < 32; l++ {
				dst[out+l] = d1*float32(q[l]&0x0F) - mn1
			}
			for l := 0; l < 32; l++ {
				dst[out+32+l] = d2*float32(q[l]>>4) - mn2
			}
			out += 64
			q = q[32:]
			is += 2
		}
	}
}

func dequantQ5K(src []byte, dst []float32) {
	for i := 0; i+QK_K <= len(dst); i += QK_K {
		b := src[i/QK_K*176:]
		d := halfAt(b[0:2])
		min := halfAt(b[2:4])
		scales := b[4:16]
		qh := b[16:48]
		ql := b[48:176]
		is := 0
		u1, u2 := uint8(1), uint8(2)
		out := i
		for j := 0; j < QK_K; j += 64 {
			sc1, m1 := getScaleMinK4(is, scales)
			sc2, m2 := getScaleMinK4(is+1, scales)
			d1, mn1 := d*float32(sc1), min*float32(m1)
			d2, mn2 := d*float32(sc2), min*float32(m2)
			for l := 0; l < 32; l++ {
				hi := 0
				if qh[l]&u1 != 0 {
					hi = 16
				}
				dst[out+l] = d1*float32(int(ql[l]&0x0F)+hi) - mn1
			}
			for l := 0; l < 32; l++ {
				hi := 0
				if qh[l]&u2 != 0 {
					hi = 16
				}
				dst[out+32+l] = d2*float32(int(ql[l]>>4)+hi) - mn2
			}
			out += 64
			ql = ql[32:]
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
}

func dequantQ6K(src []byte, dst []float32) {
	for i := 0; i+QK_K <= len(dst); i += QK_K {
		b := src[i/QK_K*210:]
		ql := b[0:128]
		qh := b[128:192]
		sc := b[192:208]
		d := halfAt(b[208:210])
		out := i
		for n := 0; n < QK_K; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := ((int(ql[l]) & 0xF) | (((int(qh[l]) >> 0) & 3) << 4)) - 32
				q2 := ((int(ql[l+32]) & 0xF) | (((int(qh[l]) >> 2) & 3) << 4)) - 32
				q3 := ((int(ql[l]) >> 4) | (((int(qh[l]) >> 4) & 3) << 4)) - 32
				q4 := ((int(ql[l+32]) >> 4) | (((int(qh[l]) >> 6) & 3) << 4)) - 32
				dst[out+l] = d * float32(int8(sc[is+0])) * float32(q1)
				dst[out+l+32] = d * float32(int8(sc[is+2])) * float32(q2)
				dst[out+l+64] = d * float32(int8(sc[is+4])) * float32(q3)
				dst[out+l+96] = d * float32(int8(sc[is+6])) * float32(q4)
			}
			out += 128
			ql = ql[64:]
			qh = qh[32:]
			sc = sc[8:]
		}
	}
}

func dequantIQ4NL(src []byte, dst []float32) {
	const qk = 32
	for i := 0; i+qk <= len(dst); i += qk {
		b := src[i/qk*18:]
		d := halfAt(b[0:2])
		qs := b[2:18]
		for j := 0; j < 16; j++ {
			dst[i+j] = d * iq4nlValues[qs[j]&0x0F]
			dst[i+j+16] = d * iq4nlValues[qs[j]>>4]
		}
	}
}

func dequantIQ4XS(src []byte, dst []float32) {
	for i := 0; i+QK_K <= len(dst); i += QK_K {
		b := src[i/QK_K*136:]
		d := halfAt(b[0:2])
		scalesH := binary.LittleEndian.Uint16(b[2:4])
		scalesL := b[4:8]
		qs := b[8:136]
		out := i
		for ib := 0; ib < QK_K/32; ib++ {
			ls := int((scalesL[ib/2]>>(4*uint(ib%2)))&0xF) | int(((scalesH>>(2*uint(ib)))&3)<<4)
			dl := d * float32(ls-32)
			for j := 0; j < 16; j++ {
				dst[out+j] = dl * iq4nlValues[qs[j]&0x0F]
				dst[out+j+16] = dl * iq4nlValues[qs[j]>>4]
			}
			out += 32
			qs = qs[16:]
		}
	}
}

func dequantQ3K(src []byte, dst []float32) {
	const kmask1 = 0x03030303
	const kmask2 = 0x0f0f0f0f
	for i := 0; i+QK_K <= len(dst); i += QK_K {
		b := src[i/QK_K*110:]
		hm := b[0:32]
		q := b[32:96]
		scalesRaw := b[96:108]
		dAll := halfAt(b[108:110])

		a0 := binary.LittleEndian.Uint32(scalesRaw[0:4])
		a1 := binary.LittleEndian.Uint32(scalesRaw[4:8])
		tmp := binary.LittleEndian.Uint32(scalesRaw[8:12])
		aux0 := (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
		aux1 := (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
		aux2 := ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
		aux3 := ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)

		var sc [16]int8
		putLE := func(d []int8, v uint32) {
			d[0], d[1], d[2], d[3] = int8(v), int8(v>>8), int8(v>>16), int8(v>>24)
		}
		putLE(sc[0:4], aux0)
		putLE(sc[4:8], aux1)
		putLE(sc[8:12], aux2)
		putLE(sc[12:16], aux3)

		out := i
		is := 0
		m := uint8(1)
		for n := 0; n < QK_K; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := dAll * float32(int(sc[is])-32)
				is++
				for l := 0; l < 16; l++ {
					v := int8((q[l] >> shift) & 3)
					if hm[l]&m == 0 {
						v -= 4
					}
					dst[out] = dl * float32(v)
					out++
				}
				dl = dAll * float32(int(sc[is])-32)
				is++
				for l := 0; l < 16; l++ {
					v := int8((q[l+16] >> shift) & 3)
					if hm[l+16]&m == 0 {
						v -= 4
					}
					dst[out] = dl * float32(v)
					out++
				}
				shift += 2
				m <<= 1
			}
			q = q[32:]
		}
	}
}

func dequantIQ3S(src []byte, dst []float32) {
	for i := 0; i+QK_K <= len(dst); i += QK_K {
		b := src[i/QK_K*110:]
		d := halfAt(b[0:2])
		qs := b[2:66]
		qh := b[66:74]
		signs := b[74:106]
		scales := b[106:110]

		out := i
		for ib32 := 0; ib32 < QK_K/32; ib32 += 2 {
			db1 := d * float32(1+2*(scales[ib32/2]&0xF))
			db2 := d * float32(1+2*(scales[ib32/2]>>4))
			for l := 0; l < 4; l++ {
				idx1 := int(qs[2*l]) | ((int(qh[0]) << (8 - 2*l)) & 256)
				idx2 := int(qs[2*l+1]) | ((int(qh[0]) << (7 - 2*l)) & 256)
				g1 := iq3sGrid[idx1]
				g2 := iq3sGrid[idx2]
				s := signs[l]
				for j := 0; j < 4; j++ {
					out = writeIQ3SValue(dst, out, db1, g1, j, s, j)
				}
				for j := 0; j < 4; j++ {
					out = writeIQ3SValue(dst, out, db1, g2, j, s, j+4)
				}
			}
			qs = qs[8:]
			signs = signs[4:]
			for l := 0; l < 4; l++ {
				idx1 := int(qs[2*l]) | ((int(qh[1]) << (8 - 2*l)) & 256)
				idx2 := int(qs[2*l+1]) | ((int(qh[1]) << (7 - 2*l)) & 256)
				g1 := iq3sGrid[idx1]
				g2 := iq3sGrid[idx2]
				s := signs[l]
				for j := 0; j < 4; j++ {
					out = writeIQ3SValue(dst, out, db2, g1, j, s, j)
				}
				for j := 0; j < 4; j++ {
					out = writeIQ3SValue(dst, out, db2, g2, j, s, j+4)
				}
			}
			qh = qh[2:]
			qs = qs[8:]
			signs = signs[4:]
		}
	}
}

// writeIQ3SValue writes one IQ3_S output and returns the next index. The grid
// byte index and sign-mask bit index are given separately.
func writeIQ3SValue(dst []float32, out int, db float32, grid uint32, gridByte int, signMask uint8, signBit int) int {
	v := float32(byte(grid >> (8 * uint(gridByte))))
	if signMask&kmaskIq2xs[signBit] != 0 {
		v = -v
	}
	dst[out] = db * v
	return out + 1
}
