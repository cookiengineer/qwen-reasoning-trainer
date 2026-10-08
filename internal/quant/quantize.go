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

// quantizeQ4K encodes src (a multiple of QK_K elements) into Q4_K blocks. The
// scheme is a valid, self-consistent K-quant layout (not bit-identical to
// llama.cpp's quantizer): each 256-element super-block is split into eight
// 32-element sub-blocks with a 6-bit scale and min, plus a shared fp16
// super-scale and super-min.
func quantizeQ4K(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: q4_K requires a multiple of %d elements", QK_K)
	}
	nb := len(src) / QK_K
	for i := 0; i < nb; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*144:]

		var scale, min float64
		scales := [8]float64{}
		mins := [8]float64{}
		for j := 0; j < 8; j++ {
			lo, hi := minMax(block[j*32 : (j+1)*32])
			scales[j] = (hi - lo) / 15
			mins[j] = -lo
			if mins[j] < 0 {
				mins[j] = 0
			}
			if scales[j] > scale {
				scale = scales[j]
			}
			if mins[j] > min {
				min = mins[j]
			}
		}
		d := scale / 63
		dmin := min / 63
		putHalf(b[0:2], float32(d))
		putHalf(b[2:4], float32(dmin))
		d = float64(F16ToF32(F32ToF16(float32(d))))
		dmin = float64(F16ToF32(F32ToF16(float32(dmin))))

		var ls, lm [8]uint8
		for j := 0; j < 8; j++ {
			sc, mn := 0, 0
			if d > 0 {
				sc = clampInt(int(scaleOf(scales[j], d)), 0, 63)
			}
			if dmin > 0 {
				mn = clampInt(int(scaleOf(mins[j], dmin)), 0, 63)
			}
			ls[j] = uint8(sc)
			lm[j] = uint8(mn)
		}
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
		copy(b[4:16], packed[:])

		var L [QK_K]uint8
		for j := 0; j < 8; j++ {
			scf := d * float64(ls[j])
			mf := dmin * float64(lm[j])
			for l := 0; l < 32; l++ {
				val := 0
				if scf > 0 {
					val = clampInt(roundHalf((float64(block[j*32+l])+mf)/scf), 0, 15)
				}
				L[j*32+l] = uint8(val)
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
	for i := 0; i < len(src)/QK_K; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*176:]
		var scale, min float64
		var scales, mins [8]float64
		for j := 0; j < 8; j++ {
			lo, hi := minMax(block[j*32 : (j+1)*32])
			scales[j] = (hi - lo) / 31
			mins[j] = -lo
			if mins[j] < 0 {
				mins[j] = 0
			}
			if scales[j] > scale {
				scale = scales[j]
			}
			if mins[j] > min {
				min = mins[j]
			}
		}
		d := scale / 63
		dmin := min / 63
		putHalf(b[0:2], float32(d))
		putHalf(b[2:4], float32(dmin))
		d = float64(F16ToF32(F32ToF16(float32(d))))
		dmin = float64(F16ToF32(F32ToF16(float32(dmin))))
		var ls, lm [8]uint8
		for j := 0; j < 8; j++ {
			sc, mn := 0, 0
			if d > 0 {
				sc = clampInt(roundHalf(scales[j]/d), 0, 63)
			}
			if dmin > 0 {
				mn = clampInt(roundHalf(mins[j]/dmin), 0, 63)
			}
			ls[j] = uint8(sc)
			lm[j] = uint8(mn)
		}
		packScalesK4(b[4:16], ls, lm)
		var L [QK_K]uint8
		for j := 0; j < 8; j++ {
			scf := d * float64(ls[j])
			mf := dmin * float64(lm[j])
			for l := 0; l < 32; l++ {
				val := 0
				if scf > 0 {
					val = clampInt(roundHalf((float64(block[j*32+l])+mf)/scf), 0, 31)
				}
				L[j*32+l] = uint8(val)
			}
		}
		qs := b[48:176]
		qh := b[16:48]
		for l := range qh {
			qh[l] = 0
		}
		for g := 0; g < 4; g++ {
			u1 := uint8(1 << (2 * g))
			u2 := uint8(2 << (2 * g))
			for l := 0; l < 32; l++ {
				lo := L[g*64+l]
				hi := L[g*64+32+l]
				qs[g*32+l] = (lo & 0xF) | ((hi & 0xF) << 4)
				if lo >= 16 {
					qh[l] |= u1
				}
				if hi >= 16 {
					qh[l] |= u2
				}
			}
		}
	}
	return nil
}

// quantizeQ6K encodes src into Q6_K blocks (210 bytes / 256 elements).
func quantizeQ6K(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: q6_K requires a multiple of %d elements", QK_K)
	}
	for i := 0; i < len(src)/QK_K; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*210:]
		ql := b[0:128]
		qh := b[128:192]
		sc := b[192:208]
		for j := range ql {
			ql[j] = 0
		}
		for j := range qh {
			qh[j] = 0
		}
		var scale [16]float64
		var smax float64
		for j := 0; j < 16; j++ {
			scale[j] = minMaxAbs(block[j*16:(j+1)*16]) / 32
			if scale[j] > smax {
				smax = scale[j]
			}
		}
		d := smax / 127
		putHalf(b[208:210], float32(d))
		d = float64(F16ToF32(F32ToF16(float32(d))))
		var scv [16]int
		for j := 0; j < 16; j++ {
			v := 1
			if d > 0 {
				v = clampInt(roundHalf(scale[j]/d), 1, 127)
			}
			scv[j] = v
			sc[j] = byte(int8(v))
		}
		q := func(x float64, s int) uint8 {
			den := d * float64(s)
			if den == 0 {
				return 32
			}
			v := clampInt(roundHalf(x/den), -32, 31)
			return uint8(v + 32)
		}
		for g := 0; g < 2; g++ {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := q(float64(block[g*128+l]), scv[g*8+is])
				q2 := q(float64(block[g*128+l+32]), scv[g*8+is+2])
				q3 := q(float64(block[g*128+l+64]), scv[g*8+is+4])
				q4 := q(float64(block[g*128+l+96]), scv[g*8+is+6])
				ql[g*64+l] = (q1 & 0xF) | ((q3 & 0xF) << 4)
				ql[g*64+l+32] = (q2 & 0xF) | ((q4 & 0xF) << 4)
				qh[g*32+l] = ((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) |
					(((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6)
			}
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
func quantizeQ3K(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: q3_K requires a multiple of %d elements", QK_K)
	}
	for i := 0; i < len(src)/QK_K; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*110:]
		hmask := b[0:32]
		qs := b[32:96]
		for j := range hmask {
			hmask[j] = 0
		}
		for j := range qs {
			qs[j] = 0
		}
		var amax [16]float64
		var gmax float64
		for j := 0; j < 16; j++ {
			amax[j] = minMaxAbs(block[j*16 : (j+1)*16])
			if amax[j] > gmax {
				gmax = amax[j]
			}
		}
		d := gmax / 3 / 31
		putHalf(b[108:110], float32(d))
		d = float64(F16ToF32(F32ToF16(float32(d))))
		var scv [16]int
		for j := 0; j < 16; j++ {
			sc := 32
			if d > 0 && amax[j] > 0 {
				sc = 32 + clampInt(roundHalf((amax[j]/3)/d), 0, 31)
			}
			scv[j] = sc
		}
		packQ3K(b[96:108], scv)
		for e := 0; e < QK_K; e++ {
			n := e / 128
			r := e % 128
			j := r / 32
			within := r % 32
			hi := within >= 16
			sIdx := n*8 + j*2
			if hi {
				sIdx++
			}
			dl := d * float64(scv[sIdx]-32)
			var q int
			if dl != 0 {
				q = roundHalf(float64(block[e]) / dl)
			}
			qsIdx := n*32 + within
			if q < 0 {
				q = clampInt(q+4, 0, 3)
				qs[qsIdx] |= uint8(q) << (2 * j)
			} else {
				q = clampInt(q, 0, 3)
				qs[qsIdx] |= uint8(q) << (2 * j)
				hmask[within] |= uint8(1 << (n*4 + j))
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

// quantizeIQ4NL encodes src into IQ4_NL blocks (18 bytes / 32 elements).
func quantizeIQ4NL(src []float32, dst []byte) error {
	if len(src)%32 != 0 {
		return fmt.Errorf("quant: iq4_nl requires a multiple of 32 elements")
	}
	for i := 0; i < len(src)/32; i++ {
		block := src[i*32 : (i+1)*32]
		b := dst[i*18:]
		d := minMaxAbs(block) / 127
		putHalf(b[0:2], float32(d))
		d = float64(F16ToF32(F32ToF16(float32(d))))
		for j := 0; j < 16; j++ {
			lo := nearestKV(divOr0(block[j], d))
			hi := nearestKV(divOr0(block[j+16], d))
			b[2+j] = lo | (hi << 4)
		}
	}
	return nil
}

// quantizeIQ4XS encodes src into IQ4_XS blocks (136 bytes / 256 elements).
func quantizeIQ4XS(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: iq4_xs requires a multiple of %d elements", QK_K)
	}
	for i := 0; i < len(src)/QK_K; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*136:]
		var amax [8]float64
		var gmax float64
		for ib := 0; ib < 8; ib++ {
			amax[ib] = minMaxAbs(block[ib*32 : (ib+1)*32])
			if amax[ib] > gmax {
				gmax = amax[ib]
			}
		}
		d := gmax / 127 / 31
		putHalf(b[0:2], float32(d))
		d = float64(F16ToF32(F32ToF16(float32(d))))
		var ls [8]int
		for ib := 0; ib < 8; ib++ {
			rel := 0
			if d > 0 && amax[ib] > 0 {
				rel = clampInt(roundHalf((amax[ib]/127)/d), 1, 31)
			}
			ls[ib] = 32 + rel
		}
		var scalesH uint16
		var scalesL [4]byte
		for ib := 0; ib < 8; ib++ {
			v := ls[ib]
			scalesL[ib/2] |= byte(v&0xF) << (4 * uint(ib%2))
			scalesH |= uint16((v>>4)&3) << (2 * uint(ib))
		}
		binary.LittleEndian.PutUint16(b[2:4], scalesH)
		copy(b[4:8], scalesL[:])
		qs := b[8:136]
		for ib := 0; ib < 8; ib++ {
			dl := d * float64(ls[ib]-32)
			for j := 0; j < 16; j++ {
				lo := nearestKV(divOr0(block[ib*32+j], dl))
				hi := nearestKV(divOr0(block[ib*32+j+16], dl))
				qs[ib*16+j] = lo | (hi << 4)
			}
		}
	}
	return nil
}

// bestGrid finds the IQ3_S grid entry and sign mask best matching four targets.
func bestGrid(v []float32, db float64) (int, byte) {
	var t [4]float64
	for j := 0; j < 4; j++ {
		t[j] = divOr0(v[j], db)
	}
	best := 0
	bestCost := math.Inf(1)
	for gi := 0; gi < 512; gi++ {
		g := iq3sGrid[gi]
		cost := 0.0
		for j := 0; j < 4; j++ {
			gv := float64((g >> uint(8*j)) & 0xFF)
			cost += math.Abs(gv - math.Abs(t[j]))
		}
		if cost < bestCost {
			bestCost = cost
			best = gi
		}
	}
	var signs byte
	for j := 0; j < 4; j++ {
		if t[j] < 0 {
			signs |= 1 << uint(j)
		}
	}
	return best, signs
}

// quantizeIQ3S encodes src into IQ3_S blocks (110 bytes / 256 elements).
func quantizeIQ3S(src []float32, dst []byte) error {
	if len(src)%QK_K != 0 {
		return fmt.Errorf("quant: iq3_s requires a multiple of %d elements", QK_K)
	}
	for i := 0; i < len(src)/QK_K; i++ {
		block := src[i*QK_K : (i+1)*QK_K]
		b := dst[i*110:]
		qsBase := b[2:66]
		qhBase := b[66:74]
		signsBase := b[74:106]
		scBase := b[106:110]
		for j := range qsBase {
			qsBase[j] = 0
		}
		for j := range qhBase {
			qhBase[j] = 0
		}
		var amax [8]float64
		var gmax float64
		for ib := 0; ib < 8; ib++ {
			amax[ib] = minMaxAbs(block[ib*32 : (ib+1)*32])
			if amax[ib] > gmax {
				gmax = amax[ib]
			}
		}
		d := gmax / 15 / 31
		putHalf(b[0:2], float32(d))
		d = float64(F16ToF32(F32ToF16(float32(d))))
		var sc4 [8]int
		for ib := 0; ib < 8; ib++ {
			v := 0
			if d > 0 && amax[ib] > 0 {
				f := (amax[ib] / 15) / d
				v = clampInt(roundHalf((f-1)/2), 0, 15)
			}
			sc4[ib] = v
		}
		for p := 0; p < 4; p++ {
			scBase[p] = byte(sc4[2*p]) | byte(sc4[2*p+1]<<4)
			db1 := d * float64(1+2*sc4[2*p])
			db2 := d * float64(1+2*sc4[2*p+1])
			encIQ3Half(block[2*p*32:], qsBase[p*16:p*16+8], &qhBase[p*2], signsBase[p*8:p*8+4], db1)
			encIQ3Half(block[(2*p+1)*32:], qsBase[p*16+8:p*16+16], &qhBase[p*2+1], signsBase[p*8+4:p*8+8], db2)
		}
	}
	return nil
}

func encIQ3Half(block []float32, qsHalf []byte, qhByte *byte, signs []byte, db float64) {
	for l := 0; l < 4; l++ {
		g1, s1 := bestGrid(block[l*8:l*8+4], db)
		g2, s2 := bestGrid(block[l*8+4:l*8+8], db)
		qsHalf[2*l] = byte(g1 & 0xFF)
		qsHalf[2*l+1] = byte(g2 & 0xFF)
		if g1 >= 256 {
			*qhByte |= 1 << uint(2*l)
		}
		if g2 >= 256 {
			*qhByte |= 1 << uint(2*l+1)
		}
		signs[l] = s1 | (s2 << 4)
	}
}
