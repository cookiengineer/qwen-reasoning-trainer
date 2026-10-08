package quant

import "math"

// F16ToF32 converts an IEEE 754 half precision float to a float32.
func F16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	var bits uint32
	switch {
	case exp == 0:
		if mant == 0 {
			bits = sign << 31
		} else {
			// Subnormal half: normalize.
			e := uint32(127 - 15 + 1)
			for mant&0x400 == 0 {
				mant <<= 1
				e--
			}
			mant &= 0x3ff
			bits = sign<<31 | e<<23 | mant<<13
		}
	case exp == 0x1f:
		bits = sign<<31 | 0xff<<23 | mant<<13
	default:
		bits = sign<<31 | (exp-15+127)<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}

// F32ToF16 converts a float32 to an IEEE 754 half precision float. It is a
// direct port of ggml_compute_fp32_to_fp16 from
// references/llama_cpp/ggml/src/ggml-impl.h for bit-exact behavior.
func F32ToF16(f float32) uint16 {
	const scaleToInf = float32(0x1.0p+112)
	const scaleToZero = float32(0x1.0p-110)
	base := (float32(math.Abs(float64(f))) * scaleToInf) * scaleToZero

	w := math.Float32bits(f)
	shl1w := w + w
	sign := w & 0x80000000
	bias := shl1w & 0xFF000000
	if bias < 0x71000000 {
		bias = 0x71000000
	}
	base = math.Float32frombits((bias>>1)+0x07800000) + base
	bits := math.Float32bits(base)
	expBits := (bits >> 13) & 0x00007C00
	mantBits := bits & 0x00000FFF
	nonsign := expBits + mantBits

	half := uint16(sign>>16) | uint16(nonsign)
	if shl1w > 0xFF000000 {
		half = uint16(sign>>16) | 0x7E00
	}
	return half
}

// BF16ToF32 converts a bfloat16 value stored in the low 16 bits to float32.
func BF16ToF32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

// F32ToBF16 converts a float32 to bfloat16 (truncated, matching ggml).
func F32ToBF16(f float32) uint16 {
	return uint16(math.Float32bits(f) >> 16)
}
