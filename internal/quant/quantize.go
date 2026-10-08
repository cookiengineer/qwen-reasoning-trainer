package quant

import (
	"encoding/binary"
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
