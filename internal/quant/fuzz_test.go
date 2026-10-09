package quant

import "testing"

// FuzzDequant asserts that the dequantizers never panic on arbitrary input,
// regardless of the declared element count (truncated/oversized buffers must
// yield an error, not a crash or out-of-bounds access).
func FuzzDequant(f *testing.F) {
	types := []Type{
		TypeF32, TypeF16, TypeBF16, TypeQ4_0, TypeQ5_0, TypeQ8_0,
		TypeQ3_K, TypeQ4_K, TypeQ5_K, TypeQ6_K, TypeIQ4_NL, TypeIQ4_XS, TypeIQ3_S,
	}
	f.Add(byte(0), []byte{0, 0, 0, 0}, int64(0))
	f.Add(byte(3), []byte{1, 2, 3, 4, 5, 6, 7, 8}, int64(8))
	f.Fuzz(func(t *testing.T, sel byte, data []byte, n int64) {
		typ := types[int(sel)%len(types)]
		if n < 0 {
			n = -n
		}
		if n > 1<<20 {
			n = 1 << 20
		}
		_, _ = Dequant(typ, data, n)
	})
}
