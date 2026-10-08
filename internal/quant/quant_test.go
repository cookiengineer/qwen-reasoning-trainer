package quant

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// Reference values are hand-derived from the dequantization formulas in
// references/llama_cpp/ggml/src/ggml-quants.c and the block layouts in
// references/llama_cpp/ggml/src/ggml-common.h.

func TestTypeTableMatchesReference(t *testing.T) {
	// block size, type size pairs from ggml-common.h static_asserts.
	cases := []struct {
		typ       Type
		name      string
		blockSize int
		typeSize  int
	}{
		{TypeF32, "f32", 1, 4},
		{TypeF16, "f16", 1, 2},
		{TypeBF16, "bf16", 1, 2},
		{TypeQ4_0, "q4_0", 32, 18},
		{TypeQ5_0, "q5_0", 32, 22},
		{TypeQ8_0, "q8_0", 32, 34},
		{TypeQ2_K, "q2_K", 256, 84},
		{TypeQ3_K, "q3_K", 256, 110},
		{TypeQ4_K, "q4_K", 256, 144},
		{TypeQ5_K, "q5_K", 256, 176},
		{TypeQ6_K, "q6_K", 256, 210},
		{TypeQ8_K, "q8_K", 256, 292},
		{TypeIQ2_XXS, "iq2_xxs", 256, 66},
		{TypeIQ2_XS, "iq2_xs", 256, 74},
		{TypeIQ3_XXS, "iq3_xxs", 256, 98},
		{TypeIQ1_S, "iq1_s", 256, 50},
		{TypeIQ4_NL, "iq4_nl", 32, 18},
		{TypeIQ3_S, "iq3_s", 256, 110},
		{TypeIQ2_S, "iq2_s", 256, 82},
		{TypeIQ4_XS, "iq4_xs", 256, 136},
		{TypeIQ1_M, "iq1_m", 256, 56},
		{TypeQ1_0, "q1_0", 128, 18},
		{TypeQ2_0, "q2_0", 64, 18},
	}
	for _, c := range cases {
		if got := c.typ.String(); got != c.name {
			t.Errorf("%d: name = %q, want %q", c.typ, got, c.name)
		}
		if got := c.typ.BlockSize(); got != c.blockSize {
			t.Errorf("%s: BlockSize = %d, want %d", c.name, got, c.blockSize)
		}
		if got := c.typ.TypeSize(); got != c.typeSize {
			t.Errorf("%s: TypeSize = %d, want %d", c.name, got, c.typeSize)
		}
	}
}

func TestSizeRejectsMisaligned(t *testing.T) {
	if _, err := TypeQ4_0.Size(31); err == nil {
		t.Fatal("expected error for misaligned element count")
	}
	got, err := TypeQ4_K.Size(512)
	if err != nil {
		t.Fatal(err)
	}
	if got != 288 {
		t.Fatalf("Size(512) = %d, want 288", got)
	}
}

func TestFP16RoundTrip(t *testing.T) {
	values := []float32{0, 1, -1, 0.5, 2, 65504, -65504, 0.00006103515625}
	for _, v := range values {
		h := F32ToF16(v)
		got := F16ToF32(h)
		if got != v {
			t.Errorf("round trip %v -> 0x%04x -> %v", v, h, got)
		}
	}
	if F32ToF16(1.0) != 0x3C00 {
		t.Errorf("F32ToF16(1) = 0x%04x, want 0x3c00", F32ToF16(1.0))
	}
	if F32ToF16(0.5) != 0x3800 {
		t.Errorf("F32ToF16(0.5) = 0x%04x, want 0x3800", F32ToF16(0.5))
	}
}

func TestBF16RoundTrip(t *testing.T) {
	for _, v := range []float32{0, 1, -1, 2.5, 100} {
		got := BF16ToF32(F32ToBF16(v))
		if got != v {
			t.Errorf("bf16 round trip %v -> %v", v, got)
		}
	}
}

func TestDequantQ4_0(t *testing.T) {
	// d = 1.0, qs[0] = 0x21 -> low nibble 1 -> -7, high nibble 2 -> -6.
	blk := make([]byte, 18)
	binary.LittleEndian.PutUint16(blk[0:], 0x3C00)
	blk[2] = 0x21
	blk[3] = 0xFF // low 15 -> 7, high 15 -> 7
	out, err := Dequant(TypeQ4_0, blk, 32)
	if err != nil {
		t.Fatal(err)
	}
	if out[0] != -7 || out[16] != -6 {
		t.Errorf("q4_0 out[0]=%v out[16]=%v, want -7 -6", out[0], out[16])
	}
	if out[1] != 7 || out[17] != 7 {
		t.Errorf("q4_0 out[1]=%v out[17]=%v, want 7 7", out[1], out[17])
	}
}

func TestDequantQ8_0(t *testing.T) {
	blk := make([]byte, 34)
	binary.LittleEndian.PutUint16(blk[0:], F32ToF16(0.5))
	blk[2] = 0x02 // +2
	blk[3] = 0xFE // -2
	out, err := Dequant(TypeQ8_0, blk, 32)
	if err != nil {
		t.Fatal(err)
	}
	if out[0] != 1.0 || out[1] != -1.0 {
		t.Errorf("q8_0 out[0]=%v out[1]=%v, want 1 -1", out[0], out[1])
	}
}

func TestDequantQ4K(t *testing.T) {
	blk := make([]byte, 144)
	binary.LittleEndian.PutUint16(blk[0:], 0x3C00) // d = 1
	binary.LittleEndian.PutUint16(blk[2:], 0x0000) // dmin = 0
	// scales: d for is 0..3, mins for is 0..3, then high nibble scales.
	blk[4] = 1  // is0 d
	blk[5] = 2  // is1 d
	blk[6] = 4  // is2 d
	blk[7] = 5  // is3 d
	blk[8] = 0  // is0 m
	blk[9] = 0  // is1 m
	blk[10] = 0 // is2 m
	blk[11] = 0 // is3 m
	blk[12] = 3 // is4 d (low nibble)
	blk[13] = 0 // is5 d
	blk[14] = 0 // is6 d
	blk[15] = 0 // is7 d
	qs := blk[16:]
	qs[0] = 0x21
	qs[32] = 0x43
	qs[64] = 0x05
	out, err := Dequant(TypeQ4_K, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	want := map[int]float32{0: 1, 32: 4, 64: 12, 96: 20, 128: 15}
	for idx, w := range want {
		if out[idx] != w {
			t.Errorf("q4_K out[%d] = %v, want %v", idx, out[idx], w)
		}
	}
	for i := range out {
		if _, ok := want[i]; !ok && out[i] != 0 {
			t.Fatalf("q4_K out[%d] = %v, want 0", i, out[i])
		}
	}
}

func TestDequantQ5K(t *testing.T) {
	blk := make([]byte, 176)
	binary.LittleEndian.PutUint16(blk[0:], 0x3C00) // d = 1
	blk[4] = 1                                     // is0 d
	blk[5] = 2                                     // is1 d
	qh := blk[16:48]
	ql := blk[48:176]
	qh[0] = 0x03 // set high bits for low and high nibble of lane 0
	ql[0] = 0x21
	out, err := Dequant(TypeQ5_K, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	// lane 0: low = 1 + 16 = 17 -> 17 ; high = 2 + 16 = 18 -> d2=2 -> 36.
	if out[0] != 17 || out[32] != 36 {
		t.Errorf("q5_K out[0]=%v out[32]=%v, want 17 36", out[0], out[32])
	}
}

func TestDequantQ6K(t *testing.T) {
	blk := make([]byte, 210)
	ql := blk[0:128]
	qh := blk[128:192]
	sc := blk[192:208]
	binary.LittleEndian.PutUint16(blk[208:], 0x3C00) // d = 1
	sc[0] = 1
	ql[0] = 0x01
	qh[0] = 0x02 // q1 = (1 | (2<<4)) - 32 = 33 - 32 = 1
	out, err := Dequant(TypeQ6_K, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	if out[0] != 1 {
		t.Errorf("q6_K out[0] = %v, want 1", out[0])
	}
}

func TestDequantIQ4XS(t *testing.T) {
	blk := make([]byte, 136)
	binary.LittleEndian.PutUint16(blk[0:], 0x3C00) // d = 1
	binary.LittleEndian.PutUint16(blk[2:], 0)      // scales_h = 0
	blk[4] = 0x22                                  // scales_l[0]: ls = 2
	qs := blk[8:136]
	qs[0] = 0x21
	out, err := Dequant(TypeIQ4_XS, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	// dl = 1 * (2 - 32) = -30. kvalues[1] = -104, kvalues[2] = -83.
	if got := out[0]; got != -30*-104 {
		t.Errorf("iq4_xs out[0] = %v, want %v", got, -30*-104)
	}
	if got := out[16]; got != -30*-83 {
		t.Errorf("iq4_xs out[16] = %v, want %v", got, -30*-83)
	}
}

func TestDequantUnsupported(t *testing.T) {
	_, err := Dequant(TypeQ2_K, make([]byte, 84), 256)
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("expected ErrUnsupported, got %v", err)
	}
}

func TestQuantizeRoundTrip(t *testing.T) {
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(math.Sin(float64(i))) * 3
	}
	for _, typ := range []Type{TypeF32, TypeF16, TypeQ8_0, TypeQ4_0} {
		enc, err := Quantize(typ, src)
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		dec, err := Dequant(typ, enc, 256)
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		if len(dec) != len(src) {
			t.Fatalf("%s: length mismatch", typ)
		}
		maxErr := 0.0
		for i := range src {
			d := math.Abs(float64(src[i] - dec[i]))
			if d > maxErr {
				maxErr = d
			}
		}
		var bound float64
		switch typ {
		case TypeF32:
			bound = 0
		case TypeF16:
			bound = 0.01
		case TypeQ8_0:
			bound = 0.02
		case TypeQ4_0:
			bound = 0.6
		}
		if maxErr > bound {
			t.Errorf("%s: max error %v exceeds bound %v", typ, maxErr, bound)
		}
	}
}

func TestDequantQ3K(t *testing.T) {
	blk := make([]byte, 110)
	// d = 1.0
	binary.LittleEndian.PutUint16(blk[108:], 0x3C00)
	// First scale byte -> 6-bit value 33 (low nibble 1, high bits from tmp).
	blk[96] = 0x01
	blk[104] = 0x02
	// hmask lane 0 present, qs lane 0 low 2 bits = 1.
	blk[0] = 0x01
	blk[32] = 0x01
	out, err := Dequant(TypeQ3_K, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	if out[0] != 1 {
		t.Errorf("q3_K out[0] = %v, want 1", out[0])
	}
	// Second element: q=0, hmask bit0 clear -> (0 - 4) * dl = -4.
	if out[1] != -4 {
		t.Errorf("q3_K out[1] = %v, want -4", out[1])
	}
}

func TestDequantIQ3S(t *testing.T) {
	blk := make([]byte, 110)
	binary.LittleEndian.PutUint16(blk[0:], 0x3C00) // d = 1
	// qs, qh, scales all zero -> grid index 0 = 0x01010101 -> all bytes 1.
	out, err := Dequant(TypeIQ3_S, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		if v != 1 {
			t.Fatalf("iq3_s out[%d] = %v, want 1", i, v)
		}
	}

	// Set the first sign bit: output 0 becomes negative.
	blk[74] = 0x01 // signs[0]
	out2, err := Dequant(TypeIQ3_S, blk, 256)
	if err != nil {
		t.Fatal(err)
	}
	if out2[0] != -1 {
		t.Errorf("iq3_s signed out[0] = %v, want -1", out2[0])
	}
	if out2[1] != 1 {
		t.Errorf("iq3_s out[1] = %v, want 1", out2[1])
	}
}

func TestQuantizeUnsupported(t *testing.T) {
	_, err := Quantize(TypeQ4_K, make([]float32, 256))
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("expected ErrUnsupported, got %v", err)
	}
}
