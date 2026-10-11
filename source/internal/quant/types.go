// Package quant implements the GGML tensor type system and the subset of
// quantization block codecs needed to read and write GGUF files.
//
// Block layouts and dequantization formulas are ported from the reference
// implementation in references/llama_cpp/ggml/src/ggml-quants.c and
// references/llama_cpp/ggml/src/ggml-common.h. Type identifiers mirror the
// ggml_type enum in references/llama_cpp/ggml/include/ggml.h.
package quant

import "fmt"

// Type is a GGML tensor element type.
type Type uint32

// ggml_type enum values.
const (
	TypeF32     Type = 0
	TypeF16     Type = 1
	TypeQ4_0    Type = 2
	TypeQ4_1    Type = 3
	TypeQ5_0    Type = 6
	TypeQ5_1    Type = 7
	TypeQ8_0    Type = 8
	TypeQ8_1    Type = 9
	TypeQ2_K    Type = 10
	TypeQ3_K    Type = 11
	TypeQ4_K    Type = 12
	TypeQ5_K    Type = 13
	TypeQ6_K    Type = 14
	TypeQ8_K    Type = 15
	TypeIQ2_XXS Type = 16
	TypeIQ2_XS  Type = 17
	TypeIQ3_XXS Type = 18
	TypeIQ1_S   Type = 19
	TypeIQ4_NL  Type = 20
	TypeIQ3_S   Type = 21
	TypeIQ2_S   Type = 22
	TypeIQ4_XS  Type = 23
	TypeI8      Type = 24
	TypeI16     Type = 25
	TypeI32     Type = 26
	TypeI64     Type = 27
	TypeF64     Type = 28
	TypeIQ1_M   Type = 29
	TypeBF16    Type = 30
	TypeTQ1_0   Type = 34
	TypeTQ2_0   Type = 35
	TypeMXFP4   Type = 39
	TypeNVFP4   Type = 40
	TypeQ1_0    Type = 41
	TypeQ2_0    Type = 42
)

// QK_K is the super-block size for K-quants.
const QK_K = 256

// typeInfo describes a GGML type: human name, elements per block, and bytes
// per block. Block size 1 denotes a non-quantized type.
type typeInfo struct {
	name      string
	blockSize int
	typeSize  int
}

// typeTable is indexed by the numeric type value. Gaps are zero-valued.
var typeTable = []typeInfo{
	TypeF32:     {"f32", 1, 4},
	TypeF16:     {"f16", 1, 2},
	TypeQ4_0:    {"q4_0", 32, 18},
	TypeQ4_1:    {"q4_1", 32, 20},
	TypeQ5_0:    {"q5_0", 32, 22},
	TypeQ5_1:    {"q5_1", 32, 24},
	TypeQ8_0:    {"q8_0", 32, 34},
	TypeQ8_1:    {"q8_1", 32, 36},
	TypeQ2_K:    {"q2_K", 256, 84},
	TypeQ3_K:    {"q3_K", 256, 110},
	TypeQ4_K:    {"q4_K", 256, 144},
	TypeQ5_K:    {"q5_K", 256, 176},
	TypeQ6_K:    {"q6_K", 256, 210},
	TypeQ8_K:    {"q8_K", 256, 292},
	TypeIQ2_XXS: {"iq2_xxs", 256, 66},
	TypeIQ2_XS:  {"iq2_xs", 256, 74},
	TypeIQ3_XXS: {"iq3_xxs", 256, 98},
	TypeIQ1_S:   {"iq1_s", 256, 50},
	TypeIQ4_NL:  {"iq4_nl", 32, 18},
	TypeIQ3_S:   {"iq3_s", 256, 110},
	TypeIQ2_S:   {"iq2_s", 256, 82},
	TypeIQ4_XS:  {"iq4_xs", 256, 136},
	TypeI8:      {"i8", 1, 1},
	TypeI16:     {"i16", 1, 2},
	TypeI32:     {"i32", 1, 4},
	TypeI64:     {"i64", 1, 8},
	TypeF64:     {"f64", 1, 8},
	TypeIQ1_M:   {"iq1_m", 256, 56},
	TypeBF16:    {"bf16", 1, 2},
	TypeTQ1_0:   {"tq1_0", 256, 54},
	TypeTQ2_0:   {"tq2_0", 256, 66},
	TypeMXFP4:   {"mxfp4", 32, 17},
	TypeNVFP4:   {"nvfp4", 64, 36},
	TypeQ1_0:    {"q1_0", 128, 18},
	TypeQ2_0:    {"q2_0", 64, 18},
}

// info returns the metadata for t, or a zero typeInfo if unknown.
func (t Type) info() typeInfo {
	if int(t) < len(typeTable) {
		return typeTable[t]
	}
	return typeInfo{}
}

// Known reports whether t is a recognized GGML type.
func (t Type) Known() bool {
	return t.info().blockSize != 0
}

// String returns the canonical short name (e.g. "q4_K").
func (t Type) String() string {
	if i := t.info(); i.name != "" {
		return i.name
	}
	return fmt.Sprintf("unknown(%d)", uint32(t))
}

// BlockSize returns the number of elements covered by one storage block.
func (t Type) BlockSize() int { return t.info().blockSize }

// TypeSize returns the number of bytes in one storage block.
func (t Type) TypeSize() int { return t.info().typeSize }

// IsQuantized reports whether the type packs multiple elements per block.
func (t Type) IsQuantized() bool { return t.info().blockSize > 1 }

// Size returns the number of bytes required to store n elements of type t.
// n must be a multiple of the block size.
func (t Type) Size(n int64) (int64, error) {
	i := t.info()
	if i.blockSize == 0 {
		return 0, fmt.Errorf("quant: unknown type %d", uint32(t))
	}
	if n%int64(i.blockSize) != 0 {
		return 0, fmt.Errorf("quant: %s: element count %d not a multiple of block size %d", t, n, i.blockSize)
	}
	return n / int64(i.blockSize) * int64(i.typeSize), nil
}

// RowSize returns the number of bytes in a row of n elements, matching
// ggml_row_size. n must be a multiple of the block size.
func (t Type) RowSize(n int64) (int64, error) { return t.Size(n) }
