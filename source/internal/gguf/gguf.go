// Package gguf implements reading and writing of the GGUF file format (v3).
//
// The layout follows the reference implementation in
// references/llama_cpp/ggml/src/gguf.cpp and the constants in
// references/llama_cpp/gguf-py/gguf/constants.py.
package gguf

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// Magic is the GGUF file magic as a little-endian uint32 ("GGUF").
const Magic uint32 = 0x46554747

// Supported version.
const Version uint32 = 3

// DefaultAlignment is the default value of general.alignment.
const DefaultAlignment uint64 = 32

// ValueType enumerates the GGUF metadata value types.
type ValueType uint32

const (
	ValueTypeUint8   ValueType = 0
	ValueTypeInt8    ValueType = 1
	ValueTypeUint16  ValueType = 2
	ValueTypeInt16   ValueType = 3
	ValueTypeUint32  ValueType = 4
	ValueTypeInt32   ValueType = 5
	ValueTypeFloat32 ValueType = 6
	ValueTypeBool    ValueType = 7
	ValueTypeString  ValueType = 8
	ValueTypeArray   ValueType = 9
	ValueTypeUint64  ValueType = 10
	ValueTypeInt64   ValueType = 11
	ValueTypeFloat64 ValueType = 12
)

func (t ValueType) String() string {
	switch t {
	case ValueTypeUint8:
		return "uint8"
	case ValueTypeInt8:
		return "int8"
	case ValueTypeUint16:
		return "uint16"
	case ValueTypeInt16:
		return "int16"
	case ValueTypeUint32:
		return "uint32"
	case ValueTypeInt32:
		return "int32"
	case ValueTypeFloat32:
		return "float32"
	case ValueTypeBool:
		return "bool"
	case ValueTypeString:
		return "string"
	case ValueTypeArray:
		return "array"
	case ValueTypeUint64:
		return "uint64"
	case ValueTypeInt64:
		return "int64"
	case ValueTypeFloat64:
		return "float64"
	default:
		return fmt.Sprintf("valueType(%d)", uint32(t))
	}
}

// Value is a GGUF metadata value. Exactly one field is populated according to
// Type. Arrays are stored in Array with scalar elements; nested arrays are not
// permitted by the format.
type Value struct {
	Type   ValueType
	Uint   uint64
	Int    int64
	Float  float64
	Bool   bool
	String string
	Array  []Value
}

// AsInt returns the value coerced to int64 when it is an integer type.
func (v Value) AsInt() (int64, bool) {
	switch v.Type {
	case ValueTypeUint8, ValueTypeUint16, ValueTypeUint32, ValueTypeUint64:
		return int64(v.Uint), true
	case ValueTypeInt8, ValueTypeInt16, ValueTypeInt32, ValueTypeInt64:
		return v.Int, true
	default:
		return 0, false
	}
}

// AsUint returns the value coerced to uint64 when it is an unsigned integer.
func (v Value) AsUint() (uint64, bool) {
	switch v.Type {
	case ValueTypeUint8, ValueTypeUint16, ValueTypeUint32, ValueTypeUint64:
		return v.Uint, true
	case ValueTypeInt8, ValueTypeInt16, ValueTypeInt32, ValueTypeInt64:
		if v.Int < 0 {
			return 0, false
		}
		return uint64(v.Int), true
	default:
		return 0, false
	}
}

// AsFloat returns the value coerced to float64 for numeric types.
func (v Value) AsFloat() (float64, bool) {
	switch v.Type {
	case ValueTypeFloat32, ValueTypeFloat64:
		return v.Float, true
	case ValueTypeUint8, ValueTypeUint16, ValueTypeUint32, ValueTypeUint64:
		return float64(v.Uint), true
	case ValueTypeInt8, ValueTypeInt16, ValueTypeInt32, ValueTypeInt64:
		return float64(v.Int), true
	default:
		return 0, false
	}
}

// KV is a key/value metadata pair.
type KV struct {
	Key   string
	Value Value
}

// TensorInfo describes one tensor in the file.
type TensorInfo struct {
	Name   string
	Dims   []uint64 // GGML order: Dims[0] is the row length
	Type   quant.Type
	Offset uint64 // relative to the start of the tensor data section
}

// NumElements returns the product of the dimensions.
func (t *TensorInfo) NumElements() int64 {
	n := int64(1)
	for _, d := range t.Dims {
		n *= int64(d)
	}
	return n
}

// RowLen returns the number of elements in Dims[0].
func (t *TensorInfo) RowLen() int64 {
	if len(t.Dims) == 0 {
		return 1
	}
	return int64(t.Dims[0])
}

// ByteSize returns the number of bytes occupied by the tensor data. It detects
// dimension overflow so a hostile header cannot produce a bogus (negative or
// wrapped) size that would later be handed to make.
func (t *TensorInfo) ByteSize() (int64, error) {
	elems := int64(1)
	for _, d := range t.Dims {
		if d == 0 {
			elems = 0
			break
		}
		if d > math.MaxInt64 {
			return 0, fmt.Errorf("gguf: tensor %q dimension %d overflows", t.Name, d)
		}
		if elems > math.MaxInt64/int64(d) {
			return 0, fmt.Errorf("gguf: tensor %q element count overflows", t.Name)
		}
		elems *= int64(d)
	}
	return t.Type.Size(elems)
}

func (t *TensorInfo) String() string {
	return fmt.Sprintf("%s type=%s dims=%v", t.Name, t.Type, t.Dims)
}
