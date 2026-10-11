package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// WriteTensor is a tensor to be serialized.
type WriteTensor struct {
	Name string
	Dims []uint64
	Type quant.Type
	Data []byte
}

// WriteFile serializes a GGUF v3 file to path. The metadata order is preserved.
// Tensor offsets are assigned automatically with the file alignment.
func WriteFile(path string, kvs []KV, tensors []WriteTensor) error {
	buf, err := Encode(kvs, tensors)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Encode serializes a GGUF v3 file into memory.
func Encode(kvs []KV, tensors []WriteTensor) ([]byte, error) {
	alignment := DefaultAlignment
	for _, kv := range kvs {
		if kv.Key == "general.alignment" {
			if u, ok := kv.Value.AsUint(); ok && u > 0 {
				alignment = u
			}
		}
	}

	var prefix bytes.Buffer
	writeU32(&prefix, Magic)
	writeU32(&prefix, Version)
	writeU64(&prefix, uint64(len(tensors)))
	writeU64(&prefix, uint64(len(kvs)))

	for _, kv := range kvs {
		writeString(&prefix, kv.Key)
		if err := writeValue(&prefix, kv.Value); err != nil {
			return nil, err
		}
	}

	// Assign data offsets.
	type placement struct {
		offset uint64
	}
	placements := make([]placement, len(tensors))
	var cursor uint64
	for i, t := range tensors {
		cursor = align(cursor, alignment)
		placements[i] = placement{offset: cursor}
		cursor += uint64(len(t.Data))
	}

	for i, t := range tensors {
		if len(t.Dims) == 0 || len(t.Dims) > 4 {
			return nil, fmt.Errorf("gguf: tensor %q has %d dimensions", t.Name, len(t.Dims))
		}
		writeString(&prefix, t.Name)
		writeU32(&prefix, uint32(len(t.Dims)))
		for _, d := range t.Dims {
			writeU64(&prefix, d)
		}
		writeU32(&prefix, uint32(t.Type))
		writeU64(&prefix, placements[i].offset)
	}

	dataStart := align(uint64(prefix.Len()), alignment)
	total := dataStart + cursor

	out := make([]byte, total)
	copy(out, prefix.Bytes())
	for i, t := range tensors {
		off := dataStart + placements[i].offset
		copy(out[off:], t.Data)
	}
	return out, nil
}

func writeU32(b *bytes.Buffer, v uint32) {
	var tmp [4]byte
	binary.LittleEndian.PutUint32(tmp[:], v)
	b.Write(tmp[:])
}

func writeU64(b *bytes.Buffer, v uint64) {
	var tmp [8]byte
	binary.LittleEndian.PutUint64(tmp[:], v)
	b.Write(tmp[:])
}

func writeString(b *bytes.Buffer, s string) {
	writeU64(b, uint64(len(s)))
	b.WriteString(s)
}

func writeValue(b *bytes.Buffer, v Value) error {
	writeU32(b, uint32(v.Type))
	return writeValueBody(b, v)
}

func writeValueBody(b *bytes.Buffer, v Value) error {
	switch v.Type {
	case ValueTypeUint8, ValueTypeBool:
		if v.Type == ValueTypeBool {
			if v.Bool {
				b.WriteByte(1)
			} else {
				b.WriteByte(0)
			}
		} else {
			b.WriteByte(byte(v.Uint))
		}
	case ValueTypeInt8:
		b.WriteByte(byte(int8(v.Int)))
	case ValueTypeUint16:
		var tmp [2]byte
		binary.LittleEndian.PutUint16(tmp[:], uint16(v.Uint))
		b.Write(tmp[:])
	case ValueTypeInt16:
		var tmp [2]byte
		binary.LittleEndian.PutUint16(tmp[:], uint16(int16(v.Int)))
		b.Write(tmp[:])
	case ValueTypeUint32:
		writeU32(b, uint32(v.Uint))
	case ValueTypeInt32:
		writeU32(b, uint32(int32(v.Int)))
	case ValueTypeFloat32:
		writeU32(b, math.Float32bits(float32(v.Float)))
	case ValueTypeString:
		writeString(b, v.String)
	case ValueTypeUint64:
		writeU64(b, v.Uint)
	case ValueTypeInt64:
		writeU64(b, uint64(v.Int))
	case ValueTypeFloat64:
		writeU64(b, math.Float64bits(v.Float))
	case ValueTypeArray:
		if len(v.Array) == 0 {
			return fmt.Errorf("gguf: cannot write empty array without element type")
		}
		elemType := v.Array[0].Type
		writeU32(b, uint32(elemType))
		writeU64(b, uint64(len(v.Array)))
		for _, e := range v.Array {
			if e.Type != elemType {
				return fmt.Errorf("gguf: array element type mismatch")
			}
			if err := writeValueBody(b, e); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("gguf: cannot write value type %d", uint32(v.Type))
	}
	return nil
}

// Helpers for constructing typed metadata values.

// Uint32Value builds a uint32 metadata value.
func Uint32Value(v uint32) Value { return Value{Type: ValueTypeUint32, Uint: uint64(v)} }

// Uint64Value builds a uint64 metadata value.
func Uint64Value(v uint64) Value { return Value{Type: ValueTypeUint64, Uint: v} }

// Int32Value builds an int32 metadata value.
func Int32Value(v int32) Value { return Value{Type: ValueTypeInt32, Int: int64(v)} }

// Float32Value builds a float32 metadata value.
func Float32Value(v float32) Value { return Value{Type: ValueTypeFloat32, Float: float64(v)} }

// StringValue builds a string metadata value.
func StringValue(v string) Value { return Value{Type: ValueTypeString, String: v} }

// BoolValue builds a bool metadata value.
func BoolValue(v bool) Value { return Value{Type: ValueTypeBool, Bool: v} }

// StringArray builds a string-array metadata value.
func StringArray(vals []string) Value {
	arr := make([]Value, len(vals))
	for i, s := range vals {
		arr[i] = StringValue(s)
	}
	return Value{Type: ValueTypeArray, Array: arr}
}
