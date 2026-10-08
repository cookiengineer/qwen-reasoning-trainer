package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// ErrInvalid reports a malformed GGUF file.
var ErrInvalid = errors.New("gguf: invalid file")

// File is an open GGUF file. It retains the underlying ReaderAt so tensor data
// can be read on demand.
type File struct {
	r      io.ReaderAt
	closer io.Closer
	size   int64

	Version   uint32
	Alignment uint64
	KVs       []KV
	meta      map[string]int
	Tensors   []*TensorInfo
	byName    map[string]*TensorInfo
	dataStart int64
}

// Open opens the GGUF file at path.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	g, err := readFrom(f, st.Size())
	if err != nil {
		f.Close()
		return nil, err
	}
	g.closer = f
	return g, nil
}

// Close releases the underlying file handle.
func (g *File) Close() error {
	if g.closer != nil {
		return g.closer.Close()
	}
	return nil
}

// Size returns the file size in bytes.
func (g *File) Size() int64 { return g.size }

// DataStart returns the absolute offset of the tensor data section.
func (g *File) DataStart() int64 { return g.dataStart }

// Get returns the metadata value for key.
func (g *File) Get(key string) (Value, bool) {
	i, ok := g.meta[key]
	if !ok {
		return Value{}, false
	}
	return g.KVs[i].Value, true
}

// Uint returns a metadata value coerced to uint64.
func (g *File) Uint(key string) (uint64, bool) {
	v, ok := g.Get(key)
	if !ok {
		return 0, false
	}
	return v.AsUint()
}

// Int returns a metadata value coerced to int64.
func (g *File) Int(key string) (int64, bool) {
	v, ok := g.Get(key)
	if !ok {
		return 0, false
	}
	return v.AsInt()
}

// Float returns a metadata value coerced to float64.
func (g *File) Float(key string) (float64, bool) {
	v, ok := g.Get(key)
	if !ok {
		return 0, false
	}
	return v.AsFloat()
}

// Str returns a string metadata value.
func (g *File) Str(key string) (string, bool) {
	v, ok := g.Get(key)
	if !ok || v.Type != ValueTypeString {
		return "", false
	}
	return v.String, true
}

// Strings returns a string-array metadata value.
func (g *File) Strings(key string) ([]string, bool) {
	v, ok := g.Get(key)
	if !ok || v.Type != ValueTypeArray {
		return nil, false
	}
	out := make([]string, 0, len(v.Array))
	for _, e := range v.Array {
		if e.Type != ValueTypeString {
			return nil, false
		}
		out = append(out, e.String)
	}
	return out, true
}

// Tensor returns the tensor info for name.
func (g *File) Tensor(name string) (*TensorInfo, bool) {
	t, ok := g.byName[name]
	return t, ok
}

// ReadTensor reads and returns the raw bytes of a tensor.
func (g *File) ReadTensor(t *TensorInfo) ([]byte, error) {
	n, err := t.ByteSize()
	if err != nil {
		return nil, err
	}
	buf := make([]byte, n)
	if _, err := g.r.ReadAt(buf, g.dataStart+int64(t.Offset)); err != nil {
		return nil, fmt.Errorf("gguf: read tensor %s: %w", t.Name, err)
	}
	return buf, nil
}

// ReadTensorF32 reads a tensor and dequantizes it to float32.
func (g *File) ReadTensorF32(t *TensorInfo) ([]float32, error) {
	raw, err := g.ReadTensor(t)
	if err != nil {
		return nil, err
	}
	return quant.Dequant(t.Type, raw, t.NumElements())
}

type cursor struct {
	r   *bufio.Reader
	off int64
	max int64 // file size, for sanity bounds
}

func (c *cursor) read(p []byte) error {
	if _, err := io.ReadFull(c.r, p); err != nil {
		return err
	}
	c.off += int64(len(p))
	return nil
}

func (c *cursor) u8() (uint8, error) {
	var b [1]byte
	if err := c.read(b[:]); err != nil {
		return 0, err
	}
	return b[0], nil
}

func (c *cursor) u16() (uint16, error) {
	var b [2]byte
	if err := c.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b[:]), nil
}

func (c *cursor) u32() (uint32, error) {
	var b [4]byte
	if err := c.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (c *cursor) u64() (uint64, error) {
	var b [8]byte
	if err := c.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (c *cursor) str() (string, error) {
	n, err := c.u64()
	if err != nil {
		return "", err
	}
	if c.max > 0 && int64(n) > c.max-c.off {
		return "", fmt.Errorf("%w: string length %d exceeds file", ErrInvalid, n)
	}
	b := make([]byte, n)
	if err := c.read(b); err != nil {
		return "", err
	}
	return string(b), nil
}

func align(offset, a uint64) uint64 {
	if a == 0 {
		return offset
	}
	rem := offset % a
	if rem == 0 {
		return offset
	}
	return offset + a - rem
}

// ReadFrom parses a GGUF file from a ReaderAt of the given size. It is useful
// for in-memory parsing and testing; Open is the usual entry point.
func ReadFrom(r io.ReaderAt, size int64) (*File, error) {
	return readFrom(r, size)
}

func readFrom(r io.ReaderAt, size int64) (*File, error) {
	c := &cursor{r: bufio.NewReaderSize(io.NewSectionReader(r, 0, size), 1<<16), max: size}

	magic, err := c.u32()
	if err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if magic != Magic {
		return nil, fmt.Errorf("%w: bad magic 0x%08x", ErrInvalid, magic)
	}
	version, err := c.u32()
	if err != nil {
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrInvalid, version)
	}
	tensorCount, err := c.u64()
	if err != nil {
		return nil, fmt.Errorf("gguf: read tensor count: %w", err)
	}
	kvCount, err := c.u64()
	if err != nil {
		return nil, fmt.Errorf("gguf: read kv count: %w", err)
	}

	g := &File{
		r:       r,
		size:    size,
		Version: version,
		meta:    make(map[string]int),
		byName:  make(map[string]*TensorInfo),
	}

	g.KVs = make([]KV, 0, kvCount)
	for i := uint64(0); i < kvCount; i++ {
		key, err := c.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: read kv key %d: %w", i, err)
		}
		val, err := readValue(c)
		if err != nil {
			return nil, fmt.Errorf("gguf: read kv %q: %w", key, err)
		}
		g.meta[key] = len(g.KVs)
		g.KVs = append(g.KVs, KV{Key: key, Value: val})
	}

	g.Alignment = DefaultAlignment
	if v, ok := g.Uint("general.alignment"); ok && v > 0 {
		g.Alignment = v
	}

	g.Tensors = make([]*TensorInfo, 0, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := c.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: read tensor name %d: %w", i, err)
		}
		nd, err := c.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: read tensor %q ndims: %w", name, err)
		}
		if nd == 0 || nd > 4 {
			return nil, fmt.Errorf("%w: tensor %q has %d dimensions", ErrInvalid, name, nd)
		}
		dims := make([]uint64, nd)
		for d := uint32(0); d < nd; d++ {
			dim, err := c.u64()
			if err != nil {
				return nil, fmt.Errorf("gguf: read tensor %q dim %d: %w", name, d, err)
			}
			dims[d] = dim
		}
		ttype, err := c.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: read tensor %q type: %w", name, err)
		}
		offset, err := c.u64()
		if err != nil {
			return nil, fmt.Errorf("gguf: read tensor %q offset: %w", name, err)
		}
		ti := &TensorInfo{Name: name, Dims: dims, Type: quant.Type(ttype), Offset: offset}
		if _, dup := g.byName[name]; dup {
			return nil, fmt.Errorf("%w: duplicate tensor %q", ErrInvalid, name)
		}
		g.byName[name] = ti
		g.Tensors = append(g.Tensors, ti)
	}

	g.dataStart = int64(align(uint64(c.off), g.Alignment))

	// Validate tensor bounds.
	for _, t := range g.Tensors {
		sz, err := t.ByteSize()
		if err != nil {
			return nil, fmt.Errorf("gguf: tensor %q: %w", t.Name, err)
		}
		end := g.dataStart + int64(t.Offset) + sz
		if end > size {
			return nil, fmt.Errorf("%w: tensor %q extends past end of file (%d > %d)", ErrInvalid, t.Name, end, size)
		}
	}

	return g, nil
}

func readValue(c *cursor) (Value, error) {
	t, err := c.u32()
	if err != nil {
		return Value{}, err
	}
	return readValueOfType(c, ValueType(t), 0)
}

func readValueOfType(c *cursor, t ValueType, depth int) (Value, error) {
	if depth > 2 {
		return Value{}, fmt.Errorf("%w: nested arrays too deep", ErrInvalid)
	}
	switch t {
	case ValueTypeUint8:
		b, err := c.u8()
		return Value{Type: t, Uint: uint64(b)}, err
	case ValueTypeInt8:
		b, err := c.u8()
		return Value{Type: t, Int: int64(int8(b))}, err
	case ValueTypeUint16:
		v, err := c.u16()
		return Value{Type: t, Uint: uint64(v)}, err
	case ValueTypeInt16:
		v, err := c.u16()
		return Value{Type: t, Int: int64(int16(v))}, err
	case ValueTypeUint32:
		v, err := c.u32()
		return Value{Type: t, Uint: uint64(v)}, err
	case ValueTypeInt32:
		v, err := c.u32()
		return Value{Type: t, Int: int64(int32(v))}, err
	case ValueTypeFloat32:
		v, err := c.u32()
		return Value{Type: t, Float: float64(math.Float32frombits(v))}, err
	case ValueTypeBool:
		b, err := c.u8()
		return Value{Type: t, Bool: b != 0}, err
	case ValueTypeString:
		s, err := c.str()
		return Value{Type: t, String: s}, err
	case ValueTypeUint64:
		v, err := c.u64()
		return Value{Type: t, Uint: v}, err
	case ValueTypeInt64:
		v, err := c.u64()
		return Value{Type: t, Int: int64(v)}, err
	case ValueTypeFloat64:
		v, err := c.u64()
		return Value{Type: t, Float: math.Float64frombits(v)}, err
	case ValueTypeArray:
		elemTypeRaw, err := c.u32()
		if err != nil {
			return Value{}, err
		}
		elemType := ValueType(elemTypeRaw)
		if elemType == ValueTypeArray {
			return Value{}, fmt.Errorf("%w: array of arrays", ErrInvalid)
		}
		count, err := c.u64()
		if err != nil {
			return Value{}, err
		}
		if c.max > 0 && count > uint64(c.max) {
			return Value{}, fmt.Errorf("%w: array count %d too large", ErrInvalid, count)
		}
		arr := make([]Value, count)
		for i := uint64(0); i < count; i++ {
			v, err := readValueOfType(c, elemType, depth+1)
			if err != nil {
				return Value{}, err
			}
			arr[i] = v
		}
		return Value{Type: t, Array: arr}, nil
	default:
		return Value{}, fmt.Errorf("%w: unknown value type %d", ErrInvalid, uint32(t))
	}
}
