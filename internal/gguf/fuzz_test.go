package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// validGGUF builds a minimal but complete GGUF v3 file used as a fuzz seed.
func validGGUF(t testing.TB) []byte {
	t.Helper()
	src := []float32{1, 2, 3, 4}
	data := make([]byte, len(src)*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	b, err := Encode(
		[]KV{
			{Key: "general.architecture", Value: StringValue("fuzz")},
			{Key: "fuzz.count", Value: Uint32Value(4)},
		},
		[]WriteTensor{
			{Name: "t", Dims: []uint64{2, 2}, Type: quant.TypeF32, Data: data},
		},
	)
	if err != nil {
		t.Fatalf("encode seed: %v", err)
	}
	return b
}

// FuzzReadFrom asserts that parsing arbitrary bytes never panics and that any
// successfully parsed file can be queried tensor-by-tensor without a crash.
func FuzzReadFrom(f *testing.F) {
	f.Add(validGGUF(f))
	f.Add([]byte("GGUF"))
	f.Add([]byte{0x47, 0x47, 0x55, 0x46, 3, 0, 0, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		g, err := ReadFrom(bytes.NewReader(data), int64(len(data)))
		if err != nil {
			return
		}
		defer g.Close()
		for _, ti := range g.Tensors {
			// Bounds-checked reads must return data or an error, never panic.
			if _, err := g.ReadTensor(ti); err != nil {
				continue
			}
		}
	})
}
