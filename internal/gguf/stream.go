package gguf

import (
	"bytes"
	"fmt"
	"io"
	"os"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// StreamTensor describes one tensor for the streaming writer. Write must emit
// exactly Size bytes to the provided writer.
type StreamTensor struct {
	Name  string
	Dims  []uint64
	Type  quant.Type
	Size  int64
	Write func(io.Writer) error
}

// WriteStream writes a GGUF v3 file to path without buffering all tensor data.
func WriteStreamToFile(path string, kvs []KV, tensors []StreamTensor) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := WriteStream(f, kvs, tensors); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// WriteStream serializes a GGUF v3 file to w, calling each tensor's Write
// callback to emit its data.
func WriteStream(w io.Writer, kvs []KV, tensors []StreamTensor) error {
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
			return err
		}
	}

	// Compute aligned data offsets sequentially, then serialize tensor infos.
	offsets := make([]uint64, len(tensors))
	var cursor uint64
	for i, t := range tensors {
		if len(t.Dims) == 0 || len(t.Dims) > 4 {
			return fmt.Errorf("gguf: tensor %q has %d dimensions", t.Name, len(t.Dims))
		}
		cursor = align(cursor, alignment)
		offsets[i] = cursor
		cursor += uint64(t.Size)
	}
	for i, t := range tensors {
		writeString(&prefix, t.Name)
		writeU32(&prefix, uint32(len(t.Dims)))
		for _, d := range t.Dims {
			writeU64(&prefix, d)
		}
		writeU32(&prefix, uint32(t.Type))
		writeU64(&prefix, offsets[i])
	}

	if _, err := w.Write(prefix.Bytes()); err != nil {
		return err
	}
	dataStart := align(uint64(prefix.Len()), alignment)
	if pad := dataStart - uint64(prefix.Len()); pad > 0 {
		if _, err := w.Write(make([]byte, pad)); err != nil {
			return err
		}
	}
	var written uint64
	for i, t := range tensors {
		if offsets[i] < written {
			return fmt.Errorf("gguf: tensor %q offset %d behind %d", t.Name, offsets[i], written)
		}
		if pad := offsets[i] - written; pad > 0 {
			if _, err := w.Write(make([]byte, pad)); err != nil {
				return err
			}
			written += pad
		}
		cw := &countWriter{w: w}
		if err := t.Write(cw); err != nil {
			return err
		}
		if cw.n != t.Size {
			return fmt.Errorf("gguf: tensor %q wrote %d bytes, expected %d", t.Name, cw.n, t.Size)
		}
		written += uint64(cw.n)
	}
	return nil
}

type countWriter struct {
	w io.Writer
	n int64
}

func (c *countWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	return n, err
}
