// Package fsutil provides crash-safe file writing.
//
// Long-running jobs (device training checkpoints, merged GGUF output) used to
// write large files with a plain os.Create + Write + rename. A crash or an
// out-of-memory kill between the create and the rename left a truncated temp
// file behind, and skipping fsync meant the rename could become visible before
// the data was durable. AtomicWriter centralises the safe sequence: write to a
// temp file, fsync it, rename over the target, then fsync the directory.
package fsutil

import (
	"bufio"
	"io"
	"os"
	"path/filepath"
)

// defaultBufSize is the write buffer used for atomic writers. It is large
// enough that streaming a multi-gigabyte GGUF is not syscall-bound, but small
// enough that it never becomes a meaningful part of the process footprint.
const defaultBufSize = 8 << 20

// AtomicWriter writes to path+".tmp" and atomically renames it to path on
// Commit. The temp file is fsynced before the rename and the parent directory
// after, so a crash at any point leaves either the previous file or the
// complete new one, never a torn mixture.
type AtomicWriter struct {
	f    *os.File
	bw   *bufio.Writer
	tmp  string
	path string
	done bool
}

// Create starts an atomic write to path, creating parent directories as needed.
// The caller must call Commit or Abort exactly once.
func Create(path string) (*AtomicWriter, error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, err
	}
	return &AtomicWriter{
		f:    f,
		bw:   bufio.NewWriterSize(f, defaultBufSize),
		tmp:  tmp,
		path: path,
	}, nil
}

// Write buffers p. It implements io.Writer so streaming writers can target an
// AtomicWriter directly.
func (w *AtomicWriter) Write(p []byte) (int, error) { return w.bw.Write(p) }

// Flush writes any buffered bytes to the temp file without fsyncing or
// renaming. It is useful for progress reporting; Commit does the durable part.
func (w *AtomicWriter) Flush() error {
	if w.done {
		return nil
	}
	return w.bw.Flush()
}

// Commit flushes, fsyncs, closes, renames to the final path and fsyncs the
// parent directory. After Commit the writer is spent.
func (w *AtomicWriter) Commit() error {
	if w.done {
		return nil
	}
	w.done = true
	if err := w.bw.Flush(); err != nil {
		w.cleanup()
		return err
	}
	if err := w.f.Sync(); err != nil {
		w.cleanup()
		return err
	}
	if err := w.f.Close(); err != nil {
		os.Remove(w.tmp)
		return err
	}
	if err := os.Rename(w.tmp, w.path); err != nil {
		os.Remove(w.tmp)
		return err
	}
	syncDir(filepath.Dir(w.path))
	return nil
}

// Abort discards the temp file, leaving any existing path untouched.
func (w *AtomicWriter) Abort() error {
	if w.done {
		return nil
	}
	w.done = true
	err := w.f.Close()
	os.Remove(w.tmp)
	return err
}

// cleanup closes the temp file and removes it, ignoring close errors (the write
// path already failed).
func (w *AtomicWriter) cleanup() {
	w.f.Close()
	os.Remove(w.tmp)
}

// syncDir fsyncs a directory so a rename within it is durable. Best-effort:
// some filesystems reject directory fsync, which is not fatal here.
func syncDir(dir string) {
	if dir == "" {
		dir = "."
	}
	d, err := os.Open(dir)
	if err != nil {
		return
	}
	d.Sync()
	d.Close()
}

var _ io.Writer = (*AtomicWriter)(nil)
