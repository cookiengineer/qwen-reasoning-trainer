package train

import (
	"bufio"
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/fsutil"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// deviceCheckpointVersion is bumped whenever the DeviceCheckpoint layout
// changes so an old file is rejected instead of silently misread.
//
// v1 was a single gob stream; v2 is a streaming binary format that writes one
// tensor at a time so a checkpoint save never has to hold every adapter and
// AdamW moment (order of a gigabyte) in host memory simultaneously. v1 files
// are still readable.
const deviceCheckpointVersion = 2

// checkpointMagic identifies the v2 streaming checkpoint format.
const checkpointMagic = "QWCKPT2\n"

// DeviceCheckpoint is the resumable state of a device-resident QLoRA run: the
// optimizer step (for the cosine schedule and AdamW bias correction), every
// adapter's [A,B] tensors, and the AdamW first/second moments. It is distinct
// from the adapter-only Trainer checkpoint (SaveCheckpoint) so that a
// half-finished run can be resumed exactly.
type DeviceCheckpoint struct {
	Version int
	Step    int
	Params  []*compute.Tensor // 2 per adapter (A, B) in AdapterLins order
	Moments []*compute.Tensor // 4 per adapter (mA, vA, mB, vB)
}

// checkpointStream writes a v2 checkpoint incrementally. Each tensor is written
// as it arrives and is not retained, so the caller can stream device downloads
// straight through without materialising the whole checkpoint.
type checkpointStream struct {
	aw  *fsutil.AtomicWriter
	bw  *bufio.Writer
	got int
}

func newCheckpointStream(path string, step, nParams, nMoment int) (*checkpointStream, error) {
	aw, err := fsutil.Create(path)
	if err != nil {
		return nil, err
	}
	bw := bufio.NewWriterSize(aw, 4<<20)
	if _, err := bw.WriteString(checkpointMagic); err != nil {
		aw.Abort()
		return nil, err
	}
	hdr := []uint32{
		deviceCheckpointVersion,
		uint32(step),
		uint32(nParams),
		uint32(nMoment),
	}
	for _, v := range hdr {
		if err := binary.Write(bw, binary.LittleEndian, v); err != nil {
			aw.Abort()
			return nil, err
		}
	}
	return &checkpointStream{aw: aw, bw: bw}, nil
}

// WriteTensor appends one tensor. Tensors must be written in adapter order:
// every parameter, then every moment, matching DeviceCheckpoint.
func (s *checkpointStream) WriteTensor(t *compute.Tensor) error {
	if t == nil {
		return fmt.Errorf("train: nil checkpoint tensor")
	}
	if err := binary.Write(s.bw, binary.LittleEndian, uint32(len(t.Dims))); err != nil {
		return err
	}
	for _, d := range t.Dims {
		if err := binary.Write(s.bw, binary.LittleEndian, uint64(d)); err != nil {
			return err
		}
	}
	if err := binary.Write(s.bw, binary.LittleEndian, uint32(t.Type)); err != nil {
		return err
	}
	raw := f32Bytes(t.F32)
	if err := binary.Write(s.bw, binary.LittleEndian, uint64(len(raw))); err != nil {
		return err
	}
	if _, err := s.bw.Write(raw); err != nil {
		return err
	}
	s.got++
	return nil
}

// Commit flushes and atomically publishes the checkpoint.
func (s *checkpointStream) Commit() error {
	if err := s.bw.Flush(); err != nil {
		s.aw.Abort()
		return err
	}
	return s.aw.Commit()
}

// Abort discards the partial checkpoint.
func (s *checkpointStream) Abort() error { return s.aw.Abort() }

// f32Bytes views a float32 slice as its little-endian byte representation. It
// is zero-copy and relies on the host being little-endian, like the rest of the
// tensor code.
func f32Bytes(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

// SaveDeviceCheckpoint writes ck atomically (temp file + fsync + rename). It
// streams tensors one at a time so peak host memory is bounded by the largest
// single tensor rather than the whole checkpoint.
func SaveDeviceCheckpoint(path string, ck DeviceCheckpoint) error {
	st, err := newCheckpointStream(path, ck.Step, len(ck.Params), len(ck.Moments))
	if err != nil {
		return err
	}
	for _, t := range ck.Params {
		if err := st.WriteTensor(t); err != nil {
			st.Abort()
			return err
		}
	}
	for _, t := range ck.Moments {
		if err := st.WriteTensor(t); err != nil {
			st.Abort()
			return err
		}
	}
	return st.Commit()
}

// LoadDeviceCheckpoint reads a checkpoint written by SaveDeviceState. It
// supports both the streaming v2 format and the legacy v1 gob format.
func LoadDeviceCheckpoint(path string) (DeviceCheckpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return DeviceCheckpoint{}, err
	}
	defer f.Close()

	magic := make([]byte, len(checkpointMagic))
	if _, err := io.ReadFull(f, magic); err != nil {
		return DeviceCheckpoint{}, err
	}
	if string(magic) == checkpointMagic {
		return readCheckpointV2(f)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return DeviceCheckpoint{}, err
	}
	return readCheckpointV1(f)
}

func readCheckpointV2(r io.Reader) (DeviceCheckpoint, error) {
	br := bufio.NewReaderSize(r, 4<<20)
	var hdr [4]uint32
	for i := range hdr {
		if err := binary.Read(br, binary.LittleEndian, &hdr[i]); err != nil {
			return DeviceCheckpoint{}, fmt.Errorf("train: read checkpoint header: %w", err)
		}
	}
	if hdr[0] != deviceCheckpointVersion {
		return DeviceCheckpoint{}, fmt.Errorf("train: unknown device checkpoint version %d (want %d)", hdr[0], deviceCheckpointVersion)
	}
	nParams := int(hdr[2])
	nMoments := int(hdr[3])
	read := func(n int) ([]*compute.Tensor, error) {
		out := make([]*compute.Tensor, 0, n)
		for i := 0; i < n; i++ {
			t, err := readTensorV2(br)
			if err != nil {
				return nil, err
			}
			out = append(out, t)
		}
		return out, nil
	}
	params, err := read(nParams)
	if err != nil {
		return DeviceCheckpoint{}, err
	}
	moments, err := read(nMoments)
	if err != nil {
		return DeviceCheckpoint{}, err
	}
	return DeviceCheckpoint{Version: int(hdr[0]), Step: int(hdr[1]), Params: params, Moments: moments}, nil
}

func readTensorV2(r io.Reader) (*compute.Tensor, error) {
	var nd uint32
	if err := binary.Read(r, binary.LittleEndian, &nd); err != nil {
		return nil, err
	}
	if nd > 4 {
		return nil, fmt.Errorf("train: checkpoint tensor has %d dimensions", nd)
	}
	dims := make([]int, nd)
	for i := range dims {
		var d uint64
		if err := binary.Read(r, binary.LittleEndian, &d); err != nil {
			return nil, err
		}
		dims[i] = int(d)
	}
	var typ uint32
	if err := binary.Read(r, binary.LittleEndian, &typ); err != nil {
		return nil, err
	}
	var nbytes uint64
	if err := binary.Read(r, binary.LittleEndian, &nbytes); err != nil {
		return nil, err
	}
	if nbytes%4 != 0 {
		return nil, fmt.Errorf("train: checkpoint tensor byte length %d is not a multiple of 4", nbytes)
	}
	f := make([]float32, nbytes/4)
	if _, err := io.ReadFull(r, f32Bytes(f)); err != nil {
		return nil, err
	}
	return &compute.Tensor{Dims: dims, Type: quant.Type(typ), F32: f}, nil
}

func readCheckpointV1(r io.Reader) (DeviceCheckpoint, error) {
	var ck DeviceCheckpoint
	if err := gob.NewDecoder(r).Decode(&ck); err != nil {
		return DeviceCheckpoint{}, err
	}
	if ck.Version != 1 {
		return DeviceCheckpoint{}, fmt.Errorf("train: unknown device checkpoint version %d (want 1)", ck.Version)
	}
	return ck, nil
}

// uploadInto copies a host tensor into an existing device buffer (same element
// count/type), reusing the buffer instead of reallocating so the optimizer's
// moment map stays valid.
func uploadInto(be compute.Backend, dst compute.Buffer, t *compute.Tensor) error {
	if dst.NumElements() != len(t.F32) {
		return fmt.Errorf("train: upload size mismatch %d != %d", dst.NumElements(), len(t.F32))
	}
	tmp, err := be.Upload(t)
	if err != nil {
		return err
	}
	defer be.Free(tmp)
	return be.Copy(dst, tmp)
}

// PushAdapters uploads the host LoRA tensors into the device adapter buffers.
// It is the inverse of SyncAdapters and is used to resume a checkpoint.
func (m *DeviceModel) PushAdapters() error {
	for _, l := range m.AdapterLins() {
		if err := uploadInto(m.be, l.a, l.lora.A); err != nil {
			return err
		}
		if err := uploadInto(m.be, l.b, l.lora.B); err != nil {
			return err
		}
	}
	return m.be.Sync()
}

// SaveDeviceState snapshots the current adapters (params + AdamW moments + step)
// to path so the run can be resumed. Adapters and moments are streamed to disk
// one tensor at a time, so the save does not hold the whole checkpoint in host
// memory.
func (m *DeviceModel) SaveDeviceState(path string, opt *DeviceAdamW) error {
	if err := m.SyncAdapters(); err != nil {
		return err
	}
	params := m.Params()
	st, err := newCheckpointStream(path, opt.StepCount(), len(params), 4*len(opt.order))
	if err != nil {
		return err
	}
	for _, t := range params {
		if err := st.WriteTensor(t); err != nil {
			st.Abort()
			return err
		}
	}
	if err := opt.ForEachMoment(st.WriteTensor); err != nil {
		st.Abort()
		return err
	}
	return st.Commit()
}

// LoadDeviceState restores adapters, AdamW moments and the step counter from a
// checkpoint and returns the resumed step. It expects the model to have been
// built with the same LoRA configuration (adapter count/shape).
func (m *DeviceModel) LoadDeviceState(path string, opt *DeviceAdamW) (int, error) {
	ck, err := LoadDeviceCheckpoint(path)
	if err != nil {
		return 0, err
	}
	params := m.Params()
	if len(ck.Params) != len(params) {
		return 0, fmt.Errorf("train: checkpoint has %d adapter tensors, want %d", len(ck.Params), len(params))
	}
	for i := range params {
		if len(ck.Params[i].F32) != len(params[i].F32) {
			return 0, fmt.Errorf("train: adapter tensor %d size mismatch", i)
		}
		copy(params[i].F32, ck.Params[i].F32)
	}
	if err := m.PushAdapters(); err != nil {
		return 0, err
	}
	if err := opt.Restore(ck.Step, ck.Moments); err != nil {
		return 0, err
	}
	return ck.Step, nil
}
