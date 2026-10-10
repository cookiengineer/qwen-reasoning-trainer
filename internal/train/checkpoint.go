package train

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// deviceCheckpointVersion is bumped whenever the DeviceCheckpoint layout
// changes so an old file is rejected instead of silently misread.
const deviceCheckpointVersion = 1

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

// SaveDeviceCheckpoint writes ck atomically (temp file + rename) so a crash
// during a long run cannot corrupt the resume file.
func SaveDeviceCheckpoint(path string, ck DeviceCheckpoint) error {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(f).Encode(ck); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// LoadDeviceCheckpoint reads a checkpoint written by SaveDeviceCheckpoint.
func LoadDeviceCheckpoint(path string) (DeviceCheckpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return DeviceCheckpoint{}, err
	}
	defer f.Close()
	var ck DeviceCheckpoint
	if err := gob.NewDecoder(f).Decode(&ck); err != nil {
		return DeviceCheckpoint{}, err
	}
	if ck.Version != deviceCheckpointVersion {
		return DeviceCheckpoint{}, fmt.Errorf("train: unknown device checkpoint version %d (want %d)", ck.Version, deviceCheckpointVersion)
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
// to path so the run can be resumed.
func (m *DeviceModel) SaveDeviceState(path string, opt *DeviceAdamW) error {
	if err := m.SyncAdapters(); err != nil {
		return err
	}
	moments, err := opt.Snapshot()
	if err != nil {
		return err
	}
	return SaveDeviceCheckpoint(path, DeviceCheckpoint{
		Version: deviceCheckpointVersion,
		Step:    opt.StepCount(),
		Params:  m.Params(),
		Moments: moments,
	})
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
