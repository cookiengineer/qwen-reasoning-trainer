package train

import (
	"bytes"
	"encoding/gob"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func f32Tensor(vals ...float32) *compute.Tensor {
	t := compute.NewF32(len(vals))
	copy(t.F32, vals)
	return t
}

func TestDeviceCheckpointV2RoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ck.pt")
	p1 := compute.NewF32(2, 3)
	for i := range p1.F32 {
		p1.F32[i] = float32(i) + 0.5
	}
	p2 := compute.NewF32(4)
	copy(p2.F32, []float32{1, 2, 3, 4})
	m1 := compute.NewF32(3)
	copy(m1.F32, []float32{-1, -2, -3})

	ck := DeviceCheckpoint{
		Version: deviceCheckpointVersion,
		Step:    17,
		Params:  []*compute.Tensor{p1, p2},
		Moments: []*compute.Tensor{m1},
	}
	if err := SaveDeviceCheckpoint(path, ck); err != nil {
		t.Fatal(err)
	}
	got, err := LoadDeviceCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Step != 17 || len(got.Params) != 2 || len(got.Moments) != 1 {
		t.Fatalf("loaded step=%d params=%d moments=%d", got.Step, len(got.Params), len(got.Moments))
	}
	if !reflect.DeepEqual(got.Params[0].Dims, p1.Dims) || !reflect.DeepEqual(got.Params[0].F32, p1.F32) {
		t.Fatalf("param 0 mismatch: %+v vs %+v", got.Params[0], p1)
	}
	if !reflect.DeepEqual(got.Moments[0].F32, m1.F32) {
		t.Fatalf("moment 0 mismatch: %v vs %v", got.Moments[0].F32, m1.F32)
	}
}

func TestLoadCheckpointV1Compatible(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ck-v1.pt")
	var buf bytes.Buffer
	v1 := DeviceCheckpoint{
		Version: 1,
		Step:    3,
		Params:  []*compute.Tensor{f32Tensor(1, 2, 3)},
		Moments: []*compute.Tensor{f32Tensor(4, 5, 6)},
	}
	if err := gob.NewEncoder(&buf).Encode(v1); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := LoadDeviceCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Version != 1 || got.Step != 3 {
		t.Fatalf("v1 load: version=%d step=%d", got.Version, got.Step)
	}
	if !reflect.DeepEqual(got.Params[0].F32, []float32{1, 2, 3}) {
		t.Fatalf("v1 param mismatch: %v", got.Params[0].F32)
	}
}

func TestSaveCheckpointFailureKeepsOriginal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ck.pt")
	if err := os.WriteFile(path, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := SaveDeviceCheckpoint(path, DeviceCheckpoint{Params: []*compute.Tensor{nil}}); err == nil {
		t.Fatal("want error from nil tensor")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "keep" {
		t.Fatalf("checkpoint clobbered on failure: %q", got)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("temp file left behind: %v", err)
	}
}
