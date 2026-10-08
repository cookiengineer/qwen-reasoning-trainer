package train

import (
	"math"
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// hostMLPGrads builds the same LoRA-MLP on the host autograd graph and returns
// the adapter gradients. It is the reference for the device block.
func hostMLPGrads(w1, w2 *qwen38.Weight, a1, b1, a2, b2, x, c *compute.Tensor, scale float32) (da1, db1, da2, db2 *compute.Tensor) {
	av1 := autograd.Param(a1)
	bv1 := autograd.Param(b1)
	av2 := autograd.Param(a2)
	bv2 := autograd.Param(b2)
	xv := autograd.Param(x)
	cv := autograd.Param(c)

	l1 := autograd.VAdd(
		autograd.VMatMulWeight(w1, xv),
		autograd.VScale(autograd.VMatMul(bv1, autograd.VMatMul(av1, xv)), scale),
	)
	h := autograd.VSilu(l1)
	l2 := autograd.VAdd(
		autograd.VMatMulWeight(w2, h),
		autograd.VScale(autograd.VMatMul(bv2, autograd.VMatMul(av2, h)), scale),
	)
	loss := autograd.VSum(autograd.VMul(l2, cv))
	loss.Backward()
	return av1.Grad, bv1.Grad, av2.Grad, bv2.Grad
}

func maxAbsDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var m float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

func TestDeviceMLPGradientsMatchHost(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	const In, Hid, Out, R, m = 16, 12, 10, 3, 5
	scale := float32(0.5)

	w1 := qwen38.NewWeightF32([]int{In, Hid}, autograd.Random(1, 0.3, In, Hid).F32)
	w2 := qwen38.NewWeightF32([]int{Hid, Out}, autograd.Random(2, 0.3, Hid, Out).F32)
	a1 := autograd.Random(3, 0.2, In, R)
	b1 := autograd.Random(4, 0.2, R, Hid)
	a2 := autograd.Random(5, 0.2, Hid, R)
	b2 := autograd.Random(6, 0.2, R, Out)
	x := autograd.Random(7, 0.5, In, m)
	c := autograd.Random(8, 0.5, Out, m)

	// Host reference.
	hda1, hdb1, hda2, hdb2 := hostMLPGrads(w1, w2, a1, b1, a2, b2, x, c, scale)

	// Device block.
	base1, err := v.UploadWeight(w1.Typ, w1.Raw, w1.Dims)
	if err != nil {
		t.Fatal(err)
	}
	base2, err := v.UploadWeight(w2.Typ, w2.Raw, w2.Dims)
	if err != nil {
		t.Fatal(err)
	}
	mlp, err := NewDeviceMLP(v, base1, a1, b1, base2, a2, b2, In, Hid, Out, R, scale)
	if err != nil {
		t.Fatal(err)
	}
	xb, err := v.Upload(x)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := v.Upload(c)
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := mlp.Forward(xb)
	if err != nil {
		t.Fatal(err)
	}
	grads, _, err := mlp.Backward(ctx, cb)
	if err != nil {
		t.Fatal(err)
	}
	gda1, gdb1, gda2, gdb2, err := mlp.DownloadGrads(grads)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		dev  *compute.Tensor
		host *compute.Tensor
	}{
		{"dA1", gda1, hda1}, {"dB1", gdb1, hdb1},
		{"dA2", gda2, hda2}, {"dB2", gdb2, hdb2},
	} {
		if d := maxAbsDiff(tc.dev.F32, tc.host.F32); d > 1e-4 {
			t.Errorf("%s: device vs host max diff %g", tc.name, d)
		}
	}
}

func TestDeviceMLPStepTiming(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	const In, Hid, Out, R, m = 512, 1024, 512, 16, 64
	scale := float32(0.5)

	w1 := qwen38.NewWeightF32([]int{In, Hid}, autograd.Random(11, 0.2, In, Hid).F32)
	w2 := qwen38.NewWeightF32([]int{Hid, Out}, autograd.Random(12, 0.2, Hid, Out).F32)
	a1 := autograd.Random(13, 0.1, In, R)
	b1 := autograd.Random(14, 0.1, R, Hid)
	a2 := autograd.Random(15, 0.1, Hid, R)
	b2 := autograd.Random(16, 0.1, R, Out)
	x := autograd.Random(17, 0.5, In, m)
	c := autograd.Random(18, 0.5, Out, m)

	base1, _ := v.UploadWeight(w1.Typ, w1.Raw, w1.Dims)
	base2, _ := v.UploadWeight(w2.Typ, w2.Raw, w2.Dims)
	mlp, err := NewDeviceMLP(v, base1, a1, b1, base2, a2, b2, In, Hid, Out, R, scale)
	if err != nil {
		t.Fatal(err)
	}
	xb, _ := v.Upload(x)
	cb, _ := v.Upload(c)

	// Warm up.
	ctx, _ := mlp.Forward(xb)
	mlp.Backward(ctx, cb)
	v.Sync()

	const iters = 20
	v.ResetStats()
	t0 := time.Now()
	for i := 0; i < iters; i++ {
		ctx, err := mlp.Forward(xb)
		if err != nil {
			t.Fatal(err)
		}
		grads, _, err := mlp.Backward(ctx, cb)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, _, _, err := mlp.DownloadGrads(grads); err != nil {
			t.Fatal(err)
		}
	}
	devDur := time.Since(t0) / iters
	devStats := v.Stats()

	t1 := time.Now()
	for i := 0; i < iters; i++ {
		hostMLPGrads(w1, w2, a1, b1, a2, b2, x, c, scale)
	}
	hostDur := time.Since(t1) / iters

	t.Logf("device step: %v  (dispatches=%d submits=%d down=%d bytes)",
		devDur.Round(time.Microsecond), devStats.Dispatches, devStats.Submits, devStats.BytesDownloaded)
	t.Logf("host step:   %v", hostDur.Round(time.Microsecond))
	if hostDur > 0 {
		t.Logf("device/host: %.2fx", float64(hostDur)/float64(devDur))
	}
}
