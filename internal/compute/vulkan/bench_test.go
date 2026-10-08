package vulkan_test

import (
	"testing"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// chainWeight builds a Q4_K weight [K,N] with deterministic values.
func chainWeight(t testing.TB, K, N int) []byte {
	t.Helper()
	wf := make([]float32, K*N)
	for i := range wf {
		wf[i] = float32((i%17)-8) * 0.01
	}
	raw, err := quant.Quantize(quant.TypeQ4_K, wf)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// hostMediatedChain mimics Model.mm: upload the activation, run one matmul,
// download the result, apply SiLU on the host, and repeat. One flush per layer.
func hostMediatedChain(t testing.TB, v *vulkan.Backend, w compute.Buffer, x0 []float32, K, N, m, layers int) []float32 {
	t.Helper()
	x := append([]float32(nil), x0...)
	for l := 0; l < layers; l++ {
		xb, err := v.Upload(&compute.Tensor{Dims: []int{K, m}, F32: x})
		if err != nil {
			t.Fatal(err)
		}
		ob, err := v.MatMulWeight(w, xb)
		if err != nil {
			t.Fatal(err)
		}
		out, err := v.Download(ob) // flush + wait here
		if err != nil {
			t.Fatal(err)
		}
		x = out.F32
		for i := range x {
			x[i] = compute.Silu(x[i])
		}
		v.Free(xb)
		v.Free(ob)
	}
	return x
}

// residentChain keeps the whole chain on the device: one upload at the start,
// one download at the end, and a single flush.
func residentChain(t testing.TB, v *vulkan.Backend, w compute.Buffer, x0 []float32, K, N, m, layers int) []float32 {
	t.Helper()
	cur, err := v.Upload(&compute.Tensor{Dims: []int{K, m}, F32: x0})
	if err != nil {
		t.Fatal(err)
	}
	for l := 0; l < layers; l++ {
		ob, err := v.MatMulWeight(w, cur)
		if err != nil {
			t.Fatal(err)
		}
		act, err := v.Unary(compute.UnarySilu, ob)
		if err != nil {
			t.Fatal(err)
		}
		if l > 0 {
			v.Free(cur)
		}
		v.Free(ob)
		cur = act
	}
	out, err := v.Download(cur) // single flush + wait
	if err != nil {
		t.Fatal(err)
	}
	v.Free(cur)
	return out.F32
}

func TestResidentVsHostMediated(t *testing.T) {
	v, err := vulkan.New()
	if err != nil {
		t.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()

	const K, N, m, layers = 1024, 1024, 64, 8
	raw := chainWeight(t, K, N)
	w, err := v.UploadWeight(quant.TypeQ4_K, raw, []int{K, N})
	if err != nil {
		t.Fatal(err)
	}
	x0 := make([]float32, K*m)
	for i := range x0 {
		x0[i] = float32((i%9)-4) * 0.05
	}

	// Warm up both paths (shader compilation, pool population).
	hostMediatedChain(t, v, w, x0, K, N, m, 1)
	residentChain(t, v, w, x0, K, N, m, 1)

	v.ResetStats()
	t0 := time.Now()
	hostOut := hostMediatedChain(t, v, w, x0, K, N, m, layers)
	hostDur := time.Since(t0)
	hostStats := v.Stats()

	v.ResetStats()
	t1 := time.Now()
	resOut := residentChain(t, v, w, x0, K, N, m, layers)
	resDur := time.Since(t1)
	resStats := v.Stats()

	t.Logf("chain of %d matmul+silu, K=%d N=%d m=%d", layers, K, N, m)
	t.Logf("host-mediated: %v  stats=%+v", hostDur.Round(time.Microsecond), hostStats)
	t.Logf("resident:      %v  stats=%+v", resDur.Round(time.Microsecond), resStats)
	if hostDur > 0 {
		t.Logf("speedup: %.2fx  (submits %d -> %d, downloaded %d -> %d bytes)",
			float64(hostDur)/float64(resDur), hostStats.Submits, resStats.Submits,
			hostStats.BytesDownloaded, resStats.BytesDownloaded)
	}

	// Parity: a short chain isolates real bugs from nonlinear amplification.
	hostShort := hostMediatedChain(t, v, w, x0, K, N, m, 1)
	resShort := residentChain(t, v, w, x0, K, N, m, 1)
	var shortDiff float64
	for i := range hostShort {
		d := float64(hostShort[i] - resShort[i])
		if d < 0 {
			d = -d
		}
		if d > shortDiff {
			shortDiff = d
		}
	}
	if shortDiff > 1e-4 {
		t.Fatalf("single-layer mismatch: max diff %g", shortDiff)
	}

	if len(hostOut) != len(resOut) {
		t.Fatalf("length mismatch")
	}
	var maxDiff, maxAbs float64
	for i := range hostOut {
		d := float64(hostOut[i] - resOut[i])
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
		a := float64(hostOut[i])
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	rel := maxDiff / (1 + maxAbs)
	t.Logf("final relative diff after %d layers: %.3g (max |out| %.3g)", layers, rel, maxAbs)
	if rel > 1e-4 {
		t.Fatalf("paths disagree beyond rounding: rel %g", rel)
	}
}

func BenchmarkHostMediatedChain(b *testing.B) {
	v, err := vulkan.New()
	if err != nil {
		b.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()
	const K, N, m, layers = 1024, 1024, 64, 8
	raw := chainWeight(b, K, N)
	w, _ := v.UploadWeight(quant.TypeQ4_K, raw, []int{K, N})
	x0 := make([]float32, K*m)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hostMediatedChain(b, v, w, x0, K, N, m, layers)
	}
}

func BenchmarkResidentChain(b *testing.B) {
	v, err := vulkan.New()
	if err != nil {
		b.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()
	const K, N, m, layers = 1024, 1024, 64, 8
	raw := chainWeight(b, K, N)
	w, _ := v.UploadWeight(quant.TypeQ4_K, raw, []int{K, N})
	x0 := make([]float32, K*m)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		residentChain(b, v, w, x0, K, N, m, layers)
	}
}

func BenchmarkHostMediatedDecode(b *testing.B) {
	v, err := vulkan.New()
	if err != nil {
		b.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()
	const K, N, m, layers = 4096, 4096, 1, 16
	raw := chainWeight(b, K, N)
	w, _ := v.UploadWeight(quant.TypeQ4_K, raw, []int{K, N})
	x0 := make([]float32, K*m)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hostMediatedChain(b, v, w, x0, K, N, m, layers)
	}
}

func BenchmarkResidentDecode(b *testing.B) {
	v, err := vulkan.New()
	if err != nil {
		b.Skipf("vulkan unavailable: %v", err)
	}
	defer v.Close()
	const K, N, m, layers = 4096, 4096, 1, 16
	raw := chainWeight(b, K, N)
	w, _ := v.UploadWeight(quant.TypeQ4_K, raw, []int{K, N})
	x0 := make([]float32, K*m)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		residentChain(b, v, w, x0, K, N, m, layers)
	}
}
