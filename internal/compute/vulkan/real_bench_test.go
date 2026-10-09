package vulkan_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// TestVulkanMatMulLargeParity checks the 64x64-tile GEMMs against the CPU
// oracle at shapes that cross tile boundaries and are not multiples of the tile
// (the real model's projection shapes are far larger than the unit tests).
func TestVulkanMatMulLargeParity(t *testing.T) {
	c, v := newBackends(t)
	shapes := []struct{ K, N, M int }{
		{300, 133, 99},
		{1024, 320, 200},
		{512, 257, 64},
	}
	for _, s := range shapes {
		a := autograd.Random(61, 0.1, s.K, s.N)
		b := autograd.Random(62, 1.0, s.K, s.M)
		f := func(be compute.Backend) (compute.Buffer, error) {
			return be.MatMul(up(t, be, a.Dims, a.F32), up(t, be, b.Dims, b.F32))
		}
		co, _ := f(c)
		vo, _ := f(v)
		compare(t, fmt.Sprintf("matmul K=%d N=%d M=%d", s.K, s.N, s.M), down(t, v, vo), down(t, c, co))

		// Transposed activation gradient: w[K,N], dY[N,M] -> [K,M].
		w := autograd.Random(63, 0.1, s.K, s.N)
		raw, err := quant.Quantize(quant.TypeF32, w.F32)
		if err != nil {
			t.Fatal(err)
		}
		dy := autograd.Random(64, 1.0, s.N, s.M)
		cw, _ := c.UploadWeight(quant.TypeF32, raw, w.Dims)
		vw, _ := v.UploadWeight(quant.TypeF32, raw, w.Dims)
		ft := func(be compute.Backend, wb compute.Buffer) (compute.Buffer, error) {
			return be.MatMulWeightTranspose(wb, up(t, be, dy.Dims, dy.F32))
		}
		cot, _ := ft(c, cw)
		vot, _ := ft(v, vw)
		compare(t, fmt.Sprintf("matmul_t K=%d N=%d M=%d", s.K, s.N, s.M), down(t, v, vot), down(t, c, cot))

		// Weight gradient: x[In,M], dOut[Out,M] -> [In,Out].
		x := autograd.Random(65, 1.0, s.K, s.M)
		do := autograd.Random(66, 1.0, s.N, s.M)
		fw := func(be compute.Backend) (compute.Buffer, error) {
			return be.MatMulWeightGrad(up(t, be, x.Dims, x.F32), up(t, be, do.Dims, do.F32))
		}
		cow, _ := fw(c)
		vow, _ := fw(v)
		compare(t, fmt.Sprintf("matmul_wg In=%d Out=%d M=%d", s.K, s.N, s.M), down(t, v, vow), down(t, c, cow))
	}
}

// TestVulkanAttentionNonSquareAndWideHead exercises the two fallback paths of
// Attention: a non-square (nQ != nKV) shape, which must use the host oracle, and
// a head dimension above the tile shader's shared-accumulator bound (256), which
// must use attention_naive.
func TestVulkanAttentionNonSquareAndWideHead(t *testing.T) {
	c, v := newBackends(t)

	// Incremental decode (nQ != nKV), non-causal -> host fallback.
	{
		hd, nh, nhkv, nQ, nKV := 3, 2, 1, 2, 4
		q := autograd.Random(81, 1.0, hd, nh, nQ)
		k := autograd.Random(82, 1.0, hd, nhkv, nKV)
		vv := autograd.Random(83, 1.0, hd, nhkv, nKV)
		scale := float32(1.0 / math.Sqrt(float64(hd)))
		f := func(b compute.Backend) (compute.Buffer, error) {
			return b.Attention(up(t, b, q.Dims, q.F32), up(t, b, k.Dims, k.F32),
				up(t, b, vv.Dims, vv.F32), nh, nhkv, scale, false)
		}
		co, _ := f(c)
		vo, _ := f(v)
		compare(t, "attention non-square", down(t, v, vo), down(t, c, co))
	}

	// Wide head (headDim > 256), square causal -> attention_naive.
	{
		hd, nh, nhkv, T := 300, 2, 1, 3
		q := autograd.Random(84, 1.0, hd, nh, T)
		k := autograd.Random(85, 1.0, hd, nhkv, T)
		vv := autograd.Random(86, 1.0, hd, nhkv, T)
		scale := float32(1.0 / math.Sqrt(float64(hd)))
		f := func(b compute.Backend) (compute.Buffer, error) {
			return b.Attention(up(t, b, q.Dims, q.F32), up(t, b, k.Dims, k.F32),
				up(t, b, vv.Dims, vv.F32), nh, nhkv, scale, true)
		}
		co, _ := f(c)
		vo, _ := f(v)
		compare(t, "attention wide-head", down(t, v, vo), down(t, c, co))
	}
}

// TestVulkanAttentionBackwardFD checks the attention backward shaders against
// central finite differences of the (independent) GPU forward, so a bug shared
// between the analytic backward and the CPU/autograd oracles is still caught.
func TestVulkanAttentionBackwardFD(t *testing.T) {
	_, v := newBackends(t)
	cases := []struct {
		hd, nh, nhkv, T int
		causal          bool
	}{
		{4, 2, 1, 3, true},
		{8, 4, 2, 5, true},
		{8, 4, 2, 5, false},
	}
	const step = float32(1e-2)
	for _, tc := range cases {
		scale := float32(1.0 / math.Sqrt(float64(tc.hd)))
		q := autograd.Random(71, 1.0, tc.hd, tc.nh, tc.T)
		k := autograd.Random(72, 1.0, tc.hd, tc.nhkv, tc.T)
		vv := autograd.Random(73, 1.0, tc.hd, tc.nhkv, tc.T)
		dO := autograd.Random(74, 1.0, tc.hd, tc.nh, tc.T)

		dqb, dkb, dvb, err := v.AttentionBackward(
			up(t, v, q.Dims, q.F32), up(t, v, k.Dims, k.F32),
			up(t, v, vv.Dims, vv.F32), up(t, v, dO.Dims, dO.F32),
			tc.nh, tc.nhkv, scale, tc.causal)
		if err != nil {
			t.Fatal(err)
		}
		dq, dk, dv := down(t, v, dqb), down(t, v, dkb), down(t, v, dvb)

		fwd := func(qq, kk, vvv []float32) []float32 {
			o, err := v.Attention(up(t, v, q.Dims, qq), up(t, v, k.Dims, kk),
				up(t, v, vv.Dims, vvv), tc.nh, tc.nhkv, scale, tc.causal)
			if err != nil {
				t.Fatal(err)
			}
			return down(t, v, o)
		}
		check := func(label string, grad []float32, which int) {
			for j := range grad {
				qp := append([]float32(nil), q.F32...)
				kp := append([]float32(nil), k.F32...)
				vp := append([]float32(nil), vv.F32...)
				qm := append([]float32(nil), q.F32...)
				km := append([]float32(nil), k.F32...)
				vm := append([]float32(nil), vv.F32...)
				switch which {
				case 0:
					qp[j] += step
					qm[j] -= step
				case 1:
					kp[j] += step
					km[j] -= step
				case 2:
					vp[j] += step
					vm[j] -= step
				}
				op := fwd(qp, kp, vp)
				om := fwd(qm, km, vm)
				num := (vecDot(op, dO.F32) - vecDot(om, dO.F32)) / float64(2*step)
				if d := math.Abs(num - float64(grad[j])); d > 5e-2+5e-2*math.Abs(num) {
					t.Fatalf("attention_back %s FD hd=%d nh=%d nhkv=%d T=%d causal=%v [%d]: numeric %g vs kernel %g",
						label, tc.hd, tc.nh, tc.nhkv, tc.T, tc.causal, j, num, grad[j])
				}
			}
		}
		check("dq", dq, 0)
		check("dk", dk, 1)
		check("dv", dv, 2)
	}
}

// TestVulkanAttentionTileParity checks the single-query-per-workgroup attention
// kernel against the CPU oracle across GQA, causal/non-causal, the real head
// dimension, and a sequence that spans more than one online-softmax tile.
func TestVulkanAttentionTileParity(t *testing.T) {
	c, v := newBackends(t)
	cases := []struct {
		hd, nh, nhkv, T int
		causal          bool
	}{
		{8, 4, 2, 7, true},
		{8, 4, 2, 7, false},
		{256, 24, 4, 128, true},
		{4, 2, 1, 1100, true}, // spans two TILE=1024 passes
	}
	for _, tc := range cases {
		q := autograd.Random(41, 1.0, tc.hd, tc.nh, tc.T)
		k := autograd.Random(42, 1.0, tc.hd, tc.nhkv, tc.T)
		vv := autograd.Random(43, 1.0, tc.hd, tc.nhkv, tc.T)
		scale := float32(1.0 / math.Sqrt(float64(tc.hd)))
		f := func(b compute.Backend) (compute.Buffer, error) {
			return b.Attention(up(t, b, q.Dims, q.F32), up(t, b, k.Dims, k.F32),
				up(t, b, vv.Dims, vv.F32), tc.nh, tc.nhkv, scale, tc.causal)
		}
		co, _ := f(c)
		vo, _ := f(v)
		compare(t, fmt.Sprintf("attention hd=%d nh=%d nhkv=%d T=%d causal=%v", tc.hd, tc.nh, tc.nhkv, tc.T, tc.causal),
			down(t, v, vo), down(t, c, co))
	}
}

// These benchmarks exercise the kernels at the real Qwen3.8-27B shapes so the
// GPU throughput work can be measured without loading the 16 GiB model. Run the
// slow ones with -benchtime=1x to force a single (already expensive) iteration:
//
//	go test ./internal/compute/vulkan -run '^$' -bench Attention -benchtime=1x
//
// Usage:
//   QWEN38_VK_DEVICE selects the device on multi-GPU hosts.

func realBackend(b *testing.B) *vulkan.Backend {
	b.Helper()
	v, err := vulkan.New()
	if err != nil {
		b.Skip(err)
	}
	return v
}

func benchAttentionFwd(b *testing.B, hd, nHead, nHeadKV, T int) {
	v := realBackend(b)
	defer v.Close()
	q := autograd.Random(1, 1.0, hd, nHead, T)
	k := autograd.Random(2, 1.0, hd, nHeadKV, T)
	vv := autograd.Random(3, 1.0, hd, nHeadKV, T)
	qb, _ := v.Upload(q)
	kb, _ := v.Upload(k)
	vb, _ := v.Upload(vv)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	if err := v.Sync(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, err := v.Attention(qb, kb, vb, nHead, nHeadKV, scale, true)
		if err != nil {
			b.Fatal(err)
		}
		v.Free(o)
		if err := v.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAttentionFwdReal128(b *testing.B) { benchAttentionFwd(b, 256, 24, 4, 128) }
func BenchmarkAttentionFwdReal256(b *testing.B) { benchAttentionFwd(b, 256, 24, 4, 256) }
func BenchmarkAttentionFwdReal512(b *testing.B) { benchAttentionFwd(b, 256, 24, 4, 512) }

func benchAttentionBwd(b *testing.B, hd, nHead, nHeadKV, T int) {
	v := realBackend(b)
	defer v.Close()
	q := autograd.Random(11, 1.0, hd, nHead, T)
	k := autograd.Random(12, 1.0, hd, nHeadKV, T)
	vv := autograd.Random(13, 1.0, hd, nHeadKV, T)
	dO := autograd.Random(14, 1.0, hd, nHead, T)
	qb, _ := v.Upload(q)
	kb, _ := v.Upload(k)
	vb, _ := v.Upload(vv)
	dob, _ := v.Upload(dO)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	if err := v.Sync(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dq, dk, dv, err := v.AttentionBackward(qb, kb, vb, dob, nHead, nHeadKV, scale, true)
		if err != nil {
			b.Fatal(err)
		}
		v.Free(dq)
		v.Free(dk)
		v.Free(dv)
		if err := v.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAttentionBwdReal128(b *testing.B) { benchAttentionBwd(b, 256, 24, 4, 128) }
func BenchmarkAttentionBwdReal256(b *testing.B) { benchAttentionBwd(b, 256, 24, 4, 256) }
func BenchmarkAttentionBwdReal512(b *testing.B) { benchAttentionBwd(b, 256, 24, 4, 512) }

func benchGDNFwd(b *testing.B, sv, h, T int) {
	v := realBackend(b)
	defer v.Close()
	q := autograd.Random(21, 1.0, sv, h, T)
	k := autograd.Random(22, 1.0, sv, h, T)
	vv := autograd.Random(23, 1.0, sv, h, T)
	g := autograd.Random(24, 0.3, sv, h, T) // KDA gate (gstride == sv)
	beta := autograd.Random(25, 0.5, 1, h, T)
	state := autograd.Random(26, 0.5, sv, sv, h)
	qb, _ := v.Upload(q)
	kb, _ := v.Upload(k)
	vb, _ := v.Upload(vv)
	gb, _ := v.Upload(g)
	bb, _ := v.Upload(beta)
	sb, _ := v.Upload(state)
	if err := v.Sync(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, ns, err := v.GatedDeltaNet(qb, kb, vb, gb, bb, sb)
		if err != nil {
			b.Fatal(err)
		}
		v.Free(out)
		v.Free(ns)
		if err := v.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGDNFwdReal128(b *testing.B) { benchGDNFwd(b, 128, 48, 128) }
func BenchmarkGDNFwdReal512(b *testing.B) { benchGDNFwd(b, 128, 48, 512) }

// benchMatMulFwd times the forward weight GEMM (f32 storage isolates the GEMM
// from the dequant path): out[N,M] = W[K,N] . x[K,M].
func benchMatMulFwd(b *testing.B, K, N, M int) {
	v := realBackend(b)
	defer v.Close()
	w := autograd.Random(31, 0.05, K, N)
	x := autograd.Random(32, 1.0, K, M)
	wb, _ := v.Upload(w)
	xb, _ := v.Upload(x)
	if err := v.Sync(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, err := v.MatMulWeight(wb, xb)
		if err != nil {
			b.Fatal(err)
		}
		v.Free(o)
		if err := v.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulFwdReal512(b *testing.B) { benchMatMulFwd(b, 5120, 17408, 512) }

// benchMatMulT times the transposed frozen-weight GEMM (activation gradient):
// dX[K,M] = W[K,N]^T . dY[N,M].
func benchMatMulT(b *testing.B, K, N, M int) {
	v := realBackend(b)
	defer v.Close()
	w := autograd.Random(33, 0.05, K, N)
	dy := autograd.Random(34, 1.0, N, M)
	wb, _ := v.Upload(w)
	dyb, _ := v.Upload(dy)
	if err := v.Sync(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, err := v.MatMulWeightTranspose(wb, dyb)
		if err != nil {
			b.Fatal(err)
		}
		v.Free(o)
		if err := v.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulTReal512(b *testing.B) { benchMatMulT(b, 5120, 17408, 512) }

// benchMatMulWG times the LoRA weight-gradient GEMM: dW[In,Out] = x[In,M] . dOut[Out,M]^T.
func benchMatMulWG(b *testing.B, In, Out, M int) {
	v := realBackend(b)
	defer v.Close()
	x := autograd.Random(35, 1.0, In, M)
	dOut := autograd.Random(36, 1.0, Out, M)
	xb, _ := v.Upload(x)
	dob, _ := v.Upload(dOut)
	if err := v.Sync(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, err := v.MatMulWeightGrad(xb, dob)
		if err != nil {
			b.Fatal(err)
		}
		v.Free(o)
		if err := v.Sync(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulWGReal512(b *testing.B) { benchMatMulWG(b, 5120, 17408, 512) }
