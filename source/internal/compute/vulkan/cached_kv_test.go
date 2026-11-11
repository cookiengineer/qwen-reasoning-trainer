package vulkan_test

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func TestVulkanCachedAttention(t *testing.T) {
	_, be := newBackends(t)
	vb := be.(*vulkan.Backend)
	const (
		hd      = 256
		nHead   = 4
		nHeadKV = 2
		nKV     = 5
		nQ      = 2
	)
	qFull := autograd.Random(101, 1.0, hd, nHead, nKV)
	k := autograd.Random(102, 1.0, hd, nHeadKV, nKV)
	vv := autograd.Random(103, 1.0, hd, nHeadKV, nKV)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	qStart := nKV - nQ
	qTail := &compute.Tensor{
		Dims: []int{hd, nHead, nQ},
		Type: qFull.Type,
		F32:  qFull.F32[qStart*hd*nHead:],
	}

	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ8_0} {
		t.Run(typ.String(), func(t *testing.T) {
			kc, err := vb.AllocKV(hd, nHeadKV, nKV, typ)
			if err != nil {
				t.Fatal(err)
			}
			defer vb.Free(kc)
			vc, err := vb.AllocKV(hd, nHeadKV, nKV, typ)
			if err != nil {
				t.Fatal(err)
			}
			defer vb.Free(vc)

			kraw, err := quant.Quantize(typ, k.F32)
			if err != nil {
				t.Fatal(err)
			}
			vraw, err := quant.Quantize(typ, vv.F32)
			if err != nil {
				t.Fatal(err)
			}
			if err := vb.WriteKV(kc, 0, kraw); err != nil {
				t.Fatal(err)
			}
			if err := vb.WriteKV(vc, 0, vraw); err != nil {
				t.Fatal(err)
			}

			qb := up(t, vb, qTail.Dims, qTail.F32)
			defer vb.Free(qb)
			out, err := vb.AttentionCached(qb, kc, vc, nHead, nHeadKV, nKV, scale)
			if err != nil {
				t.Fatal(err)
			}
			defer vb.Free(out)
			got := down(t, vb, out)

			// CPU reference: full causal attention, then compare the last nQ
			// query positions (the shader's absolute qStart window).
			ref, err := compute.Attention(qFull, k, vv, nHead, nHeadKV, scale, true)
			if err != nil {
				t.Fatal(err)
			}
			want := ref.F32[qStart*hd*nHead:]
			if len(got) != len(want) {
				t.Fatalf("len = %d, want %d", len(got), len(want))
			}
			for i := range want {
				tol := float32(5e-2 + 1e-1*math.Abs(float64(want[i])))
				if d := float32(math.Abs(float64(got[i] - want[i]))); d > tol {
					t.Fatalf("%s[%d]: got %v want %v (diff %v)", typ, i, got[i], want[i], d)
				}
			}
		})
	}
}

func TestVulkanCachedAttentionSplit(t *testing.T) {
	_, be := newBackends(t)
	vb := be.(*vulkan.Backend)
	const (
		hd      = 256
		nHead   = 4
		nHeadKV = 2
		nKV     = 20000
	)
	q := autograd.Random(201, 1.0, hd, nHead, 1)
	k := autograd.Random(202, 1.0, hd, nHeadKV, nKV)
	vv := autograd.Random(203, 1.0, hd, nHeadKV, nKV)
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ8_0} {
		t.Run(typ.String(), func(t *testing.T) {
			kc, err := vb.AllocKV(hd, nHeadKV, nKV, typ)
			if err != nil {
				t.Fatal(err)
			}
			defer vb.Free(kc)
			vc, err := vb.AllocKV(hd, nHeadKV, nKV, typ)
			if err != nil {
				t.Fatal(err)
			}
			defer vb.Free(vc)
			kraw, err := quant.Quantize(typ, k.F32)
			if err != nil {
				t.Fatal(err)
			}
			vraw, err := quant.Quantize(typ, vv.F32)
			if err != nil {
				t.Fatal(err)
			}
			if err := vb.WriteKV(kc, 0, kraw); err != nil {
				t.Fatal(err)
			}
			if err := vb.WriteKV(vc, 0, vraw); err != nil {
				t.Fatal(err)
			}
			qb := up(t, vb, q.Dims, q.F32)
			defer vb.Free(qb)
			out, err := vb.AttentionCached(qb, kc, vc, nHead, nHeadKV, nKV, scale)
			if err != nil {
				t.Fatal(err)
			}
			defer vb.Free(out)
			got := down(t, vb, out)

			kf, err := quant.Dequant(typ, kraw, int64(len(k.F32)))
			if err != nil {
				t.Fatal(err)
			}
			vf, err := quant.Dequant(typ, vraw, int64(len(vv.F32)))
			if err != nil {
				t.Fatal(err)
			}
			group := nHead / nHeadKV
			logits := make([]float64, nKV)
			want := make([]float32, hd*nHead)
			for h := 0; h < nHead; h++ {
				kv := h / group
				maxLogit := math.Inf(-1)
				for j := 0; j < nKV; j++ {
					kOff := (j*nHeadKV + kv) * hd
					qOff := h * hd
					var dot float64
					for d := 0; d < hd; d++ {
						dot += float64(q.F32[qOff+d]) * float64(kf[kOff+d])
					}
					dot *= float64(scale)
					logits[j] = dot
					if dot > maxLogit {
						maxLogit = dot
					}
				}
				var sum float64
				for j := 0; j < nKV; j++ {
					e := math.Exp(logits[j] - maxLogit)
					logits[j] = e
					sum += e
				}
				for d := 0; d < hd; d++ {
					var acc float64
					for j := 0; j < nKV; j++ {
						vOff := (j*nHeadKV + kv) * hd
						acc += logits[j] * float64(vf[vOff+d])
					}
					want[h*hd+d] = float32(acc / sum)
				}
			}
			for i := range want {
				tol := float32(1e-1 + 1e-1*math.Abs(float64(want[i])))
				if d := float32(math.Abs(float64(got[i] - want[i]))); d > tol {
					t.Fatalf("%s[%d]: got %v want %v (diff %v)", typ, i, got[i], want[i], d)
				}
			}
		})
	}
}
