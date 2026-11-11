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
