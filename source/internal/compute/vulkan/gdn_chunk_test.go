package vulkan_test

import (
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
)

func TestVulkanGatedDeltaNetChunkedBackward(t *testing.T) {
	c, v := newBackends(t)
	vb, ok := v.(*vulkan.Backend)
	if !ok {
		t.Fatal("vulkan backend type")
	}
	if !v.Capabilities().Float32Atomics {
		t.Skip("vulkan device lacks float32 atomics")
	}
	cases := []struct{ sv, h, nTok, chunk int }{
		{4, 3, 19, 1}, {4, 3, 19, 4}, {4, 3, 19, 7}, {4, 3, 19, 32},
		{8, 2, 70, 64}, {5, 2, 13, 5},
	}
	for _, tc := range cases {
		for _, kda := range []bool{false, true} {
			q := autograd.Random(90, 1.0, tc.sv, tc.h, tc.nTok)
			k := autograd.Random(91, 1.0, tc.sv, tc.h, tc.nTok)
			vv := autograd.Random(92, 1.0, tc.sv, tc.h, tc.nTok)
			beta := autograd.Random(94, 0.5, 1, tc.h, tc.nTok)
			state := autograd.Random(95, 0.5, tc.sv, tc.sv, tc.h)
			dOut := autograd.Random(96, 1.0, tc.sv, tc.h, tc.nTok)
			dNew := autograd.Random(97, 1.0, tc.sv, tc.sv, tc.h)
			var gT *compute.Tensor
			if kda {
				gT = autograd.Random(93, 0.3, tc.sv, tc.h, tc.nTok)
			} else {
				gT = autograd.Random(93, 0.3, 1, tc.h, tc.nTok)
			}
			vq, vk, vvv, vg, vb, vs, err := vb.GatedDeltaNetChunkedBackward(
				up(t, v, q.Dims, q.F32), up(t, v, k.Dims, k.F32), up(t, v, vv.Dims, vv.F32),
				up(t, v, gT.Dims, gT.F32), up(t, v, beta.Dims, beta.F32), up(t, v, state.Dims, state.F32),
				up(t, v, dOut.Dims, dOut.F32), up(t, v, dNew.Dims, dNew.F32), tc.chunk)
			if err != nil {
				t.Fatalf("sv=%d h=%d T=%d chunk=%d kda=%v: %v", tc.sv, tc.h, tc.nTok, tc.chunk, kda, err)
			}
			cq, ck, cv, cg, cb, cs, err := c.GatedDeltaNetBackward(
				up(t, c, q.Dims, q.F32), up(t, c, k.Dims, k.F32), up(t, c, vv.Dims, vv.F32),
				up(t, c, gT.Dims, gT.F32), up(t, c, beta.Dims, beta.F32), up(t, c, state.Dims, state.F32),
				up(t, c, dOut.Dims, dOut.F32), up(t, c, dNew.Dims, dNew.F32))
			if err != nil {
				t.Fatal(err)
			}
			name := "kda"
			if !kda {
				name = "scalar"
			}
			compare(t, name+" dV", down(t, v, vvv), down(t, c, cv))
			compare(t, name+" dState", down(t, v, vs), down(t, c, cs))
			compare(t, name+" dBeta", down(t, v, vb), down(t, c, cb))
			compare(t, name+" dV", down(t, v, vvv), down(t, c, cv))
			compare(t, name+" dState", down(t, v, vs), down(t, c, cs))
			compare(t, name+" dBeta", down(t, v, vb), down(t, c, cb))
			compare(t, name+" dQ", down(t, v, vq), down(t, c, cq))
			compare(t, name+" dK", down(t, v, vk), down(t, c, ck))
			compare(t, name+" dG", down(t, v, vg), down(t, c, cg))

			hq, hk, hvv, hg, hb, hs, err := autograd.GatedDeltaNetBackward(q, k, vv, gT, beta, state, dOut, dNew)
			if err != nil {
				t.Fatal(err)
			}
			for _, x := range []struct {
				n    string
				got  []float32
				want *compute.Tensor
			}{
				{name + " h dQ", down(t, v, vq), hq}, {name + " h dK", down(t, v, vk), hk},
				{name + " h dV", down(t, v, vvv), hvv}, {name + " h dG", down(t, v, vg), hg},
				{name + " h dBeta", down(t, v, vb), hb}, {name + " h dState", down(t, v, vs), hs},
			} {
				if d := maxAbsDiffLocal(x.got, x.want.F32); d > 1e-3 {
					t.Fatalf("sv=%d h=%d T=%d chunk=%d %s vs autograd max diff %g", tc.sv, tc.h, tc.nTok, tc.chunk, x.n, d)
				}
			}
		}
	}
}
