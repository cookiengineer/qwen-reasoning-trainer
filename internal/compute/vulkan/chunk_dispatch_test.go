package vulkan_test

import (
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// TestVulkanChunkedDispatch exercises the 1D shaders with element counts above
// the per-dispatch group limit (65535 groups * 64 = 4,194,240 elements) so the
// chunked-dispatch path (multiple dispatches with a base offset) is covered.
func TestVulkanChunkedDispatch(t *testing.T) {
	c, v := newBackends(t)

	const big = 4_300_000 // > 65535*64

	// Unary (silu) and Binary (add) over a large activation.
	x := autograd.Random(1, 1.0, big)
	y := autograd.Random(2, 1.0, big)
	ux, _ := c.Unary(compute.UnarySilu, up(t, c, x.Dims, x.F32))
	vx, _ := v.Unary(compute.UnarySilu, up(t, v, x.Dims, x.F32))
	compare(t, "unary big", down(t, v, vx), down(t, c, ux))
	ux, _ = c.Binary(compute.BinaryAdd, up(t, c, x.Dims, x.F32), up(t, c, y.Dims, y.F32))
	vx, _ = v.Binary(compute.BinaryAdd, up(t, v, x.Dims, x.F32), up(t, v, y.Dims, y.F32))
	compare(t, "binary big", down(t, v, vx), down(t, c, ux))
	ux, _ = c.SiluBack(up(t, c, x.Dims, x.F32), up(t, c, y.Dims, y.F32))
	vx, _ = v.SiluBack(up(t, v, x.Dims, x.F32), up(t, v, y.Dims, y.F32))
	compare(t, "silu_back big", down(t, v, vx), down(t, c, ux))

	// SplitQG: hd*nHead*T > limit.
	hd, nHead, T := 256, 24, 700 // 4,300,800
	qg := autograd.Random(3, 1.0, 2*hd*nHead, T)
	{
		cq, cg, _ := c.SplitQG(up(t, c, qg.Dims, qg.F32), hd, nHead, T)
		vq, vg, _ := v.SplitQG(up(t, v, qg.Dims, qg.F32), hd, nHead, T)
		compare(t, "qg_split q", down(t, v, vq), down(t, c, cq))
		compare(t, "qg_split gate", down(t, v, vg), down(t, c, cg))
	}

	// ConvInput: convDim*T > limit.
	convDim, dConv, ct := 1024, 4, 4200 // 4,300,800
	qkv := autograd.Random(4, 1.0, convDim, ct)
	{
		co, _ := c.ConvInput(up(t, c, qkv.Dims, qkv.F32), dConv, convDim, ct)
		vo, _ := v.ConvInput(up(t, v, qkv.Dims, qkv.F32), dConv, convDim, ct)
		compare(t, "conv_input big", down(t, v, vo), down(t, c, co))
	}

	// SSMConv: dInner*nT > limit.
	dInner, nT := 1024, 4200
	sx := autograd.Random(5, 1.0, nT+dConv-1, dInner, 1)
	cc := autograd.Random(6, 1.0, dConv, dInner)
	{
		co, _ := c.SSMConv(up(t, c, sx.Dims, sx.F32), up(t, c, cc.Dims, cc.F32))
		vo, _ := v.SSMConv(up(t, v, sx.Dims, sx.F32), up(t, v, cc.Dims, cc.F32))
		compare(t, "ssm_conv big", down(t, v, vo), down(t, c, co))
	}
}
