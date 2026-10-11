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

	// RoPE: nTok*nHead*(nDims/2) > limit.
	{
		rhd, rnHead, rnTok, rnDims := 4, 8, 280000, 4 // total 4,480,000
		rx := autograd.Random(7, 1.0, rhd, rnHead, rnTok)
		pos := make([]int32, rnTok)
		for i := range pos {
			pos[i] = int32(i % 16) // keep angles small to avoid pow/argument-reduction noise
		}
		co, _ := c.RoPE(up(t, c, rx.Dims, rx.F32), pos, 1e7, rnDims)
		vo, _ := v.RoPE(up(t, v, rx.Dims, rx.F32), pos, 1e7, rnDims)
		compare(t, "rope big", down(t, v, vo), down(t, c, co))
	}

	// RepeatHeads / RepeatHeadsBack: headDim*nOut*T and headDim*nIn*T > limit.
	{
		rhd, rnIn, rnOut, rT := 4, 4, 4, 280000 // 4,480,000
		rx := autograd.Random(8, 1.0, rhd, rnIn, rT)
		co, _ := c.RepeatHeads(up(t, c, rx.Dims, rx.F32), rhd, rnIn, rnOut, rT)
		vo, _ := v.RepeatHeads(up(t, v, rx.Dims, rx.F32), rhd, rnIn, rnOut, rT)
		compare(t, "repeat_heads big", down(t, v, vo), down(t, c, co))

		gy := autograd.Random(9, 1.0, rhd, rnOut, rT)
		cb, _ := c.RepeatHeadsBack(up(t, c, gy.Dims, gy.F32), rhd, rnIn, rnOut, rT)
		vb, _ := v.RepeatHeadsBack(up(t, v, gy.Dims, gy.F32), rhd, rnIn, rnOut, rT)
		compare(t, "repeat_heads_back big", down(t, v, vb), down(t, c, cb))
	}

	// GatherHeads / GatherHeadsBack: headDim*nHead*T > limit.
	{
		ghd, gnHead, gT, gConv := 4, 4, 280000, 16 // 4,480,000
		conv := autograd.Random(10, 1.0, gConv, gT)
		co, _ := c.GatherHeads(up(t, c, conv.Dims, conv.F32), 0, ghd, gnHead, gT, gConv)
		vo, _ := v.GatherHeads(up(t, v, conv.Dims, conv.F32), 0, ghd, gnHead, gT, gConv)
		compare(t, "gather_heads big", down(t, v, vo), down(t, c, co))

		gy := autograd.Random(11, 1.0, ghd, gnHead, gT)
		cb, _ := c.GatherHeadsBack(up(t, c, gy.Dims, gy.F32), 0, ghd, gnHead, gT, gConv)
		vb, _ := v.GatherHeadsBack(up(t, v, gy.Dims, gy.F32), 0, ghd, gnHead, gT, gConv)
		compare(t, "gather_heads_back big", down(t, v, vb), down(t, c, cb))
	}

	// GetRows: rowLen*count > limit.
	{
		rowLen, count := 64, 70000 // 4,480,000
		tbl := autograd.Random(12, 1.0, rowLen, 64)
		idx := make([]int32, count)
		for i := range idx {
			idx[i] = int32((i * 7) % 64)
		}
		co, _ := c.GetRows(up(t, c, tbl.Dims, tbl.F32), idx)
		vo, _ := v.GetRows(up(t, v, tbl.Dims, tbl.F32), idx)
		compare(t, "get_rows big", down(t, v, vo), down(t, c, co))
	}
}
