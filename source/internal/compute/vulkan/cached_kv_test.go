package vulkan_test

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// writeKVHeadMajor reorders token-major quantized bytes into the head-major
// runtime cache layout before uploading them.
func writeKVHeadMajor(t *testing.T, vb *vulkan.Backend, cache compute.Buffer, raw []byte, capacity, nHeadKV int) {
	t.Helper()
	per := len(raw) / (capacity * nHeadKV)
	out := make([]byte, len(raw))
	for tok := 0; tok < capacity; tok++ {
		for h := 0; h < nHeadKV; h++ {
			src := (tok*nHeadKV + h) * per
			dst := (h*capacity + tok) * per
			copy(out[dst:dst+per], raw[src:src+per])
		}
	}
	if err := vb.WriteKV(cache, 0, out); err != nil {
		t.Fatal(err)
	}
}

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
			writeKVHeadMajor(t, vb, kc, kraw, nKV, nHeadKV)
			writeKVHeadMajor(t, vb, vc, vraw, nKV, nHeadKV)

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
			writeKVHeadMajor(t, vb, kc, kraw, nKV, nHeadKV)
			writeKVHeadMajor(t, vb, vc, vraw, nKV, nHeadKV)
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

func TestVulkanConvInputState(t *testing.T) {
	_, be := newBackends(t)
	vb := be.(*vulkan.Backend)
	const dConv, convDim, T = 3, 2, 2
	histData := []float32{1, 2, 3, 4} // [2,2] row-major ch fastest
	qkv := []float32{5, 6, 7, 8}      // [convDim,T]
	hb := up(t, vb, []int{dConv - 1, convDim}, histData)
	defer vb.Free(hb)
	qb := up(t, vb, []int{convDim, T}, qkv)
	defer vb.Free(qb)
	out, err := vb.ConvInputState(qb, hb, dConv, convDim, T)
	if err != nil {
		t.Fatal(err)
	}
	defer vb.Free(out)
	got := down(t, vb, out)
	// out is [dConv-1+T, convDim], element (row,ch) at row+ch*ncs.
	ncs := dConv - 1 + T
	want := make([]float32, ncs*convDim)
	for ch := 0; ch < convDim; ch++ {
		for row := 0; row < dConv-1; row++ {
			want[row+ch*ncs] = histData[row+ch*(dConv-1)]
		}
		for tt := 0; tt < T; tt++ {
			want[(dConv-1+tt)+ch*ncs] = qkv[ch+tt*convDim]
		}
	}
	compare(t, "conv-input-state", got, want)
	gh := down(t, vb, hb)
	for ch := 0; ch < convDim; ch++ {
		for row := 0; row < dConv-1; row++ {
			if gh[row+ch*(dConv-1)] != want[ncs-(dConv-1)+row+ch*ncs] {
				t.Fatalf("history[%d,%d] = %v, want %v", row, ch, gh[row+ch*(dConv-1)], want[ncs-(dConv-1)+row+ch*ncs])
			}
		}
	}
}

func TestVulkanGatedDeltaNetChunkSelfConsistency(t *testing.T) {
	_, be := newBackends(t)
	const sv, h, T = 4, 2, 2
	q := autograd.Random(301, 1, sv, h, T)
	k := autograd.Random(302, 1, sv, h, T)
	v := autograd.Random(303, 1, sv, h, T)
	g := autograd.Random(304, 0.1, 1, h, T)
	beta := autograd.Random(305, 0.1, 1, h, T)
	state := autograd.Random(306, 0.1, sv, sv, h)

	qb := up(t, be, q.Dims, q.F32)
	defer be.Free(qb)
	kb := up(t, be, k.Dims, k.F32)
	defer be.Free(kb)
	vb := up(t, be, v.Dims, v.F32)
	defer be.Free(vb)
	gb := up(t, be, g.Dims, g.F32)
	defer be.Free(gb)
	bb := up(t, be, beta.Dims, beta.F32)
	defer be.Free(bb)
	sb := up(t, be, state.Dims, state.F32)
	defer be.Free(sb)
	out2, ns2, err := be.GatedDeltaNet(qb, kb, vb, gb, bb, sb)
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(out2)
	defer be.Free(ns2)
	got2 := down(t, be, out2)

	// Two one-token calls.
	q1 := &compute.Tensor{Dims: []int{sv, h, 1}, F32: q.F32[:sv*h]}
	k1 := &compute.Tensor{Dims: []int{sv, h, 1}, F32: k.F32[:sv*h]}
	v1 := &compute.Tensor{Dims: []int{sv, h, 1}, F32: v.F32[:sv*h]}
	g1 := &compute.Tensor{Dims: []int{1, h, 1}, F32: g.F32[:h]}
	b1 := &compute.Tensor{Dims: []int{1, h, 1}, F32: beta.F32[:h]}
	q2 := &compute.Tensor{Dims: []int{sv, h, 1}, F32: q.F32[sv*h:]}
	k2 := &compute.Tensor{Dims: []int{sv, h, 1}, F32: k.F32[sv*h:]}
	v2 := &compute.Tensor{Dims: []int{sv, h, 1}, F32: v.F32[sv*h:]}
	g2 := &compute.Tensor{Dims: []int{1, h, 1}, F32: g.F32[h:]}
	b2 := &compute.Tensor{Dims: []int{1, h, 1}, F32: beta.F32[h:]}
	st := sb
	qb1 := up(t, be, q1.Dims, q1.F32)
	defer be.Free(qb1)
	kb1 := up(t, be, k1.Dims, k1.F32)
	defer be.Free(kb1)
	vb1 := up(t, be, v1.Dims, v1.F32)
	defer be.Free(vb1)
	gb1 := up(t, be, g1.Dims, g1.F32)
	defer be.Free(gb1)
	bb1 := up(t, be, b1.Dims, b1.F32)
	defer be.Free(bb1)
	o1, n1, err := be.GatedDeltaNet(qb1, kb1, vb1, gb1, bb1, st)
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(o1)
	qb2 := up(t, be, q2.Dims, q2.F32)
	defer be.Free(qb2)
	kb2 := up(t, be, k2.Dims, k2.F32)
	defer be.Free(kb2)
	vb2 := up(t, be, v2.Dims, v2.F32)
	defer be.Free(vb2)
	gb2 := up(t, be, g2.Dims, g2.F32)
	defer be.Free(gb2)
	bb2 := up(t, be, b2.Dims, b2.F32)
	defer be.Free(bb2)
	o2, _, err := be.GatedDeltaNet(qb2, kb2, vb2, gb2, bb2, n1)
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(o2)
	defer be.Free(n1)
	got1 := append(down(t, be, o1), down(t, be, o2)...)
	for i := range got2 {
		if d := math.Abs(float64(got2[i] - got1[i])); d > 1e-4 {
			t.Fatalf("gdn chunk mismatch [%d]: %v vs %v", i, got2[i], got1[i])
		}
	}
}

func TestVulkanMatMulWeightM1VsM2(t *testing.T) {
	c, be := newBackends(t)
	const K, N = 64, 32
	w := autograd.Random(401, 0.1, K, N)
	x2 := autograd.Random(402, 1, K, 2)
	raw, err := quant.Quantize(quant.TypeF32, w.F32)
	if err != nil {
		t.Fatal(err)
	}
	wb, err := be.UploadWeight(quant.TypeF32, raw, w.Dims)
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(wb)
	out2, err := be.MatMulWeight(wb, up(t, be, x2.Dims, x2.F32))
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(out2)
	col2 := down(t, be, out2)
	for col := 0; col < 2; col++ {
		x1 := &compute.Tensor{Dims: []int{K, 1}, F32: append([]float32(nil), x2.F32[col*K:(col+1)*K]...)}
		out1, err := be.MatMulWeight(wb, up(t, be, x1.Dims, x1.F32))
		if err != nil {
			t.Fatal(err)
		}
		col1 := down(t, be, out1)
		be.Free(out1)
		for i := 0; i < N; i++ {
			if d := math.Abs(float64(col1[i] - col2[i+col*N])); d > 2e-4+2e-4*math.Abs(float64(col1[i])) {
				t.Fatalf("col %d row %d: m1 %v m2 %v diff %v", col, i, col1[i], col2[i+col*N], d)
			}
		}
	}
	_ = c
}

func TestVulkanConvInputStateChunkSelfConsistency(t *testing.T) {
	_, be := newBackends(t)
	vb := be.(*vulkan.Backend)
	const dConv, convDim = 4, 3
	h0 := autograd.Random(501, 0.1, dConv-1, convDim)
	q2 := autograd.Random(502, 0.1, convDim, 2)
	hb := up(t, vb, h0.Dims, h0.F32)
	defer vb.Free(hb)
	qb2 := up(t, vb, q2.Dims, q2.F32)
	defer vb.Free(qb2)
	out2, err := vb.ConvInputState(qb2, hb, dConv, convDim, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer vb.Free(out2)
	got2 := down(t, vb, out2)
	hist2 := down(t, vb, hb)

	// Sequential one-token calls from the same initial history.
	hb1 := up(t, vb, h0.Dims, h0.F32)
	defer vb.Free(hb1)
	ncs := dConv - 1 + 1
	q1a := &compute.Tensor{Dims: []int{convDim, 1}, F32: append([]float32(nil), q2.F32[:convDim]...)}
	q1b := &compute.Tensor{Dims: []int{convDim, 1}, F32: append([]float32(nil), q2.F32[convDim:]...)}
	qa := up(t, vb, q1a.Dims, q1a.F32)
	defer vb.Free(qa)
	oa, err := vb.ConvInputState(qa, hb1, dConv, convDim, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer vb.Free(oa)
	qb := up(t, vb, q1b.Dims, q1b.F32)
	defer vb.Free(qb)
	ob, err := vb.ConvInputState(qb, hb1, dConv, convDim, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer vb.Free(ob)
	gotA := down(t, vb, oa)
	gotB := down(t, vb, ob)
	// Combined token 0 row is the last row of oa; token 1 row is last of ob.
	for ch := 0; ch < convDim; ch++ {
		if got2[(dConv-1)+ch*(dConv-1+2)] != gotA[(ncs-1)+ch*ncs] {
			t.Fatalf("token0 ch%d: %v vs %v", ch, got2[(dConv-1)+ch*(dConv-1+2)], gotA[(ncs-1)+ch*ncs])
		}
		if got2[(dConv-1+1)+ch*(dConv-1+2)] != gotB[(ncs-1)+ch*ncs] {
			t.Fatalf("token1 ch%d: %v vs %v", ch, got2[(dConv-1+1)+ch*(dConv-1+2)], gotB[(ncs-1)+ch*ncs])
		}
	}
	hist1 := down(t, vb, hb1)
	for i := range hist2 {
		if hist2[i] != hist1[i] {
			t.Fatalf("history[%d]: %v vs %v", i, hist2[i], hist1[i])
		}
	}
}

func headMajorToTokenMajor(vals []float32, hd, nHeadKV, T int) []float32 {
	out := make([]float32, len(vals))
	for t := 0; t < T; t++ {
		for h := 0; h < nHeadKV; h++ {
			for d := 0; d < hd; d++ {
				out[d+h*hd+t*nHeadKV*hd] = vals[d+t*hd+h*T*hd]
			}
		}
	}
	return out
}

func TestVulkanAppendKV(t *testing.T) {
	_, be := newBackends(t)
	vb := be.(*vulkan.Backend)
	const (
		hd      = 256
		nHeadKV = 2
		T       = 2
	)
	k := autograd.Random(601, 1, hd, nHeadKV, T)
	v := autograd.Random(602, 1, hd, nHeadKV, T)
	kb := up(t, vb, k.Dims, k.F32)
	defer vb.Free(kb)
	vbK := up(t, vb, v.Dims, v.F32)
	defer vb.Free(vbK)
	for _, typ := range []quant.Type{quant.TypeQ4_K, quant.TypeQ8_0} {
		t.Run(typ.String(), func(t *testing.T) {
			kc, err := vb.AllocKV(hd, nHeadKV, T, typ)
			if err != nil {
				t.Fatal(err)
			}
			defer vb.Free(kc)
			vc, err := vb.AllocKV(hd, nHeadKV, T, typ)
			if err != nil {
				t.Fatal(err)
			}
			defer vb.Free(vc)
			if err := vb.AppendKV(kb, vbK, kc, vc, 0); err != nil {
				t.Fatal(err)
			}
			if err := vb.Sync(); err != nil {
				t.Fatal(err)
			}
			elements := int64(hd * nHeadKV * T)
			size, err := typ.Size(elements)
			if err != nil {
				t.Fatal(err)
			}
			rawK, err := vb.ReadRaw(kc, int(size))
			if err != nil {
				t.Fatal(err)
			}
			rawV, err := vb.ReadRaw(vc, int(size))
			if err != nil {
				t.Fatal(err)
			}
			kf, err := quant.Dequant(typ, rawK, elements)
			if err != nil {
				t.Fatal(err)
			}
			vf, err := quant.Dequant(typ, rawV, elements)
			if err != nil {
				t.Fatal(err)
			}
			kf = headMajorToTokenMajor(kf, hd, nHeadKV, T)
			vf = headMajorToTokenMajor(vf, hd, nHeadKV, T)
			tolBase := float32(2e-1 + 2e-1)
			for i := range k.F32 {
				tol := tolBase
				if typ == quant.TypeQ8_0 {
					tol = 2e-2 + 1e-2*float32(math.Abs(float64(k.F32[i])))
				} else {
					tol = 2e-1 + 2e-1*float32(math.Abs(float64(k.F32[i])))
				}
				if d := float32(math.Abs(float64(kf[i] - k.F32[i]))); d > tol {
					t.Fatalf("%s K[%d]: got %v want %v (diff %v)", typ, i, kf[i], k.F32[i], d)
				}
				if typ == quant.TypeQ8_0 {
					tol = 2e-2 + 1e-2*float32(math.Abs(float64(v.F32[i])))
				} else {
					tol = 2e-1 + 2e-1*float32(math.Abs(float64(v.F32[i])))
				}
				if d := float32(math.Abs(float64(vf[i] - v.F32[i]))); d > tol {
					t.Fatalf("%s V[%d]: got %v want %v (diff %v)", typ, i, vf[i], v.F32[i], d)
				}
			}
		})
	}
}

// TestVulkanCachedAttentionSplitGQA exercises the GQA-aware split decode
// kernels, which dequantize each KV head once for a group of six query heads.
func TestVulkanCachedAttentionSplitGQA(t *testing.T) {
	_, be := newBackends(t)
	vb := be.(*vulkan.Backend)
	const (
		hd      = 256
		nHead   = 12
		nHeadKV = 2
		nKV     = 9000
	)
	q := autograd.Random(211, 1.0, hd, nHead, 1)
	k := autograd.Random(212, 1.0, hd, nHeadKV, nKV)
	vv := autograd.Random(213, 1.0, hd, nHeadKV, nKV)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	group := nHead / nHeadKV

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
			writeKVHeadMajor(t, vb, kc, kraw, nKV, nHeadKV)
			writeKVHeadMajor(t, vb, vc, vraw, nKV, nHeadKV)
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
			probs := make([]float64, nKV)
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
					probs[j] = dot
					if dot > maxLogit {
						maxLogit = dot
					}
				}
				var sum float64
				for j := 0; j < nKV; j++ {
					e := math.Exp(probs[j] - maxLogit)
					probs[j] = e
					sum += e
				}
				for d := 0; d < hd; d++ {
					var acc float64
					for j := 0; j < nKV; j++ {
						vOff := (j*nHeadKV + kv) * hd
						acc += probs[j] * float64(vf[vOff+d])
					}
					want[h*hd+d] = float32(acc / sum)
				}
			}
			for i := range want {
				tol := float32(1e-1 + 1e-1*math.Abs(float64(want[i])))
				if d := float32(math.Abs(float64(got[i] - want[i]))); d > tol {
					t.Fatalf("[%d]: got %v want %v (diff %v)", i, got[i], want[i], d)
				}
			}
		})
	}
}
