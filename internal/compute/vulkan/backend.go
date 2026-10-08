package vulkan

import (
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// errNoGemv reports that no fused dequant+GEMV shader exists for a type.
var errNoGemv = errors.New("vulkan: no fused gemv shader")

// gemvShaders maps a quantized type to its fused dequantize+GEMV shader.
var gemvShaders = map[quant.Type]string{
	quant.TypeQ4_K:   "gemv_q4_k",
	quant.TypeQ5_K:   "gemv_q5_k",
	quant.TypeQ6_K:   "gemv_q6_k",
	quant.TypeIQ4_XS: "gemv_iq4_xs",
}

// Capabilities implements compute.Backend.
func (b *Backend) Capabilities() compute.Capabilities { return b.caps }

func (b *Backend) newF32Buffer(dims ...int) (*buffer, error) {
	n := 1
	for _, d := range dims {
		n *= d
	}
	buf, err := b.allocBuffer(uint64(n)*4, vkBufferUsageStorage|vkBufferUsageTransferSrc|vkBufferUsageTransferDst)
	if err != nil {
		return nil, err
	}
	buf.dims = dims
	buf.typ = quant.TypeF32
	return buf, nil
}

func asBuffer(buf compute.Buffer) (*buffer, error) {
	x, ok := buf.(*buffer)
	if !ok {
		return nil, fmt.Errorf("vulkan: foreign buffer")
	}
	return x, nil
}

// Alloc implements compute.Backend.
func (b *Backend) Alloc(dims []int, t quant.Type) (compute.Buffer, error) {
	buf, err := b.newF32Buffer(dims...)
	if err != nil {
		return nil, err
	}
	buf.typ = t
	return buf, nil
}

// Upload implements compute.Backend.
func (b *Backend) Upload(t *compute.Tensor) (compute.Buffer, error) {
	if len(t.F32) == 0 {
		return b.newF32Buffer(1)
	}
	return b.uploadTensor(t)
}

// Download implements compute.Backend.
func (b *Backend) Download(buf compute.Buffer) (*compute.Tensor, error) {
	x, err := asBuffer(buf)
	if err != nil {
		return nil, err
	}
	if err := b.flush(); err != nil {
		return nil, err
	}
	return b.downloadBuffer(x)
}

// Free implements compute.Backend. Buffers freed while commands are being
// recorded are destroyed after the next flush.
func (b *Backend) Free(buf compute.Buffer) {
	x, ok := buf.(*buffer)
	if !ok || x == nil {
		return
	}
	if b.recording {
		b.pending = append(b.pending, x)
		return
	}
	b.destroyBuffer(x)
}

// destroyBuffer releases a buffer back to the pool, or frees it when the pool
// is disabled or full.
func (b *Backend) destroyBuffer(x *buffer) {
	if !b.closed && b.poolPut(x) {
		return
	}
	b.freeBuffer(x)
}

// freeBuffer unmaps and destroys a buffer's Vulkan resources.
func (b *Backend) freeBuffer(x *buffer) {
	if x.mapped != nil {
		vkCall(b.vk.UnmapMemory, b.device, x.mem)
		x.mapped = nil
	}
	vkCall(b.vk.DestroyBuffer, b.device, x.buf, 0)
	vkCall(b.vk.FreeMemory, b.device, x.mem, 0)
}

// drainPool frees every pooled buffer.
func (b *Backend) drainPool() {
	for key, list := range b.pool {
		for _, buf := range list {
			b.freeBuffer(buf)
		}
		delete(b.pool, key)
	}
	b.poolBytes = 0
}

// Sync implements compute.Backend.
func (b *Backend) Sync() error {
	return b.flush()
}

// Close implements compute.Backend.
func (b *Backend) Close() error {
	if b.closed {
		return nil
	}
	if b.recording {
		b.flush()
	}
	b.closed = true
	b.drainPool()
	if b.dummy != nil {
		b.freeBuffer(b.dummy)
		b.dummy = nil
	}
	if b.device != 0 {
		vkCall(b.vk.DeviceWaitIdle, b.device)
		vkCall(b.vk.DestroyDevice, b.device, 0)
	}
	if b.instance != 0 {
		vkCall(b.vk.DestroyInstance, b.instance, 0)
	}
	return nil
}

func (b *Backend) uploadInt32(vals []int32) (*buffer, error) {
	buf, err := b.allocBuffer(uint64(len(vals))*4, vkBufferUsageStorage|vkBufferUsageTransferSrc|vkBufferUsageTransferDst)
	if err != nil {
		return nil, err
	}
	buf.dims = []int{len(vals)}
	src := unsafe.Slice((*byte)(unsafe.Pointer(&vals[0])), len(vals)*4)
	if err := b.writeBytes(buf, src); err != nil {
		b.destroyBuffer(buf)
		return nil, err
	}
	return buf, nil
}

// Unary implements compute.Backend.
func (b *Backend) Unary(op compute.UnaryOp, a compute.Buffer) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	out, err := b.newF32Buffer(x.dims...)
	if err != nil {
		return nil, err
	}
	n := uint32(x.NumElements())
	groups := [3]uint32{ceilDiv(n, 64), 1, 1}
	if err := b.dispatch("unary", []*buffer{x, out}, push(n, uint32(op), float32(0)), groups); err != nil {
		return nil, err
	}
	return out, nil
}

// Scale implements compute.Backend.
func (b *Backend) Scale(a compute.Buffer, s float32) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	out, err := b.newF32Buffer(x.dims...)
	if err != nil {
		return nil, err
	}
	n := uint32(x.NumElements())
	groups := [3]uint32{ceilDiv(n, 64), 1, 1}
	if err := b.dispatch("unary", []*buffer{x, out}, push(n, uint32(9), s), groups); err != nil {
		return nil, err
	}
	return out, nil
}

// Binary implements compute.Backend.
func (b *Backend) Binary(op compute.BinaryOp, a, c compute.Buffer) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	y, err := asBuffer(c)
	if err != nil {
		return nil, err
	}
	if x.NumElements() != y.NumElements() {
		return nil, compute.ErrShape
	}
	out, err := b.newF32Buffer(x.dims...)
	if err != nil {
		return nil, err
	}
	n := uint32(x.NumElements())
	groups := [3]uint32{ceilDiv(n, 64), 1, 1}
	if err := b.dispatch("binary", []*buffer{x, y, out}, push(n, uint32(op)), groups); err != nil {
		return nil, err
	}
	return out, nil
}

// MatMul implements compute.Backend.
func (b *Backend) MatMul(a, c compute.Buffer) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	y, err := asBuffer(c)
	if err != nil {
		return nil, err
	}
	k := x.Ne0()
	n := dimOr1(x.dims, 1)
	m := dimOr1(y.dims, 1)
	if y.Ne0() != k {
		return nil, compute.ErrShape
	}
	out, err := b.newF32Buffer(n, m)
	if err != nil {
		return nil, err
	}
	// Tiled GEMM: one 16x16 workgroup tile per output block.
	gx := ceilDiv(uint32(n), 16)
	gy := ceilDiv(uint32(m), 16)
	if gx > 65535 || gy > 65535 {
		b.Free(out)
		return nil, fmt.Errorf("vulkan: matmul dimensions too large (%d x %d)", n, m)
	}
	groups := [3]uint32{gx, gy, 1}
	if err := b.dispatch("matmul", []*buffer{x, y, out}, push(uint32(k), uint32(n), uint32(m), uint32(0)), groups); err != nil {
		b.Free(out)
		return nil, err
	}
	return out, nil
}

// RMSNorm implements compute.Backend.
func (b *Backend) RMSNorm(a, w compute.Buffer, eps float32) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	out, err := b.newF32Buffer(x.dims...)
	if err != nil {
		return nil, err
	}
	rowLen := x.Ne0()
	rows := x.NumElements() / rowLen
	var wb *buffer
	hasW := uint32(0)
	if w != nil {
		wb, err = asBuffer(w)
		if err != nil {
			return nil, err
		}
		hasW = 1
	}
	groups := [3]uint32{uint32(rows), 1, 1}
	if err := b.dispatch("rms_norm", []*buffer{x, wb, out}, push(uint32(rowLen), uint32(rows), eps, hasW), groups); err != nil {
		return nil, err
	}
	return out, nil
}

// RMSNormBack implements compute.Backend.
func (b *Backend) RMSNormBack(a, w, dOut compute.Buffer, eps float32) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	d, err := asBuffer(dOut)
	if err != nil {
		return nil, err
	}
	out, err := b.newF32Buffer(x.dims...)
	if err != nil {
		return nil, err
	}
	rowLen := x.Ne0()
	rows := x.NumElements() / rowLen
	var wb *buffer
	hasW := uint32(0)
	if w != nil {
		wb, err = asBuffer(w)
		if err != nil {
			return nil, err
		}
		hasW = 1
	}
	groups := [3]uint32{uint32(rows), 1, 1}
	if err := b.dispatch("rms_norm_back", []*buffer{x, wb, d, out}, push(uint32(rowLen), uint32(rows), eps, hasW), groups); err != nil {
		b.Free(out)
		return nil, err
	}
	return out, nil
}

// SiluBack implements compute.Backend.
func (b *Backend) SiluBack(x, dOut compute.Buffer) (compute.Buffer, error) {
	xa, err := asBuffer(x)
	if err != nil {
		return nil, err
	}
	da, err := asBuffer(dOut)
	if err != nil {
		return nil, err
	}
	out, err := b.newF32Buffer(xa.dims...)
	if err != nil {
		return nil, err
	}
	n := uint32(xa.NumElements())
	groups := [3]uint32{ceilDiv(n, 64), 1, 1}
	if err := b.dispatch("silu_back", []*buffer{xa, da, out}, push(n), groups); err != nil {
		b.Free(out)
		return nil, err
	}
	return out, nil
}

// L2Norm implements compute.Backend.
func (b *Backend) L2Norm(a compute.Buffer, eps float32) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	out, err := b.newF32Buffer(x.dims...)
	if err != nil {
		return nil, err
	}
	rowLen := x.Ne0()
	rows := x.NumElements() / rowLen
	groups := [3]uint32{uint32(rows), 1, 1}
	if err := b.dispatch("l2_norm", []*buffer{x, out}, push(uint32(rowLen), uint32(rows), eps, uint32(0)), groups); err != nil {
		b.Free(out)
		return nil, err
	}
	return out, nil
}

// L2NormBack implements compute.Backend.
func (b *Backend) L2NormBack(a, dOut compute.Buffer, eps float32) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	d, err := asBuffer(dOut)
	if err != nil {
		return nil, err
	}
	out, err := b.newF32Buffer(x.dims...)
	if err != nil {
		return nil, err
	}
	rowLen := x.Ne0()
	rows := x.NumElements() / rowLen
	groups := [3]uint32{uint32(rows), 1, 1}
	if err := b.dispatch("l2_norm_back", []*buffer{x, d, out}, push(uint32(rowLen), uint32(rows), eps, uint32(0)), groups); err != nil {
		b.Free(out)
		return nil, err
	}
	return out, nil
}

// Softmax implements compute.Backend.
func (b *Backend) Softmax(a compute.Buffer) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	out, err := b.newF32Buffer(x.dims...)
	if err != nil {
		return nil, err
	}
	rowLen := x.Ne0()
	rows := x.NumElements() / rowLen
	groups := [3]uint32{uint32(rows), 1, 1}
	if err := b.dispatch("softmax", []*buffer{x, out}, push(uint32(rowLen), uint32(rows)), groups); err != nil {
		return nil, err
	}
	return out, nil
}

// SoftmaxBack implements compute.Backend.
func (b *Backend) SoftmaxBack(out, dOut compute.Buffer) (compute.Buffer, error) {
	o, err := asBuffer(out)
	if err != nil {
		return nil, err
	}
	d, err := asBuffer(dOut)
	if err != nil {
		return nil, err
	}
	dx, err := b.newF32Buffer(o.dims...)
	if err != nil {
		return nil, err
	}
	rowLen := o.Ne0()
	rows := o.NumElements() / rowLen
	groups := [3]uint32{uint32(rows), 1, 1}
	if err := b.dispatch("soft_max_back", []*buffer{o, d, dx}, push(uint32(rowLen), uint32(rows)), groups); err != nil {
		b.Free(dx)
		return nil, err
	}
	return dx, nil
}

// GetRows implements compute.Backend.
func (b *Backend) GetRows(table compute.Buffer, indices []int32) (compute.Buffer, error) {
	tt, err := asBuffer(table)
	if err != nil {
		return nil, err
	}
	rowLen := tt.Ne0()
	nRows := dimOr1(tt.dims, 1)
	idxBuf, err := b.uploadInt32(indices)
	if err != nil {
		return nil, err
	}
	defer b.Free(idxBuf)
	out, err := b.newF32Buffer(rowLen, len(indices))
	if err != nil {
		return nil, err
	}
	total := uint32(rowLen * len(indices))
	groups := [3]uint32{ceilDiv(total, 64), 1, 1}
	if err := b.dispatch("get_rows", []*buffer{tt, idxBuf, out}, push(uint32(rowLen), uint32(nRows), uint32(len(indices))), groups); err != nil {
		return nil, err
	}
	return out, nil
}

// RoPE implements compute.Backend.
func (b *Backend) RoPE(a compute.Buffer, positions []int32, theta float64, nDims int) (compute.Buffer, error) {
	x, err := asBuffer(a)
	if err != nil {
		return nil, err
	}
	headDim := x.Ne0()
	nHead := dimOr1(x.dims, 1)
	nTok := dimOr1(x.dims, 2)
	if len(positions) != nTok {
		return nil, compute.ErrShape
	}
	posBuf, err := b.uploadInt32(positions)
	if err != nil {
		return nil, err
	}
	defer b.Free(posBuf)
	out, err := b.newF32Buffer(x.dims...)
	if err != nil {
		return nil, err
	}
	total := uint32(nTok * nHead * (nDims / 2))
	groups := [3]uint32{ceilDiv(total, 64), 1, 1}
	if err := b.dispatch("rope", []*buffer{x, posBuf, out}, push(uint32(headDim), uint32(nHead), uint32(nTok), uint32(nDims), float32(theta), uint32(0)), groups); err != nil {
		return nil, err
	}
	return out, nil
}

// Attention implements compute.Backend.
func (b *Backend) Attention(q, k, v compute.Buffer, nHead, nHeadKV int, scale float32, causal bool) (compute.Buffer, error) {
	tq, err := asBuffer(q)
	if err != nil {
		return nil, err
	}
	tk, err := asBuffer(k)
	if err != nil {
		return nil, err
	}
	tv, err := asBuffer(v)
	if err != nil {
		return nil, err
	}
	headDim := tq.Ne0()
	nTok := dimOr1(tq.dims, 2)
	out, err := b.newF32Buffer(tq.dims...)
	if err != nil {
		return nil, err
	}
	c := uint32(0)
	if causal {
		c = 1
	}
	// Chunk the dispatch to stay below the per-dimension group limit. The
	// attention shader has local_size_x = 1, so one group per element.
	const maxPerDispatch = 65535
	total := uint32(nHead * nTok)
	for start := uint32(0); start < total; start += maxPerDispatch {
		chunk := total - start
		if chunk > maxPerDispatch {
			chunk = maxPerDispatch
		}
		groups := [3]uint32{chunk, 1, 1}
		if err := b.dispatch("attention", []*buffer{tq, tk, tv, out}, push(uint32(headDim), uint32(nTok), uint32(nHead), uint32(nHeadKV), scale, c, start), groups); err != nil {
			b.Free(out)
			return nil, err
		}
	}
	return out, nil
}

// SSMConv implements compute.Backend with the causal conv1d shader. Only
// single-sequence inputs are supported on device; others fall back to the host.
func (b *Backend) SSMConv(sx, c compute.Buffer) (compute.Buffer, error) {
	s, err := asBuffer(sx)
	if err != nil {
		return nil, err
	}
	cc, err := asBuffer(c)
	if err != nil {
		return nil, err
	}
	if dimOr1(s.dims, 2) != 1 {
		return b.ssmConvHost(sx, c)
	}
	dConv := cc.Ne0()
	dInner := dimOr1(cc.dims, 1)
	ncs := s.Ne0()
	nT := ncs - dConv + 1
	if nT < 0 || dimOr1(s.dims, 1) != dInner {
		return nil, compute.ErrShape
	}
	out, err := b.newF32Buffer(dInner, nT, 1)
	if err != nil {
		return nil, err
	}
	total := uint32(dInner * nT)
	groups := [3]uint32{ceilDiv(total, 64), 1, 1}
	if err := b.dispatch("ssm_conv", []*buffer{s, cc, out}, push(uint32(dConv), uint32(dInner), uint32(ncs), uint32(nT)), groups); err != nil {
		b.Free(out)
		return nil, err
	}
	return out, nil
}

// ESSMConvHost is the multi-sequence host fallback.
func (b *Backend) ssmConvHost(sx, c compute.Buffer) (compute.Buffer, error) {
	ts, err := b.Download(sx)
	if err != nil {
		return nil, err
	}
	tc, err := b.Download(c)
	if err != nil {
		return nil, err
	}
	out, err := compute.SSMConv(ts, tc)
	if err != nil {
		return nil, err
	}
	return b.Upload(out)
}

// SSMConvBack implements compute.Backend with two atomic-free dispatches
// (one per gradient). Single-sequence only.
func (b *Backend) SSMConvBack(sx, c, dOut compute.Buffer) (compute.Buffer, compute.Buffer, error) {
	s, err := asBuffer(sx)
	if err != nil {
		return nil, nil, err
	}
	cc, err := asBuffer(c)
	if err != nil {
		return nil, nil, err
	}
	d, err := asBuffer(dOut)
	if err != nil {
		return nil, nil, err
	}
	if dimOr1(s.dims, 2) != 1 {
		return nil, nil, fmt.Errorf("vulkan: SSMConvBack supports a single sequence")
	}
	dConv := cc.Ne0()
	dInner := dimOr1(cc.dims, 1)
	ncs := s.Ne0()
	nT := ncs - dConv + 1
	if nT < 0 || dimOr1(s.dims, 1) != dInner {
		return nil, nil, compute.ErrShape
	}
	dSx, err := b.newF32Buffer(ncs, dInner, 1)
	if err != nil {
		return nil, nil, err
	}
	dC, err := b.newF32Buffer(dConv, dInner)
	if err != nil {
		b.Free(dSx)
		return nil, nil, err
	}
	pc := push(uint32(dConv), uint32(dInner), uint32(ncs), uint32(nT))
	gx := ceilDiv(uint32(dInner*ncs), 64)
	if err := b.dispatch("ssm_conv_back_sx", []*buffer{cc, d, dSx}, pc, [3]uint32{gx, 1, 1}); err != nil {
		b.Free(dSx)
		b.Free(dC)
		return nil, nil, err
	}
	gc := ceilDiv(uint32(dInner*dConv), 64)
	if err := b.dispatch("ssm_conv_back_c", []*buffer{s, d, dC}, pc, [3]uint32{gc, 1, 1}); err != nil {
		b.Free(dSx)
		b.Free(dC)
		return nil, nil, err
	}
	return dSx, dC, nil
}

// GatedDeltaNet implements compute.Backend with the fused recurrence shader.
// Inputs with S_v > 256 fall back to the host.
func (b *Backend) GatedDeltaNet(q, k, v, g, beta, state compute.Buffer) (compute.Buffer, compute.Buffer, error) {
	tq, err := asBuffer(q)
	if err != nil {
		return nil, nil, err
	}
	tk, err := asBuffer(k)
	if err != nil {
		return nil, nil, err
	}
	tv, err := asBuffer(v)
	if err != nil {
		return nil, nil, err
	}
	tg, err := asBuffer(g)
	if err != nil {
		return nil, nil, err
	}
	tb, err := asBuffer(beta)
	if err != nil {
		return nil, nil, err
	}
	ts, err := asBuffer(state)
	if err != nil {
		return nil, nil, err
	}
	sv := tq.Ne0()
	h := dimOr1(tq.dims, 1)
	nTok := dimOr1(tq.dims, 2)
	gstride := tg.Ne0()
	if sv > 256 || tv.Ne0() != sv || ts.Ne0() != sv {
		return b.gatedDeltaNetHost(q, k, v, g, beta, state)
	}
	kda := uint32(0)
	if gstride == sv {
		kda = 1
	} else if gstride != 1 {
		return nil, nil, compute.ErrShape
	}
	out, err := b.newF32Buffer(sv, h, nTok)
	if err != nil {
		return nil, nil, err
	}
	ns, err := b.newF32Buffer(sv, sv, h)
	if err != nil {
		b.Free(out)
		return nil, nil, err
	}
	scale := float32(1 / math.Sqrt(float64(sv)))
	groups := [3]uint32{uint32(h * sv), 1, 1}
	if err := b.dispatch("gated_delta_net", []*buffer{tq, tk, tv, tg, tb, ts, out, ns},
		push(uint32(sv), uint32(h), uint32(nTok), uint32(gstride), scale, kda), groups); err != nil {
		b.Free(out)
		b.Free(ns)
		return nil, nil, err
	}
	return out, ns, nil
}

// gatedDeltaNetHost is the fallback used when the dimensions exceed the shader.
func (b *Backend) gatedDeltaNetHost(q, k, v, g, beta, state compute.Buffer) (compute.Buffer, compute.Buffer, error) {
	dq, err := b.Download(q)
	if err != nil {
		return nil, nil, err
	}
	dk, err := b.Download(k)
	if err != nil {
		return nil, nil, err
	}
	dv, err := b.Download(v)
	if err != nil {
		return nil, nil, err
	}
	dg, err := b.Download(g)
	if err != nil {
		return nil, nil, err
	}
	db, err := b.Download(beta)
	if err != nil {
		return nil, nil, err
	}
	ds, err := b.Download(state)
	if err != nil {
		return nil, nil, err
	}
	out, ns, err := compute.GatedDeltaNet(dq, dk, dv, dg, db, ds)
	if err != nil {
		return nil, nil, err
	}
	ob, err := b.Upload(out)
	if err != nil {
		return nil, nil, err
	}
	nb, err := b.Upload(ns)
	if err != nil {
		return nil, nil, err
	}
	return ob, nb, nil
}

// Copy implements compute.Backend.
func (b *Backend) Copy(dst, src compute.Buffer) error {
	d, err := asBuffer(dst)
	if err != nil {
		return err
	}
	s, err := asBuffer(src)
	if err != nil {
		return err
	}
	if d.size < s.size {
		return compute.ErrShape
	}
	if err := b.beginBatch(); err != nil {
		return err
	}
	region := bufferCopy{srcOffset: 0, dstOffset: 0, size: s.size}
	vkCall(b.vk.CmdCopyBuffer, b.cmdBuffer, s.buf, d.buf, 1, uintptr(unsafe.Pointer(&region)))
	return nil
}

// Ne0 returns Dims[0] or 1.
func (x *buffer) Ne0() int { return dimOr1(x.dims, 0) }

// dequantShaders maps a quantized type to its dequant shader and block size.
var dequantShaders = map[quant.Type]struct {
	name       string
	blockElems int
}{
	quant.TypeF16:    {"dequant_f16", 1},
	quant.TypeQ8_0:   {"dequant_q8_0", 32},
	quant.TypeQ4_K:   {"dequant_q4_k", 256},
	quant.TypeQ5_K:   {"dequant_q5_k", 256},
	quant.TypeQ6_K:   {"dequant_q6_k", 256},
	quant.TypeQ3_K:   {"dequant_q3_k", 256},
	quant.TypeIQ4_XS: {"dequant_iq4_xs", 256},
	quant.TypeIQ4_NL: {"dequant_iq4_nl", 32},
	quant.TypeIQ3_S:  {"dequant_iq3_s", 256},
}

// UploadWeight implements compute.Backend. The buffer is padded to a 4-byte
// boundary so shaders can read 32-bit words safely.
func (b *Backend) UploadWeight(t quant.Type, raw []byte, dims []int) (compute.Buffer, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("vulkan: empty weight")
	}
	size := uint64((len(raw) + 3) &^ 3)
	buf, err := b.allocBuffer(size, vkBufferUsageStorage|vkBufferUsageTransferSrc|vkBufferUsageTransferDst)
	if err != nil {
		return nil, err
	}
	buf.dims = append([]int(nil), dims...)
	buf.typ = t
	padded := make([]byte, size)
	copy(padded, raw)
	if err := b.writeBytes(buf, padded); err != nil {
		b.destroyBuffer(buf)
		return nil, err
	}
	return buf, nil
}

// SetMaxScratchFloats overrides the row-block size used by MatMulWeight. It is
// intended for tests that need to force multiple row blocks.
func (b *Backend) SetMaxScratchFloats(n int) { b.maxScratchFloats = n }

// SetForceStaging makes buffer allocation prefer non-host-visible device-local
// memory so tests exercise the staging transfer path.
func (b *Backend) SetForceStaging(v bool) { b.forceStaging = v }

// dequantRange dequantizes count blocks starting at input block inBase into a
// float32 buffer, writing from output block outBase. outDims are the dims of the
// produced float32 tensor.
func (b *Backend) dequantRange(x *buffer, inBase, outBase, count uint32, outDims []int) (*buffer, error) {
	sh, ok := dequantShaders[x.typ]
	if !ok {
		return nil, fmt.Errorf("vulkan: no dequant shader for %s", x.typ)
	}
	out, err := b.newF32Buffer(outDims...)
	if err != nil {
		return nil, err
	}
	// Vulkan caps dispatch counts per dimension; process in chunks.
	const maxPerDispatch = 65535 * 64
	for start := uint32(0); start < count; start += maxPerDispatch {
		chunk := count - start
		if chunk > maxPerDispatch {
			chunk = maxPerDispatch
		}
		groups := [3]uint32{ceilDiv(chunk, 64), 1, 1}
		if err := b.dispatch(sh.name, []*buffer{x, out}, push(chunk, inBase+start, outBase+start, uint32(0)), groups); err != nil {
			b.Free(out)
			return nil, err
		}
	}
	return out, nil
}

// DequantWeight implements compute.Backend.
func (b *Backend) DequantWeight(w compute.Buffer) (compute.Buffer, error) {
	x, err := asBuffer(w)
	if err != nil {
		return nil, err
	}
	if x.typ == quant.TypeF32 {
		out, err := b.newF32Buffer(x.dims...)
		if err != nil {
			return nil, err
		}
		if err := b.Copy(out, x); err != nil {
			b.Free(out)
			return nil, err
		}
		return out, nil
	}
	sh, ok := dequantShaders[x.typ]
	if !ok {
		return nil, fmt.Errorf("vulkan: no dequant shader for %s", x.typ)
	}
	count := uint32(x.NumElements())
	if sh.blockElems > 1 {
		count = uint32(x.NumElements() / sh.blockElems)
	}
	return b.dequantRange(x, 0, 0, count, x.dims)
}

// gemv computes y = W * x for a single activation column using a fused
// dequantize+GEMV shader. It returns errNoGemv when the type has no shader.
func (b *Backend) gemv(w, x *buffer) (compute.Buffer, error) {
	name, ok := gemvShaders[w.typ]
	if !ok {
		return nil, errNoGemv
	}
	k := w.Ne0()
	n := dimOr1(w.dims, 1)
	if x.NumElements() < k {
		return nil, compute.ErrShape
	}
	out, err := b.newF32Buffer(n, 1)
	if err != nil {
		return nil, err
	}
	groups := [3]uint32{ceilDiv(uint32(n), 64), 1, 1}
	if err := b.dispatch(name, []*buffer{w, x, out}, push(uint32(k), uint32(n)), groups); err != nil {
		b.Free(out)
		return nil, err
	}
	return out, nil
}

// MatMulWeight implements compute.Backend. Quantized weights are processed in
// row blocks: each block is dequantized on the device into a bounded float32
// scratch buffer and multiplied, then the results are assembled on the host.
// This keeps every storage-buffer binding within device limits.
func (b *Backend) MatMulWeight(w, x compute.Buffer) (compute.Buffer, error) {
	wb, err := asBuffer(w)
	if err != nil {
		return nil, err
	}
	xb, err := asBuffer(x)
	if err != nil {
		return nil, err
	}
	if wb.typ == quant.TypeF32 {
		return b.MatMul(w, x)
	}
	sh, ok := dequantShaders[wb.typ]
	if !ok {
		return b.matMulHostDequant(wb, x)
	}
	k := wb.Ne0()
	n := dimOr1(wb.dims, 1)
	m := dimOr1(xb.dims, 1)
	if xb.Ne0() != k {
		return nil, compute.ErrShape
	}
	if sh.blockElems > 1 && k%sh.blockElems != 0 {
		return nil, fmt.Errorf("vulkan: row length %d not a multiple of block %d", k, sh.blockElems)
	}

	// Single-token decode: use a fused dequantize+GEMV shader when available,
	// which avoids materializing the weight as float32.
	if m == 1 {
		if out, gerr := b.gemv(wb, xb); gerr == nil {
			return out, nil
		} else if !errors.Is(gerr, errNoGemv) {
			return nil, gerr
		}
	}

	// Bound the float32 scratch to ~256 MiB and the block count per dequant
	// call below the dispatch limit.
	maxFloats := b.maxScratchFloats
	if maxFloats <= 0 {
		maxFloats = 64 * 1024 * 1024
	}
	const maxBlocks = 65535 * 64
	rowBlock := n
	if k > 0 && rowBlock*k > maxFloats {
		rowBlock = maxFloats / k
	}
	if sh.blockElems > 1 {
		if rb := int(maxBlocks) * sh.blockElems / k; rb < rowBlock {
			rowBlock = rb
		}
	} else {
		if rb := int(maxBlocks) / k; rb < rowBlock {
			rowBlock = rb
		}
	}
	if rowBlock < 1 {
		rowBlock = 1
	}

	outDev, err := b.newF32Buffer(n, m)
	if err != nil {
		return nil, err
	}
	for n0 := 0; n0 < n; n0 += rowBlock {
		rows := rowBlock
		if n0+rows > n {
			rows = n - n0
		}
		inBase := uint32(n0 * k / sh.blockElems)
		count := uint32(rows * k / sh.blockElems)
		scratch, err := b.dequantRange(wb, inBase, 0, count, []int{k, rows})
		if err != nil {
			return nil, err
		}
		blockOut, err := b.MatMul(scratch, x)
		b.Free(scratch)
		if err != nil {
			return nil, err
		}
		if err := b.copyRows(outDev, blockOut.(*buffer), rows, m, n, n0); err != nil {
			b.Free(blockOut)
			return nil, err
		}
		b.Free(blockOut)
	}
	return outDev, nil
}

// MatMulWeightTranspose computes the frozen-weight activation gradient
// dX = W^T dY on the device: w [K,N], dY [N,M] -> [K,M]. A quantized weight is
// dequantized in row blocks; each block runs the transposed GEMM and
// accumulates into the output, all in one batched submit.
func (b *Backend) MatMulWeightTranspose(w, dY compute.Buffer) (compute.Buffer, error) {
	wb, err := asBuffer(w)
	if err != nil {
		return nil, err
	}
	db, err := asBuffer(dY)
	if err != nil {
		return nil, err
	}
	k := wb.Ne0()
	n := dimOr1(wb.dims, 1)
	m := dimOr1(db.dims, 1)
	if db.Ne0() != n {
		return nil, compute.ErrShape
	}
	out, err := b.newF32Buffer(k, m)
	if err != nil {
		return nil, err
	}

	if wb.typ == quant.TypeF32 {
		if err := b.matmulT(wb, db, out, uint32(k), uint32(n), uint32(m), 0, uint32(n), false); err != nil {
			b.Free(out)
			return nil, err
		}
		return out, nil
	}

	sh, ok := dequantShaders[wb.typ]
	if !ok {
		return b.matmulTransposeHostDequant(wb, db, out, k, n, m)
	}
	if sh.blockElems > 1 && k%sh.blockElems != 0 {
		b.Free(out)
		return nil, fmt.Errorf("vulkan: row length %d not a multiple of block %d", k, sh.blockElems)
	}

	maxFloats := b.maxScratchFloats
	if maxFloats <= 0 {
		maxFloats = 64 * 1024 * 1024
	}
	const maxBlocks = 65535 * 64
	rowBlock := n
	if k > 0 && rowBlock*k > maxFloats {
		rowBlock = maxFloats / k
	}
	if sh.blockElems > 1 {
		if rb := int(maxBlocks) * sh.blockElems / k; rb < rowBlock {
			rowBlock = rb
		}
	} else {
		if rb := int(maxBlocks) / k; rb < rowBlock {
			rowBlock = rb
		}
	}
	if rowBlock < 1 {
		rowBlock = 1
	}

	for n0 := 0; n0 < n; n0 += rowBlock {
		rows := rowBlock
		if n0+rows > n {
			rows = n - n0
		}
		inBase := uint32(n0 * k / sh.blockElems)
		count := uint32(rows * k / sh.blockElems)
		scratch, err := b.dequantRange(wb, inBase, 0, count, []int{k, rows})
		if err != nil {
			b.Free(out)
			return nil, err
		}
		if err := b.matmulT(scratch, db, out, uint32(k), uint32(n), uint32(m), uint32(n0), uint32(rows), n0 > 0); err != nil {
			b.Free(scratch)
			b.Free(out)
			return nil, err
		}
		b.Free(scratch)
	}
	return out, nil
}

// matmulT dispatches the transposed GEMM tile shader for one weight row block.
func (b *Backend) matmulT(w, d, out *buffer, K, N, M, n0, rows uint32, acc bool) error {
	gx := ceilDiv(K, 16)
	gy := ceilDiv(M, 16)
	if gx > 65535 || gy > 65535 {
		return fmt.Errorf("vulkan: matmul_t dimensions too large (%d x %d)", K, M)
	}
	a := uint32(0)
	if acc {
		a = 1
	}
	groups := [3]uint32{gx, gy, 1}
	return b.dispatch("matmul_t", []*buffer{w, d, out}, push(K, N, M, n0, rows, a), groups)
}

// matmulTransposeHostDequant is the fallback for weight types without a dequant
// shader: dequantize on the host, upload, and run the transposed GEMM.
func (b *Backend) matmulTransposeHostDequant(wb, db, out *buffer, k, n, m int) (compute.Buffer, error) {
	raw, err := b.readBytes(wb, int(wb.size))
	if err != nil {
		b.Free(out)
		return nil, err
	}
	f32, err := quant.Dequant(wb.typ, raw, int64(wb.NumElements()))
	if err != nil {
		b.Free(out)
		return nil, err
	}
	tmp, err := b.Upload(&compute.Tensor{Dims: wb.dims, Type: quant.TypeF32, F32: f32})
	if err != nil {
		b.Free(out)
		return nil, err
	}
	defer b.Free(tmp)
	if err := b.matmulT(tmp.(*buffer), db, out, uint32(k), uint32(n), uint32(m), 0, uint32(n), false); err != nil {
		b.Free(out)
		return nil, err
	}
	return out, nil
}

func (b *Backend) matMulHostDequant(w *buffer, x compute.Buffer) (compute.Buffer, error) {
	raw, err := b.readBytes(w, int(w.size))
	if err != nil {
		return nil, err
	}
	f32, err := quant.Dequant(w.typ, raw, int64(w.NumElements()))
	if err != nil {
		return nil, err
	}
	wt := &compute.Tensor{Dims: append([]int(nil), w.dims...), Type: quant.TypeF32, F32: f32}
	wb, err := b.Upload(wt)
	if err != nil {
		return nil, err
	}
	defer b.Free(wb)
	return b.MatMul(wb, x)
}

func dimOr1(dims []int, i int) int {
	if i < 0 || i >= len(dims) {
		return 1
	}
	return dims[i]
}

func ceilDiv(a, b uint32) uint32 {
	if b == 0 {
		return 0
	}
	return (a + b - 1) / b
}
