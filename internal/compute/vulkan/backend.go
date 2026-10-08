package vulkan

import (
	"fmt"
	"unsafe"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

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
	return b.downloadBuffer(x), nil
}

// Free implements compute.Backend.
func (b *Backend) Free(buf compute.Buffer) {
	x, ok := buf.(*buffer)
	if !ok || x == nil {
		return
	}
	if x.mapped != nil {
		vkCall(b.vk.UnmapMemory, b.device, x.mem)
		x.mapped = nil
	}
	vkCall(b.vk.DestroyBuffer, b.device, x.buf, 0)
	vkCall(b.vk.FreeMemory, b.device, x.mem, 0)
}

// Sync implements compute.Backend.
func (b *Backend) Sync() error {
	vkCall(b.vk.QueueWaitIdle, b.queue)
	return nil
}

// Close implements compute.Backend.
func (b *Backend) Close() error {
	if b.closed {
		return nil
	}
	b.closed = true
	if b.device != 0 {
		vkCall(b.vk.DeviceWaitIdle, b.device)
		for _, p := range b.pipelines {
			// Pipelines are destroyed with the device.
			_ = p
		}
		vkCall(b.vk.DestroyDevice, b.device, 0)
	}
	if b.instance != 0 {
		vkCall(b.vk.DestroyInstance, b.instance, 0)
	}
	if b.lib != 0 {
		// Deliberately keep the library loaded for process lifetime; unloading
		// while other instances may exist is unsafe.
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
	dst := unsafe.Slice((*byte)(buf.mapped), int(buf.size))
	copy(dst, src)
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
	groups := [3]uint32{ceilDiv(uint32(n*m), 64), 1, 1}
	if err := b.dispatch("matmul", []*buffer{x, y, out}, push(uint32(k), uint32(n), uint32(m)), groups); err != nil {
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
	groups := [3]uint32{uint32(nHead * nTok), 1, 1}
	if err := b.dispatch("attention", []*buffer{tq, tk, tv, out}, push(uint32(headDim), uint32(nTok), uint32(nHead), uint32(nHeadKV), scale, c), groups); err != nil {
		return nil, err
	}
	return out, nil
}

// SSMConv implements compute.Backend using host fallback.
func (b *Backend) SSMConv(sx, c compute.Buffer) (compute.Buffer, error) {
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

// GatedDeltaNet implements compute.Backend using host fallback.
func (b *Backend) GatedDeltaNet(q, k, v, g, beta, state compute.Buffer) (compute.Buffer, compute.Buffer, error) {
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
	if res := vkCall(b.vk.ResetCommandBuffer, b.cmdBuffer, 0); res != vkSuccess {
		return fmt.Errorf("vulkan: reset command buffer failed")
	}
	begin := commandBufferBeginInfo{sType: vkStructureCommandBufferBeginInfo, flags: vkCommandBufferUsageOneTime}
	if res := vkCall(b.vk.BeginCommandBuffer, b.cmdBuffer, uintptr(unsafe.Pointer(&begin))); res != vkSuccess {
		return fmt.Errorf("vulkan: begin command buffer failed")
	}
	region := bufferCopy{srcOffset: 0, dstOffset: 0, size: s.size}
	vkCall(b.vk.CmdCopyBuffer, b.cmdBuffer, s.buf, d.buf, 1, uintptr(unsafe.Pointer(&region)))
	if res := vkCall(b.vk.EndCommandBuffer, b.cmdBuffer); res != vkSuccess {
		return fmt.Errorf("vulkan: end command buffer failed")
	}
	submit := submitInfo{sType: vkStructureSubmitInfo, commandBufferCount: 1, pCommandBuffers: uintptr(unsafe.Pointer(&b.cmdBuffer))}
	vkCall(b.vk.ResetFences, b.device, 1, uintptr(unsafe.Pointer(&b.fence)))
	if res := vkCall(b.vk.QueueSubmit, b.queue, 1, uintptr(unsafe.Pointer(&submit)), b.fence); res != vkSuccess {
		return fmt.Errorf("vulkan: queue submit failed")
	}
	if res := vkCall(b.vk.WaitForFences, b.device, 1, uintptr(unsafe.Pointer(&b.fence)), 1, ^uintptr(0)); res != vkSuccess {
		return fmt.Errorf("vulkan: wait for fence failed")
	}
	return nil
}

// Ne0 returns Dims[0] or 1.
func (x *buffer) Ne0() int { return dimOr1(x.dims, 0) }

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
