package compute

import "github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"

// Capabilities describes a backend's device limits and supported features.
type Capabilities struct {
	Name string
	FP16 bool
	BF16 bool
	// Float32Atomics reports support for shader float32 buffer atomicAdd
	// (VK_EXT_shader_atomic_float), needed by the atomic GatedDeltaNet backward.
	Float32Atomics bool
	MaxBufferSize  uint64
	// MemoryBytes is the size of the memory the backend allocates from (the
	// host-visible coherent heap for the Vulkan backend). Zero means unknown.
	MemoryBytes uint64
	// HostFallback reports whether unsupported ops are executed on the CPU.
	HostFallback bool
}

// UnaryOp enumerates elementwise unary operations.
type UnaryOp int

const (
	UnarySilu UnaryOp = iota
	UnarySigmoid
	UnarySoftplus
	UnaryGelu
	UnaryNeg
	UnaryExp
	UnarySqr
	UnarySqrt
	UnaryTanh
)

func (u UnaryOp) String() string {
	switch u {
	case UnarySilu:
		return "silu"
	case UnarySigmoid:
		return "sigmoid"
	case UnarySoftplus:
		return "softplus"
	case UnaryGelu:
		return "gelu"
	case UnaryNeg:
		return "neg"
	case UnaryExp:
		return "exp"
	case UnarySqr:
		return "sqr"
	case UnarySqrt:
		return "sqrt"
	case UnaryTanh:
		return "tanh"
	default:
		return "unary"
	}
}

// BinaryOp enumerates elementwise binary operations.
type BinaryOp int

const (
	BinaryAdd BinaryOp = iota
	BinarySub
	BinaryMul
	BinaryDiv
	// BinarySigmoidBack computes a*(1-a)*b where a is the sigmoid forward
	// output and b the upstream gradient.
	BinarySigmoidBack
)

func (b BinaryOp) String() string {
	switch b {
	case BinaryAdd:
		return "add"
	case BinarySub:
		return "sub"
	case BinaryMul:
		return "mul"
	case BinaryDiv:
		return "div"
	default:
		return "binary"
	}
}

// AdamWParams holds the hyperparameters for one AdamW step plus the
// bias-correction factors for the current step index t:
// Beta1Hat = 1/(1-beta1^t), Beta2Hat = 1/(1-beta2^t).
type AdamWParams struct {
	Alpha       float32 // learning rate
	Beta1       float32
	Beta2       float32
	Eps         float32
	WeightDecay float32
	Beta1Hat    float32
	Beta2Hat    float32
}

// Buffer is an opaque handle to tensor storage owned by a Backend.
type Buffer interface {
	Dims() []int
	NumElements() int
	Type() quant.Type
}

// Backend abstracts a compute device. The CPU reference backend and the Vulkan
// backend implement it. Ops allocate and return new buffers; callers own the
// result and must Free it. Upload and Download synchronize as needed.
type Backend interface {
	Capabilities() Capabilities

	Alloc(dims []int, t quant.Type) (Buffer, error)
	Upload(t *Tensor) (Buffer, error)
	Download(b Buffer) (*Tensor, error)
	Free(b Buffer)
	Sync() error
	Close() error

	// BeginScope/EndScope bound the lifetime of temporary buffers. EndScope
	// frees every buffer allocated since the matching BeginScope except the
	// `keep` buffers. Backends without a device allocator implement them as
	// no-ops.
	BeginScope()
	EndScope(keep ...Buffer)

	Unary(op UnaryOp, a Buffer) (Buffer, error)
	Binary(op BinaryOp, a, b Buffer) (Buffer, error)
	Scale(a Buffer, s float32) (Buffer, error)
	MatMul(a, b Buffer) (Buffer, error)
	RMSNorm(a, w Buffer, eps float32) (Buffer, error)
	// RMSNormBack computes the RMSNorm input gradient; w may be nil.
	RMSNormBack(a, w, dOut Buffer, eps float32) (Buffer, error)
	// SiluBack computes the SiLU gradient dx = dOut * silu'(x).
	SiluBack(x, dOut Buffer) (Buffer, error)
	// L2Norm normalizes each contiguous row to unit L2 norm.
	L2Norm(a Buffer, eps float32) (Buffer, error)
	// L2NormBack computes the L2Norm input gradient.
	L2NormBack(a, dOut Buffer, eps float32) (Buffer, error)
	Softmax(a Buffer) (Buffer, error)
	// SoftmaxBack computes the softmax input gradient given the forward output.
	SoftmaxBack(out, dOut Buffer) (Buffer, error)
	// CrossEntropy computes the mean cross-entropy loss over logits [V,T]
	// against per-token targets (ignoreIndex entries skipped) and its gradient.
	// It returns the loss in a scalar [1] buffer and dLogits [V,T].
	CrossEntropy(logits Buffer, targets []int32, ignoreIndex int) (loss, dLogits Buffer, err error)
	GetRows(table Buffer, indices []int32) (Buffer, error)
	// GetRowsWeight gathers rows from a quantized weight table and dequantizes
	// them on the device into a float32 [rowLen, len(indices)] tensor.
	GetRowsWeight(w Buffer, indices []int32) (Buffer, error)
	// SplitQG splits a fused Q+gate projection qg [2*hd*nHead, T] into
	// q [hd,nHead,T] and gate [nHead*hd,T].
	SplitQG(qg Buffer, hd, nHead, T int) (q, gate Buffer, err error)
	// SplitQGBack scatters dq and dgate back into the fused layout.
	SplitQGBack(dq, dgate Buffer) (dqg Buffer, err error)
	// Reshape returns a non-owning view of b with new dims (same element count).
	Reshape(b Buffer, dims []int) (Buffer, error)

	// ConvInput builds the causal-conv input [dConv-1+T, convDim, 1] from a qkv
	// projection [convDim, T] (zero history rows). ConvInputBack is its gradient.
	ConvInput(qkv Buffer, dConv, convDim, T int) (Buffer, error)
	ConvInputBack(dConvIn Buffer, dConv, convDim, T int) (Buffer, error)
	// GatherHeads extracts head-structured channels from [convDim,T] into
	// [headDim,nHead,T]; GatherHeadsBack scatters back.
	GatherHeads(convOut Buffer, offset, headDim, nHead, T, convDim int) (Buffer, error)
	GatherHeadsBack(dOut Buffer, offset, headDim, nHead, T, convDim int) (Buffer, error)
	// RepeatHeads repeats q/k heads from nIn to nOut; RepeatHeadsBack sums back.
	RepeatHeads(x Buffer, headDim, nIn, nOut, T int) (Buffer, error)
	RepeatHeadsBack(dOut Buffer, headDim, nIn, nOut, T int) (Buffer, error)
	RoPE(a Buffer, positions []int32, theta float64, nDims int) (Buffer, error)
	Attention(q, k, v Buffer, nHead, nHeadKV int, scale float32, causal bool) (Buffer, error)
	// AttentionBackward computes gradients of Attention w.r.t. q, k, v.
	// dOut has the same shape as the attention output [hd, nHead, nQ].
	AttentionBackward(q, k, v, dOut Buffer, nHead, nHeadKV int, scale float32, causal bool) (dQ, dK, dV Buffer, err error)
	SSMConv(sx, c Buffer) (Buffer, error)
	// SSMConvBack computes gradients of SSMConv with respect to sx and c.
	SSMConvBack(sx, c, dOut Buffer) (dSx, dC Buffer, err error)
	GatedDeltaNet(q, k, v, g, beta, state Buffer) (out, newState Buffer, err error)
	// GatedDeltaNetBackward computes gradients of GatedDeltaNet w.r.t. all inputs.
	// dOut and dNewState are the upstream gradients of the output and new state.
	GatedDeltaNetBackward(q, k, v, g, beta, state, dOut, dNewState Buffer) (dQ, dK, dV, dG, dBeta, dState Buffer, err error)
	Copy(dst, src Buffer) error

	// UploadWeight stores a raw quantized weight [dims] of type t on the device.
	UploadWeight(t quant.Type, raw []byte, dims []int) (Buffer, error)
	// DequantWeight dequantizes a weight buffer to float32.
	DequantWeight(w Buffer) (Buffer, error)
	// MatMulWeight multiplies a (possibly quantized) weight by an activation.
	MatMulWeight(w Buffer, x Buffer) (Buffer, error)
	// MatMulWeightTranspose computes the frozen-weight activation gradient
	// dX = W^T dY: w [K,N], dY [N,M] -> [K,M].
	MatMulWeightTranspose(w Buffer, dY Buffer) (Buffer, error)
	// MatMulWeightGrad computes the weight gradient dW = x . dOut^T:
	// x [In,M], dOut [Out,M] -> dW [In,Out].
	MatMulWeightGrad(x, dOut Buffer) (Buffer, error)

	// AdamWStep applies one decoupled-weight-decay AdamW update to param in
	// place, updating the first/second moment buffers m and v. param, grad, m
	// and v must have the same element count; grad is read-only.
	AdamWStep(param, grad, m, v Buffer, p AdamWParams) error
	// SumSquares adds sum(src[i]^2) to dst[0]. dst is a single-element float32
	// buffer; src is read-only. Used for global gradient-norm clipping.
	SumSquares(dst, src Buffer) error
}
