package compute

import "github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"

// Capabilities describes a backend's device limits and supported features.
type Capabilities struct {
	Name          string
	FP16          bool
	BF16          bool
	MaxBufferSize uint64
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
	GetRows(table Buffer, indices []int32) (Buffer, error)
	RoPE(a Buffer, positions []int32, theta float64, nDims int) (Buffer, error)
	Attention(q, k, v Buffer, nHead, nHeadKV int, scale float32, causal bool) (Buffer, error)
	// AttentionBackward computes gradients of Attention w.r.t. q, k, v.
	// dOut has the same shape as the attention output [hd, nHead, nQ].
	AttentionBackward(q, k, v, dOut Buffer, nHead, nHeadKV int, scale float32, causal bool) (dQ, dK, dV Buffer, err error)
	SSMConv(sx, c Buffer) (Buffer, error)
	// SSMConvBack computes gradients of SSMConv with respect to sx and c.
	SSMConvBack(sx, c, dOut Buffer) (dSx, dC Buffer, err error)
	GatedDeltaNet(q, k, v, g, beta, state Buffer) (out, newState Buffer, err error)
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
}
