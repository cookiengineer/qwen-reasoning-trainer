package train

import "github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"

// deviceLinear computes base*x + scale*B*(A*x) on the device. base is a frozen
// (possibly quantized) weight buffer; a and b are the uploaded F32 LoRA factors.
// It returns y and the intermediate u = A*x (needed by the backward pass).
func deviceLinear(be compute.Backend, base, a, b, x compute.Buffer, scale float32) (y, u compute.Buffer, err error) {
	baseOut, err := be.MatMulWeight(base, x)
	if err != nil {
		return nil, nil, err
	}
	u, err = be.MatMul(a, x)
	if err != nil {
		return nil, nil, err
	}
	v, err := be.MatMul(b, u)
	if err != nil {
		return nil, nil, err
	}
	vs, err := be.Scale(v, scale)
	if err != nil {
		return nil, nil, err
	}
	y, err = be.Binary(compute.BinaryAdd, baseOut, vs)
	if err != nil {
		return nil, nil, err
	}
	return y, u, nil
}

// deviceLinearBackward computes dX and the adapter gradients for a linear layer
// whose forward was deviceLinear.
func deviceLinearBackward(be compute.Backend, base, a, b, x, u, dY compute.Buffer, scale float32) (dX, dA, dB compute.Buffer, err error) {
	dBase, err := be.MatMulWeightTranspose(base, dY)
	if err != nil {
		return nil, nil, nil, err
	}
	dVs, err := be.Scale(dY, scale)
	if err != nil {
		return nil, nil, nil, err
	}
	dB, err = be.MatMulWeightGrad(u, dVs)
	if err != nil {
		return nil, nil, nil, err
	}
	dU, err := be.MatMulWeightTranspose(b, dVs)
	if err != nil {
		return nil, nil, nil, err
	}
	dA, err = be.MatMulWeightGrad(x, dU)
	if err != nil {
		return nil, nil, nil, err
	}
	dXlora, err := be.MatMulWeightTranspose(a, dU)
	if err != nil {
		return nil, nil, nil, err
	}
	dX, err = be.Binary(compute.BinaryAdd, dBase, dXlora)
	if err != nil {
		return nil, nil, nil, err
	}
	return dX, dA, dB, nil
}
