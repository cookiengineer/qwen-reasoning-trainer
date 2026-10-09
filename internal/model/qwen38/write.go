package qwen38

import (
	"fmt"
	"io"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// Override is a replacement tensor for RewriteGGUF.
type Override struct {
	Type quant.Type
	Raw  []byte
}

// TargetTensor names an abliterable weight in a *Weights.
type TargetTensor struct {
	Name string
	W    *Weight
}

// TargetWeights returns the abliterable tensors (attention / linear-attention
// out-projection and MLP down-projection) in deterministic layer order. It is
// the pointer-keyed counterpart to Overrides, used to snapshot and restore the
// affected weights across abliteration search trials.
func (w *Weights) TargetWeights() []TargetTensor {
	var out []TargetTensor
	for il := range w.Layers {
		lw := &w.Layers[il]
		if w.Cfg.IsRecurrent(il) {
			if lw.SSMOut != nil {
				out = append(out, TargetTensor{fmt.Sprintf("blk.%d.ssm_out.weight", il), lw.SSMOut})
			}
		} else if lw.AttnOutput != nil {
			out = append(out, TargetTensor{fmt.Sprintf("blk.%d.attn_output.weight", il), lw.AttnOutput})
		}
		if lw.FfnDown != nil {
			out = append(out, TargetTensor{fmt.Sprintf("blk.%d.ffn_down.weight", il), lw.FfnDown})
		}
	}
	return out
}

// Overrides returns the abliterable tensors (attention out-projection / linear
// out-projection and MLP down-projection) keyed by their GGUF tensor names.
func (w *Weights) Overrides() map[string]Override {
	out := map[string]Override{}
	for il := range w.Layers {
		lw := &w.Layers[il]
		if w.Cfg.IsRecurrent(il) {
			if lw.SSMOut != nil {
				out[fmt.Sprintf("blk.%d.ssm_out.weight", il)] = Override{lw.SSMOut.Typ, lw.SSMOut.Raw}
			}
		} else if lw.AttnOutput != nil {
			out[fmt.Sprintf("blk.%d.attn_output.weight", il)] = Override{lw.AttnOutput.Typ, lw.AttnOutput.Raw}
		}
		if lw.FfnDown != nil {
			out[fmt.Sprintf("blk.%d.ffn_down.weight", il)] = Override{lw.FfnDown.Typ, lw.FfnDown.Raw}
		}
	}
	return out
}

// RewriteGGUF writes a copy of base to outPath, replacing the tensors listed in
// overrides and streaming every other tensor unchanged.
func RewriteGGUF(base *gguf.File, overrides map[string]Override, outPath string) error {
	tensors := make([]gguf.StreamTensor, 0, len(base.Tensors))
	for _, ti := range base.Tensors {
		ti := ti
		if ov, ok := overrides[ti.Name]; ok {
			raw := ov.Raw
			tensors = append(tensors, gguf.StreamTensor{
				Name: ti.Name, Dims: ti.Dims, Type: ov.Type, Size: int64(len(raw)),
				Write: func(w io.Writer) error { _, err := w.Write(raw); return err },
			})
			continue
		}
		size, err := ti.ByteSize()
		if err != nil {
			return err
		}
		tensors = append(tensors, gguf.StreamTensor{
			Name: ti.Name, Dims: ti.Dims, Type: ti.Type, Size: size,
			Write: func(w io.Writer) error { return base.CopyTensor(ti, w) },
		})
	}
	return gguf.WriteStreamToFile(outPath, base.KVs, tensors)
}
