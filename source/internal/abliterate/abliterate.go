// Package abliterate implements directional ablation of refusal directions in
// Qwen3.8 weights, following the technique used by Heretic
// (references/heretic/src/heretic/modifiers/abliteration.py):
//
//  1. Collect per-layer residual-stream means over "good" and "bad" prompts.
//  2. Compute the difference-of-means direction per layer (optionally
//     orthogonalized against the good direction).
//  3. Orthogonalize the attention out-projection and MLP down-projection
//     matrices against that direction, with an optional magnitude-preserving
//     (MPOA) row renormalization.
package abliterate

import (
	"fmt"
	"math"
	"sort"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// ComponentParams describes the ablation strength across layers for one
// component, matching Heretic's weight distribution.
type ComponentParams struct {
	MaxWeight         float64
	MaxWeightPosition float64
	MinWeight         float64
	MinWeightDistance float64
}

// Params controls an ablation run.
type Params struct {
	// DirectionIndex selects a global direction (a float layer index); nil
	// means use each layer's own direction.
	DirectionIndex *float64
	// Attn applies to the attention out-projection (or linear-attention
	// out-projection).
	Attn ComponentParams
	// MLP applies to the MLP down-projection.
	MLP ComponentParams
	// RowNormalization is "none", "pre", or "full" (MPOA).
	RowNormalization string
}

// DefaultParams returns Heretic-like defaults (per-layer direction, MPOA).
func DefaultParams() Params {
	return Params{
		RowNormalization: "full",
		Attn:             ComponentParams{MaxWeight: 1.0, MaxWeightPosition: 0, MinWeight: 0, MinWeightDistance: 1},
		MLP:              ComponentParams{MaxWeight: 0, MaxWeightPosition: 0, MinWeight: 0, MinWeightDistance: 1},
	}
}

// CollectResiduals runs the model on each prompt and returns the mean
// residual-stream activation at the last token for every captured hidden state
// (len = layers+1: index 0 is the embedding output). winsorize in [0,1)
// clamps each per-prompt residual to that quantile of its absolute values.
func CollectResiduals(m *qwen38.Model, prompts [][]int32, winsorize float64) ([][]float64, error) {
	nHidden := m.Cfg.TrunkLayers() + 1
	hidden := m.Cfg.NEmbd
	sums := make([][]float64, nHidden)
	for i := range sums {
		sums[i] = make([]float64, hidden)
	}
	if len(prompts) == 0 {
		return sums, nil
	}
	for _, toks := range prompts {
		if len(toks) == 0 {
			continue
		}
		res, err := m.Forward(toks, qwen38.ForwardOptions{RecordHidden: true})
		if err != nil {
			return nil, err
		}
		T := len(toks)
		for k := 0; k < nHidden; k++ {
			h := res.Hidden[k]
			base := (T - 1) * hidden
			row := make([]float64, hidden)
			for j := 0; j < hidden; j++ {
				row[j] = float64(h.F32[base+j])
			}
			if winsorize > 0 && winsorize < 1 {
				winsorizeRow(row, winsorize)
			}
			for j := 0; j < hidden; j++ {
				sums[k][j] += row[j]
			}
		}
	}
	n := float64(len(prompts))
	for i := range sums {
		for j := range sums[i] {
			sums[i][j] /= n
		}
	}
	return sums, nil
}

// winsorizeRow clamps each element to the q-quantile of the absolute values.
func winsorizeRow(row []float64, q float64) {
	abs := make([]float64, len(row))
	for i, v := range row {
		abs[i] = math.Abs(v)
	}
	sort.Float64s(abs)
	idx := int(q * float64(len(abs)-1))
	th := abs[idx]
	for i := range row {
		if row[i] > th {
			row[i] = th
		} else if row[i] < -th {
			row[i] = -th
		}
	}
}

// Directions computes per-layer normalized difference-of-means directions. When
// orthogonalize is true, only the component orthogonal to the good direction is
// kept.
func Directions(good, bad [][]float64, orthogonalize bool) [][]float64 {
	n := len(good)
	out := make([][]float64, n)
	for k := 0; k < n; k++ {
		d := make([]float64, len(good[k]))
		for j := range d {
			d[j] = bad[k][j] - good[k][j]
		}
		normalize(d)
		if orthogonalize {
			g := append([]float64(nil), good[k]...)
			normalize(g)
			var proj float64
			for j := range d {
				proj += d[j] * g[j]
			}
			for j := range d {
				d[j] -= proj * g[j]
			}
			normalize(d)
		}
		out[k] = d
	}
	return out
}

func normalize(v []float64) {
	var ss float64
	for _, x := range v {
		ss += x * x
	}
	n := math.Sqrt(ss)
	if n == 0 {
		return
	}
	for i := range v {
		v[i] /= n
	}
}

// weightAt returns the ablation weight for a layer given a component's kernel.
func (c ComponentParams) weightAt(il int) float64 {
	d := math.Abs(float64(il) - c.MaxWeightPosition)
	if d > c.MinWeightDistance {
		return 0
	}
	if c.MinWeightDistance == 0 {
		return c.MaxWeight
	}
	return c.MaxWeight + (d/c.MinWeightDistance)*(c.MinWeight-c.MaxWeight)
}

// pickDirection selects the direction for a layer, interpolating for a global
// direction index.
func pickDirection(dirs [][]float64, il int, p Params) []float64 {
	if p.DirectionIndex == nil {
		idx := il + 1
		if idx >= len(dirs) {
			idx = len(dirs) - 1
		}
		return dirs[idx]
	}
	di := *p.DirectionIndex + 1
	i0 := int(math.Floor(di))
	frac := di - float64(i0)
	if i0 < 0 {
		i0 = 0
	}
	if i0 >= len(dirs) {
		i0 = len(dirs) - 1
	}
	if i0+1 >= len(dirs) || frac == 0 {
		return dirs[i0]
	}
	v := make([]float64, len(dirs[i0]))
	for j := range v {
		v[j] = dirs[i0][j] + frac*(dirs[i0+1][j]-dirs[i0][j])
	}
	normalize(v)
	return v
}

// Ablate modifies the attention out-projection and MLP down-projection weights
// of w in place, requantizing them to Q4_K. It returns the number of matrices
// changed.
func Ablate(w *qwen38.Weights, dirs [][]float64, p Params) (int, error) {
	if p.RowNormalization == "" {
		p.RowNormalization = "full"
	}
	trunk := w.Cfg.TrunkLayers()
	changed := 0
	for il := 0; il < trunk; il++ {
		lw := &w.Layers[il]
		v := pickDirection(dirs, il, p)

		type target struct {
			wt  **qwen38.Weight
			cp  ComponentParams
			set bool
		}
		targets := []target{
			{&lw.AttnOutput, p.Attn, !w.Cfg.IsRecurrent(il)},
			{&lw.SSMOut, p.Attn, w.Cfg.IsRecurrent(il)},
			{&lw.FfnDown, p.MLP, true},
		}
		for _, tg := range targets {
			if !tg.set || *tg.wt == nil {
				continue
			}
			weight := tg.cp.weightAt(il)
			if weight == 0 {
				continue
			}
			if err := ablateWeight(*tg.wt, v, weight, p.RowNormalization); err != nil {
				return changed, fmt.Errorf("layer %d: %w", il, err)
			}
			changed++
		}
	}
	return changed, nil
}

// ablateWeight dequantizes w, applies the directional ablation in place, and
// requantizes it to its original type when an encoder exists (otherwise Q4_K).
func ablateWeight(w *qwen38.Weight, v []float64, weight float64, mode string) error {
	origType := w.Typ
	t, err := w.Dequant()
	if err != nil {
		return err
	}
	if err := transformMatrix(t, v, weight, mode); err != nil {
		return err
	}
	encType := origType
	if !quant.CanQuantize(encType) {
		encType = quant.TypeQ4_K
	}
	if bs := encType.BlockSize(); bs > 1 && t.Ne(0)%bs != 0 {
		encType = quant.TypeQ4_K
	}
	raw, err := quant.Quantize(encType, t.F32)
	if err != nil {
		return err
	}
	w.Typ = encType
	w.Raw = raw
	return nil
}

// transformMatrix applies the ablation to a weight stored as [in, out] (GGML
// layout) with output dimension Dims[1] matching len(v).
func transformMatrix(t *compute.Tensor, v []float64, weight float64, mode string) error {
	in := t.Ne(0)
	out := t.Ne(1)
	if len(v) != out {
		return fmt.Errorf("abliterate: direction size %d != weight out %d", len(v), out)
	}
	// M[i][j] = weight for output i, input j; element (j, i) at j + i*in.
	get := func(i, j int) float64 { return float64(t.F32[j+i*in]) }
	set := func(i, j int, x float64) { t.F32[j+i*in] = float32(x) }

	switch mode {
	case "none":
		wv := make([]float64, in)
		for j := 0; j < in; j++ {
			var s float64
			for i := 0; i < out; i++ {
				s += v[i] * get(i, j)
			}
			wv[j] = s
		}
		for i := 0; i < out; i++ {
			vi := weight * v[i]
			for j := 0; j < in; j++ {
				set(i, j, get(i, j)-vi*wv[j])
			}
		}
	case "pre":
		rowNorms := make([]float64, out)
		for i := 0; i < out; i++ {
			var ss float64
			for j := 0; j < in; j++ {
				x := get(i, j)
				ss += x * x
			}
			rowNorms[i] = math.Sqrt(ss)
		}
		wv := make([]float64, in)
		for j := 0; j < in; j++ {
			var s float64
			for i := 0; i < out; i++ {
				s += v[i] * get(i, j) / rowNorms[i]
			}
			wv[j] = s
		}
		for i := 0; i < out; i++ {
			bi := -weight * v[i] * rowNorms[i]
			for j := 0; j < in; j++ {
				set(i, j, get(i, j)+bi*wv[j])
			}
		}
	case "full":
		rowNorms := make([]float64, out)
		for i := 0; i < out; i++ {
			var ss float64
			for j := 0; j < in; j++ {
				x := get(i, j)
				ss += x * x
			}
			rowNorms[i] = math.Sqrt(ss)
		}
		wv := make([]float64, in)
		for j := 0; j < in; j++ {
			var s float64
			for i := 0; i < out; i++ {
				rn := rowNorms[i]
				if rn == 0 {
					continue
				}
				s += v[i] * get(i, j) / rn
			}
			wv[j] = s
		}
		for i := 0; i < out; i++ {
			rn := rowNorms[i]
			newNorms := 0.0
			newRow := make([]float64, in)
			for j := 0; j < in; j++ {
				wn := 0.0
				if rn != 0 {
					wn = get(i, j) / rn
				}
				a := wn - weight*v[i]*wv[j]
				newRow[j] = a
				newNorms += a * a
			}
			newNorms = math.Sqrt(newNorms)
			scale := 1.0
			if newNorms != 0 {
				scale = rn / newNorms
			}
			for j := 0; j < in; j++ {
				set(i, j, newRow[j]*scale)
			}
		}
	default:
		return fmt.Errorf("abliterate: unknown row normalization %q", mode)
	}
	return nil
}
