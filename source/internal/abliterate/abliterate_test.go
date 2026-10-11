package abliterate

import (
	"bytes"
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func TestDirections(t *testing.T) {
	good := [][]float64{{1, 0}, {0, 1}}
	bad := [][]float64{{0, 1}, {1, 0}}
	d := Directions(good, bad, false)
	if math.Abs(d[0][0]+1/math.Sqrt2) > 1e-9 || math.Abs(d[0][1]-1/math.Sqrt2) > 1e-9 {
		t.Fatalf("layer0 direction = %v", d[0])
	}

	// Orthogonalized direction for layer 0 must be [0, 1].
	do := Directions(good, bad, true)
	if math.Abs(do[0][0]) > 1e-9 || math.Abs(do[0][1]-1) > 1e-9 {
		t.Fatalf("orthogonalized layer0 = %v", do[0])
	}
	var dot float64
	g := []float64{1, 0}
	for j := range do[0] {
		dot += do[0][j] * g[j]
	}
	if math.Abs(dot) > 1e-9 {
		t.Fatalf("not orthogonal: dot=%v", dot)
	}
}

func TestTransformNoneOrthogonalizes(t *testing.T) {
	in, out := 4, 3
	src := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	tens := &compute.Tensor{Dims: []int{in, out}, F32: append([]float32(nil), src...)}
	v := []float64{1, 2, 3}
	normalize(v)
	if err := transformMatrix(tens, v, 1.0, "none"); err != nil {
		t.Fatal(err)
	}
	for j := 0; j < in; j++ {
		var s float64
		for i := 0; i < out; i++ {
			s += v[i] * float64(tens.F32[j+i*in])
		}
		if math.Abs(s) > 1e-5 {
			t.Fatalf("column %d not orthogonal: %v", j, s)
		}
	}
}

func TestTransformFullPreservesRowNorms(t *testing.T) {
	in, out := 4, 3
	src := []float32{1, -2, 3, -4, 5, -6, 7, -8, -9, 10, 11, -12}
	tens := &compute.Tensor{Dims: []int{in, out}, F32: append([]float32(nil), src...)}
	origNorm := func(i int) float64 {
		var ss float64
		for j := 0; j < in; j++ {
			x := float64(src[j+i*in])
			ss += x * x
		}
		return math.Sqrt(ss)
	}
	v := []float64{0.5, -0.5, 0.7071}
	normalize(v)
	if err := transformMatrix(tens, v, 1.0, "full"); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < out; i++ {
		var ss float64
		for j := 0; j < in; j++ {
			x := float64(tens.F32[j+i*in])
			ss += x * x
		}
		if math.Abs(math.Sqrt(ss)-origNorm(i)) > 1e-5 {
			t.Fatalf("row %d norm changed: %v vs %v", i, math.Sqrt(ss), origNorm(i))
		}
	}
}

func TestWeightAt(t *testing.T) {
	c := ComponentParams{MaxWeight: 1.0, MaxWeightPosition: 10, MinWeight: 0.2, MinWeightDistance: 5}
	if got := c.weightAt(10); got != 1.0 {
		t.Fatalf("peak = %v", got)
	}
	if got := c.weightAt(15); math.Abs(got-0.2) > 1e-9 {
		t.Fatalf("edge = %v", got)
	}
	if got := c.weightAt(16); got != 0 {
		t.Fatalf("outside = %v", got)
	}
}

func miniConfigForAblate() *qwen38.Config {
	return &qwen38.Config{
		NEmbd: 256, NHead: 4, NHeadKV: 1, HeadDim: 64, NRot: 64, NFf: 512,
		NLayer: 4, NVocab: 128, RopeTheta: 10000, RmsEps: 1e-6,
		DConv: 4, DState: 128, DInner: 256, DtRank: 4, GroupCount: 2,
		FullAttnInterval: 4,
		LayerTypes: []modelcfg.LayerType{
			modelcfg.LayerLinearAttention, modelcfg.LayerLinearAttention,
			modelcfg.LayerLinearAttention, modelcfg.LayerFullAttention,
		},
	}
}

func TestAblateWeights(t *testing.T) {
	cfg := miniConfigForAblate()
	w := qwen38.NewRandom(cfg, 5)
	requant := func(wt *qwen38.Weight, tp quant.Type) {
		d, err := wt.Dequant()
		if err != nil {
			t.Fatal(err)
		}
		raw, err := quant.Quantize(tp, d.F32)
		if err != nil {
			t.Fatal(err)
		}
		wt.Typ = tp
		wt.Raw = raw
	}
	requant(w.Layers[3].AttnOutput, quant.TypeQ6_K)
	requant(w.Layers[0].SSMOut, quant.TypeQ5_K)
	requant(w.Layers[1].SSMOut, quant.TypeIQ4_XS)
	dirs := make([][]float64, cfg.TrunkLayers()+1)
	for k := range dirs {
		dirs[k] = make([]float64, cfg.NEmbd)
		for j := range dirs[k] {
			dirs[k][j] = math.Sin(float64(k*7 + j))
		}
		normalize(dirs[k])
	}
	p := Params{
		RowNormalization: "none",
		Attn:             ComponentParams{MaxWeight: 1, MaxWeightPosition: 3, MinWeight: 0, MinWeightDistance: 3},
		MLP:              ComponentParams{MaxWeight: 0, MaxWeightPosition: 3, MinWeightDistance: 3},
	}
	ffnRaw := append([]byte(nil), w.Layers[3].FfnDown.Raw...)
	n, err := Ablate(w, dirs, p)
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatal("expected some matrices ablated")
	}
	// Modified tensors keep their original quantization type.
	if w.Layers[3].AttnOutput.Typ != quant.TypeQ6_K {
		t.Fatalf("attn_output type = %s, want q6_K", w.Layers[3].AttnOutput.Typ)
	}
	if w.Layers[0].SSMOut.Typ != quant.TypeQ5_K {
		t.Fatalf("ssm_out type = %s, want q5_K", w.Layers[0].SSMOut.Typ)
	}
	if w.Layers[1].SSMOut.Typ != quant.TypeIQ4_XS {
		t.Fatalf("ssm_out type = %s, want iq4_xs", w.Layers[1].SSMOut.Typ)
	}
	// Layer 3 is full attention: attn_output must be orthogonalized to dirs[4].
	lw := &w.Layers[3]
	m, err := lw.AttnOutput.Dequant()
	if err != nil {
		t.Fatal(err)
	}
	in, out := m.Ne(0), m.Ne(1)
	v := dirs[4]
	for j := 0; j < in; j++ {
		var s float64
		for i := 0; i < out; i++ {
			s += v[i] * float64(m.F32[j+i*in])
		}
		if math.Abs(s) > 5e-2 {
			t.Fatalf("column %d not orthogonal: %v", j, s)
		}
	}
	// MLP max weight 0 -> ffn_down untouched (byte identical).
	if !bytes.Equal(ffnRaw, w.Layers[3].FfnDown.Raw) {
		t.Fatal("ffn_down should not have been ablated")
	}
}
