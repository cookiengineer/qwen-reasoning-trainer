package qwen38

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

type dumpTensor struct {
	Dims []int     `json:"dims"`
	Data []float32 `json:"data"`
}

type dumpDoc struct {
	Config   map[string]any        `json:"config"`
	Tokens   []int32               `json:"tokens"`
	Tensors  map[string]dumpTensor `json:"tensors"`
	GoLogits []float32             `json:"go_logits"`
	GoHidden [][]float32           `json:"go_hidden"`
}

// TestReferenceNumpy cross-checks the Go forward pass against an independent
// NumPy implementation. It only runs when QWEN38_VERIFY_REF=1 so the default
// test suite stays pure Go.
func TestReferenceNumpy(t *testing.T) {
	if os.Getenv("QWEN38_VERIFY_REF") == "" {
		t.Skip("set QWEN38_VERIFY_REF=1 to run the NumPy reference check")
	}
	cfg := tinyConfig()
	w := NewRandom(cfg, 1234)
	m := NewModel(w)
	tokens := []int32{1, 5, 2, 7, 3, 9}
	res, err := m.Forward(tokens, ForwardOptions{RecordHidden: true})
	if err != nil {
		t.Fatal(err)
	}

	doc := dumpDoc{Tokens: tokens, Tensors: map[string]dumpTensor{}, GoLogits: res.Logits.F32}
	for _, h := range res.Hidden {
		doc.GoHidden = append(doc.GoHidden, h.F32)
	}
	doc.Config = map[string]any{
		"NEmbd": cfg.NEmbd, "NHead": cfg.NHead, "NHeadKV": cfg.NHeadKV,
		"HeadDim": cfg.HeadDim, "NRot": cfg.NRot, "NFf": cfg.NFf, "NLayer": cfg.NLayer,
		"NVocab": cfg.NVocab, "RmsEps": cfg.RmsEps, "RopeTheta": cfg.RopeTheta,
		"DConv": cfg.DConv, "DState": cfg.DState, "DInner": cfg.DInner,
		"DtRank": cfg.DtRank, "GroupCount": cfg.GroupCount,
		"FullAttnInterval": cfg.FullAttnInterval, "TrunkLayers": cfg.TrunkLayers(),
		"KeyDim": cfg.KeyDim(), "ValueDim": cfg.ValueDim(), "ConvDim": cfg.ConvDim(),
		"HeadVDim": cfg.HeadVDim(),
	}
	rec := make([]bool, cfg.TrunkLayers())
	for i := range rec {
		rec[i] = cfg.IsRecurrent(i)
	}
	doc.Config["IsRecurrent"] = rec

	dumpT := func(name string, t *compute.Tensor) { doc.Tensors[name] = dumpTensor{Dims: t.Dims, Data: t.F32} }
	dumpW := func(name string, w *Weight) { dumpT(name, w.Tensor()) }
	dumpVec := func(name string, v []float32) {
		doc.Tensors[name] = dumpTensor{Dims: []int{len(v)}, Data: v}
	}
	dumpW("token_embd", w.TokenEmbd)
	dumpVec("output_norm", w.OutputNorm)
	dumpW("output", w.Output)
	for il, lw := range w.Layers {
		p := "l" + itoa(il) + "."
		dumpVec(p+"attn_norm", lw.AttnNorm)
		dumpVec(p+"post_attn_norm", lw.PostAttnNorm)
		if cfg.IsRecurrent(il) {
			dumpW(p+"attn_qkv", lw.AttnQKV)
			dumpW(p+"attn_gate", lw.AttnGate)
			dumpW(p+"ssm_out", lw.SSMOut)
			dumpW(p+"ssm_alpha", lw.SSMAlpha)
			dumpW(p+"ssm_beta", lw.SSMBeta)
			dumpT(p+"ssm_conv1d", lw.SSMConv1d)
			dumpVec(p+"ssm_norm", lw.SSMNorm)
			dumpVec(p+"ssm_a", lw.SSMA)
			dumpVec(p+"ssm_dt", lw.SSMDt)
		} else {
			dumpW(p+"attn_q", lw.AttnQ)
			dumpW(p+"attn_k", lw.AttnK)
			dumpW(p+"attn_v", lw.AttnV)
			dumpW(p+"attn_output", lw.AttnOutput)
			dumpVec(p+"attn_q_norm", lw.AttnQNorm)
			dumpVec(p+"attn_k_norm", lw.AttnKNorm)
		}
		dumpW(p+"ffn_gate", lw.FfnGate)
		dumpW(p+"ffn_up", lw.FfnUp)
		dumpW(p+"ffn_down", lw.FfnDown)
	}

	dir := t.TempDir()
	modelPath := filepath.Join(dir, "model.json")
	f, err := os.Create(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.NewEncoder(f).Encode(doc); err != nil {
		f.Close()
		t.Fatal(err)
	}
	f.Close()

	_, thisFile, _, _ := runtime.Caller(0)
	script := filepath.Join(filepath.Dir(thisFile), "ref_numpy.py")
	cmd := exec.Command("python3", script, modelPath)
	out, err := cmd.CombinedOutput()
	t.Logf("python: %s", out)
	if err != nil {
		t.Fatalf("numpy reference failed: %v", err)
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [8]byte
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	return string(b[p:])
}
