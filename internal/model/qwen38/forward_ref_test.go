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
	dumpT("token_embd", w.TokenEmbd)
	dumpT("output_norm", w.OutputNorm)
	dumpT("output", w.Output)
	for il, lw := range w.Layers {
		p := "l" + itoa(il) + "."
		dumpT(p+"attn_norm", lw.AttnNorm)
		dumpT(p+"post_attn_norm", lw.PostAttnNorm)
		if cfg.IsRecurrent(il) {
			dumpT(p+"attn_qkv", lw.AttnQKV)
			dumpT(p+"attn_gate", lw.AttnGate)
			dumpT(p+"ssm_out", lw.SSMOut)
			dumpT(p+"ssm_alpha", lw.SSMAlpha)
			dumpT(p+"ssm_beta", lw.SSMBeta)
			dumpT(p+"ssm_conv1d", lw.SSMConv1d)
			dumpT(p+"ssm_norm", lw.SSMNorm)
			dumpT(p+"ssm_a", lw.SSMA)
			dumpT(p+"ssm_dt", lw.SSMDt)
		} else {
			dumpT(p+"attn_q", lw.AttnQ)
			dumpT(p+"attn_k", lw.AttnK)
			dumpT(p+"attn_v", lw.AttnV)
			dumpT(p+"attn_output", lw.AttnOutput)
			dumpT(p+"attn_q_norm", lw.AttnQNorm)
			dumpT(p+"attn_k_norm", lw.AttnKNorm)
		}
		dumpT(p+"ffn_gate", lw.FfnGate)
		dumpT(p+"ffn_up", lw.FfnUp)
		dumpT(p+"ffn_down", lw.FfnDown)
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
