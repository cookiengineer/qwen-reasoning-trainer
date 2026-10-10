// Command verify-ref cross-checks the Go forward pass against an independent
// NumPy implementation (ref_numpy.py) on a tiny model. It is an opt-in external
// check, run as:
//
//	go run ./e2e-tests/verify-ref          # or: make verify-ref
//
// It exits non-zero on any mismatch. The interpreter defaults to python3; set
// QWEN38_REF_PYTHON (e.g. e2e-tests/.venv/bin/python) to use the project venv.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
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

func tinyConfig() *qwen38.Config {
	return &qwen38.Config{
		NEmbd: 16, NHead: 4, NHeadKV: 2, HeadDim: 4, NRot: 4, NFf: 24, NLayer: 4, NVocab: 20,
		RopeTheta: 10000, RmsEps: 1e-6,
		DConv: 4, DState: 4, DInner: 16, DtRank: 4, GroupCount: 2,
		FullAttnInterval: 4, NextNPredict: 0,
		LayerTypes: []modelcfg.LayerType{
			modelcfg.LayerLinearAttention,
			modelcfg.LayerLinearAttention,
			modelcfg.LayerLinearAttention,
			modelcfg.LayerFullAttention,
		},
	}
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "verify-ref:", err)
		os.Exit(1)
	}
}

func run() error {
	cfg := tinyConfig()
	w := qwen38.NewRandom(cfg, 1234)
	m := qwen38.NewModel(w)
	tokens := []int32{1, 5, 2, 7, 3, 9}
	res, err := m.Forward(tokens, qwen38.ForwardOptions{RecordHidden: true})
	if err != nil {
		return err
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
	dumpW := func(name string, w *qwen38.Weight) { dumpT(name, w.Tensor()) }
	dumpVec := func(name string, v []float32) {
		doc.Tensors[name] = dumpTensor{Dims: []int{len(v)}, Data: v}
	}
	dumpW("token_embd", w.TokenEmbd)
	dumpVec("output_norm", w.OutputNorm)
	dumpW("output", w.Output)
	for il, lw := range w.Layers {
		p := "l" + strconv.Itoa(il) + "."
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

	dir, err := os.MkdirTemp("", "qwen38-ref-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	modelPath := filepath.Join(dir, "model.json")
	f, err := os.Create(modelPath)
	if err != nil {
		return err
	}
	if err := json.NewEncoder(f).Encode(doc); err != nil {
		f.Close()
		return err
	}
	f.Close()

	out, err := exec.Command(refPython(), scriptPath("ref_numpy.py"), modelPath).CombinedOutput()
	fmt.Printf("python:\n%s\n", out)
	if err != nil {
		return fmt.Errorf("numpy reference failed: %w", err)
	}
	fmt.Println("verify-ref: OK (Go forward matches ref_numpy.py)")
	return nil
}

// refPython returns the interpreter to use for the reference scripts.
func refPython() string {
	if p := os.Getenv("QWEN38_REF_PYTHON"); p != "" {
		return p
	}
	return "python3"
}

// scriptPath resolves a reference script next to this source file.
func scriptPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), name)
}
