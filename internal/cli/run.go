package cli

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/abliterate"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/dataset"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/evaluate"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/hf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/train"
)

const usage = `qwen-trainer - Qwen3.8-27B reasoning trainer

Usage:
  qwen-trainer [global flags] <command> [command flags]

Commands:
  download   Download the model GGUF from Hugging Face
  inspect    Print model architecture, tensor types, and size
  gpu-info   Print Vulkan device capabilities
  run        Load the model and greedily generate tokens
  abliterate Remove refusal directions and write an abliterated GGUF
  evaluate   Score a model's refusal rate and KL divergence from a base
  dataset    Build tokenized, loss-masked examples from extractor JSONL
  train      QLoRA fine-tune on the extractor dataset (reference host path)
  help       Show this help

Global flags:
  --model <path>       Model GGUF path (default <cwd>/models/` + DefaultModelFile + `)
  --models-dir <dir>   Directory for downloaded models (default <cwd>/models)
  --repo <repo>        Hugging Face repo (default ` + hf.DefaultRepo + `)
  --file <file>        Hugging Face file (default ` + DefaultModelFile + `)
  --no-download        Do not auto-download a missing model
  --log-level <level>  quiet|info|debug (default info)
`

// Run executes the CLI and returns a process exit code.
func Run(args []string, stdout, stderr io.Writer) int {
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	cfg, err := ParseArgs(args, cwd)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 2
	}

	switch cfg.Command {
	case "", "help":
		fmt.Fprint(stdout, usage)
		return 0
	case "download":
		return cmdDownload(cfg, stdout, stderr)
	case "inspect":
		return cmdInspect(cfg, stdout, stderr)
	case "gpu-info":
		return cmdGPUInfo(cfg, stdout, stderr)
	case "run":
		return cmdRun(cfg, stdout, stderr)
	case "abliterate":
		return cmdAbliterate(cfg, stdout, stderr)
	case "evaluate":
		return cmdEvaluate(cfg, stdout, stderr)
	case "dataset":
		return cmdDataset(cfg, stdout, stderr)
	case "train":
		return cmdTrain(cfg, stdout, stderr)
	default:
		fmt.Fprintf(stderr, "error: unknown command %q\n\n", cfg.Command)
		fmt.Fprint(stderr, usage)
		return 2
	}
}

func newHFClient(cfg *Config, stderr io.Writer) *hf.Client {
	c := &hf.Client{}
	if cfg.LogLevel != "quiet" {
		c.Progress = func(done, total int64) {
			if total > 0 {
				fmt.Fprintf(stderr, "\r  %s / %s (%.1f%%)", humanBytes(done), humanBytes(total), float64(done)/float64(total)*100)
			} else {
				fmt.Fprintf(stderr, "\r  %s", humanBytes(done))
			}
		}
	}
	return c
}

func cmdDownload(cfg *Config, stdout, stderr io.Writer) int {
	c := newHFClient(cfg, stderr)
	ctx := context.Background()

	size, sha, err := c.FileInfo(ctx, cfg.Repo, cfg.File)
	if err != nil {
		// Fall back to the pinned checksum for the default artifact.
		if cfg.File == hf.DefaultFile {
			size, sha = hf.DefaultSize, hf.DefaultSHA256
		} else {
			fmt.Fprintln(stderr, "error:", err)
			return 1
		}
	}
	if size > 0 {
		fmt.Fprintf(stderr, "Downloading %s/%s (%s) -> %s\n", cfg.Repo, cfg.File, humanBytes(size), cfg.Model)
	}
	if err := c.Download(ctx, cfg.Repo, cfg.File, cfg.Model, sha); err != nil {
		fmt.Fprintln(stderr, "\nerror:", err)
		return 1
	}
	fmt.Fprintf(stderr, "\nDone: %s\n", cfg.Model)
	fmt.Fprintln(stdout, cfg.Model)
	return 0
}

func cmdAbliterate(cfg *Config, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("abliterate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	goodFile := fs.String("good", "", "file with one desirable prompt per line")
	badFile := fs.String("bad", "", "file with one undesirable prompt per line")
	out := fs.String("out", "", "output GGUF path")
	system := fs.String("system", "You are a helpful assistant.", "system prompt")
	rowNorm := fs.String("row-norm", "full", "row normalization: none|pre|full")
	orthogonalize := fs.Bool("orthogonalize", true, "orthogonalize against the good direction")
	directionIndex := fs.Float64("direction-index", -1, "global direction layer index (-1 = per layer)")
	attnMax := fs.Float64("attn-max-weight", 1.0, "attention ablation max weight")
	attnMin := fs.Float64("attn-min-weight", 0.0, "attention ablation min weight")
	mlpMax := fs.Float64("mlp-max-weight", 0.0, "MLP ablation max weight")
	maxPosFrac := fs.Float64("max-weight-position-frac", 0.75, "max weight position / layer count")
	minDistFrac := fs.Float64("min-weight-distance-frac", 0.5, "min weight distance / layer count")
	limit := fs.Int("limit", 0, "max prompts per set (0 = all)")
	winsorize := fs.Float64("winsorize", 0, "winsorization quantile in [0,1) (0 = off)")
	if err := fs.Parse(cfg.Args); err != nil {
		return 2
	}
	if *goodFile == "" || *badFile == "" || *out == "" {
		fmt.Fprintln(stderr, "error: --good, --bad and --out are required")
		return 2
	}
	goodLines := readLines(*goodFile, *limit)
	badLines := readLines(*badFile, *limit)
	if len(goodLines) == 0 || len(badLines) == 0 {
		fmt.Fprintln(stderr, "error: need at least one prompt in --good and --bad")
		return 2
	}

	path, err := resolveModel(cfg, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	g, err := gguf.Open(path)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	defer g.Close()
	mc, err := modelcfg.FromGGUF(g)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	vocab, err := tokenizer.FromGGUF(g)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stderr, "loading weights from %s...\n", path)
	w, err := qwen38.LoadFromGGUF(g, mc)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	m := qwen38.NewModel(w)
	if be := resolveGPU(modelBytes(g)+gpuHeadroomBytes, cfg, stderr); be != nil {
		defer func() { m.FreeDevice(); be.Close() }()
		m.Backend = be
	}

	good := encodePrompts(vocab, *system, goodLines)
	bad := encodePrompts(vocab, *system, badLines)
	fmt.Fprintf(stderr, "collecting residuals (%d good, %d bad prompts)...\n", len(good), len(bad))
	gm, err := abliterate.CollectResiduals(m, good, *winsorize)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	bm, err := abliterate.CollectResiduals(m, bad, *winsorize)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	dirs := abliterate.Directions(gm, bm, *orthogonalize)

	L := m.Cfg.TrunkLayers()
	p := abliterate.Params{
		RowNormalization: *rowNorm,
		Attn: abliterate.ComponentParams{
			MaxWeight: *attnMax, MaxWeightPosition: *maxPosFrac * float64(L),
			MinWeight: *attnMin, MinWeightDistance: math.Max(1, *minDistFrac*float64(L)),
		},
		MLP: abliterate.ComponentParams{
			MaxWeight: *mlpMax, MaxWeightPosition: *maxPosFrac * float64(L),
			MinWeightDistance: math.Max(1, *minDistFrac*float64(L)),
		},
	}
	if *directionIndex >= 0 {
		di := *directionIndex
		p.DirectionIndex = &di
	}
	changed, err := abliterate.Ablate(w, dirs, p)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stderr, "abliterated %d weight matrices; writing %s...\n", changed, *out)
	if err := qwen38.RewriteGGUF(g, w.Overrides(), *out); err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote %s (%d matrices ablated)\n", *out, changed)
	return 0
}

func readLines(path string, limit int) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		out = append(out, line)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

func encodePrompts(vocab *tokenizer.Vocab, system string, lines []string) [][]int32 {
	out := make([][]int32, 0, len(lines))
	for _, line := range lines {
		out = append(out, vocab.ChatPrompt(system, line))
	}
	return out
}

func cmdRun(cfg *Config, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tokenList := fs.String("tokens", "", "comma-separated prompt token ids")
	prompt := fs.String("prompt", "", "text prompt (encoded with the model tokenizer)")
	system := fs.String("system", "", "optional system message for --prompt")
	gen := fs.Int("generate", 8, "number of tokens to generate")
	if err := fs.Parse(cfg.Args); err != nil {
		return 2
	}

	path, err := resolveModel(cfg, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	g, err := gguf.Open(path)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	defer g.Close()
	mc, err := modelcfg.FromGGUF(g)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	vocab, err := tokenizer.FromGGUF(g)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}

	var tokens []int32
	if *prompt != "" {
		tokens = vocab.ChatPrompt(*system, *prompt)
	} else {
		if tokens, err = parseTokens(*tokenList); err != nil {
			fmt.Fprintln(stderr, "error:", err)
			return 2
		}
	}
	if len(tokens) == 0 {
		fmt.Fprintln(stderr, "error: provide --tokens or --prompt")
		return 2
	}
	fmt.Fprintf(stderr, "loading weights from %s (block_count=%d)...\n", path, mc.BlockCount)
	t0 := time.Now()
	w, err := qwen38.LoadFromGGUF(g, mc)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stderr, "weights loaded in %s\n", time.Since(t0).Round(time.Millisecond))

	m := qwen38.NewModel(w)
	if be := resolveGPU(modelBytes(g)+gpuHeadroomBytes, cfg, stderr); be != nil {
		defer func() { m.FreeDevice(); be.Close() }()
		m.Backend = be
	}

	st := qwen38.NewState(m.Cfg)
	t1 := time.Now()
	res, err := m.Forward(tokens, qwen38.ForwardOptions{State: st, LastTokenOnly: true})
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stderr, "prompt %d tokens in %s\n", len(tokens), time.Since(t1).Round(time.Millisecond))

	generated := make([]int32, 0, *gen)
	pos := len(tokens)
	for i := 0; i < *gen; i++ {
		next := argmax(res.Logits)
		if vocab.IsEOS(next) {
			break
		}
		generated = append(generated, next)
		step := time.Now()
		res, err = m.Forward([]int32{next}, qwen38.ForwardOptions{
			State:         st,
			Positions:     []int32{int32(pos)},
			LastTokenOnly: true,
		})
		if err != nil {
			fmt.Fprintln(stderr, "error:", err)
			return 1
		}
		fmt.Fprintf(stderr, "token %d: id=%d in %s\n", i, next, time.Since(step).Round(time.Millisecond))
		pos++
	}

	fmt.Fprintf(stdout, "prompt_tokens: %v\n", tokens)
	fmt.Fprintf(stdout, "generated_ids: %v\n", generated)
	fmt.Fprintf(stdout, "decoded: %q\n", vocab.Decode(generated))
	return 0
}

func parseTokens(s string) ([]int32, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	out := make([]int32, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid token %q", p)
		}
		out = append(out, int32(v))
	}
	return out, nil
}

func argmax(logits *compute.Tensor) int32 {
	best := 0
	n := logits.Ne(0)
	for i := 1; i < n; i++ {
		if logits.F32[i] > logits.F32[best] {
			best = i
		}
	}
	return int32(best)
}

func cmdEvaluate(cfg *Config, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("evaluate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	promptsFile := fs.String("prompts", "", "file with one prompt per line")
	baseFile := fs.String("base", "", "optional base GGUF to compute KL divergence against")
	system := fs.String("system", "You are a helpful assistant.", "system prompt")
	maxTokens := fs.Int("max-tokens", 32, "tokens to generate per prompt for refusal detection")
	kw := fs.String("keywords", "", "comma-separated refusal keywords (default built-in)")
	if err := fs.Parse(cfg.Args); err != nil {
		return 2
	}
	if *promptsFile == "" {
		fmt.Fprintln(stderr, "error: --prompts is required")
		return 2
	}
	lines := readLines(*promptsFile, 0)
	if len(lines) == 0 {
		fmt.Fprintln(stderr, "error: no prompts loaded")
		return 2
	}

	newModel := func(path string, be compute.Backend) (*qwen38.Model, *tokenizer.Vocab, func(), error) {
		g, err := gguf.Open(path)
		if err != nil {
			return nil, nil, nil, err
		}
		mc, err := modelcfg.FromGGUF(g)
		if err != nil {
			g.Close()
			return nil, nil, nil, err
		}
		vocab, err := tokenizer.FromGGUF(g)
		if err != nil {
			g.Close()
			return nil, nil, nil, err
		}
		w, err := qwen38.LoadFromGGUF(g, mc)
		if err != nil {
			g.Close()
			return nil, nil, nil, err
		}
		m := qwen38.NewModel(w)
		m.Backend = be
		cleanup := func() {
			m.FreeDevice()
			g.Close()
		}
		return m, vocab, cleanup, nil
	}

	path, err := resolveModel(cfg, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	need := gpuHeadroomBytes + fileModelBytes(path)
	if *baseFile != "" {
		need += fileModelBytes(*baseFile)
	}
	be := resolveGPU(need, cfg, stderr)
	defer func() {
		if be != nil {
			be.Close()
		}
	}()
	m, vocab, cleanup, err := newModel(path, be)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	defer cleanup()
	keywords := evaluate.DefaultKeywords
	if *kw != "" {
		keywords = strings.Split(*kw, ",")
	}
	prompts := encodePrompts(vocab, *system, lines)
	rate, err := evaluate.RefusalRate(m, vocab, prompts, *maxTokens, keywords)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stdout, "refusal_rate: %.4f (%d prompts, up to %d tokens)\n", rate, len(prompts), *maxTokens)

	if *baseFile != "" {
		base, _, baseCleanup, err := newModel(*baseFile, be)
		if err != nil {
			fmt.Fprintln(stderr, "error:", err)
			return 1
		}
		defer baseCleanup()
		kl, err := evaluate.KLDivergence(base, m, prompts)
		if err != nil {
			fmt.Fprintln(stderr, "error:", err)
			return 1
		}
		fmt.Fprintf(stdout, "kl_divergence: %.5f\n", kl)
	}
	return 0
}

// cmdDataset tokenizes extractor JSONL with the model chat template and reports
// assistant-only loss-mask statistics.
func cmdDataset(cfg *Config, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("dataset", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "extractor output directory")
	seqLen := fs.Int("seq-len", 0, "truncate sequences to N tokens (0 = no limit)")
	minTarget := fs.Int("min-target", 1, "drop examples with fewer trainable tokens")
	valRatio := fs.Float64("val-ratio", 0.02, "validation split ratio")
	subagents := fs.Bool("include-subagents", false, "include the subagents/ tree")
	effort := fs.String("reasoning-effort", "xhigh", "xhigh|medium|low")
	noThink := fs.Bool("no-thinking", false, "render with enable_thinking=false")
	if err := fs.Parse(cfg.Args); err != nil {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "error: --input is required")
		return 2
	}
	if *valRatio < 0 || *valRatio >= 1 {
		fmt.Fprintln(stderr, "error: --val-ratio must be in [0, 1)")
		return 2
	}

	path, err := resolveModel(cfg, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	g, err := gguf.Open(path)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	defer g.Close()
	vocab, err := tokenizer.FromGGUF(g)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}

	d, err := dataset.Open(*input, dataset.WithSubagents(*subagents))
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	recs, err := d.Sessions()
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}

	opt := dataset.DefaultBuildOptions()
	opt.MaxSeqLen = *seqLen
	opt.MinTargetTokens = *minTarget
	opt.Render.ReasoningEffort = *effort
	opt.Render.EnableThinking = !*noThink
	opt.Render.Tools = d.Manifest.Tools
	exs, err := dataset.BuildExamples(vocab, recs, opt)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	train, val := dataset.Split(exs, *valRatio)

	fmt.Fprintf(stdout, "input:          %s\n", d.Root())
	fmt.Fprintf(stdout, "manifest:       version=%s strict=%v\n", d.Manifest.Version, d.Manifest.Strict)
	fmt.Fprintf(stdout, "tokenizer:      pre=%s tools=%d\n", vocab.Pre, len(d.Manifest.Tools))
	fmt.Fprintf(stdout, "sessions:       %d\n", len(recs))
	fmt.Fprintf(stdout, "examples:       %d (dropped %d)\n", len(exs), len(recs)-len(exs))
	fmt.Fprintf(stdout, "tokens:         %d total, avg %.1f\n", dataset.TotalTokens(exs), avg(dataset.TotalTokens(exs), len(exs)))
	fmt.Fprintf(stdout, "target tokens:  %d total, avg %.1f\n", dataset.TotalLossTokens(exs), avg(dataset.TotalLossTokens(exs), len(exs)))
	fmt.Fprintf(stdout, "mask ratio:     %.4f\n", ratio(dataset.TotalLossTokens(exs), dataset.TotalTokens(exs)))
	fmt.Fprintf(stdout, "split:          train=%d val=%d\n", len(train), len(val))
	return 0
}

// cmdTrain runs QLoRA fine-tuning on the extractor dataset. This is the
// reference host path: the quantized base is frozen and only F32 LoRA adapters
// are trained. It is correct-first and slow on the real 27B; the device-resident
// Vulkan graph is future work.
func cmdTrain(cfg *Config, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "extractor output directory")
	steps := fs.Int("steps", 50, "optimizer steps")
	seqLen := fs.Int("seq-len", 512, "max sequence length (front truncation)")
	lr := fs.Float64("lr", 1e-4, "peak learning rate")
	rank := fs.Int("rank", 16, "LoRA rank")
	alpha := fs.Float64("alpha", 32, "LoRA alpha")
	accum := fs.Int("accum", 1, "gradient accumulation steps")
	warmup := fs.Int("warmup", 10, "warmup steps")
	maxGrad := fs.Float64("max-grad-norm", 1.0, "gradient clipping norm (0 = off)")
	seed := fs.Int64("seed", 1, "seed")
	out := fs.String("out", "", "output merged GGUF path")
	adapterOut := fs.String("adapter-out", "", "LoRA adapter checkpoint path")
	valRatio := fs.Float64("val-ratio", 0.0, "validation split ratio (0 = train on all)")
	subagents := fs.Bool("include-subagents", false, "include the subagents/ tree")
	if err := fs.Parse(cfg.Args); err != nil {
		return 2
	}
	if *input == "" {
		fmt.Fprintln(stderr, "error: --input is required")
		return 2
	}
	if *rank <= 0 || *alpha <= 0 {
		fmt.Fprintln(stderr, "error: --rank and --alpha must be positive")
		return 2
	}

	path, err := resolveModel(cfg, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	g, err := gguf.Open(path)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	defer g.Close()
	mc, err := modelcfg.FromGGUF(g)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	vocab, err := tokenizer.FromGGUF(g)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stderr, "loading weights from %s...\n", path)
	w, err := qwen38.LoadFromGGUF(g, mc)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}

	d, err := dataset.Open(*input, dataset.WithSubagents(*subagents))
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	recs, err := d.Sessions()
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	bo := dataset.DefaultBuildOptions()
	bo.MaxSeqLen = *seqLen
	bo.Render.Tools = d.Manifest.Tools
	exs, err := dataset.BuildExamples(vocab, recs, bo)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	if len(exs) == 0 {
		fmt.Fprintln(stderr, "error: no training examples built")
		return 1
	}
	toTrain := func(src []dataset.Example) []train.Example {
		out := make([]train.Example, len(src))
		for i, e := range src {
			mask := make([]bool, len(e.Mask))
			for j, m := range e.Mask {
				mask[j] = m != 0
			}
			out[i] = train.Example{Tokens: e.IDs, LossMask: mask}
		}
		return out
	}
	trainSet, valSet := dataset.Split(exs, *valRatio)
	data := toTrain(trainSet)

	loCfg := train.DefaultLoRA()
	loCfg.Rank = *rank
	loCfg.Alpha = float32(*alpha)
	tm := train.NewQwenModel(w.Cfg, w, loCfg, uint64(*seed))
	fmt.Fprintf(stderr, "training: %d examples (%d val), %d adapters, %d trainable tensors\n",
		len(data), len(valSet), len(tm.Adapters()), len(tm.Params()))

	if len(valSet) > 0 {
		fmt.Fprintf(stderr, "val loss before: %.4f\n", meanLoss(tm, toTrain(valSet)))
	}

	params := tm.Params()
	o := train.NewAdamW(train.DefaultAdamW(float32(*lr)), params)
	sched := train.CosineSchedule(float32(*lr), float32(*lr)*0.1, *warmup, *steps)
	tr := &train.Trainer{Cfg: train.TrainerConfig{Steps: *steps, Accum: *accum, MaxGradNorm: float32(*maxGrad), LogEvery: 1, Seed: uint64(*seed)}}
	hist, err := tr.Run(tm, o, sched, data, func(step int, loss float32) {
		if cfg.LogLevel != "quiet" {
			fmt.Fprintf(stderr, "step %d/%d: loss %.4f lr %.2e\n", step+1, *steps, loss, sched(step))
		}
	})
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stdout, "steps: %d\n", len(hist))
	fmt.Fprintf(stdout, "initial_loss: %.4f\n", hist[0])
	fmt.Fprintf(stdout, "final_loss: %.4f\n", hist[len(hist)-1])
	if len(valSet) > 0 {
		fmt.Fprintf(stdout, "val_loss: %.4f\n", meanLoss(tm, toTrain(valSet)))
	}
	if *adapterOut != "" {
		if err := train.SaveCheckpoint(*adapterOut, params); err != nil {
			fmt.Fprintln(stderr, "error:", err)
			return 1
		}
		fmt.Fprintf(stdout, "wrote adapters: %s\n", *adapterOut)
	}
	if *out != "" {
		if err := tm.WriteMerged(g, *out); err != nil {
			fmt.Fprintln(stderr, "error:", err)
			return 1
		}
		fmt.Fprintf(stdout, "wrote merged model: %s\n", *out)
	}
	return 0
}

func meanLoss(tm *train.QwenModel, data []train.Example) float32 {
	if len(data) == 0 {
		return 0
	}
	var sum float32
	for _, ex := range data {
		l, _, err := tm.ForwardBackward(ex)
		if err != nil {
			continue
		}
		sum += l
	}
	return sum / float32(len(data))
}

func avg(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}

func ratio(n, d int) float64 {
	if d == 0 {
		return 0
	}
	return float64(n) / float64(d)
}

func cmdGPUInfo(cfg *Config, stdout, stderr io.Writer) int {
	b, err := vulkan.New()
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	defer b.Close()
	caps := b.Capabilities()
	fmt.Fprintf(stdout, "backend:        %s\n", caps.Name)
	fmt.Fprintf(stdout, "fp16:           %v\n", caps.FP16)
	fmt.Fprintf(stdout, "bf16:           %v\n", caps.BF16)
	fmt.Fprintf(stdout, "host fallback:  %v\n", caps.HostFallback)
	fmt.Fprintf(stdout, "max buffer:     %d\n", caps.MaxBufferSize)
	fmt.Fprintf(stdout, "memory:         %s (%d bytes)\n", humanBytes(int64(caps.MemoryBytes)), caps.MemoryBytes)
	return 0
}

func cmdInspect(cfg *Config, stdout, stderr io.Writer) int {
	path, err := resolveModel(cfg, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	g, err := gguf.Open(path)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	defer g.Close()

	rep, err := modelcfg.NewReport(g)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	fmt.Fprintf(stdout, "File: %s\n\n", path)
	fmt.Fprint(stdout, rep.String())
	if len(rep.Unsupported) > 0 {
		return 3
	}
	return 0
}

// resolveModel returns the model path, downloading the default artifact when
// missing and downloads are enabled.
func resolveModel(cfg *Config, stderr io.Writer) (string, error) {
	if st, err := os.Stat(cfg.Model); err == nil && !st.IsDir() {
		return cfg.Model, nil
	}
	if !cfg.Download {
		return "", fmt.Errorf("model %s not found and downloads are disabled", cfg.Model)
	}
	size, sha := int64(0), ""
	if cfg.File == hf.DefaultFile {
		size, sha = hf.DefaultSize, hf.DefaultSHA256
	}
	c := newHFClient(cfg, stderr)
	ctx := context.Background()
	if size == 0 || sha == "" {
		s, h, err := c.FileInfo(ctx, cfg.Repo, cfg.File)
		if err != nil {
			return "", fmt.Errorf("model %s not found and file info lookup failed: %w", cfg.Model, err)
		}
		size, sha = s, h
	}
	fmt.Fprintf(stderr, "Model not found, downloading %s/%s (%s) -> %s\n", cfg.Repo, cfg.File, humanBytes(size), cfg.Model)
	if err := c.Download(ctx, cfg.Repo, cfg.File, cfg.Model, sha); err != nil {
		return "", err
	}
	fmt.Fprintln(stderr)
	return cfg.Model, nil
}

func humanBytes(b int64) string {
	f := float64(b)
	const unit = 1024.0
	switch {
	case f >= unit*unit*unit:
		return fmt.Sprintf("%.2f GiB", f/(unit*unit*unit))
	case f >= unit*unit:
		return fmt.Sprintf("%.2f MiB", f/(unit*unit))
	case f >= unit:
		return fmt.Sprintf("%.2f KiB", f/unit)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

// gpuHeadroomBytes is reserved on top of the model size for activations, scratch
// and the KV/recurrent state when deciding whether the GPU has enough memory.
const gpuHeadroomBytes = 2 << 30

// modelBytes returns the total on-disk size of a GGUF's tensor data.
func modelBytes(g *gguf.File) uint64 {
	var n uint64
	for _, t := range g.Tensors {
		if sz, err := t.ByteSize(); err == nil {
			n += uint64(sz)
		}
	}
	return n
}

func fileModelBytes(path string) uint64 {
	g, err := gguf.Open(path)
	if err != nil {
		return 0
	}
	defer g.Close()
	return modelBytes(g)
}

// resolveGPU returns a Vulkan backend when a device is present and can hold at
// least needBytes of weights, otherwise nil (CPU fallback). The caller owns a
// non-nil backend and must Close it.
func resolveGPU(needBytes uint64, cfg *Config, stderr io.Writer) compute.Backend {
	be, err := vulkan.New()
	if err != nil {
		if cfg.LogLevel != "quiet" {
			fmt.Fprintf(stderr, "gpu: unavailable (%v); using CPU\n", err)
		}
		return nil
	}
	caps := be.Capabilities()
	if caps.MemoryBytes > 0 && caps.MemoryBytes < needBytes {
		if cfg.LogLevel != "quiet" {
			fmt.Fprintf(stderr, "gpu: %s has %s, need ~%s; using CPU\n",
				caps.Name, humanBytes(int64(caps.MemoryBytes)), humanBytes(int64(needBytes)))
		}
		be.Close()
		return nil
	}
	if cfg.LogLevel != "quiet" {
		fmt.Fprintf(stderr, "gpu: using %s (%s)\n", caps.Name, humanBytes(int64(caps.MemoryBytes)))
	}
	return be
}
