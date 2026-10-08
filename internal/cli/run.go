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
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/evaluate"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/hf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
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
	useGPU := fs.Bool("gpu", false, "run matmuls on the Vulkan backend")
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
	if *useGPU {
		be, err := vulkan.New()
		if err != nil {
			fmt.Fprintln(stderr, "error: vulkan backend:", err)
			return 1
		}
		defer func() { m.FreeDevice(); be.Close() }()
		m.Backend = be
		fmt.Fprintf(stderr, "using backend %s\n", be.Capabilities().Name)
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
	useGPU := fs.Bool("gpu", false, "run matmuls on the Vulkan backend")
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
	if *useGPU {
		be, err := vulkan.New()
		if err != nil {
			fmt.Fprintln(stderr, "error: vulkan backend:", err)
			return 1
		}
		defer func() { m.FreeDevice(); be.Close() }()
		m.Backend = be
		fmt.Fprintf(stderr, "using backend %s\n", be.Capabilities().Name)
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
	useGPU := fs.Bool("gpu", false, "run matmuls on the Vulkan backend")
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

	newModel := func(path string) (*qwen38.Model, *tokenizer.Vocab, func(), error) {
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
		var be *vulkan.Backend
		if *useGPU {
			be, err = vulkan.New()
			if err != nil {
				g.Close()
				return nil, nil, nil, err
			}
			m.Backend = be
		}
		cleanup := func() {
			if be != nil {
				m.FreeDevice()
				be.Close()
			}
			g.Close()
		}
		return m, vocab, cleanup, nil
	}

	path, err := resolveModel(cfg, stderr)
	if err != nil {
		fmt.Fprintln(stderr, "error:", err)
		return 1
	}
	m, vocab, cleanup, err := newModel(path)
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
		base, _, baseCleanup, err := newModel(*baseFile)
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
