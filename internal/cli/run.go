package cli

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/hf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
)

const usage = `qwen-trainer - Qwen3.8-27B reasoning trainer

Usage:
  qwen-trainer [global flags] <command> [command flags]

Commands:
  download   Download the model GGUF from Hugging Face
  inspect    Print model architecture, tensor types, and size
  gpu-info   Print Vulkan device capabilities
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
