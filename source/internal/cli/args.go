// Package cli implements the qwen-trainer command line interface.
package cli

import (
	"flag"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/hf"
)

// Config is the parsed global configuration.
type Config struct {
	// Model is the resolved path to the model GGUF.
	Model string
	// ModelsDir is the directory used for downloaded models.
	ModelsDir string
	// LogLevel controls verbosity (quiet, info, debug).
	LogLevel string
	// Download enables automatic download of the default model when missing.
	Download bool
	// Repo and File identify the default model artifact.
	Repo string
	File string

	// StatusFile overrides the progress status file path (empty = default).
	StatusFile string

	// Command is the subcommand name.
	Command string
	// Args are the remaining arguments after the command.
	Args []string
}

// stringList is a flag.Value that accumulates repeated flag occurrences, so a
// flag can be supplied more than once (e.g. --dataset a --dataset b).
type stringList []string

func (s *stringList) String() string { return strings.Join(*s, ",") }

func (s *stringList) Set(v string) error {
	*s = append(*s, v)
	return nil
}

// DefaultModelFile is the default downloaded artifact.
const DefaultModelFile = hf.DefaultFile

// DefaultModelPath returns <cwd>/models/<DefaultModelFile>.
func DefaultModelPath(cwd string) string {
	return filepath.Join(cwd, "models", DefaultModelFile)
}

// ParseArgs parses the command line. The command must be the first argument
// (argv[1]); every flag, including --model, follows it and is parsed by the
// command's own flag set. This keeps the syntax uniform:
//
//	qwen-trainer <command> --model <path> [command flags]
func ParseArgs(args []string, cwd string) (*Config, error) {
	cfg := &Config{
		ModelsDir: filepath.Join(cwd, "models"),
		LogLevel:  "info",
		Download:  true,
		Repo:      hf.DefaultRepo,
		File:      hf.DefaultFile,
	}

	if len(args) == 0 {
		return cfg, nil
	}
	switch args[0] {
	case "-h", "--help", "-help":
		cfg.Command = "help"
		return cfg, nil
	}
	if strings.HasPrefix(args[0], "-") {
		return nil, fmt.Errorf("the command must be the first argument (got flag %q); usage: qwen-trainer <command> [flags]", args[0])
	}
	cfg.Command = args[0]
	cfg.Args = args[1:]
	return cfg, nil
}

// RegisterCommonFlags adds the flags shared by the commands that operate on a
// model. --model is required by those commands; call RequireModel after parsing
// to enforce it. It is registered with an empty default (not the historical
// default path) so a missing --model is an explicit error rather than a silent
// download. -m is accepted as an alias.
func (cfg *Config) RegisterCommonFlags(fs *flag.FlagSet) {
	fs.StringVar(&cfg.Model, "model", "", "model GGUF path (required)")
	fs.StringVar(&cfg.Model, "m", "", "alias for --model")
	fs.StringVar(&cfg.ModelsDir, "models-dir", cfg.ModelsDir, "directory for downloaded models")
	fs.StringVar(&cfg.Repo, "repo", cfg.Repo, "Hugging Face repo for a missing model")
	fs.StringVar(&cfg.File, "file", cfg.File, "Hugging Face file for a missing model")
	fs.StringVar(&cfg.StatusFile, "status-file", cfg.StatusFile, "progress status file")
	fs.StringVar(&cfg.LogLevel, "log-level", cfg.LogLevel, "quiet|info|debug")
	fs.BoolFunc("no-download", "do not auto-download a missing model", func(string) error {
		cfg.Download = false
		return nil
	})
	fs.BoolFunc("download", "auto-download a missing model (default)", func(string) error {
		cfg.Download = true
		return nil
	})
}

// RequireModel returns an error unless --model was supplied.
func (cfg *Config) RequireModel() error {
	if cfg.Model == "" {
		return fmt.Errorf("--model <path> is required")
	}
	return nil
}
