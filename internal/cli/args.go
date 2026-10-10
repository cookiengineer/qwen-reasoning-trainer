// Package cli implements the qwen-trainer command line interface.
package cli

import (
	"fmt"
	"path/filepath"

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

// DefaultModelFile is the default downloaded artifact.
const DefaultModelFile = hf.DefaultFile

// DefaultModelPath returns <cwd>/models/<DefaultModelFile>.
func DefaultModelPath(cwd string) string {
	return filepath.Join(cwd, "models", DefaultModelFile)
}

// ParseArgs parses global flags up to the first non-flag argument (the
// command). The remaining arguments are returned untouched for the command's
// own flag parser.
func ParseArgs(args []string, cwd string) (*Config, error) {
	cfg := &Config{
		ModelsDir: filepath.Join(cwd, "models"),
		LogLevel:  "info",
		Download:  true,
		Repo:      hf.DefaultRepo,
		File:      hf.DefaultFile,
	}
	modelExplicit := false

	i := 0
	for i < len(args) {
		a := args[i]
		if a == "" || a[0] != '-' {
			break
		}
		name, value, hasValue := splitFlag(a)
		switch name {
		case "--model", "-model", "-m":
			v, err := flagValue(args, &i, value, hasValue, name)
			if err != nil {
				return nil, err
			}
			cfg.Model = v
			modelExplicit = true
		case "--models-dir":
			v, err := flagValue(args, &i, value, hasValue, name)
			if err != nil {
				return nil, err
			}
			cfg.ModelsDir = v
		case "--log-level":
			v, err := flagValue(args, &i, value, hasValue, name)
			if err != nil {
				return nil, err
			}
			cfg.LogLevel = v
		case "--repo":
			v, err := flagValue(args, &i, value, hasValue, name)
			if err != nil {
				return nil, err
			}
			cfg.Repo = v
		case "--file":
			v, err := flagValue(args, &i, value, hasValue, name)
			if err != nil {
				return nil, err
			}
			cfg.File = v
		case "--status-file":
			v, err := flagValue(args, &i, value, hasValue, name)
			if err != nil {
				return nil, err
			}
			cfg.StatusFile = v
		case "--no-download":
			cfg.Download = false
		case "--download":
			cfg.Download = true
		case "-h", "--help", "-help":
			cfg.Command = "help"
			i++
			cfg.Args = args[i:]
			return cfg, nil
		default:
			return nil, fmt.Errorf("unknown flag %q", name)
		}
		i++
	}

	if i < len(args) {
		cfg.Command = args[i]
		cfg.Args = args[i+1:]
	}

	if cfg.Model == "" {
		cfg.Model = filepath.Join(cfg.ModelsDir, cfg.File)
	}
	_ = modelExplicit
	return cfg, nil
}

func splitFlag(a string) (name, value string, hasValue bool) {
	for j := 0; j < len(a); j++ {
		if a[j] == '=' {
			return a[:j], a[j+1:], true
		}
	}
	return a, "", false
}

func flagValue(args []string, i *int, value string, hasValue bool, name string) (string, error) {
	if hasValue {
		return value, nil
	}
	if *i+1 >= len(args) {
		return "", fmt.Errorf("flag %s requires a value", name)
	}
	*i++
	return args[*i], nil
}
