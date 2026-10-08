package cli

import (
	"path/filepath"
	"testing"
)

func TestParseArgsDefaults(t *testing.T) {
	cfg, err := ParseArgs([]string{"inspect"}, "/work")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Command != "inspect" {
		t.Errorf("command = %q", cfg.Command)
	}
	want := filepath.Join("/work", "models", DefaultModelFile)
	if cfg.Model != want {
		t.Errorf("model = %q, want %q", cfg.Model, want)
	}
	if !cfg.Download {
		t.Error("download should default to true")
	}
}

func TestParseArgsModelFlag(t *testing.T) {
	cfg, err := ParseArgs([]string{"--model", "/tmp/x.gguf", "inspect", "--extra"}, "/work")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Model != "/tmp/x.gguf" {
		t.Errorf("model = %q", cfg.Model)
	}
	if cfg.Command != "inspect" {
		t.Errorf("command = %q", cfg.Command)
	}
	if len(cfg.Args) != 1 || cfg.Args[0] != "--extra" {
		t.Errorf("args = %v", cfg.Args)
	}
}

func TestParseArgsEqualsForm(t *testing.T) {
	cfg, err := ParseArgs([]string{"--model=/a/b.gguf", "--log-level=debug", "download"}, "/work")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Model != "/a/b.gguf" || cfg.LogLevel != "debug" {
		t.Errorf("cfg = %+v", cfg)
	}
}

func TestParseArgsNoDownload(t *testing.T) {
	cfg, err := ParseArgs([]string{"--no-download", "inspect"}, "/work")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Download {
		t.Error("download should be false")
	}
}

func TestParseArgsUnknownFlag(t *testing.T) {
	if _, err := ParseArgs([]string{"--bogus"}, "/work"); err == nil {
		t.Fatal("expected error")
	}
}

func TestParseArgsMissingValue(t *testing.T) {
	if _, err := ParseArgs([]string{"--model"}, "/work"); err == nil {
		t.Fatal("expected error")
	}
}

func TestParseArgsHelp(t *testing.T) {
	cfg, err := ParseArgs([]string{"--help"}, "/work")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Command != "help" {
		t.Errorf("command = %q", cfg.Command)
	}
}
