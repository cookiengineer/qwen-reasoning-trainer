package cli

import (
	"flag"
	"io"
	"testing"
)

func TestParseArgsCommandFirst(t *testing.T) {
	cfg, err := ParseArgs([]string{"inspect", "--model", "/tmp/x.gguf"}, "/work")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Command != "inspect" {
		t.Errorf("command = %q", cfg.Command)
	}
	if len(cfg.Args) != 2 || cfg.Args[0] != "--model" || cfg.Args[1] != "/tmp/x.gguf" {
		t.Errorf("args = %v", cfg.Args)
	}
	if cfg.Model != "" {
		t.Errorf("model should be unset until the command parses its flags, got %q", cfg.Model)
	}
	if !cfg.Download {
		t.Error("download should default to true")
	}
}

func TestParseArgsNoArgs(t *testing.T) {
	cfg, err := ParseArgs(nil, "/work")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Command != "" {
		t.Errorf("command = %q, want empty", cfg.Command)
	}
}

func TestParseArgsLeadingFlagRejected(t *testing.T) {
	if _, err := ParseArgs([]string{"--model", "/tmp/x.gguf", "inspect"}, "/work"); err == nil {
		t.Fatal("expected error for a flag before the command")
	}
}

func TestParseArgsUnknownCommandIsNotFlag(t *testing.T) {
	cfg, err := ParseArgs([]string{"bogus", "--x"}, "/work")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Command != "bogus" {
		t.Errorf("command = %q", cfg.Command)
	}
}

func TestParseArgsHelp(t *testing.T) {
	for _, arg := range []string{"--help", "-h", "-help"} {
		cfg, err := ParseArgs([]string{arg}, "/work")
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Command != "help" {
			t.Errorf("%s: command = %q, want help", arg, cfg.Command)
		}
	}
}

func TestCommonFlagsParse(t *testing.T) {
	cfg, err := ParseArgs([]string{"train", "--model", "/tmp/x.gguf", "--log-level=debug", "--no-download"}, "/work")
	if err != nil {
		t.Fatal(err)
	}
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	cfg.RegisterCommonFlags(fs)
	if err := fs.Parse(cfg.Args); err != nil {
		t.Fatal(err)
	}
	if err := cfg.RequireModel(); err != nil {
		t.Fatalf("model should be set: %v", err)
	}
	if cfg.Model != "/tmp/x.gguf" {
		t.Errorf("model = %q", cfg.Model)
	}
	if cfg.LogLevel != "debug" {
		t.Errorf("log level = %q", cfg.LogLevel)
	}
	if cfg.Download {
		t.Error("download should be false after --no-download")
	}
}

func TestCommonFlagsModelAlias(t *testing.T) {
	cfg := &Config{Download: true}
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	cfg.RegisterCommonFlags(fs)
	if err := fs.Parse([]string{"-m", "/tmp/y.gguf"}); err != nil {
		t.Fatal(err)
	}
	if cfg.Model != "/tmp/y.gguf" {
		t.Errorf("model = %q, want /tmp/y.gguf", cfg.Model)
	}
}

func TestStringListFlag(t *testing.T) {
	fs := flag.NewFlagSet("train", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	var datasets stringList
	fs.Var(&datasets, "dataset", "extractor output directory (repeatable)")
	if err := fs.Parse([]string{"--dataset", "a", "--dataset", "b", "--dataset=c"}); err != nil {
		t.Fatal(err)
	}
	if len(datasets) != 3 || datasets[0] != "a" || datasets[1] != "b" || datasets[2] != "c" {
		t.Fatalf("datasets = %v, want [a b c]", datasets)
	}
	if got := datasets.String(); got != "a,b,c" {
		t.Fatalf("String() = %q, want a,b,c", got)
	}
}

func TestRequireModel(t *testing.T) {
	cfg := &Config{}
	if err := cfg.RequireModel(); err == nil {
		t.Fatal("expected --model required error")
	}
	cfg.Model = "/tmp/x.gguf"
	if err := cfg.RequireModel(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
