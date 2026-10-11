// Command qwen-trainer is the CLI for the Qwen3.8-27B reasoning trainer.
package main

import (
	"os"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
