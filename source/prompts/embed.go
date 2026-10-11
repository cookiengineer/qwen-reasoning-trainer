// Package prompts embeds the bundled Heretic prompt sets so the abliteration
// defaults work regardless of the working directory.
package prompts

import "embed"

//go:embed *.jsonl
var FS embed.FS
