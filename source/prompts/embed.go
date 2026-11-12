// Package prompts embeds the bundled abliteration prompt catalogs so the
// defaults work regardless of the working directory.
package prompts

import "embed"

//go:embed *.jsonl
var FS embed.FS
