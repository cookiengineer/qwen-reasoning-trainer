package train

import (
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// Overrides merges every adapter into its frozen base weight and returns the
// abliteration-style tensor overrides keyed by GGUF tensor name. Each modified
// tensor is requantized to its original storage type (or Q4_K when no encoder
// exists).
func (m *QwenModel) Overrides() (map[string]qwen38.Override, error) {
	out := make(map[string]qwen38.Override, len(m.entries))
	for _, e := range m.entries {
		t, raw, err := e.l.MergeToQuant(e.w.Typ, e.w.Raw, e.w.Dims)
		if err != nil {
			return nil, err
		}
		out[e.name] = qwen38.Override{Type: t, Raw: raw}
	}
	return out, nil
}

// WriteMerged streams a copy of the base GGUF to outPath with all adapters
// merged and requantized.
func (m *QwenModel) WriteMerged(base *gguf.File, outPath string) error {
	ov, err := m.Overrides()
	if err != nil {
		return err
	}
	return qwen38.RewriteGGUF(base, ov, outPath)
}
