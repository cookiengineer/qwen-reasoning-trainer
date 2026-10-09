package train

import (
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// mergeItem is one adapter together with the frozen base weight it updates.
type mergeItem struct {
	name string
	l    *LoRA
	w    *qwen38.Weight
}

// mergeSource implements qwen38.OverrideSource. It merges each adapter into its
// base weight lazily, so a full-model rewrite only holds one merged tensor in
// memory at a time (rather than all of them).
type mergeSource struct {
	items map[string]mergeItem
}

func newMergeSource(items []mergeItem) *mergeSource {
	m := &mergeSource{items: make(map[string]mergeItem, len(items))}
	for _, it := range items {
		m.items[it.name] = it
	}
	return m
}

// OverrideMeta reports the merged tensor's type and byte size without
// materializing it. The type policy matches LoRA.MergeToQuant.
func (s *mergeSource) OverrideMeta(name string, dims []uint64, baseType quant.Type) (quant.Type, int64, bool) {
	if _, ok := s.items[name]; !ok {
		return 0, 0, false
	}
	typ := baseType
	if !quant.CanQuantize(typ) {
		typ = quant.TypeQ4_K
	}
	n := int64(1)
	for _, d := range dims {
		n *= int64(d)
	}
	size, err := typ.Size(n)
	if err != nil {
		return 0, 0, false
	}
	return typ, size, true
}

// OverrideBytes materializes the merged, requantized tensor.
func (s *mergeSource) OverrideBytes(name string) ([]byte, error) {
	it := s.items[name]
	_, raw, err := it.l.MergeToQuant(it.w.Typ, it.w.Raw, it.w.Dims)
	return raw, err
}

// Overrides merges every adapter into its frozen base weight and returns the
// abliteration-style tensor overrides keyed by GGUF tensor name. Each modified
// tensor is requantized to its original storage type (or Q4_K when no encoder
// exists). Prefer WriteMerged for a full-model rewrite: it streams the merged
// tensors one at a time instead of holding them all at once.
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
// merged and requantized, materializing one tensor at a time.
func (m *QwenModel) WriteMerged(base *gguf.File, outPath string) error {
	items := make([]mergeItem, 0, len(m.entries))
	for _, e := range m.entries {
		items = append(items, mergeItem{name: e.name, l: e.l, w: e.w})
	}
	return qwen38.RewriteGGUFStream(base, newMergeSource(items), outPath)
}
