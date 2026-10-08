package dataset

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

// Dataset is an opened extractor output root.
type Dataset struct {
	root     string
	Manifest Manifest

	includeSubagents bool
}

// Option configures Open.
type Option func(*options)

type options struct {
	includeSubagents bool
}

// WithSubagents includes the subagents/ tree alongside sft/.
func WithSubagents(include bool) Option {
	return func(o *options) { o.includeSubagents = include }
}

// Open reads manifest.json from root and validates the output layout.
func Open(root string, opts ...Option) (*Dataset, error) {
	var o options
	for _, opt := range opts {
		opt(&o)
	}
	raw, err := os.ReadFile(filepath.Join(root, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("dataset: read manifest: %w", err)
	}
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("dataset: parse manifest: %w", err)
	}
	if m.Version == "" {
		return nil, fmt.Errorf("dataset: manifest has no version")
	}
	d := &Dataset{root: root, Manifest: m, includeSubagents: o.includeSubagents}
	return d, nil
}

// Root returns the extractor output root.
func (d *Dataset) Root() string { return d.root }

// Sessions returns the deduplicated session records. Sessions that the
// extractor wrote into multiple topic files are read once.
func (d *Dataset) Sessions() ([]SessionRecord, error) {
	var out []SessionRecord
	if err := d.WalkSessions(func(rec *SessionRecord) error {
		out = append(out, *rec)
		return nil
	}); err != nil {
		return nil, err
	}
	return out, nil
}

// WalkSessions calls fn once per unique session record, streaming the JSONL
// files without buffering them. Duplicates (a session assigned several topics)
// are skipped using the record id, or a content hash when the id is absent
// (strict/openai output).
func (d *Dataset) WalkSessions(fn func(*SessionRecord) error) error {
	seen := map[string]bool{}
	for _, tree := range d.trees() {
		topics, err := d.topics(tree)
		if err != nil {
			return err
		}
		for _, topic := range topics {
			path := filepath.Join(d.root, tree, topic, "sessions.jsonl")
			err := walkJSONL(path, func(raw json.RawMessage) error {
				var rec SessionRecord
				if err := json.Unmarshal(raw, &rec); err != nil {
					return fmt.Errorf("dataset: %s: %w", path, err)
				}
				rec.Tree = tree
				key := rec.ID
				if key == "" {
					key = hashMessages(rec.Messages)
				}
				if seen[key] {
					return nil
				}
				seen[key] = true
				return fn(&rec)
			})
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// WalkRL calls fn once per reward metadata entry in rl/metadata.jsonl. A
// missing file is not an error.
func (d *Dataset) WalkRL(fn func(*RLEntry) error) error {
	path := filepath.Join(d.root, "rl", "metadata.jsonl")
	return walkJSONL(path, func(raw json.RawMessage) error {
		var e RLEntry
		if err := json.Unmarshal(raw, &e); err != nil {
			return fmt.Errorf("dataset: %s: %w", path, err)
		}
		return fn(&e)
	})
}

func (d *Dataset) trees() []string {
	if d.includeSubagents {
		return []string{"sft", "subagents"}
	}
	return []string{"sft"}
}

// topics lists the topic subdirectories of a tree, sorted for determinism.
func (d *Dataset) topics(tree string) ([]string, error) {
	dir := filepath.Join(d.root, tree)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("dataset: read %s: %w", dir, err)
	}
	var topics []string
	for _, e := range entries {
		if e.IsDir() {
			topics = append(topics, e.Name())
		}
	}
	sort.Strings(topics)
	return topics, nil
}

// walkJSONL decodes a stream of newline-delimited JSON values. A missing file
// is treated as empty.
func walkJSONL(path string, fn func(json.RawMessage) error) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	dec := json.NewDecoder(bufio.NewReaderSize(f, 1<<20))
	for {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("dataset: decode %s: %w", path, err)
		}
		if err := fn(raw); err != nil {
			return err
		}
	}
}

func hashMessages(msgs []ChatMessage) string {
	b, _ := json.Marshal(msgs)
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}
