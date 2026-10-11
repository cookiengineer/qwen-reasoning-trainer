package dataset

import (
	"context"
	"fmt"
	"hash/fnv"
	"math/rand"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
)

// Example is a tokenized training sequence with an assistant-only loss mask.
type Example struct {
	IDs        []int32
	Mask       []uint8 // 1 = train, 0 = context
	Source     string  // session key
	Topic      []string
	LossTokens int
}

// BuildOptions controls example construction.
type BuildOptions struct {
	Render RenderOptions
	// MaxSeqLen truncates the front of a sequence to this many tokens. 0 means
	// no truncation.
	MaxSeqLen int
	// MinTargetTokens drops examples with fewer trainable (loss=1) tokens.
	MinTargetTokens int
	// Prefetch is the streaming iterator buffer size.
	Prefetch int
}

// DefaultBuildOptions returns the default pipeline options.
func DefaultBuildOptions() BuildOptions {
	return BuildOptions{
		Render:          DefaultRenderOptions(),
		MaxSeqLen:       0,
		MinTargetTokens: 1,
		Prefetch:        64,
	}
}

// SessionKey returns the stable identifier used for deduplication and splits.
func SessionKey(rec *SessionRecord) string {
	if rec.ID != "" {
		return rec.ID
	}
	return hashMessages(rec.Messages)
}

// BuildExample renders and tokenizes one session. It returns (nil, nil) when
// the session has no trainable tokens (or is otherwise unusable).
func BuildExample(v *tokenizer.Vocab, rec *SessionRecord, opt BuildOptions) (*Example, error) {
	if len(opt.Render.Tools) == 0 && len(rec.Meta.Tools) > 0 {
		opt.Render.Tools = rec.Meta.Tools
	}
	segs, err := RenderSegments(rec.Messages, opt.Render)
	if err != nil {
		return nil, err
	}
	ids, mask := EncodeSegments(v, segs)
	if opt.MaxSeqLen > 0 && len(ids) > opt.MaxSeqLen {
		ids = ids[len(ids)-opt.MaxSeqLen:]
		mask = mask[len(mask)-opt.MaxSeqLen:]
	}
	loss := 0
	for _, m := range mask {
		if m == 1 {
			loss++
		}
	}
	if loss < opt.MinTargetTokens {
		return nil, nil
	}
	return &Example{
		IDs:        ids,
		Mask:       mask,
		Source:     SessionKey(rec),
		Topic:      rec.Topic,
		LossTokens: loss,
	}, nil
}

// BuildExamples builds all examples from an in-memory session slice.
func BuildExamples(v *tokenizer.Vocab, recs []SessionRecord, opt BuildOptions) ([]Example, error) {
	var out []Example
	for i := range recs {
		ex, err := BuildExample(v, &recs[i], opt)
		if err != nil {
			return nil, fmt.Errorf("dataset: session %s: %w", SessionKey(&recs[i]), err)
		}
		if ex != nil {
			out = append(out, *ex)
		}
	}
	return out, nil
}

// Examples builds all examples, auto-loading the run-level tool registry from
// the manifest when the options do not set one.
func (d *Dataset) Examples(v *tokenizer.Vocab, opt BuildOptions) ([]Example, error) {
	if len(opt.Render.Tools) == 0 {
		opt.Render.Tools = d.Manifest.Tools
	}
	recs, err := d.Sessions()
	if err != nil {
		return nil, err
	}
	return BuildExamples(v, recs, opt)
}

// Iter streams examples from a dataset. The caller must drain ex until closed;
// any build error is delivered on errs. Cancel ctx to stop early.
func Iter(ctx context.Context, d *Dataset, v *tokenizer.Vocab, opt BuildOptions) (<-chan Example, <-chan error) {
	if len(opt.Render.Tools) == 0 {
		opt.Render.Tools = d.Manifest.Tools
	}
	buf := opt.Prefetch
	if buf <= 0 {
		buf = 1
	}
	out := make(chan Example, buf)
	errs := make(chan error, 1)
	go func() {
		defer close(out)
		defer close(errs)
		err := d.WalkSessions(func(rec *SessionRecord) error {
			ex, err := BuildExample(v, rec, opt)
			if err != nil {
				return fmt.Errorf("dataset: session %s: %w", SessionKey(rec), err)
			}
			if ex == nil {
				return nil
			}
			select {
			case out <- *ex:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		if err != nil {
			errs <- err
		}
	}()
	return out, errs
}

// Shuffle randomizes examples in place using a deterministic seed.
func Shuffle(ex []Example, seed int64) {
	r := rand.New(rand.NewSource(seed))
	r.Shuffle(len(ex), func(i, j int) { ex[i], ex[j] = ex[j], ex[i] })
}

// Split partitions examples deterministically by session key hash. The same
// example always lands in the same partition regardless of input order.
func Split(ex []Example, valRatio float64) (train, val []Example) {
	for _, e := range ex {
		h := fnv.New64a()
		_, _ = h.Write([]byte(e.Source))
		f := float64(h.Sum64()%100000) / 100000.0
		if f < valRatio {
			val = append(val, e)
		} else {
			train = append(train, e)
		}
	}
	return train, val
}

// TotalLossTokens returns the sum of trainable tokens across examples.
func TotalLossTokens(ex []Example) int {
	n := 0
	for _, e := range ex {
		n += e.LossTokens
	}
	return n
}

// TotalTokens returns the sum of sequence lengths across examples.
func TotalTokens(ex []Example) int {
	n := 0
	for _, e := range ex {
		n += len(e.IDs)
	}
	return n
}
