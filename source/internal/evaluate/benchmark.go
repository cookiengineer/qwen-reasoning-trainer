package evaluate

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
)

// MCItem is one multiple-choice task. Context is scored as a prefix and each
// Choice as a continuation; Answer is the index of the correct choice.
type MCItem struct {
	ID      string   `json:"id,omitempty"`
	Context string   `json:"context"`
	Choices []string `json:"choices"`
	Answer  int      `json:"answer"`
}

// MCResult is the scored outcome for one item.
type MCResult struct {
	ID        string
	Predicted int
	Correct   bool
	// Scores holds the (optionally length-normalized) log-likelihood of each
	// choice under the model.
	Scores []float64
}

// maxTaskLine bounds a single JSONL task record.
const maxTaskLine = 16 << 20

// LoadTasks reads a JSON Lines file of MCItem records (one JSON object per
// line, blank lines ignored). It validates that every item has at least two
// choices and an in-range answer, reporting the offending line number.
func LoadTasks(path string) ([]MCItem, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), maxTaskLine)
	var items []MCItem
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var it MCItem
		if err := json.Unmarshal([]byte(raw), &it); err != nil {
			return nil, fmt.Errorf("evaluate: tasks line %d: %w", line, err)
		}
		if len(it.Choices) < 2 {
			return nil, fmt.Errorf("evaluate: tasks line %d: need at least 2 choices, got %d", line, len(it.Choices))
		}
		if it.Answer < 0 || it.Answer >= len(it.Choices) {
			return nil, fmt.Errorf("evaluate: tasks line %d: answer %d out of range [0,%d)", line, it.Answer, len(it.Choices))
		}
		items = append(items, it)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return items, nil
}

// MultipleChoice scores each item by the log-likelihood of every choice given
// the context and returns the per-item results plus the overall accuracy. When
// normalize is true each choice score is divided by its number of scored tokens
// (length normalization); otherwise the raw summed log-likelihood is used.
func MultipleChoice(m *qwen38.Model, vocab *tokenizer.Vocab, items []MCItem, normalize bool) ([]MCResult, float64, error) {
	nVocab := m.Cfg.NVocab
	results := make([]MCResult, len(items))
	correct := 0
	for i, it := range items {
		ctx := vocab.Encode(it.Context)
		scores := make([]float64, len(it.Choices))
		best := 0
		for ci, choice := range it.Choices {
			full := vocab.Encode(it.Context + choice)
			prefix := len(ctx)
			if prefix > len(full) {
				prefix = len(full)
			}
			res, err := m.Forward(full, qwen38.ForwardOptions{})
			if err != nil {
				return nil, 0, err
			}
			if got := res.Logits.Ne(0); got != nVocab {
				return nil, 0, fmt.Errorf("evaluate: logits vocab %d, want %d", got, nVocab)
			}
			var sum float64
			count := 0
			for t := prefix - 1; t >= 0 && t+1 < len(full); t++ {
				sum += logProbAt(res.Logits.F32, nVocab, t, int(full[t+1]))
				count++
			}
			if count == 0 {
				scores[ci] = math.Inf(-1)
			} else if normalize {
				scores[ci] = sum / float64(count)
			} else {
				scores[ci] = sum
			}
			if scores[ci] > scores[best] {
				best = ci
			}
		}
		results[i] = MCResult{ID: it.ID, Predicted: best, Correct: best == it.Answer, Scores: scores}
		if results[i].Correct {
			correct++
		}
	}
	acc := 0.0
	if len(items) > 0 {
		acc = float64(correct) / float64(len(items))
	}
	return results, acc, nil
}
