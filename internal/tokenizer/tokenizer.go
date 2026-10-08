// Package tokenizer provides GGUF vocabulary encoding/decoding for Qwen models.
//
// Qwen uses byte-level BPE (GPT-2 style): text is pre-tokenized with a regex,
// each byte of a piece is mapped to a printable unicode character, and adjacent
// symbols are merged according to the merge ranks stored in the GGUF file.
package tokenizer

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
)

// Vocab is a decoded GGUF tokenizer vocabulary.
type Vocab struct {
	Tokens []string
	EOS    []int32
	BOS    []int32

	rev     map[rune]byte
	enc     map[byte]rune
	tokenID map[string]int32
	merges  map[[2]string]int
}

// pretokRe approximates the Qwen/GPT-4 pre-tokenization pattern. Go's regexp is
// RE2 and does not support the trailing-whitespace lookahead, which only
// affects how a run of trailing spaces is grouped.
var pretokRe = regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`)

// NewVocab builds a vocabulary from raw token strings (used by tests).
func NewVocab(tokens []string) *Vocab {
	return newVocab(tokens, nil)
}

func newVocab(tokens []string, merges []string) *Vocab {
	v := &Vocab{
		Tokens:  tokens,
		rev:     map[rune]byte{},
		enc:     map[byte]rune{},
		tokenID: make(map[string]int32, len(tokens)),
		merges:  make(map[[2]string]int, len(merges)),
	}
	buildByteAlphabet(v.enc, v.rev)
	for i, t := range tokens {
		if _, ok := v.tokenID[t]; !ok {
			v.tokenID[t] = int32(i)
		}
	}
	for rank, m := range merges {
		parts := strings.SplitN(m, " ", 2)
		if len(parts) != 2 {
			continue
		}
		v.merges[[2]string{parts[0], parts[1]}] = rank
	}
	return v
}

// FromGGUF reads the token list, merges, and special token ids.
func FromGGUF(g *gguf.File) (*Vocab, error) {
	toks, ok := g.Strings("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("tokenizer: missing tokenizer.ggml.tokens")
	}
	merges, _ := g.Strings("tokenizer.ggml.merges")
	v := newVocab(toks, merges)
	if id, ok := g.Int("tokenizer.ggml.eos_token_id"); ok {
		v.EOS = append(v.EOS, int32(id))
	}
	if id, ok := g.Int("tokenizer.ggml.bos_token_id"); ok {
		v.BOS = append(v.BOS, int32(id))
	}
	return v, nil
}

// Token returns the raw string for a token id.
func (v *Vocab) Token(id int32) string {
	if id < 0 || int(id) >= len(v.Tokens) {
		return fmt.Sprintf("<invalid:%d>", id)
	}
	return v.Tokens[id]
}

// ID returns the token id for a token string.
func (v *Vocab) ID(tok string) (int32, bool) {
	id, ok := v.tokenID[tok]
	return id, ok
}

// SpecialID returns the id of a special token such as "<|im_start|>".
func (v *Vocab) SpecialID(tok string) (int32, bool) { return v.ID(tok) }

// IsEOS reports whether id is an end-of-sequence token.
func (v *Vocab) IsEOS(id int32) bool {
	for _, e := range v.EOS {
		if e == id {
			return true
		}
	}
	return false
}

// Decode converts token ids to text using the GPT-2 byte-level alphabet.
func (v *Vocab) Decode(ids []int32) string {
	var buf []byte
	for _, id := range ids {
		if id < 0 || int(id) >= len(v.Tokens) {
			continue
		}
		for _, r := range v.Tokens[id] {
			if b, ok := v.rev[r]; ok {
				buf = append(buf, b)
			} else {
				buf = append(buf, []byte(string(r))...)
			}
		}
	}
	return string(buf)
}

// Encode converts text to token ids using byte-level BPE.
func (v *Vocab) Encode(text string) []int32 {
	var out []int32
	for _, piece := range pretokRe.FindAllString(text, -1) {
		out = append(out, v.encodePiece(piece)...)
	}
	return out
}

func (v *Vocab) encodePiece(piece string) []int32 {
	// Map each byte to its printable unicode symbol.
	symbols := make([]string, 0, len(piece))
	for i := 0; i < len(piece); i++ {
		r, ok := v.enc[piece[i]]
		if !ok {
			r = rune(piece[i])
		}
		symbols = append(symbols, string(r))
	}
	// Greedily apply the lowest-rank adjacent merge.
	for len(symbols) > 1 {
		bestRank := -1
		bestIdx := -1
		for i := 0; i+1 < len(symbols); i++ {
			if rank, ok := v.merges[[2]string{symbols[i], symbols[i+1]}]; ok {
				if bestRank < 0 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		symbols = append(symbols[:bestIdx], append([]string{merged}, symbols[bestIdx+2:]...)...)
	}
	out := make([]int32, 0, len(symbols))
	for _, s := range symbols {
		if id, ok := v.tokenID[s]; ok {
			out = append(out, id)
		}
	}
	return out
}

// ChatPrompt builds the Qwen chat-formatted token ids for a system and user
// message plus the assistant generation prompt.
func (v *Vocab) ChatPrompt(system, user string) []int32 {
	imStart, ok1 := v.SpecialID("<|im_start|>")
	imEnd, ok2 := v.SpecialID("<|im_end|>")
	if !ok1 || !ok2 {
		return v.Encode(user)
	}
	var out []int32
	role := func(name, content string) {
		out = append(out, imStart)
		out = append(out, v.Encode(name)...)
		out = append(out, v.Encode("\n"+content)...)
		out = append(out, imEnd)
		out = append(out, v.Encode("\n")...)
	}
	if system != "" {
		role("system", system)
	}
	role("user", user)
	out = append(out, imStart)
	out = append(out, v.Encode("assistant")...)
	out = append(out, v.Encode("\n")...)
	return out
}

func buildByteAlphabet(enc map[byte]rune, rev map[rune]byte) {
	bs := make([]int, 0, 256)
	for b := '!'; b <= '~'; b++ {
		bs = append(bs, int(b))
	}
	for b := 0xA1; b <= 0xAC; b++ {
		bs = append(bs, b)
	}
	for b := 0xAE; b <= 0xFF; b++ {
		bs = append(bs, b)
	}
	cs := append([]int(nil), bs...)
	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, x := range bs {
			if x == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	for i := range bs {
		enc[byte(bs[i])] = rune(cs[i])
		rev[rune(cs[i])] = byte(bs[i])
	}
}
