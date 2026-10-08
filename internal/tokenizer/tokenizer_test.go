package tokenizer

import "testing"

func TestDecodeASCII(t *testing.T) {
	v := NewVocab([]string{"H", "e", "l", "l", "o"})
	if got := v.Decode([]int32{0, 1, 2, 3, 4}); got != "Hello" {
		t.Fatalf("Decode = %q, want %q", got, "Hello")
	}
}

func TestDecodeByteAlphabet(t *testing.T) {
	// The GPT-2 byte alphabet maps the space byte (0x20) to U+0120.
	v := NewVocab([]string{"Hello", "\u0120world"})
	if got := v.Decode([]int32{0, 1}); got != "Hello world" {
		t.Fatalf("Decode = %q, want %q", got, "Hello world")
	}
}

func TestDecodeSpecials(t *testing.T) {
	v := NewVocab([]string{"a", "<|im_start|>", "b"})
	if got := v.Decode([]int32{0, 1, 2}); got != "a<|im_start|>b" {
		t.Fatalf("Decode = %q", got)
	}
	if v.Token(99) != "<invalid:99>" {
		t.Fatalf("Token(99) = %q", v.Token(99))
	}
}

func TestEncodeBPE(t *testing.T) {
	v := NewVocab([]string{"a", "b", "c", "d", "ab", "cd", "abcd"})
	v.merges = map[[2]string]int{
		{"a", "b"}:   0,
		{"c", "d"}:   1,
		{"ab", "cd"}: 2,
	}
	ids := v.Encode("abcd")
	if len(ids) != 1 || ids[0] != 6 {
		t.Fatalf("Encode(abcd) = %v, want [6]", ids)
	}
	// Without the final merge the pieces merge pairwise.
	w := NewVocab([]string{"a", "b", "c", "d", "ab", "cd"})
	w.merges = map[[2]string]int{{"a", "b"}: 0, {"c", "d"}: 1}
	if ids := w.Encode("abcd"); len(ids) != 2 || ids[0] != 4 || ids[1] != 5 {
		t.Fatalf("Encode(abcd) = %v, want [4 5]", ids)
	}
}

func TestEncodeDecodeRoundTrip(t *testing.T) {
	// Build a vocab containing every byte's symbol plus a merge, so all input
	// bytes are representable.
	syms := map[rune]bool{}
	for b := 0; b < 256; b++ {
		enc := map[byte]rune{}
		rev := map[rune]byte{}
		buildByteAlphabet(enc, rev)
		syms[enc[byte(b)]] = true
	}
	toks := make([]string, 0, len(syms))
	for r := range syms {
		toks = append(toks, string(r))
	}
	v := NewVocab(toks)
	text := "Hello, world! 123"
	ids := v.Encode(text)
	if got := v.Decode(ids); got != text {
		t.Fatalf("round trip = %q, want %q (ids %v)", got, text, ids)
	}
}

func TestChatPrompt(t *testing.T) {
	toks := []string{"<|im_start|>", "<|im_end|>", "a", "b", "s", "u", "e", "r", "n", "t", "m"}
	v := NewVocab(toks)
	ids := v.ChatPrompt("", "ab")
	start, _ := v.SpecialID("<|im_start|>")
	end, _ := v.SpecialID("<|im_end|>")
	if len(ids) == 0 || ids[0] != start {
		t.Fatalf("prompt should start with im_start: %v", ids)
	}
	found := false
	for _, id := range ids {
		if id == end {
			found = true
		}
	}
	if !found {
		t.Fatalf("prompt missing im_end: %v", ids)
	}
}
