package tokenizer

import (
	"reflect"
	"testing"
)

func TestSplitQwen35(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"don't", []string{"don", "'t"}},
		{"we've", []string{"we", "'ve"}},
		{"a  b", []string{"a", " ", " b"}},
		{"a   ", []string{"a", "   "}},
		{"123", []string{"1", "2", "3"}},
		{"line\nnext", []string{"line", "\n", "next"}},
	}
	for _, c := range cases {
		if got := SplitQwen35(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("SplitQwen35(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestEncodeSpecials(t *testing.T) {
	v := NewVocab([]string{"a", "b", "<|im_start|>", "<|im_end|>"})
	v.Pre = "qwen35"
	for i, s := range []string{"<|im_start|>", "<|im_end|>"} {
		id := int32(i + 2)
		v.specials[s] = id
		v.specialsFirst[s[0]] = append(v.specialsFirst[s[0]], specialToken{s: s, id: id})
	}

	got := v.Encode("a<|im_start|>b<|im_end|>")
	want := []int32{0, 2, 1, 3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode = %v, want %v", got, want)
	}
}

func TestEncodeWithoutSpecialsUnchanged(t *testing.T) {
	v := NewVocab([]string{"a", "b"})
	if got := v.Encode("ab"); !reflect.DeepEqual(got, []int32{0, 1}) {
		t.Fatalf("Encode = %v, want [0 1]", got)
	}
}
