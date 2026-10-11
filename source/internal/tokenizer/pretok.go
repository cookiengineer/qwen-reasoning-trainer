package tokenizer

import "unicode"

// outOfRange marks a code point index past the end of the input. It is not a
// valid Unicode code point, matching llama.cpp's 0xFFFFFFFF sentinel.
const outOfRange rune = -1

// cptFlags caches the character classes used by the qwen35 pre-tokenizer.
type cptFlags struct {
	valid      bool // in-range; llama.cpp's flags.as_uint() != 0
	letter     bool // \p{L}
	number     bool // \p{N}
	whitespace bool // \s
	accentMark bool // \p{M}
}

func flagsOf(r rune) cptFlags {
	if r == outOfRange {
		return cptFlags{}
	}
	return cptFlags{
		valid:      true,
		letter:     unicode.IsLetter(r),
		number:     unicode.IsNumber(r),
		whitespace: unicode.IsSpace(r),
		accentMark: unicode.Is(unicode.M, r),
	}
}

// SplitQwen35 pre-tokenizes text with the Qwen3.5 regex. It is a direct port of
// llama.cpp's unicode_regex_split_custom_qwen35 (src/unicode.cpp), which
// implements:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// It is a hand-written port because Go's RE2 does not support the trailing
// whitespace lookahead (?!\S). Compared to Qwen2, letter runs also consume
// Unicode combining marks (\p{M}).
func SplitQwen35(text string) []string {
	cpts := []rune(text)

	getCpt := func(pos int) rune {
		if pos >= 0 && pos < len(cpts) {
			return cpts[pos]
		}
		return outOfRange
	}
	getFlags := func(pos int) cptFlags {
		return flagsOf(getCpt(pos))
	}

	var out []string
	prevEnd := 0
	addToken := func(end int) int {
		length := end - prevEnd
		if length > 0 {
			out = append(out, string(cpts[prevEnd:end]))
		}
		prevEnd = end
		return length
	}

	for pos := 0; pos < len(cpts); {
		cpt := getCpt(pos)
		flags := getFlags(pos)

		// (?i:'s|'t|'re|'ve|'m|'ll|'d)
		if cpt == '\'' && pos+1 < len(cpts) {
			n1 := unicode.ToLower(getCpt(pos + 1))
			if n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd' {
				pos += addToken(pos + 2)
				continue
			}
			if pos+2 < len(cpts) {
				n2 := unicode.ToLower(getCpt(pos + 2))
				if (n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l') {
					pos += addToken(pos + 3)
					continue
				}
			}
		}

		// [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
		if !(cpt == '\r' || cpt == '\n' || flags.number) {
			if flags.letter || flags.accentMark || getFlags(pos+1).accentMark || getFlags(pos+1).letter {
				pos++
				for getFlags(pos).letter || getFlags(pos).accentMark {
					pos++
				}
				addToken(pos)
				continue
			}
		}

		// \p{N}
		if flags.number {
			pos++
			addToken(pos)
			continue
		}

		//  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
		f2 := flags
		if cpt == ' ' {
			f2 = getFlags(pos + 1)
		}
		if !(f2.whitespace || f2.letter || f2.accentMark || f2.number) && flags.valid {
			if cpt == ' ' {
				pos++
			}
			for !(f2.whitespace || f2.letter || f2.accentMark || f2.number) && f2.valid {
				pos++
				f2 = getFlags(pos)
			}
			cpt2 := getCpt(pos)
			for cpt2 == '\r' || cpt2 == '\n' {
				pos++
				cpt2 = getCpt(pos)
			}
			addToken(pos)
			continue
		}

		numWhitespaces := 0
		lastEndROrN := 0
		for getFlags(pos + numWhitespaces).whitespace {
			cpt2 := getCpt(pos + numWhitespaces)
			if cpt2 == '\r' || cpt2 == '\n' {
				lastEndROrN = pos + numWhitespaces + 1
			}
			numWhitespaces++
		}

		// \s*[\r\n]+
		if lastEndROrN > 0 {
			pos = lastEndROrN
			addToken(pos)
			continue
		}

		// \s+(?!\S)
		if numWhitespaces > 1 && getCpt(pos+numWhitespaces) != outOfRange {
			pos += numWhitespaces - 1
			addToken(pos)
			continue
		}

		// \s+
		if numWhitespaces > 0 {
			pos += numWhitespaces
			addToken(pos)
			continue
		}

		// no match
		pos++
		addToken(pos)
	}
	return out
}
