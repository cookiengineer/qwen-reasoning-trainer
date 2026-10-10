#!/usr/bin/env python3
"""Independent Qwen3.5 pre-tokenizer reference.

Uses the `regex` module, which supports the trailing-whitespace lookahead that
Go's RE2 cannot, to split text with the exact qwen35 pattern. Reads a JSON array
of strings from stdin and writes a JSON array of piece arrays to stdout.

Unmatched code points are emitted one per piece to mirror llama.cpp's
unicode_regex_split_custom_qwen35 no-match branch.
"""
import json
import sys

import regex

PATTERN = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def split(text):
    pieces = []
    last = 0
    for m in PATTERN.finditer(text):
        start, end = m.span()
        if start > last:
            pieces.extend(text[i] for i in range(last, start))
        pieces.append(text[start:end])
        last = end
    if last < len(text):
        pieces.extend(text[i] for i in range(last, len(text)))
    return pieces


def main():
    texts = json.load(sys.stdin)
    json.dump([split(t) for t in texts], sys.stdout)


if __name__ == "__main__":
    main()
