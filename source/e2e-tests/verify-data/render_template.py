#!/usr/bin/env python3
"""Independent chat-template reference for the Qwen3.8 GGUF.

Renders the model's own tokenizer.chat_template with Jinja2 (the engine
transformers/HF uses) so the Go port in internal/dataset/template.go can be
cross-checked. Requires jinja2.

Usage: render_template.py <model.gguf> <cases.json> [trim|notrim]
Writes a JSON array of rendered strings to stdout.
"""
import json
import struct
import sys

import jinja2


def load_template(path):
    f = open(path, "rb")
    _magic, _ver, _ntensors, n_kv = struct.unpack("<IIQQ", f.read(24))

    def rd(t):
        if t == 8:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rd(et) for _ in range(n)]
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t]
        return f.read(sz)

    for _ in range(n_kv):
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen).decode()
        vt = struct.unpack("<I", f.read(4))[0]
        val = rd(vt)
        if key == "tokenizer.chat_template":
            return val
    raise SystemExit("no tokenizer.chat_template in GGUF")


def make_env(trim):
    def raise_exception(message):
        raise RuntimeError(message)

    env = jinja2.Environment(trim_blocks=trim, lstrip_blocks=trim)
    env.globals["raise_exception"] = raise_exception
    return env


def main():
    model, cases_path = sys.argv[1], sys.argv[2]
    trim = (len(sys.argv) < 4) or (sys.argv[3] == "trim")
    template = load_template(model)
    cases = json.load(open(cases_path))
    env = make_env(trim)
    compiled = env.from_string(template)

    out = []
    for case in cases:
        o = case.get("options", {})
        out.append(
            compiled.render(
                messages=case["messages"],
                tools=o.get("tools"),
                enable_thinking=o.get("enable_thinking", True),
                reasoning_effort=o.get("reasoning_effort", "xhigh"),
                preserve_thinking=o.get("preserve_thinking", True),
                add_generation_prompt=o.get("add_generation_prompt", False),
            )
        )
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
