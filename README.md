# Qwen3.8-27B Reasoning Trainer

A pure-Go, cgo-free toolkit for the **Qwen3.8 27B** model in its **Q4_K_M** GGUF
quantization. It implements a complete GGUF/quant stack, a CPU + Vulkan compute
backend, the Qwen3.8 forward pass (full attention + GatedDeltaNet), abliteration,
and a dataset pipeline for SFT Training (QLoRA).

## Requirements

- Go 1.27 (build and tests).
- `glslc` and `spirv-val` to rebuild the committed SPIR-V shaders.
- A Vulkan device for GPU acceleration and the GPU tests (the CLI selects it automatically when it has enough memory).
- `python3` with venv support only for the external `verify-*` tests.

## Prerequisites

1. Download **Qwen3.8-27B** (Unsloth Dynamic **Q4_K_M**) from Hugging Face:

- Repository: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- File: in the **Files and versions** list on the right side of the repo page,
  download `Qwen3.8-27B-UD-Q4_K_M.gguf`

Place it at `models/Qwen3.8-27B-UD-Q4_K_M.gguf`, or let the CLI fetch the pinned
artifact (with resume + SHA-256 verification) for you:

```sh
./build/qwen-trainer download;
```

2. Export the training data from your Opencode environment and use the session
   history as the teacher model. This writes the `manifest.json`,
   `sft/<topic>/sessions.jsonl`, `sft/<topic>/turns.jsonl`, `subagents/<topic>/...`,
   and `rl/metadata.jsonl`.

```sh
opencode-reasoning-extractor ~/.local/share/opencode /my/own/reasoning-traces;
```

3. Point the Qwen Trainer at that output dataset directory (see [Usage](#usage)).

## Usage

```sh
# compile shaders, then build build/qwen-trainer
make;

# run the Qwen Trainer
./build/qwen-trainer inspect;
./build/qwen-trainer run --prompt "The capital of France is" --generate 8;

# tokenize/validate the extractor output and report loss-mask statistics
./build/qwen-trainer dataset \
  --input /path/to/my/own/reasoning-traces/from/opencode;
```

The compute backend is chosen automatically: Vulkan is used when a device is
present and it has enough device memory for the model, otherwise the CPU
reference runs. `gpu-info` reports the detected device and memory.

See [Abliteration](#abliteration) for removing refusal directions and for the
parameter search.

## Abliteration

Remove the refusal/safety direction from the weights with Heretic-style
directional ablation (projected / magnitude-preserving orthogonal ablation,
MPOA), then optionally optimize the ablation hyperparameters.

```sh
# 1. Ablate with the default parameters and write a full Q4_K_M GGUF.
./build/qwen-trainer abliterate --out models/abliterated.gguf;

# 2. (optional) search direction index + per-component weight kernels and write
#    the best trial. Trials persist to the study file and resume automatically.
./build/qwen-trainer search \
  --trials 25 \
  --sampler tpe \
  --study models/search.jsonl \
  --out models/search-best.gguf;

# 3. Score the result: refusal-keyword rate and KL divergence from the base.
./build/qwen-trainer --model models/abliterated.gguf evaluate \
  --prompts prompts/harmful_behaviors_test.jsonl \
  --max-tokens 32;
```

### Prompt sets and limits

`prompts/` bundles the same datasets Heretic uses, pre-converted to JSON Lines
(one `{"text": "..."}` object per line), so no download is needed. `abliterate`
and `search` use them by default; override with `--good`/`--bad`
(`--eval-good`/`--eval-bad` for `search`). Files may be plain text (one prompt
per line) or JSONL (`--prompt-col` selects the column, default `text`).

| File | Rows | Role (default) |
| --- | --- | --- |
| `harmless_alpaca_train.jsonl` | 25058 | good prompts / residual directions (`--good`) |
| `harmful_behaviors_train.jsonl` | 416 | bad prompts / residual directions (`--bad`) |
| `harmless_alpaca_test.jsonl` | 6265 | KL divergence evaluation (`--eval-good`) |
| `harmful_behaviors_test.jsonl` | 104 | refusal evaluation (`--eval-bad`) |

The full files are large, so the limits default to Heretic's: `--limit 400`
(residual prompts per set) and `--eval-limit 100` (evaluation prompts per set).
Pass `--limit 0` / `--eval-limit 0` to use an entire file. Every prompt costs one
model forward, so raising the limits directly increases runtime.

These mirror `mlabonne/harmless_alpaca` and `mlabonne/harmful_behaviors` as
published under `heretic-org` on Hugging Face.

### Key flags

| Flag | Command | Meaning |
| --- | --- | --- |
| `--row-norm none\|pre\|full` | `abliterate`, `search` | MPOA row normalization (default `full`) |
| `--orthogonalize` | `abliterate`, `search` | project the direction orthogonal to the good direction (default true) |
| `--direction-index` | `abliterate` | single global direction layer index (`-1` = per layer, default) |
| `--winsorize` | `abliterate`, `search` | clamp residual components to a quantile in `[0,1)` |
| `--sampler random\|halton\|tpe` | `search` | optimizer (default `random`) |
| `--lambda` | `search` | KL weight in the objective `refusal + lambda*KL` (default `1.0`) |
| `--seed`, `--study` | `search` | reproducible seed and resumable JSON Lines checkpoint |

Note: the first-token KL is a weak signal for a thinking model (it emits
`<think>` first), so small divergence scores are expected on short generations.

## Evaluation

`evaluate` scores a model along three independent axes; at least one input is
required.

```sh
# Refusal keyword rate (and KL divergence from a base with --base).
./build/qwen-trainer --model models/abliterated.gguf evaluate \
  --prompts prompts/harmful_behaviors_test.jsonl --max-tokens 32 \
  --base models/Qwen3.8-27B-UD-Q4_K_M.gguf;

# Corpus perplexity over a text file, scored in windows of N tokens.
./build/qwen-trainer evaluate --perplexity corpus.txt --window 512;

# Multiple-choice accuracy over JSONL {context, choices, answer} records.
./build/qwen-trainer evaluate --tasks tasks.jsonl --tasks-verbose;
```

Perplexity uses non-overlapping windows (default 512 tokens, `--window 0` for
the whole input); the window bounds peak memory at `vocab * window * 4` bytes,
so it is the quality/memory knob. Because it computes full LM-head logits for
every window it is considerably slower per token than `run`. Task accuracy is
scored by the length-normalized log-likelihood of each choice given the context
(`--tasks-normalize=false` for raw sums).

## Testing

The default suite is pure Go and needs neither a GPU nor Python:

```sh
make test;      # go test ./...
make check;     # fmt-check + vet + test (run this before calling anything done)
```

Tests that need hardware or external data skip automatically, so `make check`
works on any machine:

- Vulkan tests skip when no device is present.
- Real-model tests (tokenizer specials, example building) skip when no GGUF is
  present under `models/`.

### External verification checks

Independent cross-checks live under [e2e-tests/](e2e-tests/), one isolated `main`
program per check. They shell out to reference implementations and are never run
by `make check`:

```sh
make verify-ref;    # NumPy forward pass vs the Go tiny model (~3e-8 match)
make verify-data;   # Jinja2 render of the model chat template + Python `regex`
                    # evaluation of the qwen35 pre-tokenizer
make verify-quants; # our block decoders + encoders vs llama.cpp's ggml-quants.c
                    # (needs a C compiler)
```

Each is also runnable directly, e.g. `go run ./e2e-tests/verify-ref`.

The Python references use a project-local virtualenv at `e2e-tests/.venv`,
created and populated on demand from each check's own `requirements.txt`
(`e2e-tests/verify-ref/requirements.txt`, `e2e-tests/verify-data/requirements.txt`
— `numpy`, `jinja2`, `regex`). The venv is gitignored.

- `make e2e-venv` creates/populates the venv explicitly.
- `SYSTEM_PYTHON=python3.x` selects the base interpreter used to build it.
- `E2E_PYTHON=/path/to/python` runs the references against your own environment.

The programs can also be driven directly with `QWEN38_REF_PYTHON`:

```sh
QWEN38_REF_PYTHON=/path/to/python go run ./e2e-tests/verify-ref
QWEN38_REF_PYTHON=/path/to/python go run ./e2e-tests/verify-data
go run ./e2e-tests/verify-quants     # CC=... to pick a compiler
```

The dataset reader is additionally checked against a fixture emitted by the
`opencode-reasoning-extractor`'s own writer
(`internal/dataset/testdata/extractor-writer/`).

## License

Released under the MIT License. See [LICENSE.txt](LICENSE.txt).

Note: this project is a clean-room reimplementation; the bundled read-only
reference checkouts (`references/llama_cpp`, `references/heretic`) are not part
of the build and keep their own licenses (Heretic is AGPL-3.0).

