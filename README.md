# Qwen3.8 27B Reasoning Trainer

This is a Pure Go Abliteration, Reasoning Training, and Inference stack.
It is implemented with a CPU backend and a Vulkan backend, which uses compute
shaders and GPU kernels. It implements a complete GGUF/quant stack for the Qwen
3.8 27B model in its `Q4_K_M` GGUF quantization.

## Motivation

This project aims to make Qwen3.8 self-training feasible for consumer-grade
hardware, open source, for everyone to reproduce by themselves.

It is recommended to use the Qwen Reasoning Trainer stack with `48GB` VRAM GPUs.

Abliteration, Reasoning Training, and Inference is tested actively on a machine
with an `AMD Radeon Pro W7800 48GB` GPU and with `96GB` DDR5 SODIMM CPU memory
available. The costs for the hardware of self training a Qwen3.8 model is at
a budget of around `5000 EUR`, even in this ongoing RAM crisis.

Hardware requirements depend on the amount of context-length, the memory calculation
is like this:

- Around 17GB for the model weights
- A couple of MB for GPU kernels (negligible)
- Additional memory for K/V and cache (`~10GB` for `512k`, `~20GB` for `1M`)

That is why it's recommended to use a 32GB VRAM GPU for 512k context length,
and a 48GB VRAM GPU for 1M context length. And yes, self-hosted, local, on
consumer-grade hardware. Pretty awesome, huh?

The Qwen Trainer project started as an experiment of making `Qwen3.8` think like
`DeepSeek`. The training pipeline meanwhile is reproducible and confirmed by others.
The reason for doing so is that `DeepSeek v4.1 Flash` is very good at planning of
coding tasks, but isn't good at cyber security or exploit validation tasks anymore
because the model has been censored heavily compared to the previous `DeepSeek v4 Pro`
model. As v4 Pro is now as overpriced as 'murican models, I decided to go this route.

The CPU backend kernel stack and the K/V cache implementations have been partially
reused from my own experimental [gonano SIMD model](https://github.com/cookiengineer/gonano)
that focuses on better SIMD inference and scales with CPUs (but has its own limitations
due to Go not having lower precisions than `float32` available as of today, sadly).

## Requirements

- Go 1.27 (build and tests).
- `glslc` and `spirv-val` to rebuild the committed SPIR-V shaders.
- A Vulkan device for GPU acceleration and the GPU tests (the CLI selects it automatically when it has enough memory).
- `python3` with venv support only for the external `verify-*` tests.

## Prerequisites

1. Download the **Qwen3.8-27B** model from Hugging Face.

Unsloth Dynamic **Q4_K_M** quantization is the best model for this due to
its weights-in-memory usage of around 17GB of VRAM.

Repository: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>

File: in the **Files and versions** list on the right side of the repo page,
download `Qwen3.8-27B-UD-Q4_K_M.gguf` and place it at `models/Qwen3.8-27B-UD-Q4_K_M.gguf`.
Alternatively let the CLI fetch the pinned artifact for you:

```sh
# Download the Qwen3.8 27B Q4_K_M model from huggingface:
./build/qwen-trainer download;
```

2. Export the training data from your Opencode environment.

Use the session history as the teacher model's reasoning traces. The [opencode-reasoning-extractor](https://github.com/cookiengineer/opencode-reasoning-extractor)
tool writes the following files for an exported dataset:

- `manifest.json`
- `sft/<topic>/sessions.jsonl`
- `sft/<topic>/turns.jsonl`
- `subagents/<topic>/...`
- `rl/metadata.jsonl`.

```sh
# Export Reasoning Traces from your previous Opencode Sessions
opencode-reasoning-extractor ~/.local/share/opencode /path/to/qwen-reasoning-trainer/datasets/my-own-reasoning-traces;
```

3. Point the Qwen Trainer at that output dataset directory using the `--dataset` CLI parameter.

## Usage: Model and Dataset Interaction

```sh
# compile Vulkan compute shaders, then build the build/qwen-trainer binary
make;

# Command is always the first argument
# Flags (including --model) have to come afterwards

# Inspect the GGUF file for Tensor quantization details
./build/qwen-trainer inspect \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf;

# Run the Qwen Trainer as a quick generation test
./build/qwen-trainer run \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --prompt "The capital of France is" \
  --generate 8;

# Tokenize/validate the extractor output and report loss-mask statistics
./build/qwen-trainer dataset \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --dataset /path/to/my/own/reasoning-traces/from/opencode;
```

- `--model` is required by every model command (`inspect`, `run`, `abliterate`, `search`, `evaluate`, `dataset`, `train`)
- `--dataset` is repeatable. Use it multiple times to include multiple (exported) reasoning traces datasets

The CPU/GPU compute backend is chosen automatically.
Vulkan is used when a device is present and it has enough device memory for the
model, otherwise the CPU backend and kernel runs.

Use `gpu-info` to see the detected device and available memory.

## Usage: Qwen Inference Server

The [qwen-server](./source/cmds/qwen-server) exposes the same Vulkan/CPU inference
pipeline through a minimal OpenAI-compatible API. You can build it with `make server`:

```sh
# Build inference server
make server;

# Execute inference server
./build/qwen-server --model models/Qwen3.8-27B-UD-Q4_K_M.gguf --port 8080;
```

The inference server is intentionally kept minimalistic. It serves the following endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /health`.

Up to `--max-concurrent` requests are allowed at the same time. Each request
has their own concurrency-safe Vulkan kernel running in-VRAM. Each vulkan kernel
requires around 1MB of memory in-VRAM.

```sh
# Example request that returns ReasoningContent via HTTP Stream
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-UD-Q4_K_M",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": true
  }';
```

Supported request features include `stream` (SSE), `reasoning_content` deltas,
`tools`/`tool_calls`, `temperature`, `top_p`, `top_k`, `seed`, `stop`,
`max_tokens`/`max_completion_tokens`, `reasoning_effort`, and `chat_template_kwargs`.

Thinking is enabled by default. Tool calls use the Qwen XML format and are returned
as OpenAI API compatible `tool_calls` with JSON arguments.

### Inference Flags

| Flag                     | Values                         | Default         |
|--------------------------|--------------------------------|-----------------|
| `--device`               | `auto`, `cpu`, `vulkan`        | `auto`          |
| `--context-length`       | `128k`, `256k`, `384k`, `512k` | (GGUF context)  |
| `--kv-cache-type`        | `q4_k`, `q8_0`                 | `q4_k`          |
| `--max-concurrent`       | (int)                          | `1`             |
| `--default-max-tokens`   | (int)                          | `1024`          |
| `--speculative`          | `true`, `false`                | `true`          |
| `--spec-draft-max`       | (int)                          | `4`             |
| `--spec-draft-min-prob`  | (float64)                      | `0.6`           |
| `--model-id`             | (string)                       | (GGUF filename) |
| `--api-key`              | (string)                       |                 |

### Inference Environment Variables

| Variable              | Description     |
|-----------------------|-----------------|
| `QWEN_SERVER_MODEL`   | Model GGUF Path |
| `QWEN_SERVER_API_KEY` | Bearer API Key  |

On Vulkan the full kernel and inference forward shaders run in-VRAM on the GPU.
The K/V are also stored in the quantized cache and appended with GPU shaders.
The prefill chunk size is selected automatically from the context length and
available device memory.

- Streaming usage is included when `stream_options.include_usage` is true.
- MTP speculative decoding is enabled by default on both Vulkan and CPU.
- Context values above the natively supported GGUF context (256k) select
  YaRN automatically, with the factor derived from `context / native_context`.

See the [ARCHITECTURE](docs/ARCHITECTURE.md) for more details.

## Usage: Qwen Model Abliteration

The [Qwen Trainer](./source/cmds/qwen-trainer) also removes the refusal/safety
direction from the weights with Heretic-style directional ablation by using the
same magnitude-preserving orthogonal ablation (called MPOA). It then optionally
optimizes the ablation hyperparameters.

```sh
# 1. Abliterate with default parameters and write a new full Q4_K_M GGUF afterwards
./build/qwen-trainer abliterate \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --out models/my-own-abliterated.gguf;

# 2. Search direction index and per-component weight kernels, and write the trial results.
./build/qwen-trainer search \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --trials 25 \
  --sampler tpe \
  --study models/search.jsonl \
  --out models/search-best.gguf;

# 3. Score the abliteration result via measuring refusal-keyword rate and KL divergence from base model
./build/qwen-trainer evaluate \
  --model models/abliterated.gguf \
  --prompts source/prompts/harmful_behaviors_test.jsonl \
  --max-tokens 32;
```

### Prompt Sets and Prompt Limits

The [prompts](./source/prompts) folder bundles the same datasets that Heretic uses
behind the scenes, but the dataset has been pre-converted to JSON Lines where
one object per line represents one question, essentially.

These prompts are embedded into the binary with `go:embed`, so that `abliterate`
and `search` use them by default with no external files being needed.

You can override them with `--good` or `--bad` for `abliterate` and with `--eval-good`
or `--eval-bad` for `search`. It would be heavily welcomed if you would contribute
your custom abliteration questions catalog via Pull Request to this repository though!

The abliteration catalog files may be either plain text files (one prompt per line)
or JSONL files. The `--prompt-col` CLI parameter selects the column, default `text`.

| File                            | Rows  | Role (default)                                |
|---------------------------------|-------|-----------------------------------------------|
| `harmless_alpaca_train.jsonl`   | 25058 | good prompts / residual directions (`--good`) |
| `harmful_behaviors_train.jsonl` | 416   | bad prompts / residual directions (`--bad`)   |
| `harmless_alpaca_test.jsonl`    | 6265  | KL divergence evaluation (`--eval-good`)      |
| `harmful_behaviors_test.jsonl`  | 104   | refusal evaluation (`--eval-bad`)             |

The full files are quite large, so the limits are defaulted to Heretic's limits,
with `--limit 400` and `--eval-limit 100`. Set `--limit 0` or `--eval-limit 0`
to use an entire file. Note that more prompts will cost one complete model forward,
so abliteration time increases linearly.

The embedded datasets mirror `mlabonne/harmless_alpaca` and `mlabonne/harmful_behaviors`
from the `heretic-org` organization on Hugging Face.

### Abliteration Flags

| Flag                | Values                    | (for) Commands            | Description                                                             |
|---------------------|---------------------------|---------------------------|-------------------------------------------------------------------------|
| `--row-norm`        | `none`, `pre`, `full`     | `abliterate`, `search`    | MPOA row normalization (default `full`)                                 |
| `--orthogonalize`   |                           | `abliterate`, `search`    | project the direction orthogonal to the good direction (default `true`) |
| `--direction-index` |                           | `abliterate`              | single global direction layer index (default `-1` per layer)            |
| `--winsorize`       |                           | `abliterate`, `search`    | clamp residual components to a quantile in `[0,1)`                      |
| `--sampler`         | `random`, `halton`, `tpe` | `search`                  | optimizer (default `random`)                                            |
| `--lambda`          |                           | `search`                  | KL weight in the objective `refusal + lambda*KL` (default `1.0`)        |
| `--seed`, `--study` |                           | `search`                  | reproducible seed and resumable JSON Lines checkpoint                   |

Note: the first-token for KL divergence measurement is a weak signal for a thinking model
like this, because it emits `<think>` first. That's why small divergence scores are expected
on short generations.

## Usage: Qwen Abliteration Evaluation

`evaluate` scores a model along three independent axes; at least one input is
required.

```sh
# Evaluates Refusal keyword rate and KL divergence from a base model
./build/qwen-trainer evaluate \
  --model models/abliterated.gguf \
  --prompts source/prompts/harmful_behaviors_test.jsonl \
  --max-tokens 32 \
  --base models/Qwen3.8-27B-UD-Q4_K_M.gguf;

# Evaluates Corpus Perplexity over a text file, scored in specified windows of N tokens
./build/qwen-trainer evaluate \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --perplexity corpus.txt \
  --window 512;

# Evaluates multiple-choice accuracy over JSONL {context, choices, answer} records
./build/qwen-trainer evaluate \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --tasks tasks.jsonl \
  --tasks-verbose;
```

The Corpus Perplexity evaluation uses non-overlapping windows and is defaulted to `512` tokens.
Use `--window 0` for the whole input. The window bounds peak memory at around `vocab * window * 4`
bytes, so it is the quality/memory limit and because it computes full LM-head logits for every
window it is considerably slower per token than `run`.

The Task accuracy is scored by the length-normalized log-likelihood of each choice given the context.
Use `--tasks-normalize=false` for raw sums.

## Testing

The default suite is pure Go and needs neither a GPU nor Python:

```sh
make test;     # go test ./...
make check;    # fmt-check + vet + test (run this before calling anything done)
```

In order to run the suite on a remote GPU server, stream the working tree over
SSH and run a `make` target there:

```sh
bash run-tests.sh my-48gb-vram-server;          # remote make check
bash run-tests.sh my-48gb-vram-server test;     # remote make test
```

The unit and end-to-end tests that need hardware or external data skip automatically.
This means that `make check` works on any machine:

- Vulkan tests skip when no device is present.
- Real-model tests (tokenizer specials, example building) skip when no GGUF is
  present under `models/`.

### External end-to-end verification tests

Independent cross-checks live under [e2e-tests](source/e2e-tests/). Each end-to-end
test has an isolated `main` program per check. They reuse reference implementations
from other implementations and are never run directly by `make check`. Instead you
have to run them like this (and they will install also python/cpp dependencies):

```sh
make verify-ref;    # Compares NumPy forward pass against the Go tiny model
make verify-data;   # Compares Jinja2 render of the model chat template and Python regex
make verify-quants; # Evaluates our quant implementations against llama.cpp's ggml-quants.c
```

Each of the end-to-end tests is also runnable directly. Execute them with the `go -C` flag,
for example with `go -C source run ./e2e-tests/verify-ref`.

The Python references use a project-local virtualenv at `source/e2e-tests/.venv`, which is
created on demand and installs dependencies from each test's own `requirements.txt` file.
The venv itself is gitignored on purpose.

- `make e2e-venv` creates the venv with all dependencies explicitly.
- `SYSTEM_PYTHON=python3.x` selects the base interpreter used to build it.
- `E2E_PYTHON=/path/to/python` runs the references against your own environment.

The verification programs can also be run directly with the `QWEN38_REF_PYTHON`
environment variable:

```sh
QWEN38_REF_PYTHON=/path/to/python go -C source run ./e2e-tests/verify-ref;
QWEN38_REF_PYTHON=/path/to/python go -C source run ./e2e-tests/verify-data;
CC=/usr/bin/gcc go -C source run ./e2e-tests/verify-quants;
```

The reasoning traces dataset reader is additionally checked against an example
fixture emitted by the `opencode-reasoning-extractor`'s own writer.

## License

This project is released under the MIT License.
See [LICENSE.txt](LICENSE.txt) for details.

