# Qwen3.8-27B Reasoning Trainer

A pure-Go, cgo-free toolkit for the **Qwen3.8 27B** model in its **Q4_K_M** GGUF
quantization. It implements a complete GGUF/quant stack, a CPU + Vulkan compute
backend, the Qwen3.8 forward pass (full attention + GatedDeltaNet), abliteration,
and a dataset pipeline for SFT Training (QLoRA).

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

# remove refusal directions and write an abliterated GGUF
./build/qwen-trainer abliterate \
  --good prompts/good.txt \
  --bad prompts/bad.txt \
  --out models/abliterated.gguf;

# search abliteration parameters (refusal + lambda*KL) and write the best trial
./build/qwen-trainer search \
  --good prompts/good.txt \
  --bad prompts/bad.txt \
  --trials 25 --sampler tpe --study models/search.jsonl \
  --out models/search-best.gguf;

# score a model's refusal rate and KL divergence from the base
./build/qwen-trainer \
  --model models/abliterated.gguf \
  evaluate \
  --prompts prompts/bad.txt \
  --max-tokens 32;
```

The compute backend is chosen automatically: Vulkan is used when a device is
present and it has enough device memory for the model, otherwise the CPU
reference runs. `gpu-info` reports the detected device and memory.

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

### External reference checks

Independent cross-checks live in the isolated [e2e-tests/](e2e-tests/)
directory (Go drivers plus Python reference scripts). They are opt-in via
environment variables and never run as part of `make check`:

```sh
make verify-ref;    # NumPy forward pass vs the Go tiny model (~3e-8 match)
make verify-data;   # Jinja2 render of the model chat template + Python `regex`
                    # evaluation of the qwen35 pre-tokenizer
```

These checks use a project-local virtualenv at `e2e-tests/.venv`, created and
populated on demand from [e2e-tests/requirements.txt](e2e-tests/requirements.txt)
(`numpy`, `jinja2`, `regex`). The venv is gitignored.

- `make e2e-venv` creates/populates the venv explicitly.
- `SYSTEM_PYTHON=python3.x` selects the base interpreter used to build it.
- `E2E_PYTHON=/path/to/python` runs the references against your own environment.

When a `verify-*` target is run, a missing interpreter, dependency, or model is a
hard failure, not a skip - an explicitly requested check never silently passes.

The reference scripts can also be driven directly with the usual Go env vars:

```sh
QWEN38_VERIFY_REF=1  QWEN38_REF_PYTHON=/path/to/python go test -run TestReferenceNumpy -v ./e2e-tests/...;
QWEN38_VERIFY_DATA=1 QWEN38_REF_PYTHON=/path/to/python go test -run TestReferenceData  -v ./e2e-tests/...;
```

The dataset reader is additionally checked against a fixture emitted by the
`opencode-reasoning-extractor`'s own writer
(`internal/dataset/testdata/extractor-writer/`).

## License

Released under the MIT License. See [LICENSE.txt](LICENSE.txt).

Note: this project is a clean-room reimplementation; the bundled read-only
reference checkouts (`references/llama_cpp`, `references/heretic`) are not part
of the build and keep their own licenses (Heretic is AGPL-3.0).

## Requirements

- Go 1.27 (build and tests).
- `glslc` + `spirv-val` only to rebuild the committed SPIR-V shaders.
- A Vulkan device for GPU acceleration and the GPU tests (the CLI selects it
  automatically when it has enough memory).
- `python3` with venv support only for the external reference checks.
