# Training Guide

End-to-end operational guide for abliterating and fine-tuning **Qwen3.8 27B**
(Q4_K_M GGUF). For the design behind these commands see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for open work see [`TASKS.md`](TASKS.md).

The pipeline produces three artifacts:

| Artifact | Produced by |
| --- | --- |
| `Qwen3.8-27B-UD-Q4_K_M.gguf` | `download` (the base) |
| `Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf` | `abliterate` |
| `Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf` | `train` (SFT on reasoning traces) |

`run-pipeline.sh` runs the whole sequence; the sections below explain each step.

---

## 1. Requirements

- **Go 1.27** to build and test.
- **`glslc` + `spirv-val`** only to rebuild shaders. The `.spv` files are
  committed and embedded, so a plain `go build`/`go test` does not need them.
- **A Vulkan device** for GPU acceleration (the CLI selects it automatically
  when it has enough device-local memory; otherwise the CPU reference runs).
- **`python3` with venv support** only for the external `verify-*` checks.
- **`opencode-reasoning-extractor`** (a separate project) to export the SFT
  dataset from an OpenCode session database.

Target hardware for the real 27B: one Vulkan GPU with ~48 GB VRAM and ~96 GB
system RAM. Check the device and capabilities first:

```sh
make build
./build/qwen-trainer gpu-info
```

For the device trainer the GPU must report `fp32 atomics: true`.

---

## 2. Build

```sh
make;            # compiles shaders, then builds build/qwen-trainer
make check;      # fmt-check + vet + full test suite (no GPU/Python needed)
```

Optional independent cross-checks (never run by `make check`):

```sh
make verify-ref;     # NumPy forward reference
make verify-data;    # Jinja2 chat template + regex pre-tokenizer
make verify-quants;  # decoders/encoders vs llama.cpp ggml-quants.c (needs cc)
```

---

## 3. Export the reasoning-trace dataset

The dataset is the JSONL output of the separate `opencode-reasoning-extractor`
project. Point it at an OpenCode data directory and an output directory:

```sh
# build the extractor, then export one profile per run
( cd ../opencode-reasoning-extractor/source \
  && go build -o ../build/opencode-reasoning-extractor ./cmd/opencode-reasoning-extractor )

../opencode-reasoning-extractor/build/opencode-reasoning-extractor \
  --force datasets/opencode-hydraone datasets/reasoning-hydraone
```

This writes `manifest.json`, `sft/<topic>/sessions.jsonl`,
`sft/<topic>/turns.jsonl`, optional `subagents/<topic>/...`, and
`rl/metadata.jsonl`. Refusing sessions are dropped automatically.

Validate and inspect the loss masks before training:

```sh
./build/qwen-trainer dataset \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --dataset datasets/reasoning-hydraone
```

`--dataset` is repeatable; each occurrence is one exported dataset directory.

---

## 4. Download the base model

```sh
./build/qwen-trainer download --model models/Qwen3.8-27B-UD-Q4_K_M.gguf
```

`download` defaults to `unsloth/Qwen3.8-27B-GGUF` /
`Qwen3.8-27B-UD-Q4_K_M.gguf` with a pinned SHA-256, resumable into a `.part`
file. Any model command also auto-downloads a missing default model unless
`--no-download` is given.

```sh
./build/qwen-trainer inspect --model models/Qwen3.8-27B-UD-Q4_K_M.gguf
```

---

## 5. Abliterate

Remove the refusal direction (Heretic-style MPOA) and stream a type-preserving
GGUF; every tensor keeps its original quantization type.

```sh
./build/qwen-trainer abliterate \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --out   models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf
```

Good/bad prompt sets default to the bundled `source/prompts/` files (Heretic's
sets, `--limit 400`); override with `--good`/`--bad` (text or JSONL). Useful
flags: `--row-norm none|pre|full` (default `full`), `--orthogonalize`,
`--direction-index` (`-1` = per layer), per-component `--attn-*`/`--mlp-*`
weights, `--winsorize`.

### Optional: parameter search

```sh
./build/qwen-trainer search \
  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --trials 25 --sampler tpe \
  --study models/search.jsonl \
  --out   models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf
```

`search` minimizes `refusal_rate + lambda*KL`; `--sampler random|halton|tpe`.
Trials persist to `--study` (JSONL) and resume exactly if re-run. Evaluation
prompt sets default to the bundled test splits (`--eval-limit 100`).

---

## 6. Evaluate

`evaluate` scores a model on three independent axes; at least one input is
required.

```sh
# refusal keyword rate (+ KL vs a base with --base)
./build/qwen-trainer evaluate --model models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf \
  --prompts source/prompts/harmful_behaviors_test.jsonl --max-tokens 32 \
  --base models/Qwen3.8-27B-UD-Q4_K_M.gguf

# corpus perplexity, scored in windows of N tokens
./build/qwen-trainer evaluate --model models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf \
  --perplexity corpus.txt --window 512

# multiple-choice accuracy over JSONL {context, choices, answer}
./build/qwen-trainer evaluate --model models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf \
  --tasks tasks.jsonl --tasks-verbose
```

Note: a thinking model emits `<think>` first, so the first-token KL is a weak
signal; use short generations or later positions. Perplexity computes full
LM-head logits per window, so it is much slower than `run`.

---

## 7. Train (QLoRA SFT)

Two interchangeable paths: the **host autograd reference** (default) and the
fully **device-resident Vulkan** graph (`--device`). Use `--device` on the real
27B; the host path is correct-first and host-bound (very slow at scale).

```sh
# device-resident, resumable, merging only after the final step
./build/qwen-trainer train \
  --model models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf \
  --device \
  --dataset datasets/reasoning-hydraone \
  --dataset datasets/reasoning-thinky \
  --include-subagents \
  --seq-len 1024 \
  --steps 1002 \
  --lr 1e-4 --warmup 100 --max-grad-norm 1.0 --seed 1 \
  --resume models/sft-abliterated.ckpt \
  --save-every 25 \
  --adapter-out models/sft-abliterated-adapters.gob \
  --out models/Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf
```

Key flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dataset` | (required) | extractor output directory; repeat per dataset |
| `--steps` | 50 | optimizer steps |
| `--seq-len` | 512 | max sequence length (front truncation) |
| `--lr` | 1e-4 | peak learning rate |
| `--rank` / `--alpha` | 16 / 32 | LoRA rank / alpha |
| `--accum` | 1 | gradient accumulation steps |
| `--warmup` | 10 | warmup steps |
| `--max-grad-norm` | 1.0 | global-norm clipping (0 = off) |
| `--val-ratio` | 0 | validation split (0 = train on all) |
| `--include-subagents` | false | include the `subagents/` tree |
| `--limit` | 0 | cap the sessions used (0 = all) |
| `--device` | false | device-resident Vulkan graph |
| `--checkpoint` | true | activation checkpointing (device path) |
| `--resume` | - | device checkpoint to load and save to |
| `--save-every` | 0 | checkpoint cadence in steps (0 = only at the end) |
| `--adapter-out` | - | LoRA adapter checkpoint path |
| `--out` | - | merged GGUF output path |

Resuming is exact: the device checkpoint stores the step, adapter parameters,
and AdamW moments, so a long run can be interrupted and continued.

### Offline merge from a checkpoint

```sh
./build/qwen-trainer train \
  --model models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf \
  --eval-checkpoint models/sft-abliterated.ckpt \
  --out models/Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf
```

This merges the checkpoint's adapters into the base on the CPU (no GPU
contention) and exits.

---

## 8. Verify the output

A type-preserving merge leaves the per-tensor type histogram identical to the
base:

```sh
./build/qwen-trainer inspect --model models/Qwen3.8-27B-UD-Q4_K_M.gguf    | sed -n '/^Tensors/,$p' > /tmp/hist-base.txt
./build/qwen-trainer inspect --model models/Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf | sed -n '/^Tensors/,$p' > /tmp/hist-train.txt
diff -u /tmp/hist-base.txt /tmp/hist-train.txt
```

Then generate a sanity sample:

```sh
./build/qwen-trainer run \
  --model models/Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf \
  --prompt "The capital of France is" --generate 8
```

---

## 9. The pipeline driver

`run-pipeline.sh` runs build -> extract -> download -> abliterate -> train ->
verify, and can be customized entirely from the environment:

```sh
STEPS=300 LR=5e-5 ./run-pipeline.sh
```

Relevant variables include `EXTRACTOR`, `TRAINER`, `MODELS`, `DATASETS`, and
`EXTRACTOR_REPO`. It exits non-zero if the merged type histogram differs from
the base.

---

## 10. Monitoring long runs

`train`, `abliterate`, and `search` write a JSON progress file (default
`$XDG_STATE_HOME/qwen-trainer/status.json`, override with `--status-file` or
`$QWEN38_STATUS_FILE`) so a run can be queried from another terminal:

```sh
./build/qwen-trainer status          # percent, done/total, loss, ETA
./build/qwen-trainer status --json   # raw record
```

---

## 11. Troubleshooting

- **GPU not used.** Run `gpu-info`; the backend is chosen only when the
  device-local heap can hold the model plus headroom. `QWEN38_VK_DEVICE`
  overrides device selection by index or name.
- **Sequence length.** T=1024 is the real-27B sweet spot on the 48 GB target
  (~51.6 s/step, ~22.7 GiB peak). T=1536 runs but is much slower; **T=2048
  soft-recovers the Vulkan driver** (reported as a clean submit error, not a
  crash). Keep `--seq-len` at or below 1024 unless you have measured headroom.
- **Memory.** Weights stay quantized (~16 GiB resident). Use `--checkpoint`
  (on by default) and `--accum` to bound activation memory; the device trainer
  reclaims per-layer temporaries and logs peak device memory per step.
- **Disk/RAM during merge.** Merging streams one tensor at a time, so it needs
  bounded memory and roughly the output-file size in free disk.
- **Abliteration quality.** Use the bundled prompt sets and Heretic's limits;
  very small `--limit` values are for smoke tests only. The first-token KL is a
  weak signal for a thinking model.
- **No device present.** `run`, `evaluate`, `abliterate`, and `search` fall
  back to the CPU reference; GPU tests and `--device` training skip/fail
  accordingly.

---

## 12. Quick command reference

```sh
make                                             # build
make check                                       # fmt-check + vet + test
./build/qwen-trainer gpu-info
./build/qwen-trainer download --model models/Qwen3.8-27B-UD-Q4_K_M.gguf
./build/qwen-trainer inspect  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf
./build/qwen-trainer dataset  --model models/Qwen3.8-27B-UD-Q4_K_M.gguf --dataset datasets/reasoning-hydraone
./build/qwen-trainer abliterate --model models/Qwen3.8-27B-UD-Q4_K_M.gguf --out models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf
./build/qwen-trainer search   --model models/Qwen3.8-27B-UD-Q4_K_M.gguf --trials 25 --sampler tpe --study models/search.jsonl --out models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf
./build/qwen-trainer train    --model models/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf --device --dataset datasets/reasoning-hydraone --seq-len 1024 --steps 1002 --resume models/sft-abliterated.ckpt --save-every 25 --out models/Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf
./build/qwen-trainer status
./build/qwen-trainer evaluate --model models/Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf --prompts source/prompts/harmful_behaviors_test.jsonl --max-tokens 32
./build/qwen-trainer run      --model models/Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf --prompt "The capital of France is" --generate 8
```
