#!/usr/bin/env bash
#
# End-to-end pipeline for the Qwen3.8-27B reasoning trainer:
#
#   download -> abliterate -> extract reasoning traces -> SFT -> verify types
#
# Run from anywhere; it operates on the repository it lives in. Override any
# variable from the environment, e.g.:
#
#   STEPS=300 LR=5e-5 ./run-pipeline.sh
#
# The three model artifacts produced:
#
#   Qwen3.8-27B-UD-Q4_K_M.gguf                       (downloaded base)
#   Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf           (abliterated base)
#   Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf  (SFT on reasoning traces)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

EXTRACTOR_REPO="${EXTRACTOR_REPO:-$REPO_ROOT/../opencode-reasoning-extractor}"

# --- configuration ----------------------------------------------------------
EXTRACTOR="${EXTRACTOR:-$EXTRACTOR_REPO/build/opencode-reasoning-extractor}"
TRAINER="${TRAINER:-$REPO_ROOT/build/qwen-trainer}"

MODELS="${MODELS:-$REPO_ROOT/models}"
DATASETS="${DATASETS:-$REPO_ROOT/datasets}"

MODELS_BASE="$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf"
MODELS_ABLATED="$MODELS/Qwen3.8-27B-abliterated-UD-Q4_K_M.gguf"
MODELS_TRAINED="$MODELS/Qwen3.8-27B-abliterated-DeepSeek-UD-Q4_K_M.gguf"

# --- 0. build the trainer ---------------------------------------------------
# Compiles the shaders and builds build/qwen-trainer (needs glslc/spirv-val).
make -C "$REPO_ROOT";

# --- 1. extract the reasoning-trace dataset from opencode -------------------
# Builds the extractor, then turns the opencode session database into
# topic-partitioned JSONL (sft/<topic>/sessions.jsonl, subagents/, rl/,
# manifest.json). Refusing sessions are dropped automatically.
mkdir -p "$(dirname "$EXTRACTOR")";
( cd "$EXTRACTOR_REPO/source" && go build -o "$EXTRACTOR" ./cmd/opencode-reasoning-extractor )

"$EXTRACTOR" --force "$DATASETS/opencode-hydraone" "$DATASETS/reasoning-hydraone"
"$EXTRACTOR" --force "$DATASETS/opencode-thinky" "$DATASETS/reasoning-thinky"

# --- 2. download the Qwen3.8-27B UD Q4_K_M base -----------------------------
# Defaults to unsloth/Qwen3.8-27B-GGUF / Qwen3.8-27B-UD-Q4_K_M.gguf (pinned
# sha256). --model is the output file the GGUF is saved as.
"$TRAINER" download --model "$MODELS_BASE";

# --- 3. abliterate the base -------------------------------------------------
# Removes the refusal direction (Heretic-style MPOA) and streams a
# type-preserving GGUF: every tensor keeps its original quantization type.
"$TRAINER" abliterate --model "$MODELS_BASE" --out "$MODELS_ABLATED";

# --- 4. SFT the abliterated model on the reasoning traces -------------------
# QLoRA (r=16/alpha=32) on the frozen quantized base, fully device-resident on
# the Vulkan GPU. The run is resumable: re-running resumes from --resume, and
# the merged GGUF is only written after the final step.
"$TRAINER" train \
	--model "$MODELS_ABLATED" \
	--device \
	--dataset "$DATASETS/reasoning-hydraone" \
	--dataset "$DATASETS/reasoning-thinky" \
	--include-subagents \
	--seq-len "1024" \
	--steps "1002" \
	--lr "1e-4" \
	--warmup "100" \
	--max-grad-norm 1.0 \
	--seed 1 \
	--resume "$MODELS/sft-abliterated.ckpt" \
	--save-every "25" \
	--adapter-out "$MODELS/sft-abliterated-adapters.gob" \
	--out "$MODELS_TRAINED";

# --- 5. verify the tensor type histogram is unchanged -----------------------
# `inspect` prints the per-type tensor histogram; a type-preserving merge must
# leave it identical to the downloaded base.
"$TRAINER" inspect --model "$MODELS_BASE"    | sed -n '/^Tensors/,$p' > /tmp/qwen-histogram-base.txt
"$TRAINER" inspect --model "$MODELS_TRAINED" | sed -n '/^Tensors/,$p' > /tmp/qwen-histogram-trained.txt

if diff -u /tmp/qwen-histogram-base.txt /tmp/qwen-histogram-trained.txt; then
  echo "OK: tensor type histogram is unchanged. Training successful!"
else
  echo "FAIL: tensor type histogram changed! Training unsuccessful!" >&2
  exit 1
fi
