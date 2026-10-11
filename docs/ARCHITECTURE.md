# Architecture

Design reference for the Qwen3.8-27B Reasoning Trainer: a pure-Go, cgo-free
toolkit that ablates and fine-tunes the **Qwen3.8 27B** model in its
**Q4_K_M** GGUF quantization. For the end-to-end workflow see
[`TRAINING_GUIDE.md`](TRAINING_GUIDE.md); for open work see [`TASKS.md`](TASKS.md).

---

## 1. Purpose and scope

Two goals, both on the **Qwen3.8 27B** model in **Q4_K_M** GGUF form:

1. **Abliteration** - remove the refusal/safety direction from the weights
   (Heretic-style directional/projected/magnitude-preserving orthogonal
   ablation), producing a standalone, type-preserving quantized GGUF.
2. **SFT on reasoning traces** - QLoRA fine-tune the frozen quantized base on
   the dataset produced by the separate `opencode-reasoning-extractor` project.

Hard constraints:

- **Go 1.27, pure Go, no cgo.** One self-contained binary; no container, no
  messy GPU-passthrough setup on the training machine.
- **Vulkan only** (no ROCm, no CUDA). GPU access loads `libvulkan.so.1`
  dynamically with `purego`; SPIR-V shaders are precompiled and embedded.
- Target hardware: a single Vulkan GPU (48 GB VRAM) with 96 GB system RAM.

The only external dependency is `github.com/ebitengine/purego`.

---

## 2. Confirmed decisions

| Item | Decision |
| --- | --- |
| Compute | Pure-Go native only. No llama.cpp/cgo at runtime or for validation. |
| Inference GPU | Vulkan via `purego`; SPIR-V embedded (`go:embed`). |
| Training | QLoRA/LoRA on a frozen quantized base, F32 adapters. |
| SFT gradients | GatedDeltaNet backward implemented for full-depth QLoRA. |
| Abliteration output | Merged weights, requantized to each modified tensor's original type. |
| Model | `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-UD-Q4_K_M.gguf`. |
| Quant types | Enumerate per tensor at runtime. |
| Scope | Text-only. Ignore mmproj (vision) and the MTP head. |
| Naming | Call the model **Qwen3.8**. On-disk identifiers stay: HF `model_type: qwen3_5`, GGUF arch `qwen35`. |

---

## 3. Target model (Qwen3.8-27B)

A hybrid, multi-modal architecture; only the text part is in scope.

- **GGUF arch `qwen35`; HF architecture `Qwen3_5ForConditionalGeneration`.**
- **65 blocks in the GGUF = 64 trunk layers + 1 MTP layer** (the trailing block
  carries `blk.64.nextn.*` and is full-attention). Only the trunk is run/trained.
- **Hybrid attention**: `full_attention_interval = 4`; pattern 3 linear-attention
  (GatedDeltaNet) then 1 full-attention. Full layers at indices 3, 7, ..., 63
  (**16 full, 48 linear**).
- Text dims: hidden 5120, FFN 17408, 24 query heads / 4 KV heads,
  `head_dim = 256`, `partial_rotary_factor = 0.25` (rotary dim 64),
  `attn_output_gate = true` (swish gate packed into the Q projection),
  `mrope_interleaved = true`, vocab 248320, `rope_theta = 1e7`, context 262144,
  untied embeddings, `rms_norm_eps = 1e-6`.
- GatedDeltaNet: conv1d kernel 4, `linear_num_key_heads = 16` x 128,
  `linear_num_value_heads = 48` x 128, `mamba_ssm_dtype = float32`.
- Vision (27-layer ViT) and MTP are out of scope.

### Abliteration targets

| Component | Tensor name | Present on |
| --- | --- | --- |
| Attention out-projection | `blk.N.attn_output.weight` | full layers |
| Linear-attention out-projection | `blk.N.ssm_out.weight` | linear layers |
| MLP down-projection | `blk.N.ffn_down.weight` | all layers |

### Quantization types in `Qwen3.8-27B-UD-Q4_K_M.gguf`

`q5_K`, `iq4_xs`, `q4_K`, `q6_K`, `iq4_nl`, `q3_K`, `iq3_s`, `q8_0`, `f32`.
Unsloth Dynamic (UD) picks the type per tensor; never assume a uniform type.

### Pinned artifact

- Repo: `unsloth/Qwen3.8-27B-GGUF`
- File: `Qwen3.8-27B-UD-Q4_K_M.gguf`
- Size: `16464440224` bytes
- SHA-256: `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482`
- Default path: `./models/Qwen3.8-27B-UD-Q4_K_M.gguf`

---

## 4. Repository layout

```
qwen-reasoning-trainer/
  Makefile                      build/test/shaders entry points
  run-pipeline.sh               end-to-end pipeline driver
  source/                       Go module root
    go.mod
    cmds/qwen-trainer/main.go   CLI entrypoint
    internal/
      cli/                      argument parsing + command implementations
      hf/                       Hugging Face downloader (range resume + sha256)
      gguf/                     GGUF reader, writer, streaming writer
      quant/                    GGML type table + dequant + quant encoders
      modelcfg/                 Qwen3.8 hyperparameters + inspect report
      tokenizer/                byte-level BPE encoder/decoder + chat template
      compute/                  Backend interface, host Tensor, CPU oracle ops
        cpu/                    CPU Backend
        vulkan/                 Vulkan Backend (+ embedded spirv/*.spv)
      model/qwen38/             config, weights, forward pass, GGUF rewrite
      autograd/                 reference backward ops + finite-difference checker
      train/                    LoRA, AdamW/schedule, host + device trainers, merge
      abliterate/               residual collection, directions, search, weights
      evaluate/                 refusal rate + KL, perplexity, task accuracy
      dataset/                  extractor JSONL reader, chat template, loss masks
      status/                   JSON progress file for long-running jobs
    shaders/                    GLSL compute shaders (compiled to SPIR-V)
    prompts/                    embedded good/bad prompt sets (go:embed)
    e2e-tests/                  opt-in external cross-checks, one isolated program
      verify-ref/               NumPy forward pass
      verify-data/              Jinja2 template + regex pre-tokenizer
      verify-quants/            llama.cpp ggml-quants.c decoders + encoders
  models/                       downloaded/abliterated GGUF (gitignored)
  datasets/                     extractor output (gitignored)
  docs/                         this document, the training guide, the task list
```

### Package dependency sketch

```
cli --> hf, gguf, modelcfg, tokenizer, compute, compute/vulkan,
        model/qwen38, abliterate, evaluate, dataset, train, status
model/qwen38 --> compute, gguf, modelcfg, quant
abliterate   --> compute, model/qwen38, quant
evaluate     --> model/qwen38, tokenizer
dataset      --> tokenizer
autograd     --> compute (reference backward ops; oracle for the Vulkan trainer)
train        --> autograd, compute, model/qwen38, quant, gguf
compute/vulkan --> compute, quant, purego, embedded spirv
compute/cpu    --> compute, quant
gguf         --> quant
quant        --> (std lib only)
```

---

## 5. GGUF layer (`source/internal/gguf`)

GGUF v3, byte-compatible with llama.cpp.

- `gguf.go` - constants, `ValueType`, `Value`, `KV`, `TensorInfo`.
- `reader.go` - `Open`/`ReadFrom`; parses header, metadata KVs (all value
  types incl. arrays), tensor infos, `general.alignment`; computes `dataStart`,
  validates bounds. Exposes `Get`/`Uint`/`Int`/`Float`/`Str`/`Strings`,
  `Tensor`, `ReadTensor`, `ReadTensorF32`, `CopyTensor` (streams to an `io.Writer`).
- `writer.go` - `Encode`/`WriteFile` (in-memory round-trip).
- `stream.go` - `WriteStream`/`WriteStreamToFile`: writes header/metadata/tensor
  infos, then **streams** each tensor via a callback, so a 15 GiB model can be
  rewritten without buffering it.

Gotchas: magic `0x46554747`; version 3; tensor dims stored `ne[0]` first (GGML
order); `TensorInfo.Dims[0]` is contiguous; offsets are relative to the aligned
data section; metadata arrays cannot be nested. Always use `CopyTensor` +
`WriteStream` for full-model rewrites.

---

## 6. Quantization layer (`source/internal/quant`)

Numerical core for reading/writing GGML tensors.

- `types.go` - `Type` (`ggml_type` values) plus a `blockSize`/`typeSize` table
  matching `ggml-common.h` static asserts; helpers `BlockSize`, `TypeSize`,
  `Size`, `RowSize`, `IsQuantized`, `String`.
- `dequant.go` - `Dequant`/`DequantTo` to `[]float32`, exact ports of
  `ggml-quants.c`: `f32`, `f16`, `bf16`, `q4_0`, `q5_0`, `q8_0`, `q3_K`, `q4_K`,
  `q5_K`, `q6_K`, `iq4_nl`, `iq4_xs`, `iq3_s` (grids in `iq3s_grid.go`).
- `quantize.go` - `Quantize`/`QuantizeTo` and `CanQuantize` for the same set.
  The K-quant/IQ encoders are **byte-identical** to llama.cpp's
  `quantize_row_*_ref` (shared helpers in `kquant.go`, IQ3_S grid search in
  `iq3s_quant.go`); `make verify-quants` asserts exact equality.
- `fp.go` - `F16ToF32`/`F32ToF16` (port of `ggml_compute_fp32_to_fp16`) and BF16.

### The GGML tensor layout (critical)

- `Dims` are in **GGML order**: `Dims[0]` contiguous; element
  `(i0,i1,...)` at `i0 + i1*d0 + i2*d0*d1 + ...`.
- A linear weight is `[in, out]`: `Dims[0] = in` (reduction dim), `Dims[1] = out`.
- `MatMul(a[K,N], b[K,M])` is `[N,M]` with `(n,m)` at `n + m*N` (GGML layout),
  **not** row-major. This holds across CPU ops, shaders, and the model.
- NumPy reference tensors use `reshape(order="F")`.

### Q3_K gotcha

The `hmask` high-bit index is `n*4 + j` (n = 128-group, j = sub-index) while the
qs nibble shift is `2*j` and the qs byte index is `n*32 + within`. Encoding with
`1<<j` silently corrupts the sign of the second 128-group.

---

## 7. Compute layer

### 7.1 Host `Tensor` and CPU oracle (`source/internal/compute`)

`Tensor` is a dense `[]float32` with GGML-order `Dims` and a `quant.Type`.
`cpu/backend.go` implements `compute.Backend` on host memory. The free
functions in `ops.go`/`ops_ext.go` are the **numerical oracle** used by GPU
parity tests and when no backend is set: `MatMul` (float64 accumulation),
`Add`/`Sub`/`Mul`/`Scale`, `RMSNorm`, `Softmax`, `CrossEntropy`, `Silu`,
`GetRows`, `RoPENeoX`, `L2Norm`, `Attention` (GQA + causal, supports
`nQuery != nKV`), `SSMConv` (causal depthwise conv1d), `GatedDeltaNet` (gated
delta rule, transposed state layout).

### 7.2 Backend interface (`compute/backend.go`)

```go
type Backend interface {
    Capabilities() Capabilities
    Alloc(dims []int, t quant.Type) (Buffer, error)
    Upload(t *Tensor) (Buffer, error)
    Download(b Buffer) (*Tensor, error)
    Free(b Buffer)
    Sync() error
    Close() error

    Unary(op UnaryOp, a Buffer) (Buffer, error)
    Binary(op BinaryOp, a, b Buffer) (Buffer, error)
    Scale(a Buffer, s float32) (Buffer, error)
    MatMul(a, b Buffer) (Buffer, error)
    RMSNorm(a, w Buffer, eps float32) (Buffer, error)
    Softmax(a Buffer) (Buffer, error)
    GetRows(table Buffer, indices []int32) (Buffer, error)
    RoPE(a Buffer, positions []int32, theta float64, nDims int) (Buffer, error)
    Attention(q, k, v Buffer, nHead, nHeadKV int, scale float32, causal bool) (Buffer, error)
    SSMConv(sx, c Buffer) (Buffer, error)
    GatedDeltaNet(q, k, v, g, beta, state Buffer) (out, newState Buffer, err error)
    Copy(dst, src Buffer) error

    UploadWeight(t quant.Type, raw []byte, dims []int) (Buffer, error)
    DequantWeight(w Buffer) (Buffer, error)
    MatMulWeight(w Buffer, x Buffer) (Buffer, error)
}
```

Everything is out-of-place (each op returns a new `Buffer`), so the CPU and
Vulkan backends share the same tests. Ops are queued and executed by the
backend; `Download`/`Sync` flush.

### 7.3 Vulkan backend (`source/internal/compute/vulkan`)

No cgo. `vulkan.go` loads `libvulkan.so.1` with `purego.Dlopen`, resolves
`vkGetInstanceProcAddr`, and defines the minimal Vulkan structs/entry points.

- **Memory**: allocation prefers device-local memory. Integrated GPUs with a
  device-local host-visible coherent heap are mapped directly; otherwise
  (discrete GPUs) transfers go through a host-visible staging buffer recorded
  into the same command batch (`writeBytes`/`readBytes`). `SetForceStaging`
  (tests) forces the staging path. `Capabilities.MemoryBytes` reports the
  device-local heap total, which drives automatic GPU selection.
- **Pipelines**: one descriptor-set layout with 16 storage bindings
  (`maxBindings`) and a 128-byte push-constant range; pipelines are created
  lazily and cached.
- **Batching**: `dispatch`/`copyRows` **record** into a single command buffer;
  `flush` (from `Download`/`Sync`) submits and waits once. Buffers freed while
  recording are deferred and destroyed after the flush.
- **Weights**: `UploadWeight` stores raw quantized bytes (padded to 4 bytes).
  `MatMulWeight` single-token decode (`m == 1`) uses a **fused
  dequantize+GEMV** shader; otherwise the fused quantized weight GEMMs
  (`matmul_wq_<type>`/`matmul_t_wq_<type>`) decode the weight in registers
  inside the tile loader (shared `quant_decode.glsl`), so the weight is read
  once and never materialized as float32.

### 7.4 Shaders (`source/shaders/*.comp` -> embedded `spirv/*.spv`)

- Elementwise/view: `unary`, `binary`, `copy_rows`, `qg_split`/`qg_merge`,
  `conv_input(_back)`, `gather_heads(_back)`, `repeat_heads(_back)`.
- Linear algebra: `matmul` (64x64 tile, 4x4 register micro-tile), `matmul_t`
  (`dX = W^T dY`), `matmul_wg` (`dW = x dOut^T`), fused quantized
  `matmul_wq_<type>`/`matmul_t_wq_<type>`, `rms_norm(_back)`, `silu_back`,
  `softmax`/`soft_max_back`, `cross_entropy`, `l2_norm(_back)`, `rope`,
  `get_rows`, `sum_squares`.
- Attention: `attention` (one workgroup per (head, query), 64 lanes,
  online-softmax over 1024-key tiles), `attention_naive` (fallback for
  `headDim > 256`; non-square `nQ != nKV` falls back to the host oracle),
  `attn_back_dq_tile`/`attn_back_dkv_tile` (+ `attn_back_dq`/`attn_back_dkv`).
- Recurrent: `ssm_conv(_back_sx/_back_c)`, `gated_delta_net`, `gdn_back`
  (atomic fallback, needs `VK_EXT_shader_atomic_float`), and the chunked
  `gdn_chunk_{cumsum,ab,fwd,bwd,bwd_local}`.
- Dequantize/GEMV: `dequant_<type>`, `gemv_<type>` for all nine model types.
- Optimizer: `opt_step_adamw`.

The `.spv` files are committed and embedded (`go:embed`); end users only need
the Go toolchain. Rebuild with `make shaders` (needs `glslc` and `spirv-val`).

### 7.5 Vulkan/dispatch limits (critical)

- **`maxComputeWorkGroupCount` (typically 65535 per dim).** An over-limit
  dispatch is invalid and can hang the GPU (fence never signals; the process
  blocks in `drm_syncobj_array_wait_timeout`). Every size-dependent dispatch
  must **chunk**: dequant/GEMV/matmul take a base offset via push constant;
  elementwise/`get_rows`/`rope`/head-view ops go through `Backend.dispatch1D`;
  `attention` is one workgroup per query. A shader's group count is its
  workgroup count, not its element count.
- **`maxStorageBufferRange`.** Never bind a full large weight as float32 (the
  ~5 GiB output-projection scratch exceeds the limit). Use the fused weight
  GEMMs/GEMV or bounded row blocks.
- **GLSL reserved words** (`out`, `half`, ...) cannot be identifiers.
- **Word reads**: quantized data is read as `uint` words via
  `(data[off>>2] >> ((off&3)*8)) & 0xFF`; buffers are padded to 4 bytes.
- **`vkCall` must be `//go:uintptrescapes`.** It passes
  `uintptr(unsafe.Pointer(&local))` to Vulkan; without the directive a stack
  growth during the call leaves the driver a stale pointer (rare driver-side
  SIGSEGV only on deep chains).
- **Staged transfers need pipeline barriers.** `writeBytes`/`readBytes`/`Copy`
  record `vkCmdCopyBuffer` in the same batch as compute; a transfer<->compute
  `vkCmdPipelineBarrier` must bracket the copy or the driver may read stale data.
- **Alignment**: push-constant fields 4-byte aligned and in the same order as
  the Go `push(...)`.
- **A failed submit must clear the recording flag.** `flush` resets `recording`
  before returning on failure; otherwise a later `Close`->`flush` acts on an
  already-submitted command buffer and SIGSEGVs in the driver. Long sequences
  (T≈2048 for the real 27B) still soft-recover; `--seq-len` is capped below that.

---

## 8. Model layer (`source/internal/model/qwen38`)

- `qwen38.go` - `Config` (derived dims) from `modelcfg.Config` via
  `FromModelcfg`, plus `Model` and the `Model.mm` matmul dispatcher.
- `weights.go` / `weight.go` - `LayerWeights`, `Weights`, `LoadFromGGUF`,
  `NewRandom`; `Weight` keeps raw on-disk quantized bytes + type + dims with
  `Dequant`, `DequantRows`/`GetRows`, and blocked `MatMul`. Large matrices stay
  quantized in RAM/VRAM (~16 GiB resident, not ~108 GiB f32).
- `forward.go` - forward on host tensors with matmuls through `Model.mm`. Per
  trunk layer: RMSNorm -> full-attention (fused Q+gate, QK RMSNorm, partial
  GPT-NeoX RoPE, causal GQA, sigmoid-gate, out-projection) **or** linear
  attention (wqkv/gate, beta/alpha gates, causal conv1d + SiLU, GDN L2-norm,
  head repeat, gated delta rule, gated RMSNorm, out-projection) -> residual,
  post-attention norm, SwiGLU FFN, residual. Then final RMSNorm and LM head.
  `ForwardOptions` controls `RecordHidden` (abliteration), `State` (KV +
  recurrent), `Positions`, `LastTokenOnly`.
- `write.go` - `Weights.Overrides()` (abliterable tensors by GGUF name) and
  `RewriteGGUF`/`RewriteGGUFStream`, which stream a copy of the base replacing
  those tensors and carrying everything else (including MTP) unchanged.

Notes:
- mRoPE with identical positions on all axes equals ordinary NeoX RoPE, so
  text-only inference uses `RoPENeoX` with the rotary dim (64). The `rope`
  shader must copy the non-rotary head dims `[nDims, headDim)`; the model's
  `nDims=64 < headDim=256` makes this a real region.
- The GatedDeltaNet state is stored transposed (`M[j*S_v+i] = S[i][j]`) to match
  llama.cpp.

---

## 9. Abliteration (`source/internal/abliterate`)

A clean-room port of Heretic's technique (AGPL-3.0; do not copy its code):

1. **`CollectResiduals`** - capture the last-token residual stream for every
   captured hidden state; mean over prompts; optional winsorization.
2. **`Directions`** - `bad_means - good_means`, per-layer normalized, optionally
   orthogonalized against the good direction and renormalized. `pickDirection`
   supports per-layer or a global index with interpolation.
3. **`Ablate`** - per layer/component, compute the ablation weight from the
   component's kernel (`MaxWeight`, `MaxWeightPosition`, `MinWeight`,
   `MinWeightDistance`) and transform the matrix: `none`
   (`W <- W - w * v (v^T W)`), `pre` (row-normalize/ablate/restore norms), or
   `full`/MPOA (row-normalize, ablate, renormalize, multiply back original norms,
   default). Requantized to the original type when an encoder exists.

`search` (`search.go`, `sampler.go`, `tpe.go`, `study.go`) adds a
dependency-free optimizer over Heretic's sampled space: `SearchSpace` ->
`TrialParams`, a `Sampler` interface (`RandomSampler`, `HaltonSampler`, compact
`TPESampler`), JSONL study checkpoints with exact resume, and a scalar objective
`refusal_rate + lambda*KL`. Trials snapshot the target tensors' raw bytes and
restore them before each trial; `Model.InvalidateDevice` refreshes only the
changed device buffers; `evaluate.BaselineLogits` caches the KL baseline.

Prompt sets are bundled pre-converted in `source/prompts/`
(`harmless_alpaca_{train,test}.jsonl`, `harmful_behaviors_{train,test}.jsonl`),
embedded with `go:embed`, and are the defaults for `abliterate`/`search` with
Heretic's limits (`--limit 400`, `--eval-limit 100`). The first-token KL is a
weak signal for a thinking model (it emits `<think>` first).

---

## 10. Evaluation (`source/internal/evaluate`)

- `RefusalRate` - greedy-generate up to N tokens per prompt and count refusal
  keywords (`DefaultKeywords`).
- `KLDivergence` - mean KL(base || candidate) of last-token next-token
  distributions over a prompt set.
- `PerplexityWindowed` - length-normalized corpus perplexity scored in
  non-overlapping windows (`--window 0` = whole input), bounding peak logits
  memory. Much slower per token than `run` (full LM-head logits per window).
- `LoadTasks`/`MultipleChoice` - JSONL `{context, choices, answer}` benchmark;
  each choice scored by summed (optionally length-normalized) log-likelihood.
- `Generate` - shared greedy decoder.

---

## 11. Tokenizer, downloader, dataset

### `source/internal/tokenizer`

- `FromGGUF` reads `tokenizer.ggml.tokens`, `.merges`, `.token_type`, `.pre`,
  and BOS/EOS ids.
- `Decode` uses the GPT-2 byte-level alphabet. `Encode` splits on added tokens
  (GGML types 2/3/4) and emits their ids verbatim; the rest is byte-level BPE.
  For `pre=qwen35` the pre-tokenization is a direct port of llama.cpp's
  `unicode_regex_split_custom_qwen35` (`pretok.go`); other `pre` values use an
  RE2 approximation without the trailing-space lookahead.
- `ChatPrompt` builds Qwen's `<|im_start|>role ... <|im_end|>` format plus the
  assistant generation prefix.

### `source/internal/hf`

A minimal stdlib Hugging Face client: URL resolve, redirects, HTTP `Range`
resume into `<file>.part`, atomic rename, SHA-256 verification from the tree
API, `HF_TOKEN`, `HF_ENDPOINT`.

### `source/internal/dataset`

Turns extractor JSONL into tokenized, loss-masked examples (the extractor output
is the canonical dataset; no intermediate on-disk format).

- `schema.go`/`reader.go` - mirrors the extractor records and streams
  `sft/<topic>/sessions.jsonl` (and optional `subagents/`), deduplicating
  sessions across topic files. `OpenAll` takes several roots (the CLI's
  repeatable `--dataset`), de-duplicates across them, and unions their tool
  registries.
- `template.go` - segment-based port of the model's GGUF
  `tokenizer.chat_template`: merged system/developer messages,
  `reasoning_effort`, `preserve_thinking`, assistant `<think>` blocks, and
  tool-call XML (JSON-string `arguments` parsed to ordered pairs). Emits
  `[]Segment{Text, Loss}`; `EncodeSegments` produces ids plus a per-token mask
  (assistant blocks `Loss=true`).
- `sample.go` - builds one truncated sequence per session (front truncation
  keeps the target), drops empty examples, deterministic shuffle and hash-based
  train/val split, streaming `Iter`. Packing is deferred (would need
  block-diagonal attention).

The template port follows the model's own GGUF `tokenizer.chat_template`, **not**
`references/llama_cpp/models/templates/Qwen3.5-4B.jinja` (a different revision).
Tool schemas are not persisted by opencode, so the extractor ships a static
registry (`manifest.tools`/`meta.tools`) that `Dataset.Examples`/`Iter`
auto-load.

---

## 12. Training (`source/internal/autograd`, `source/internal/train`)

Two interchangeable paths: a host autograd reference and a fully
device-resident Vulkan graph. `train` uses the host path by default and the
device path with `--device`.

### 12.1 Host reference (`source/internal/autograd`)

- Reference backward ops (`backward.go`, `backward_ops.go`,
  `attention_backward.go`) validated by central finite differences (`GradCheck`);
  the novel `GatedDeltaNetBackward` plus `GatedDeltaNetBackwardChunked` (bounded
  memory) and `GatedDeltaNetChunkedForward`/`GatedDeltaNetChunked`
  (`gdn_chunked.go`, the FLA-style matrix oracle).
- A small host autograd graph (`Value`, `VMatMul`, `VRMSNorm`, ...) in
  `value.go`/`value_ops.go`/`value_weight.go` (`VMatMulWeight` freezes a
  quantized weight, only the activation gradient flows).

### 12.2 Host trainer (`source/internal/train`)

- `lora.go` - adapters ([K,r] and [r,N], zero-init B), apply, merge,
  `MergeToQuant` (parallel rank-r row update, parallel codecs).
- `optim.go` - AdamW, cosine schedule with warmup, global-norm clipping.
- `qwen38.go` - trainable Qwen3.8 graph (LoRA on attention q/k/v/o or
  linear fused qkv/gate/ssm_out and MLP gate/up/down); `Weight.MatMulTranspose`
  gives `dX = W^T dY` in bounded row blocks.
- `trainer.go` - the loop (accumulation, clipping, schedule) and gob
  checkpointing.
- `merge.go` - `Overrides`/`WriteMerged` reuse `qwen38.RewriteGGUFStream` with a
  lazy override source, materializing one merged tensor at a time.

### 12.3 Device-resident trainer

The same graph runs entirely on the Vulkan backend with one submit per step:

- `device.go` - resident LoRA-MLP block.
- `device_attention.go` - resident full-attention block (fused Q+gate split,
  RoPE, QK-norm, attention backward).
- `device_gdn.go` - resident GatedDeltaNet block (conv, L2-norm, head
  gather/repeat, gated delta rule forward/backward, gated RMSNorm).
- `device_model.go` - the full 3:1 layer stack + frozen quantized embedding
  gather (`GetRowsWeight`) + final norm + frozen LM head + fused on-device
  softmax cross-entropy; activation checkpointing stores layer inputs and
  recomputes in backward; per-layer `BeginScope`/`EndScope` bound peak memory.
- `device_optim.go` - `DeviceAdamW` and the on-device optimizer.
- `checkpoint.go` - resumable device checkpoint (step + adapter params + AdamW
  moments).
- `device_ops.go` - device helpers (`MatMulWeight`, `MatMulWeightTranspose`,
  `MatMulWeightGrad`, GLU, scopes, sum-squares clipping).

The device GatedDeltaNet backward uses the matrix-based chunked kernels
(`gdn_chunk_*.comp` + `Backend.GatedDeltaNetChunkedBackward`), which removes the
atomic kernel's sequence-length cap; the atomic `gdn_back` remains the
fallback. Device gradients match the host autograd oracle, and a real-27B
resident step runs at T=1024 on the target GPU. The remaining throughput lever
is the `O(T*S_v^2)` GDN row reduction (see [`TASKS.md`](TASKS.md)).

---

## 13. CLI reference

```
qwen-trainer <command> [flags]
```

The command is always the first argument; every flag (including `--model`)
follows it and is parsed by the command's own flag set.

Common flags (model commands; `--model` required): `--model/-m`, `--models-dir`,
`--repo`/`--file`, `--status-file`, `--no-download`, `--log-level`.

| Command | Purpose |
| --- | --- |
| `download` | Fetch the model GGUF (resume + checksum). |
| `inspect` | Arch, hyperparameters, layer types, tensor type histogram. |
| `gpu-info` | Vulkan device name, capabilities, device-local memory. |
| `run` | Load and generate (`--prompt/--system`, `--tokens`, `--generate N`). |
| `abliterate` | Ablate refusals and write a GGUF (`--good/--bad`, `--out`, `--row-norm`, `--direction-index`, per-component `--attn-*`/`--mlp-*`, `--limit`, `--winsorize`). |
| `search` | Automated abliteration search (`--trials`, `--sampler`, `--lambda`, `--seed`, `--study`, `--out`). |
| `evaluate` | `--prompts`/`--base`; `--perplexity`/`--window`; `--tasks`/`--tasks-normalize`/`--tasks-verbose`. |
| `dataset` | Tokenize/validate extractor output and report loss-mask stats (`--dataset`, repeatable). |
| `train` | QLoRA SFT (`--dataset` repeatable, `--steps`, `--seq-len`, `--lr`, `--rank`, `--alpha`, `--accum`, `--max-grad-norm`, `--adapter-out`, `--out`, `--val-ratio`, `--include-subagents`, `--limit`; `--device`, `--checkpoint`, `--resume`, `--save-every`, `--eval-checkpoint`). |
| `status` | Progress of a running/last `train`/`abliterate`/`search` job (`--file`, `--json`). |

`train`, `abliterate`, and `search` write a JSON progress file (default
`$XDG_STATE_HOME/qwen-trainer/status.json`, override with `--status-file` or
`$QWEN38_STATUS_FILE`) queryable with `qwen-trainer status`.

`run`, `abliterate`, `search`, `evaluate`, and `train --device` select the
backend automatically: Vulkan when a device has enough device-local memory for
the model, otherwise the CPU reference.

---

## 14. Build and tooling

All Go tooling runs against the `source/` module (`go -C source ...`).

| Target | Action |
| --- | --- |
| `make` / `make build` | compile shaders, then build `build/qwen-trainer` |
| `make shaders` | compile each `source/shaders/*.comp` to embedded `.spv` + `spirv-val` |
| `make test` | `go -C source test ./...` |
| `make vet` / `make fmt` / `make fmt-check` / `make tidy` | the obvious |
| `make check` | `fmt-check` + `vet` + `test` |
| `make verify-ref` | NumPy forward-pass cross-check |
| `make verify-data` | Jinja2 chat-template + `regex` pre-tokenizer cross-checks |
| `make verify-quants` | diff decoders + encoders against llama.cpp `ggml-quants.c` (needs `cc`) |
| `make e2e-venv` | create `source/e2e-tests/.venv` (gitignored) |
| `make inspect` / `make gpu-info` | build then run the command |
| `make clean` | remove `build/` and generated `.spv` |

Toolchain: Go 1.27; `glslc` + `spirv-val` only to rebuild shaders; `python3`
venv only for `verify-ref`/`verify-data`; `cc` only for `verify-quants`.
The committed `.spv` files mean a plain `go build`/`go test` needs neither
`glslc` nor Vulkan (GPU tests skip automatically when no device is present).

---

## 15. Testing and verification

- `go test ./...` is pure Go; Vulkan tests skip without a device. The
  `source/e2e-tests/<name>` programs are separate `main` packages, never run by
  `go test`.
- **GGUF/quant**: round-trip and reference-value tests; sizes checked against
  `ggml-common.h`; Go native fuzz targets (`gguf/fuzz_test.go`,
  `quant/fuzz_test.go`) with minimized crashers under `testdata/fuzz/`.
- **Compute**: CPU ops have hand-computed expected values; Vulkan is validated
  op-by-op against the CPU oracle. Backward kernels add an independent **GPU
  finite-difference** check. `real_bench_test.go` holds real-shape
  micro-benchmarks and large-shape GEMM/attention parity tests.
- **Model**: shape/determinism, incremental decode == full prompt, and an
  independent **NumPy reference** (`make verify-ref`, ~2.5e-8).
- **Quant codecs**: `make verify-quants` compiles llama.cpp's own
  `ggml-quants.c` and diffs every decoder (<1e-6) and asserts every encoder is
  byte-identical. The quantized shaders are validated transitively via
  `quant.Dequant`.
- **Dataset**: extractor-writer fixture, golden chat-template renders, Jinja2 and
  `regex` cross-checks (`make verify-data`).
- **Abliteration**: direction parity with Heretic, orthogonality, row-norm
  preservation, type preservation, GGUF rewrite round-trip.
- **Real model**: `inspect` parses the 16 GiB file; `run` generates on CPU and
  GPU; `abliterate` produces a valid type-preserving GGUF; the resident device
  training step is opt-in via `QWEN38_REAL_DEVICE=1` (`QWEN38_REAL_T` sets the
  sequence length) and checks finite loss and non-zero adapter gradients.
