# Tasks

Remaining work only. Everything else (GGUF/quant stack, CPU + Vulkan backends,
Qwen3.8 forward inference, abliteration + search, evaluation, dataset pipeline,
host and device-resident QLoRA training, streaming merge, and end-to-end
abliteration + SFT on the real 27B) is implemented and verified; see
[`ARCHITECTURE.md`](ARCHITECTURE.md) and [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md).

Status: `[ ]` not started, `[~]` in progress, `[x]` done.

---

### T1 - GatedDeltaNet chunked-backward throughput `[ ]`

- **Goal:** remove the `O(T*S_v^2)` per-token/per-row reductions that dominate
  device training time in the chunked GatedDeltaNet backward.
- **Starting point:** `source/shaders/gdn_chunk_{cumsum,ab,fwd,bwd,bwd_local}.comp`
  driven by `Backend.GatedDeltaNetChunkedBackward` and wired into
  `source/internal/train/device_gdn.go`. The matrix-based oracle is
  `source/internal/autograd/gdn_chunked.go`.
- **Approach:** pack/vectorize the row reductions. A head's value rows are
  already packed into a 128-lane workgroup; the remaining cost is the row-wise
  reduction itself (tiled or warp-cooperative). Keep the shared packed scratch.
- **Acceptance:**
  - Device gradients still match `autograd.GatedDeltaNetChunked` / the CPU
    oracle for scalar and KDA gates (existing parity tests stay green).
  - Measured speedup on the target GPU for the real shapes
    (h=48, S_v=128, T=1024), recorded in the task.
  - The atomic `gdn_back` remains the correctness fallback.

---

### T2 - Real-27B tuning on the 48 GB target `[ ]`

- **Goal:** raise the usable sequence length and step throughput beyond the
  current T=1024 sweet spot without destabilizing the driver.
- **Starting point:** `TestRealDeviceStep` (`source/internal/train/device_real_test.go`,
  opt-in via `QWEN38_REAL_DEVICE=1`, `QWEN38_REAL_T` sets T). Current target
  numbers: T=1024 ~51.6 s/step, ~22.7 GiB peak; T=1536 runs but is very slow;
  **T=2048 soft-recovers the Vulkan driver.**
- **Approach:** investigate the T=2048 submission-size soft recovery (submission
  splitting, mid-layer syncs, or smaller per-submit batches); profile forward vs
  backward vs optimizer with `Stats.DeviceBytes` / RSS; confirm memory headroom
  on the 48 GB card.
- **Acceptance:**
  - A documented, stable maximum `--seq-len` above 1024 on the target GPU (or a
    clear, reproducible explanation of the hardware/driver limit).
  - Finite loss and finite, non-zero adapter gradients at the chosen length.
  - Memory measurement recorded.

---

### T3 - Optional RL stage (DPO/GRPO) `[ ]`

- **Goal:** optional preference/reward optimization on top of SFT, using the
  extractor's `rl/metadata.jsonl`.
- **Starting point:** no `source/internal/rl` package exists yet; the dataset
  reader already has a `WalkRL` path (`source/internal/dataset/reader.go`) and
  the device trainer / optimizer machinery (`source/internal/train/device_*`)
  is reusable.
- **Approach:** implement DPO or GRPO as a new package fed by `rl/metadata.jsonl`,
  reusing the frozen quantized base, LoRA adapters, and the device graph;
  evaluate with the existing `evaluate` command.
- **Acceptance:**
  - A `train`-style command (or flag) drives a short RL run end to end.
  - Loss/reward and gradient checks pass on a small model before scaling to the
    27B.
  - Documented quality comparison (refusal rate / KL / perplexity) against the
    SFT checkpoint.
