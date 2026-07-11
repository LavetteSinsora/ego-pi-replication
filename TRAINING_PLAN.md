# ego2g1 training plan — pi05 fine-tune with custom loader, norm, multi-t, train-RTC

Status: **implemented 2026-07-11** (fork commits 320086c + a8184a6: src/ reverted
to stock upstream/main, `ego2g1/` package added with 26 tests passing; the offline
`validate.py` eval entrypoint and the E003 multi-t path remain unimplemented).
Companion to [OPENPI_EDITS.md](OPENPI_EDITS.md)
(E001 floored per-slot norm, E002 per-token timestep, E003 multi-timestep training).
This document adds the two things E00x entries don't cover: (a) the concrete
file-by-file layout of the self-contained training package, and (b) the
train-time vs deploy-time classification of every modification.

Prior decisions this plan builds on (do not silently change):
- π0.5 (`pi05_base`) full fine-tune.
- Self-contained `ego2g1/` package at the **openpi fork top level**
  (`third_party/openpi/ego2g1/`, tracked on branch `ego2g1-data`), imports
  openpi as a library, does **not** import `data_extraction` — it carries its
  own copy of the loader math, pinned by the existing equivalence test.
- Dataset: LeRobot dataset written by `data_extraction` (30-dim state/action:
  2 hands × [flange pos 3 + rot6d 6 + BrainCo cmd 6]), anchor-at-obs-tick,
  absolute per-tick poses stored, deltas built in the loader,
  `extraction_meta.json` sidecar with `config_hash` asserted at train time,
  `anchor_bad` ticks never anchor a datapoint.

---

## 0. Control-mode text in the prompt — where it lives, where to set it

**There is no control-mode mechanism anywhere in openpi.** `grep -ri
"control_mode"` over `src/`, `scripts/`, `examples/` returns nothing. It is a
*data convention from π0.5 pretraining*, not a code feature. The π0.5 paper
(arXiv 2504.16054, §appendix on action spaces) says verbatim:

> "To differentiate the two, we add '<<<control_mode>>> joint/end effector
> <<<control_mode>>>' to the text prompt."

(π0.7 keeps the same idea: "we include both joint-level and end-effector
actions during training and use a text identifier c ∈ {joint, ee} to designate
the control mode in the prompt".)

So `pi05_base` was pretrained seeing that marker string inside the task text,
and the only way to "set" it is to make it part of the prompt string. The full
prompt path in openpi:

1. Train time: the LeRobot per-episode task string becomes `data["prompt"]`
   via `PromptFromLeRobotTask`, applied in
   [data_loader.py:148-149](third_party/openpi/src/openpi/training/data_loader.py#L148-L149)
   when `DataConfig.prompt_from_task=True`.
2. `InjectDefaultPrompt` ([transforms.py:105](third_party/openpi/src/openpi/transforms.py#L105))
   only fires when `prompt` is **absent** — it will not touch our data, so it
   is *not* the injection point.
3. `TokenizePrompt` ([transforms.py:248](third_party/openpi/src/openpi/transforms.py#L248)),
   with `discrete_state_input=True` for pi05
   (default: [pi0_config.py:42-43](third_party/openpi/src/openpi/models/pi0_config.py#L42-L43)),
   calls `PaligemmaTokenizer.tokenize`
   ([tokenizer.py:22-29](third_party/openpi/src/openpi/models/tokenizer.py#L22-L29)),
   which builds the final string:
   `Task: {cleaned_text}, State: {discretized_state};\nAction: `.
   Note `cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")` —
   underscores become spaces, so what the model actually sees is
   `<<<control mode>>> end effector <<<control mode>>>`. Since openpi's
   tokenizer mirrors PI's internal pipeline, passing the literal string with
   underscores and letting this cleaning run is the closest possible match to
   pretraining.

**Where we set it**: a tiny `AppendControlMode` transform in our own config's
`model_transforms.inputs`, placed before `TokenizePrompt`:

```python
prompt = f"{task_text} <<<control_mode>>> end effector <<<control_mode>>>"
```

Rationale for a transform over baking it into the dataset task strings:
- Train/inference symmetry is automatic: `create_trained_policy`
  ([policy_config.py](third_party/openpi/src/openpi/policies/policy_config.py))
  re-assembles the *same* transform stack from the train config, so the robot
  client keeps sending the plain task text and the marker is appended
  identically on both sides.
- Changing the mode string doesn't require regenerating the dataset.
- The dataset task string stays human-readable.

Caveats to record: exact placement (before vs after the task text) inside
π0.5's pretraining prompts is not published; we choose "append after task
text" and keep it fixed. Our action space is a custom 30-dim bimanual
EEF+hand layout, so the marker buys prior alignment ("this is EEF-delta-like
data"), not an exact pretraining match. pi05 `max_token_len=200`
([pi0_config.py:41](third_party/openpi/src/openpi/models/pi0_config.py#L41))
comfortably fits task + marker (~15 tokens) + 30 state tokens.

---

## 1. Train-time RTC — what the paper actually specifies

Source: **"Training-Time Action Conditioning for Efficient Real-Time
Chunking"** (Physical Intelligence, arXiv 2512.05964). Distilled recipe:

- **Prefix length d is sampled per training example** because real inference
  delay varies: simulated experiments sample d ∈ {0..4} with exponentially
  decreasing weights ("higher delays need less supervision"); real-robot
  experiments sample **d ~ Uniform{0..10} at 50 Hz** (= 200 ms max latency
  budget).
- **Prefix = ground-truth, non-noisy first d actions of the target chunk**,
  fed at the *clean* flow timestep; remaining H−d tokens carry the sampled
  noise level. This requires a **per-token flow timestep** — exactly E002. In
  a DiT-style adaLN/adaRMS architecture "simply allow the scale, shift, and
  gate to differ between tokens. This does not change the number of learnable
  parameters."
- **Loss is masked to the postfix only.**
- **Inference**: the prefix is the tail of the currently-executing chunk (the
  actions that will run during the d-step inference delay); the model
  interface takes `(A_prefix, d)`. No guidance, no backprop at inference —
  this replaces inference-time RTC's pseudoinverse-guidance/soft-masking
  machinery.
- An RTC-trained model **degrades gracefully to d = 0** (plain sampling),
  since d = 0 is in the training distribution.

Two adaptation subtleties specific to us:

1. **Timestep convention flip.** The paper says "set the corresponding flow
   matching timesteps to 1" under a τ=1-is-clean convention. openpi uses the
   opposite: `x_t = t·noise + (1−t)·actions`, `u = noise − actions`, so
   **clean = t = 0** in openpi
   ([pi0.py:198-207](third_party/openpi/src/openpi/models/pi0.py#L198-L207)).
   OPENPI_EDITS E002 already states it correctly ("frozen at t = 0"). Any
   silent mix-up here trains garbage; the equivalence test below pins it.
2. **Anchor-relative re-anchoring at inference.** The paper conditions on the
   previous chunk's actions directly because its action space is absolute per
   embodiment. Our actions are deltas relative to the anchor at the obs tick,
   so at deployment the executing chunk's tail (relative to the *old* anchor)
   must be re-expressed relative to the *new* anchor before it can serve as
   the prefix: `Δ_new_k = T_new_anchor⁻¹ · T_old_anchor · Δ_old_{s+k}` (hand
   command dims are absolute — pass through). Both anchors are known on the
   robot side. Training needs no such step (the prefix is just the first d
   ground-truth actions of the sampled chunk, already anchor-relative), but
   train/deploy consistency of this convention must be documented in the
   deployment spec.
3. **Interaction with E001**: the prefix must be conditioned in the model's
   action space, i.e. pooled-quantile-normalized *and* per-slot-rescaled with
   the slot-0..d−1 gains. Training constructs it that way for free (it slices
   the already-transformed target chunk); the deployment client must apply the
   same forward transform to the re-anchored tail.

### Latency estimate → d distribution (inference device: RTX 4060, wired link)

Decided assumption (2026-07-11): inference runs on an **RTX 4060** connected
to the robot by cable. Estimate, to be replaced by a measurement before the
final run:

- Per-chunk work for pi05 (1 real camera + masked pads, ~800-token prefix,
  10 Euler steps): SigLIP ~0.6 TFLOP + 2B prefix expert ~3.2 TFLOP + 10 ×
  suffix (~0.03 TFLOP) ≈ **~4 TFLOP**, prefix-dominated. Cross-check: at the
  ~30-40 % utilization VLAs typically reach, a 4090 (165 bf16 TFLOPS) gives
  60-80 ms — consistent with the ~100 ms-class figures reported for
  π0-family models on 4090-class GPUs (openpi's own README benchmarks
  inference on a 4090).
- 4060 ≈ 30 bf16 dense TFLOPS, 272 GB/s → same utilization ⇒
  **~300-500 ms per chunk**. Cutting Euler steps 10→5 (the RTC paper uses 5)
  saves almost nothing here — the suffix is ~0.3 of 4 TFLOP; the prefix
  dominates. The bandwidth floor (6.6 GB bf16 weights / 272 GB/s ≈ 24 ms) is
  not binding; compute is.
- Wired GbE websocket: JPEG obs up + 50×30 f32 chunk back ≲ 10 ms —
  negligible against compute.
- **VRAM is the bigger 4060 risk, not speed**: ~6.6 GB of bf16 weights alone
  on an 8 GB card, before activations and XLA workspace; openpi's stated
  inference requirement is "> 8 GB". It may only fit with
  `XLA_PYTHON_CLIENT_MEM_FRACTION` tuning, or not at all. **Validate fit +
  measure latency on the actual 4060 before freezing d** — if it doesn't
  fit, the fallback is serving from a bigger GPU over the wire, which also
  changes the latency budget.

At 30 Hz, 300-500 ms ⇒ d ∈ [9, 15]. **Provisional choice: d ~ Uniform{0..16}**
(533 ms budget, 32 % of the H=50 chunk — supervision on slots ≥ d_max is
unaffected; d=0 stays in-support, so the checkpoint still serves in plain
mode). Tighten to Uniform{0..12} if the measured latency lands ≤ ~350 ms.
Cadence sanity: chunks cover 1.67 s, so a ~0.5 s inference consumes ≤ 16 of
50 slots per cycle — sustainable with margin.

---

## 2. Classification: what is train-time-only vs what touches deployment

Tier definitions the rest of the plan refers to:

- **Tier A — pure train-time.** Checkpoint has a stock param tree and stock
  inference semantics; existing serving code + existing model code run it
  unchanged.
- **Tier B — transform-level.** No model-code change, but the serving side
  must run our config's transform stack (i.e. the `ego2g1` package must be
  importable where the policy is created). Existing model architecture
  untouched.
- **Tier C — model/inference-code.** Requires the fork's model edits (E002)
  and/or a modified sampling loop + robot-client changes to realize the
  benefit.

| Modification | Tier | Deployment consequence |
|---|---|---|
| Loss masking to 30 real action dims (`action_dim_actual`) | A | none — loss-only |
| E003 multi-timestep training (K suffix blocks) | A | none — `compute_loss` only, checkpoint stock-servable |
| RTC *training* (random d, clean prefix, masked loss) | A→C | checkpoint still runs at d=0 on the existing pipeline (Tier A fallback); realizing RTC needs Tier C pieces |
| Fresh norm stats (pooled quantile, ours) | B | stats ship inside the checkpoint `assets/` automatically ([checkpoints.py:71-76](third_party/openpi/src/openpi/training/checkpoints.py#L71-L76)); serving must use them (stock `create_trained_policy` already does) |
| E001 floored per-slot rescale | B | inverse rescale is an output transform; per-slot gain grid must be loaded at serving — comes free iff policy is built from our config via our wrapper |
| Control-mode prompt marker | B | same string appended at inference — free via shared transform stack |
| E002 per-token adaRMS (RMSNorm rank branch) | C | lives in `ego2g1/gemma_patch.py` (runtime rebind, §3.7a) — serving must import `ego2g1` and apply the patch; param tree unchanged → **silent** wrong-semantics risk on unpatched code, hence checkpoint feature-stamping + load guard |
| Per-token `embed_suffix` + RTC `sample_actions` | C | lives in our `Pi0` subclass → serving must construct the model through `ego2g1`, not stock `Pi0Config.create()` |
| RTC serving loop (prefix re-anchor + forward-transform, async chunk swap) | C | robot client + policy server changes; separate deployment doc |

Key structural decision (confirmed 2026-07-11): **the entire feature set
lives in `ego2g1/` — `src/openpi` stays bit-stock in every phase.**
`compute_loss`, `embed_suffix`, and `sample_actions` are methods on `Pi0`
and are overridden in our subclass; the one thing that isn't subclassable —
the pi05 adaRMS path inside `gemma.py` — is handled by a **runtime patch**
(`gemma_patch.py`, §3.7a) that rebinds two module-level symbols before model
construction, instead of a fork commit. Consequences:

- Every phase, including E002/RTC, runs on **bit-stock openpi src** — even a
  fresh upstream clone (see §6). The fork branch remains only as the place
  the `ego2g1/` folder is version-tracked, not as a source diff.
- The fork's existing `action_dim_actual` edit
  ([pi0.py:212-218](third_party/openpi/src/openpi/models/pi0.py) on
  `ego2g1-data`) is **migrated into the subclass and reverted from `src/`**.
- All deviating logic has a single home (`ego2g1/`), instead of being split
  between fork edits and package code.

Deviations from E002-as-written (its re-derivation note licenses both):
`embed_suffix` moves from a fork edit into the subclass (copies ~40 lines of
stock code — drift risk on upstream pulls, covered by the golden tests), and
the RMSNorm edit becomes a runtime patch (trade-offs in §3.7a). OPENPI_EDITS
E002/E003 are updated to record the vehicle change.

---

## 3. Package layout — `third_party/openpi/ego2g1/`

Self-contained; imports `openpi.*` as a library; **never** imports
`data_extraction`. Run as `uv run python -m ego2g1.<entrypoint>` from the
openpi root. One file per concern:

```
ego2g1/
  __init__.py
  config.py               # frozen dataclass: every knob in one place
  chunk_math.py           # RelativeChunkActions + delta_timestamps (copy, pinned)
  dataset.py              # LeRobot dataset wrapper: boundary remap, sidecar assert
  transforms.py           # Ego2G1Inputs/Outputs, AppendControlMode, PerSlotRescale(+inv)
  data_config.py          # DataConfigFactory -> openpi DataConfig
  norm.py                 # stats containers, load/save, degenerate-dim policy
  compute_norm_stats.py   # entrypoint: pooled NormStats + (50,30) per-slot grid
  model.py                # Ego2G1Pi0Config / Ego2G1Pi0 subclass (all loss/sampling logic)
  gemma_patch.py          # E002: per-token adaRMS via symbol rebind (§3.7a)
  train.py                # entrypoint: mirrors scripts/train.py main()
  policy.py               # create_policy wrapper (loads both stats files from ckpt)
  stamp.py                # checkpoint feature-flag stamping + load guard
  tests/
    test_chunk_math_equivalence.py   # vs data_extraction (runs in outer repo CI only)
    test_golden_stock.py             # subclass == stock Pi0 when all features off
    test_gemma_patch.py              # patch: stock-path identity + fingerprint guard
    test_multi_t_blocks.py           # E003 block-equivalence
    test_rtc_loss.py                 # prefix/mask/timestep-convention pins
    test_per_slot_rescale.py         # roundtrip + c=1 identity
    test_stamp_guard.py
```

### 3.1 `config.py`

Single frozen dataclass `Ego2G1TrainConfig` that *produces* an
`openpi.training.config.TrainConfig` (it does not get registered in openpi's
`_CONFIGS`; our entrypoints take ours directly — no dependence on openpi's
tyro CLI or config registry). Fields, grouped:

- data: `dataset_root`, `repo_id`, `expected_config_hash` (asserted against
  the sidecar), `val_real_episodes` (held-out **real** episode indices — the
  sidecar records the real-episode ↔ LeRobot-episode mapping, so the split
  can't leak across a filter-split boundary).
- model: `action_dim=32`, `action_dim_actual=30`, `action_horizon=50`,
  `pi05=True`, `discrete_state_input=True` (default for pi05).
- prompt: `control_mode="end effector"`, marker template.
- E001: `per_slot_floor_c=0.1` (`c=1` ⇒ bitwise stock pooled behavior).
- E003: `num_flow_samples=1` (K; raise only after the profiling gate).
- RTC: `rtc_enabled=False`, `rtc_d_max=10`, `rtc_d_dist="uniform"`.
- training: batch size, steps, lr schedule, `weight_loader` pointing at
  `gs://openpi-assets/checkpoints/pi05_base/params`, wandb project, seed.
- provenance: property `feature_flags()` → dict written by `stamp.py`.

### 3.2 `chunk_math.py` — the loader-math copy

Byte-for-byte port of `data_extraction/loader/relative_actions.py` +
`boundary.py` semantics (vec9 ↔ SE3, anchor-relative deltas, hand cmds
absolute, `make_delta_timestamps`). Already pinned by the equivalence test
from commit b2d0c93; the test stays in the **outer** repo (it needs
`data_extraction`), and `tests/test_chunk_math_equivalence.py` here is a thin
re-export/skip-if-unavailable so both repos can run it.

### 3.3 `dataset.py`

- Opens the LeRobot dataset at `dataset_root/repo_id` with
  `delta_timestamps = make_delta_timestamps(H, fps)` (poses gather ticks
  0..H — row 0 is the anchor; hands gather 1..H).
- Reads `extraction_meta.json`; hard-asserts `config_hash ==
  expected_config_hash` and `datasets`/schema readability (fail loud before
  any GPU time; also catches the datasets≥5 parquet incompatibility).
- **Boundary-aware index remap**: valid datapoint indices are those whose
  anchor tick is not `anchor_bad` and whose chunk window stays inside the
  episode; pi0 ignores `action_is_pad`, so windows that would cross a
  boundary are excluded at the index level, exactly like the reverted
  57b322f implementation and the loader-equivalence spec. (As of 2026-07-11
  bridging is removed — episodes split strictly at bad ticks, so
  `anchor_bad` is always empty — but the loader keeps honoring it so the
  semantics don't change if bridging ever returns.)
- Emits gathered `pose.*` (H+1, 9) / `hand.*` (H, 6) chunks; the
  anchor-relative differencing (`RelativeChunkActions`) runs as the **first
  data transform** (§3.5), so norm-stats computation sees the exact training
  distribution through stock `data_transforms` semantics.
- Exposes train/val subsets keyed by the sidecar's `source_episode` (real
  episode), so a filter-split cannot leak one real episode across the split.

### 3.4 `transforms.py`

- `Ego2G1Inputs`: key mapping to openpi names; images → `base_0_rgb` = ego
  camera, `left_wrist_0_rgb`/`right_wrist_0_rgb` = zeros with
  `image_mask=False` (pi05 masking convention, cf.
  [droid_policy.py](third_party/openpi/src/openpi/policies/droid_policy.py)).
- `Ego2G1Outputs`: slice model output `(50,32)` → `(50,30)`.
- `AppendControlMode` (§0).
- `PerSlotRescale` / `PerSlotRescaleInverse` (E001): multiply/divide by a
  precomputed `(50,30)` gain grid `g[k,d] = sigma_pooled[d] / max(
  sigma_slot[k,d], c·sigma_pooled[d])`, computed once in `norm.py` from the
  per-slot stats file; `c=1` ⇒ `g≡1`. Applied to `actions` only (not state).
  RTC note: the same grid is what the deployment client uses on the
  re-anchored prefix (§1.3).

### 3.5 `data_config.py` — transform ordering (normalization-critical)

Produces an `openpi DataConfig` consumed by stock
`data_loader.transform_dataset`
([data_loader.py:184-189](third_party/openpi/src/openpi/training/data_loader.py#L184-L189)),
which fixes the order: `repack → data_transforms.inputs → Normalize →
model_transforms.inputs`. Our assignment:

- `repack_transforms`: identity (our dataset already emits final keys).
- `data_transforms.inputs = [RelativeChunkActions, Ego2G1Inputs]` — runs
  **before** Normalize, and this is also exactly what norm-stats computation
  sees.
- `norm_stats`: loaded by our factory from our stats file;
  `use_quantile_norm=True` (pi05 path,
  [config.py:188](third_party/openpi/src/openpi/training/config.py#L188)).
- `model_transforms.inputs = [PerSlotRescale, AppendControlMode,
  ResizeImages(224,224), TokenizePrompt(..., discrete_state_input=True),
  PadStatesAndActions(32)]` — i.e. stock pi05 `ModelTransformFactory` list
  with our two transforms prepended. **PerSlotRescale must sit after
  Normalize** (it's defined in pooled-normalized units) **and before
  padding**; **TokenizePrompt must see the normalized 30-dim state** (it
  digitizes state into 256 bins over [−1,1] —
  [tokenizer.py:26](third_party/openpi/src/openpi/models/tokenizer.py#L26) —
  so state quantile stats directly determine the discrete state tokens).
- outputs (inference, applied in reverse): `model_transforms.outputs =
  [PerSlotRescaleInverse]` → stock `Unnormalize` → `data_transforms.outputs =
  [Ego2G1Outputs]`.

### 3.6 `norm.py` + `compute_norm_stats.py`

Why not openpi's `scripts/compute_norm_stats.py`: it can't see our dataset
wrapper, computes only pooled stats, and stamps nothing. Ours:

- Iterates the **boundary-aware** dataset through `data_transforms.inputs`
  only (mirroring stock semantics,
  [compute_norm_stats.py:36-44](third_party/openpi/scripts/compute_norm_stats.py#L36-L44)),
  over the **train split only** (val must not leak into stats).
- Writes two artifacts, both stamped with the dataset `config_hash`, the
  ego2g1 config hash, and the git commits of fork+outer repo:
  1. `norm_stats.json` in openpi's native format
     (`openpi.shared.normalize.save`) — pooled per-dim mean/std/q01/q99 for
     `state` and `actions`. This is what stock `Normalize`/`Unnormalize` and
     the checkpoint-assets machinery consume, unchanged.
  2. `per_slot_stats.npz` — `sigma_slot (50,30)`, plus the derived gain grid
     for the configured `c` (grid recomputed at load, sigma is the artifact).
- Sanity gates (E001 eval item 7): assert `q99−q01 > eps` and
  `sigma_slot > 0` outside an explicit allowlist `{rot6d diagonal dims at
  slot 1, left-hand fingers (unused so far), pad dims}`; exactly-constant
  dims get divisor 1 and are recorded in the artifact (openpi's bare `1e-6`
  eps in
  [transforms.py:145](third_party/openpi/src/openpi/transforms.py#L145)
  would otherwise amplify near-constant dims explosively).
- Prints the `(50,30)` sigma grid and effective boost per (slot, dim) for
  eyeballing before any training run.

Decision recorded: we do **not** reuse any pretraining norm-stat asset
(`docs/norm_stats.md` reuse only applies when the embodiment matches a
pretraining action space; our 30-dim layout matches none).

### 3.7 `model.py`

`Ego2G1Pi0Config(Pi0Config)` with extra fields (`action_dim_actual`,
`num_flow_samples`, `rtc_*`), whose `create()` returns `Ego2G1Pi0(Pi0)`:

- `compute_loss` override — single home for, in order:
  1. loss masking to `action_dim_actual` (migrated from the fork edit);
  2. E003 K-block construction (per OPENPI_EDITS E003 spec: flat
     `(b, K·S)` suffix, explicit block-diagonal mask bypassing
     `make_attn_mask`, **repeated RoPE positions per block**, mean over K);
  3. RTC training (per §1): sample d, overwrite the first d suffix action
     tokens with clean actions at per-token t=0 (openpi convention), mask
     loss to the postfix. Composition rule with E003: each of the K blocks
     draws its own d (they're independent regression targets).
- `embed_suffix` override — accepts `(b,)` or per-token `(b, s)` timestep
  (E002's pi0.py half). **Scalar input delegates to `super().embed_suffix`**
  (the stock annotation accepts it), so the stock path is literally stock
  code — zero copied lines, no drift risk; only the per-token branch is new.
  Per-token requires pi05 (asserted); `posemb_sincos` is rank-1-typechecked,
  so the per-token path flattens to `(b·s,)` and reshapes back.
- `sample_actions_rtc` — new method (stock `sample_actions` untouched):
  takes `(observation, prefix_actions, d)`, holds prefix tokens clean at
  t=0 through the Euler loop, integrates only the postfix. Tier C: used by
  deployment phase only.
- Everything off (`action_dim_actual=None, K=1, rtc_enabled=False`) must be
  **bitwise stock** — that's `test_golden_stock.py`.

The pi05 adaRMS per-token branch cannot live here (RMSNorm is instantiated
deep inside gemma's `Block`) — it lives in `gemma_patch.py`:

### 3.7a `gemma_patch.py` — E002 without touching `src/openpi`

The two stock objects E002 needs to change are module-level symbols in
`openpi.models.gemma`, and both are **looked up late** — `Block.__call__`
resolves `RMSNorm` as a module global
([gemma.py:303,318](third_party/openpi/src/openpi/models/gemma.py#L303)), and
`Pi0.__init__` resolves `_gemma.Module` at construction
([pi0.py:74](third_party/openpi/src/openpi/models/pi0.py#L74)). So a rebind
applied before model construction is fully effective, with no source edit:

1. `PerTokenRMSNorm(nn.Module)` — a normal `@nn.compact` class defined in our
   package: `cond.ndim == 2` → byte-for-byte the stock body (including
   `modulation[:, None, :]`); `cond.ndim == 3` → same `nn.Dense` (features
   inferred from the last dim ⇒ identical kernel, zero new params), split
   without the `[:, None]`. Then `gemma.RMSNorm = PerTokenRMSNorm`.
2. `PerTokenModule(gemma.Module)` — subclass whose `__call__` is a copy of
   the ~17-line stock body with the `adarms_cond` annotation widened from
   `"b _d"` to `"b _d" | "b _s _d"` (the stock `at.typecheck` on
   [gemma.py:395](third_party/openpi/src/openpi/models/gemma.py#L395) is the
   only thing that would reject rank-3 cond — the layer scan itself passes
   it through opaquely, and `_gated_residual` is pure broadcasting). Then
   `gemma.Module = PerTokenModule`, with `__name__` forced to `"Module"`.

Safety properties, each with a test in `test_gemma_patch.py`:

- **Param-path invariance**: the gemma `Module` is the *top level* of its
  linen tree (wrapped directly by `nnx_bridge.ToNNX`), so its class name
  never enters parameter paths; every inner RMSNorm gets an explicit
  `name=`, and the Dense inside keeps auto-name `Dense_0`. Asserted by the
  existing param-pytree-invariance test (E002 eval 3).
- **Stock-path identity**: golden `compute_loss`/`sample_actions` outputs
  bitwise-equal with and without the patch applied for rank-2 cond
  (E002 eval 1); the patched rank-2 branch *is* the stock code.
- **Source fingerprint guard** (replaces the fork-commit pin): `apply()`
  hashes `inspect.getsource(gemma.RMSNorm)` and
  `inspect.getsource(gemma.Module.__call__)` against pinned digests of the
  stock code it replaces, and **refuses to patch** on mismatch. If an
  upstream pull ever rewrites gemma.py, the patch fails loud at import
  instead of silently patching changed code.
- **Idempotent, applied at one choke point**: `ego2g1.model` calls
  `gemma_patch.apply()` at import, before any config `create()`; entrypoints
  (`train.py`, `policy.py`) import through `ego2g1.model` only.

Trade-off vs the fork-commit vehicle E002 originally specced: we lose "the
deviation is a reviewable git diff in the code it changes" and accept
monkeypatch indirection; we gain a truly stock `src/` (upstream pulls are
conflict-free, any openpi checkout works, §6) and one home for all
deviations. The stamp guard (§3.9) is unchanged and still mandatory: an
RTC-trained checkpoint loaded by code that never imported `ego2g1` would run
with silently wrong semantics — the flags file is what makes that loud.

### 3.8 `train.py`

Mirror of [scripts/train.py](third_party/openpi/scripts/train.py) `main()`
(280 lines; copy, then trim): same sharding/optimizer/checkpoint code paths,
differences only:

- takes `Ego2G1TrainConfig` (tyro over our dataclass);
- builds the data loader from our `dataset.py` + `data_config.py` (stock
  `transform_dataset`/`TorchDataLoader`/`DataLoaderImpl` reused; only dataset
  construction is ours — no `create_torch_dataset` edit needed, unlike the
  reverted 57b322f which patched `data_loader.py` for this);
- reuses `init_train_state`/`init_logging`/`init_wandb` from
  `scripts/train.py` via dynamic import (no copy); carries its own
  `train_step` copy extended with the E001 per-slot loss decomposition
  (eval item 3) in the logged info;
- validation per-slot real-unit error vs the two analytic baselines
  (hold-anchor, dataset-mean chunk) lives in a separate offline entrypoint
  `validate.py` run against a checkpoint — not inside the train loop;
- on checkpoint save: stock assets hook already writes pooled norm stats
  ([checkpoints.py:71-76](third_party/openpi/src/openpi/training/checkpoints.py#L71-L76));
  we additionally copy `per_slot_stats.npz` into the same
  `assets/<asset_id>/` dir and call `stamp.py`.

### 3.9 `policy.py` + `stamp.py`

- `policy.py`: thin wrapper around
  `openpi.policies.policy_config.create_trained_policy` that (a) builds the
  openpi `TrainConfig` from our saved ego2g1 config, (b) loads **both** stats
  artifacts from the checkpoint `assets/` (never from a live assets dir — the
  stock function already prefers checkpoint stats for the pooled file; we
  extend that guarantee to the per-slot grid), (c) runs the stamp guard.
- `stamp.py`: writes `feature_flags.json` + fork/outer commit hashes into the
  checkpoint dir at save; at load, refuses to serve if the running code does
  not declare support for every recorded flag (E002 item 5 — the param tree
  stays stock-shaped, so without this guard stock code would run an
  RTC-trained checkpoint with silently wrong semantics).

### 3.10 Tests (gates before GPU time)

Ordered by what they protect:

1. `test_chunk_math_equivalence` — loader copy == `data_extraction` reference
   (already exists; re-point at `ego2g1.chunk_math`).
2. `test_golden_stock` — all features off ⇒ `compute_loss`/`sample_actions`
   bitwise-match stock `Pi0` on fixed rng (small pi05 + pi0 config), and
   param pytree identical (stock ckpt loads into subclass and vice versa).
3. `test_per_slot_rescale` — rescale∘inverse == identity; `c=1` ⇒ no-op;
   degenerate-dim allowlist honored.
3a. `test_gemma_patch` — stock-path bitwise identity with patch applied;
   per-token cond filled with a repeated scalar allclose-matches the scalar
   path (E002 eval 2); param pytree invariant (E002 eval 3); fingerprint
   guard raises on tampered source.
4. `test_multi_t_blocks` — E003 eval item 1: block k's `v_t` allclose to a
   stock single-sample forward at the same `(t_k, ε_k)`.
5. `test_rtc_loss` — prefix tokens receive t=0 and ground-truth actions
   (convention pin!), loss exactly zero on prefix positions, d=0 reduces to
   the non-RTC path, and (post-E002) per-token adaRMS broadcast-equivalence
   (E002 eval item 2).
6. `test_stamp_guard` — undeclared flag ⇒ load raises.

---

## 4. Normalization end-to-end (the part that bites later)

Forward (train == inference input path, guaranteed by shared config):

```
real-unit actions (50,30), state (30,)
  → Normalize(quantile, ours)         # → roughly [-1,1], pooled per-dim
  → PerSlotRescale(g[k,d])            # actions only, E001
  → TokenizePrompt digitizes state    # 256 bins over [-1,1]  ← state stats gate this
  → PadStatesAndActions → (50,32)/(32,)
```

Inverse (inference output path): model `(50,32)` → `PerSlotRescaleInverse` →
`Unnormalize(quantile)` → `Ego2G1Outputs` slice → real units. Deployment on
the existing pipeline needs *nothing else* as long as the policy object is
built by `ego2g1/policy.py`; a fully custom serving stack would need to
re-implement: quantile unnormalize + per-slot inverse gain + 30-dim slice —
all three read from the checkpoint's `assets/` dir, nowhere else.

**⚠ The inverse rescale is not optional.** An E001-trained model emits early
slots at *boosted* scale (a slot-1 action the pooled scheme would put at
±0.05 comes out at up to ±0.5, by design — that is the whole point of the
training-signal fix). Feeding model output to stock pooled `Unnormalize`
alone would interpret those boosted values as pooled-normalized and inflate
early-slot actions by up to `1/c = 10×` in real units — the robot would
overshoot hardest on precisely the slots it always executes. The
`PerSlotRescaleInverse` (divide by the same `g[k,d]` grid used in training)
must run first, always, and the stamp guard treats `per_slot_norm: c=0.1`
as a feature flag so a serving stack that didn't load the grid refuses the
checkpoint instead of executing 10×-scaled motions.

Single source of truth: both stats artifacts live in the checkpoint;
`config_hash` chains dataset → stats → checkpoint → policy load. Any
mismatch fails loud at load, not silently at rollout.

---

## 5. Phasing

**Phase 1 — Tier A+B training, stock model code.** Revert the
`action_dim_actual` edit from fork `src/` (moves into subclass); implement
§3 minus RTC/E002; K=1. Gates: tests 1-3, 6; one short smoke run (loss
decreases, val metrics wired, checkpoint round-trips through `policy.py`).
Deliverable: a checkpoint deployable on the existing pipeline today
(Tier B: serving imports `ego2g1` for transforms).

**Phase 2 — E002 + RTC training.** `gemma_patch.py` (§3.7a — no fork
commit); `embed_suffix` override + RTC loss in subclass; tests
`test_gemma_patch` + 4* (*if K>1 gets un-gated) + 5; retrain or continue
fine-tuning with `rtc_enabled=True`. Checkpoint still serves at d=0 on the
Phase-1 deployment path.

**Phase 3 — RTC deployment (separate doc, out of scope here).**
`sample_actions_rtc` serving endpoint; robot client: measure real latency
distribution → fix d budget; executed-tail re-anchoring + forward transform
(§1.2-1.3); async chunk swap; chunk-boundary continuity metric (E001 eval
item 5) as the acceptance test.

**E003 stays gated** on the profiling evidence spelled out in
OPENPI_EDITS.md (measure duty cycle / prefix-vs-suffix split first). It
slots into `compute_loss` later without touching anything else in this plan.

Decisions settled in review (2026-07-11):
1. **E001 starts at `c = 0.1`** from the first run (no c=1 smoke run); the
   `c=1`-is-stock property is still pinned by `test_per_slot_rescale`.
2. **RTC d ~ Uniform{0..16}**, provisional from the 4060 estimate (§1);
   revisit only after the on-device latency/VRAM check.
3. **E002 vehicle = `gemma_patch.py`** (runtime rebind in our package), not
   a fork commit; `src/openpi` stays bit-stock in all phases.
4. **Fine-tune sizing (batch/steps/lr) deferred** — decide when the training
   GPU is pinned down; start from openpi's pi05 fine-tune defaults.

5. **Phase 1 trains with `rtc_enabled=False`, but all RTC/E002 code ships
   now, toggleable by config** (decided 2026-07-11): `gemma_patch`, per-token
   `embed_suffix`, RTC loss branch, and `sample_actions_rtc` are implemented
   and tested up front; the first training run simply keeps the flag off
   (which also keeps the rng stream and loss bitwise-stock modulo the
   action-dim mask).

Still open:
1. Val split: which real episodes (propose ~10% by real-episode count,
   fixed list in config, never re-rolled).
2. Dataset regen: `lerobot_datasets/` is stale vs the current extraction
   config (cleanliness-layer rerun pending) — training waits for the regen;
   the config-hash assert enforces this mechanically.

---

## 6. Running on the training device: fresh clone vs fork, and venv reuse

**Can training run on a freshly cloned *stock* openpi?** **Yes — all
phases.** With the §2 restructuring plus `gemma_patch.py` (§3.7a), `src/`
carries zero deviations: any openpi checkout at a compatible commit + the
`ego2g1/` folder on top runs everything, including E002/RTC. The fork branch
survives only as where the `ego2g1/` folder is version-tracked (and as a
place to pin the exact upstream commit the fingerprint guard was computed
against); it is no longer required at runtime. Compatibility is enforced,
not assumed: on an incompatible upstream commit the patch's source
fingerprint check fails loud before anything runs.

**Recommended flow on the device with the already-configured env** (no
re-forking, no re-downloading):

```bash
cd <existing-openpi-checkout-on-device>
git remote add mine https://github.com/LavetteSinsora/openpi   # if absent
git fetch mine ego2g1-data
git switch ego2g1-data          # or: git switch -c ego2g1-data mine/ego2g1-data
```

The existing venv keeps working **because openpi is installed editable into
that same directory** (uv installs the project editable on `uv sync`);
switching branches changes the source in place — no reinstall, no new venv.
Do run `uv sync` once after switching only if the fork's `uv.lock` diverges
in deps that matter (currently it doesn't beyond dev noise; `uv` resolves
from its global wheel cache anyway, so this is seconds-to-minutes, not a
re-download).

**If a separate directory is unavoidable** (e.g. the existing checkout is
shared or dirty): clone the fork next to it, then rebind the *existing* venv
to the new source tree instead of creating a new one:

```bash
source <old-checkout>/.venv/bin/activate
uv pip install -e <new-clone> --no-deps     # seconds; only re-points openpi
```

Every dependency stays as-is; only the editable pointer moves. (Equivalent:
`UV_PROJECT_ENVIRONMENT=<old>/.venv uv sync` from the new clone.) The one
thing to verify afterwards: `python -c "import openpi, pathlib;
print(pathlib.Path(openpi.__file__).resolve())"` points into the new clone —
a stale editable pointer is exactly the silent-wrong-code failure the stamp
guard exists for.

Device preflight checklist (fail before GPU time):
- `import openpi` resolves to the intended checkout (above).
- `datasets` version can read the dataset parquet (the ==3.6.0-pin issue):
  open one episode end-to-end through `ego2g1/dataset.py`.
- Sidecar `config_hash` matches `expected_config_hash`.
- `pi05_base` weights reachable (gs:// download or pre-staged).
- `compute_norm_stats` artifacts present and stamped for this dataset hash.
