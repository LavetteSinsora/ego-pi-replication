# openpi edits — deviation log

Every place our training setup deviates from stock `third_party/openpi`, so a
reader (or a future debugging session) can tell deliberate changes from drift.
One numbered item per deviation: the problem in stock openpi, the fix, the
trade-off reasoning, and the metrics that will tell us whether it helped.

---

## E001 — Floored per-slot action normalization

**Status**: proposed (not yet implemented)
**Touches**: our loader transform (`data_extraction/loader/`), norm-stats
computation; openpi's `Normalize`/`Unnormalize` and stats file stay stock.

### Problem in stock openpi

Norm stats are pooled over the action horizon: `RunningStats.update` flattens
everything but the last dim (`normalize.py:37`, `batch.reshape(-1, D)`), so
one (mean, std, q01, q99) per action dimension is shared by all 50 slots.

Our actions are anchor-relative deltas, so per-slot std grows roughly
linearly with slot index by construction. Measured on `put_bottle_in_box`
episode_1 (268 datapoints), e.g. right-eef ty: slot-1 std 0.0027 m, slot-10
0.024 m, slot-50 0.078 m, pooled 0.054 m. Two consequences:

1. **Faint training signal on early slots.** The flow-matching target
   `u = x1 - x0` is dominated by the unit-scale noise term; the informative
   residual for slot k has normalized scale `sigma_k / sigma_pooled`, so its
   loss share is the square of that: **0.25 % at slot 1**, ~20 % at slot 10
   (right-eef ty). The first few slots' objective is >99 % "cancel the
   noise", and the part that distinguishes a good slot-2 action from a bad
   one is a rounding error in the loss.
2. **Sampler error is flat in real units, so early slots are drowned.** The
   10-step Euler integration error is roughly uniform per dim in normalized
   units; denormalizing with the pooled std gives every slot the same
   absolute error in meters. Slot 50 (signal ~6-8 cm) barely notices; slot 2
   (signal ~2 mm) can be swamped by the sampler's own noise. With
   receding-horizon execution the **early slots are the ones the robot
   actually executes**, so precision is misallocated to exactly the wrong
   end of the chunk.

Why stock openpi gets away with it: pi0's own delta actions have the same
structure and it trains — the model can still drive early-slot error below
the pooled floor because early slots are highly predictable. The claim here
is not "pooled fails", it is "pooled is the wrong resting point for
anchor-relative chunks", to be verified by the metrics below.

### Fix

Keep openpi's pooled per-dim quantile normalization unchanged, then apply a
**floored per-slot rescale** in our own chunk transform (right after
`RelativeChunkActions`), with inverse applied at inference before openpi's
stock unnormalize:

```
divisor[k, d] = max(sigma_slot[k, d], c * sigma_pooled[d])     # c = 0.1
actions_norm[k, d] = actions_pooled_norm[k, d] * sigma_pooled[d] / divisor[k, d]
```

equivalently: boost each slot toward unit scale, but never by more than
`1/c`. `sigma_slot` is the (50, 30) per-slot std computed over all valid
datapoints (loader-emulated chunks, boundary rules applied); stats are
stamped with `config_hash` like every other stage output. `c` is a config
field; **`c = 1` reproduces stock pooled behavior exactly**, `c -> 0` is
full per-slot normalization.

### Trade-off logic (why floored, not full per-slot)

- Slot-1 deltas (~1-3 mm at 30 Hz) are substantially Pico tracking jitter,
  not intentional motion. Full per-slot norm would amplify slot 1 by
  `sigma_pooled / sigma_1` ≈ **20-50x** (measured), promoting jitter to
  unit scale: irreducible loss the model cannot predict, and unit-scale
  jitter faithfully *generated* at sampling time. It also multiplies the
  degenerate-stats problem by 50 (q01 ≈ q99 per (slot, dim): rot6d
  diagonals at slot 1, unused left fingers, pad dims).
- The floor caps amplification at `1/c = 10x`. Jitter sits at ~0.01-0.03 in
  pooled-normalized units, so worst-case amplified jitter is ~0.1-0.3 —
  visible, still clearly sub-unit. Meanwhile slot-1's informative loss share
  rises from 0.25 % to ~20 % (100x more gradient), and mid slots
  (k ≈ 5-25, where signal is 1-4 cm vs ~1 mm jitter) get fully equalized to
  unit scale — the region where the argument is strongest and the data is
  clean.
- **Checkpoint compatibility**: base pi0 saw quantile-normalized actions in
  [-1, 1]. The rescale moves early slots from ±0.05 to at most ±0.5 —
  inside the distribution the pretrained action expert knows.
- `c` tunes "equalize slots" vs "inflate sensor noise". 0.1 keeps worst-case
  jitter below signal scale; drop toward 0.05 only if the measured jitter
  floor is < 1 mm.

### Evaluation plan (single run — no compute for A/B or grid search)

We train **one** configuration only: `c = 0.1`, everything else stock.
There is no baseline run to difference against, so the modification is
judged from that run's own stats against (a) absolute physical yardsticks
and (b) **analytic baselines that cost no training**: the "hold anchor"
predictor (all deltas = identity, hands = anchor command) and the
"dataset mean chunk" predictor (per-slot mean action), both evaluated
offline on the same validation split. These give per-slot reference error
levels any useful model must beat, and the jitter floor gives the level
below which improvement is impossible.

Record:

1. **Per-slot validation error in real units** — headline. Denormalized
   MAE/RMSE vs ground-truth chunks, per slot k = 1..50, per group: eef
   translation (mm), eef rotation (deg, geodesic from the 6d columns),
   hand commands (cmd units in [0, 1]); per hand. Plotted against the two
   analytic baselines. *Success = clearly below both baselines at every
   slot, and slots 1-5 approaching the ~1 mm jitter floor rather than
   sitting orders of magnitude above it.*
2. **Per-slot, per-dim validation error (full 50x30 grid)** — heatmap, to
   catch a single dim failing inside a healthy group average.
3. **Training loss decomposed by slot** (normalized units, per slot group)
   — the mechanism check available without a baseline: early-slot loss
   should keep improving over training. Failure signature of a wrong `c`:
   slots 1-3 flatline high and early (irreducible amplified jitter).
4. **Sampled-chunk noise at early slots**: std of slot 1-3 actions across
   repeated samples at the same observation, in mm. Should be on the order
   of the jitter floor; if it is several x larger, the model is generating
   amplified noise — evidence the floor `c` is too low.
5. **Chunk-boundary continuity at rollout**: executed-action jump (mm, deg)
   between the last executed slot of chunk i and slot 1 of chunk i+1,
   compared against the teleop data's own tick-to-tick motion at the same
   speed. Early-slot precision is the claimed benefit; this is where it
   should show up physically.
6. **Sampler-step sensitivity** (cheap, same checkpoint): early-slot
   validation error at 10 vs 50 Euler steps. Small gap = integration error
   is not the early-slot bottleneck under the rescale, consistent with the
   fix doing its job.
7. **Stats sanity at computation time**: assert `q99 - q01 > eps` and
   `sigma_slot > 0` outside the known-degenerate allowlist (left fingers
   until an episode uses them, 2 pad dims); log the (50, 30) sigma_slot
   grid and the effective boost `sigma_pooled / divisor` per (slot, dim).

Decision rule, one run: **keep** if metric 1 passes and metrics 3/4 show no
amplified-jitter failure signature. **Revert to c = 1** (stock pooled) if
early-slot loss flatlines high, sampled early-slot noise exceeds a few mm,
or rollouts show early-slot dithering — those signatures are specific to
this modification, not to general undertraining. If the run merely looks
"fine everywhere", the mod stays (it is theoretically motivated and cheap),
but this item stays open until a baseline run is someday affordable —
absent a failure signature we cannot claim the improvement is *proven*,
only that the mechanism's predictions were not falsified.

---

## E002 — Per-token timestep conditioning (per-token adaRMS)

**Status**: proposed (not yet implemented)
**Touches** (vehicle revised 2026-07-11, see TRAINING_PLAN.md §3.7a):
`ego2g1/gemma_patch.py` (runtime rebind of `gemma.RMSNorm` and
`gemma.Module`), `ego2g1/model.py` (`embed_suffix` override on the `Pi0`
subclass), `ego2g1/tests/`, checkpoint save/load path (feature stamping).
**`src/openpi` is not edited** — the original plan to commit these changes
to the fork's `gemma.py`/`pi0.py` was replaced by the package-level patch so
that stock openpi checkouts stay usable end to end; the numbered spec below
still describes the exact code semantics, only the vehicle changed. First
entry that changes model *behavior* — everything before this was
loader-side.

### Problem in stock openpi

The flow-matching timestep is a single scalar per batch element
everywhere: `compute_loss` samples `time` of shape `(b,)` (`pi0.py:198`),
`embed_suffix` takes `timestep: (b,)`, and for pi05 the resulting adaRMS
conditioning vector is `(b, emb)`, broadcast over the whole suffix
sequence inside `RMSNorm` via the hardcoded `modulation[:, None, :]`
(`gemma.py:129`). Two planned features need *different timesteps for
different suffix tokens in the same forward pass*:

1. **E003** — multiple flow timesteps per trajectory sample (K noisy
   suffix blocks, each at its own t, sharing one prefix).
2. **Train-based real-time chunking (future E00x)** — denoise a chunk
   whose first d actions are frozen at t = 0 (the tail of the currently
   executing chunk) while the rest carry a sampled noise level: a
   per-token timestep vector by definition.

For the non-pi05 variant this is almost free (time is already mixed in
per-token via `einops.repeat` + MLP, `pi0.py:173`); the pi05 adaRMS path
is the only real blocker. This entry adds the shared primitive; E003 and
train-RTC build on it without touching `gemma.py` again.

### Fix (implementation spec)

Core principle: **timestep becomes a per-suffix-token field `(b, s)`;
scalar `(b,)` remains accepted and takes the stock code path bit-for-bit.**

1. RMSNorm (via `gemma_patch.py`: define `PerTokenRMSNorm` and rebind
   `gemma.RMSNorm` — `Block` resolves it as a module global at
   `gemma.py:303/318`, so the rebind is fully effective): branch on cond
   rank. `cond.ndim == 2` (stock): unchanged code, including
   `modulation[:, None, :]`. `cond.ndim == 3` (`(b, s, emb)`): apply the
   same `nn.Dense` (features are inferred from the last dim, so the kernel
   stays `(width, 3*width)` — **zero new parameters, checkpoint pytree
   unchanged**) and split without the `[:, None]`. Gate becomes
   `(b, s, emb)` instead of `(b, 1, emb)`; `_gated_residual`
   (`gemma.py:453`) is `x + y * gate`, pure broadcasting — no change. The
   layer scan passes `adarms_cond` through as an opaque broadcast
   argument — no change. Param paths are safe: the gemma `Module` is
   top-level under `ToNNX` (class name never enters param paths) and every
   RMSNorm instance is explicitly `name=`d.
2. `Module.__call__` typecheck (via `gemma_patch.py`: subclass with copied
   `__call__` body, `__name__` forced to `"Module"`, rebind
   `gemma.Module` — `pi0.py:74` resolves it at construction time): widen
   `adarms_cond` element annotation from `"b _d"` to `"b _d" | "b _s _d"`.
   Init path (`gemma.py:420`, `zeros((1, width))`) unchanged.
   `gemma_patch.apply()` is idempotent, runs at `ego2g1.model` import
   (before any model construction), and **fingerprint-guards** the stock
   source (`inspect.getsource` digests of both replaced objects) — on an
   unexpected upstream gemma.py it refuses to patch rather than patching
   drifted code.
3. `embed_suffix` (override on our `Pi0` subclass in `ego2g1/model.py`, not
   a `pi0.py` edit): accept `timestep: (b,) | (b, s_actions)`.
   Scalar input → exactly the current code (stock configs stay bitwise
   identical). Per-token input → compute `posemb_sincos` on the flattened
   `(b*s,)` view and reshape back (leave `posemb_sincos` itself
   untouched); the pi05 time-MLP is `nnx.Linear` on the last dim and works
   on `(b, s, emb)` unchanged, yielding `adarms_cond: (b, s, emb)`; the
   non-pi05 path drops the `einops.repeat` when time is already
   per-token.
4. `compute_loss` / `sample_actions`: **not touched** in E002. They keep
   passing `(b,)`.
5. **Deployment guard** (part of this entry's scope, because this is the
   first model-code divergence): at checkpoint save, stamp the checkpoint
   dir with the openpi fork commit hash and a feature-flags file (e.g.
   `{"time_conditioning": "per_token"}`); at policy load, refuse to serve
   a checkpoint whose flags the running code does not declare support
   for. Rationale: E002-based checkpoints have a stock-compatible param
   tree, so stock openpi would load and run them *silently* with wrong
   semantics once a feature like train-RTC is actually used in training.
   The failure must be loud. Robot-side deployment must import `ego2g1`
   (which applies `gemma_patch` and declares the flags) on an openpi
   checkout whose gemma.py passes the fingerprint guard; the recorded
   upstream commit lives on the fork branch alongside the `ego2g1/`
   folder.

### Trade-off logic

- Branch-on-rank (two code paths) was chosen over "always 3-D internally"
  (one code path) to guarantee **bitwise identity** for every existing
  config — the strongest possible property for real-robot trust in the
  first model-code edit. Cost: a rank branch in `RMSNorm` and in
  `embed_suffix`, i.e. mild permanent complexity.
- Accepting both `(b,)` and `(b, s)` at the `embed_suffix` boundary (vs
  forcing all callers to broadcast) keeps `compute_loss`/`sample_actions`
  diffs at zero for this entry — least-edit, at the cost of a looser
  signature.
- Per-token modulation Dense does s× more FLOPs than scalar (per layer,
  ~s × width→3·width), negligible against attention + FFN.

### ⚠ Note on optimality — read before extending

This spec is one defensible point in the design space, **not** a claim of
optimality. Known alternatives deliberately not taken: (a) a single
always-3-D code path (simpler invariants, but loses bitwise identity with
stock and edits more call sites); (b) refactoring timestep conditioning
into its own module upstream-PR-style (cleanest long-term, largest diff);
(c) strict `(b, s)`-only signature (harder to misuse from future RTC
code, touches both call sites). A future implementer should re-derive the
best trade-off for *their* feature set rather than pattern-match this
entry, with two standing constraints: maximize extensibility for the
features actually planned, and **minimize the diff against stock openpi**
(every deviating line is merge burden on upstream pulls and a deployment
hazard on the robot).

### Evaluation plan

Pure-refactor entry: correctness gates, no training-quality metrics.

1. **Golden test, stock path**: fixed-rng `compute_loss` and
   `sample_actions` outputs for a small pi0 and pi05 config, asserted
   equal before/after the change (bitwise on same hardware).
2. **Broadcast-equivalence test**: per-token time filled with a repeated
   scalar matches the scalar path within tight `allclose` tolerance, pi05
   config (exercises the 3-D adaRMS branch end to end).
3. **Param-tree invariance**: assert identical pytree structure/shapes so
   stock checkpoints load into the fork and vice versa.
4. **Guard test**: loading a checkpoint stamped with an undeclared feature
   flag raises.
5. Throughput spot-check: stock-config step time unchanged (the branch is
   trace-time in JAX, so any regression indicates a mistake).

---

## E003 — Multi-timestep flow-matching training (K suffix blocks per sample)

**Status**: proposed — **gated on profiling evidence** (only worth doing
if training is GPU-compute-bound and prefix-dominated; measure first:
`nvidia-smi` duty cycle, cached-batch A/B, then a 3-step
`jax.profiler` trace for the prefix/suffix op-time split).
**Depends on**: E002 (per-token timestep conditioning).
**Touches**: `ego2g1/model.py` (`compute_loss` override on the `Pi0`
subclass, plus a config field — no `src/openpi` edit; vehicle revised
2026-07-11 with E002), `ego2g1/tests/`. Inference (`sample_actions`) and
checkpoints are **fully stock-compatible** — this changes only how
training FLOPs are spent.

### Problem in stock openpi

Per training step, the expensive prefix — SigLIP on every camera plus
~600–850 tokens through the 2B PaliGemma expert, forward and backward —
is amortized over exactly **one** flow-matching regression target: one
`(t, ε)` draw, 50 suffix tokens through the 300M action expert
(`compute_loss`, `pi0.py:190-218`). The suffix is a sliver of the step
FLOPs but carries all of the supervision. Replicating the noisy suffix K
times — each block at its own `(t_k, ε_k)`, attending only to itself and
the prefix — yields K regression targets per prefix at small marginal
cost, exactly equivalent to K independent forward passes.

### Fix (implementation spec)

New `Pi0Config` field `num_flow_samples: int = 1` (K). `K = 1` must
reproduce stock behavior exactly. All changes inside `compute_loss`:

1. **Sampling**: `time: (b, K)` from the same Beta(1.5, 1) mapping;
   `noise: (b, K, ah, ad)`; `x_t`, `u_t` vectorized over K. Optional
   later refinement: stratify t over K via inverse-CDF at jittered
   quantiles (strictly lower-variance estimator, ~3 lines); ship
   unstratified first to keep the diff minimal.
2. **Suffix embedding — flat, no vmap, no batch-folding**: reshape `x_t`
   to `(b, K*ah, ad)` and build per-token time `(b, K*ah)` by repeating
   each `t_k` over its block, then call `embed_suffix` **once** — this is
   why E002 exists. All suffix ops are per-token (Linear/MLP), so one
   flat call is the efficient shape: single kernel launches, no K-loop,
   no prefix duplication. For non-pi05, the state token is replicated
   per block (block size S = ah + 1; pi05: S = ah).
3. **Attention mask — built explicitly, bypassing `make_attn_mask`**: the
   cumsum construction (`pi0.py:19-44`) can only express block-*causal*
   masks (block 2 would see block 1), so it cannot express this. Build:
   prefix→prefix submask via stock `make_attn_mask` on the prefix alone;
   suffix rows = (valid-prefix broadcast) ⊕ block-diagonal, where the
   block-diagonal is `block_id[:, None] == block_id[None, :]` with
   `block_id = arange(K*S) // S`. Prefix rows never attend suffix
   (unchanged). O(N²) bool mask, constructed once per step — negligible.
4. **Positions — every block reuses the same RoPE positions** `P .. P+S-1`
   (`prefix_len + tile(arange(S), K)`), *not* a continued cumsum. This is
   the correctness subtlety: it is what makes each block exactly match a
   single-sample forward pass and match inference-time geometry.
5. **Loss**: `v_t` → reshape `(b, K, ah, ad)`, MSE against `u_t`, mean
   over K (and action dim, respecting `action_dim_actual`), returning the
   stock `(*b, ah)` shape — downstream `train_step` reduction and E001's
   per-slot loss decomposition keep working unchanged.

Cost model (to check against the profiler): sequence grows from `P + S`
to `P + K·S`; attention FLOPs/memory scale with the square (e.g. P ≈ 816,
S = 50, K = 4 → ~1.38×); suffix-expert FFN scales ×K but that expert is
~300M vs 2B; SigLIP and prefix-expert costs are unchanged. Expected:
~1.2–1.4× step time for K = 4–5 → ~3–4× flow supervision per GPU-hour.
Keep K ≤ 8: the dense-attention waste on masked cross-block pairs grows
as (K·S)².

### Trade-off logic

- The gain is **variance reduction on the (t, ε) expectation only** — the
  K losses share one observation and one ground-truth chunk, so this is
  *not* K× batch size, and its value is largest in few-epoch/pretraining
  regimes. In our many-epoch small-data fine-tuning, fresh (t, ε) draws
  already accumulate across epochs; the honest claim is "faster
  convergence per GPU-hour", not "better final model". Hence the
  profiling gate: if the step is dataloader-bound or suffix-heavy, skip.
- Mean-over-K (vs sum) keeps loss scale and lr schedule semantics
  unchanged at any K.
- Checkpoints remain stock-servable (identical params and inference
  semantics) — deployment risk profile is *lower* than E002's future
  users like train-RTC; no new serving-side requirements beyond the E002
  stamp.

### ⚠ Note on optimality — read before extending

The dense masked-attention formulation wastes compute on masked-out
cross-block pairs and was chosen because it is the **smallest correct
diff**, not the fastest possible design. Alternatives a future
implementer should genuinely reconsider instead of inheriting this one:
(a) segment-id/block-sparse attention kernels (flash-attention with
segment IDs, TPU splash attention) that skip masked blocks entirely —
right answer if K needs to grow; (b) computing the prefix once and
running suffix blocks folded into the *batch* dimension against a shared
(gradient-carrying) KV cache — no quadratic waste, but requires
restructuring the joint-forward training path and taking gradients
through the cache path, a much larger deviation from stock; (c) doing
nothing and raising batch size, which is the better use of the same
memory whenever data diversity, not (t, ε) variance, dominates gradient
noise. Same standing constraints as E002: re-derive extensibility for the
actual feature set, and minimize the diff against stock openpi.

### Evaluation plan

1. **Exact-equivalence test (the load-bearing one)**: at K = 4, fixed
   weights and inputs, block k's `v_t` must `allclose` (tight tol) the
   stock single-sample forward at the same `(t_k, ε_k)` — catches every
   plausible silent bug (mask leaking across blocks, wrong positions,
   adaRMS misalignment). Plus K = 1 golden test against stock.
2. **Step-time ratio** vs K ∈ {1, 4, 8} on the real config, after the
   E00x profiling instrumentation — validates the cost model; abort if
   K = 4 costs > ~1.6×.
3. **Loss-vs-wall-clock** on the fine-tuning run: same-config runs K = 1
   vs K = 4 to matched GPU-hours, compare validation per-slot error
   (E001 metric 1). Decision rule: **keep** if K = 4 reaches the K = 1
   endpoint quality in measurably less wall-clock (or better quality at
   equal wall-clock); **revert to K = 1** (config-only, no code removal
   needed) if curves are indistinguishable — the mechanism predicts a
   throughput win, and if it fails to materialize, the added sequence
   length is pure cost.
4. Sanity: per-slot loss decomposition (E001 metric 3) unchanged in
   *shape* at K > 1 — a distorted profile would indicate the loss
   reduction over K is wrong.
