# Audit: `ego2g1/deploy` + `ego2g1/serve`

Scope: `third_party/openpi/ego2g1/deploy/{loop,chunk,trajectory,client,safety,kinematics,dds,camera,check,__main__}.py`
and `third_party/openpi/ego2g1/serve/{policy,rtc,__main__}.py`, traced through
`ego2g1/{model,transforms,data_config,config,common}.py` and the openpi Policy/websocket layer.

Baseline: the 48 existing tests in `test_deploy.py` / `test_deploy_loop.py` / `test_serve.py` / `test_common.py` all pass.
Every finding below was confirmed by execution, not just by reading.

---

## Summary

| # | Severity | Finding | Where |
|---|---|---|---|
| 1 | **Critical** | Cold-start e-stop: the watchdog damps the robot during the first inference | `loop.py:187`, `safety.py:42` |
| 2 | **Critical** | RTC guidance never covers the slot the client actually splices at | `rtc.py:71`, `loop.py:322` |
| 3 | High | `DelayBudget.d` is unbounded; a slow inference bricks the loop | `client.py:64` |
| 4 | High | The state's hand block is ~3 ticks in the future relative to its eef block | `loop.py:222` |
| 5 | High | Camera staleness is never checked — a dead stream runs the policy open-loop | `camera.py:83`, `loop.py:245` |
| 6 | Medium | `serve --record` crashes on startup (`AttributeError`) | `serve/__main__.py:94` |
| 7 | Medium | Server `--rtc False` + async client ⇒ hard discontinuity at every seam | `rtc.py:109`, `loop.py:259` |
| 8 | Medium | The RTC prefix's zero-padding can carry non-zero guidance weight | `chunk.py:140` |
| 9 | Medium | Guided RTC applies its guidance error to the 2 *untrained* padding dims | `model.py:334` |
| 10 | Low | `check.replay` builds a safety `Clamp` and never applies it | `check.py:229` |
| 11 | Low | `LoopConfig.startup_ramp_s` is dead config; the first chunk snaps | `loop.py:59` |
| 12 | Low | `G1DDS._msg` is mutated from multiple threads without a lock | `dds.py:254` |
| 13 | **High** | Splicing after a hold lurches: the clamp bounds step magnitude but not rate | `trajectory.py:57`, `loop.py:338` |
| 14 | **High** | The pinned (train-time RTC) sampler can pin zero-padding as ground truth | `model.py:207` |

> Finding #14 was surfaced while designing the fix; it is written up in
> [DEPLOY_SERVE_FIXES.md](DEPLOY_SERVE_FIXES.md) §0, alongside the change that closes it.

---

## 1. Cold-start e-stop — the robot damps before it ever moves  **[Critical]**

`DeployLoop.start()` seeds the trajectory with a *single* knot, so `traj_arm.runway(now)` is `0.0`
from the very first control tick. `_control` calls `watchdog.check_starvation(runway, now)` on
every tick ([loop.py:187](third_party/openpi/ego2g1/deploy/loop.py#L187)), and `check_starvation`
is duration-based with `max_starvation = 1.0 s`
([safety.py:42](third_party/openpi/ego2g1/deploy/safety.py#L42)).

Nothing has been spliced yet, so the starvation timer is already running while the **first**
inference is in flight. A cold policy server JIT-compiles π0.5 on its first request — tens of
seconds, not milliseconds. The loop trips the e-stop and damps the arm before a single action is
executed.

Reproduced against the real `DeployLoop` with the repo's own fakes, using a 1.3 s first inference
(far *shorter* than a real cold start):

```
TRIPPED after 1.3s first inference: no trajectory for 1.0s — planner is dead
lp.stats["chunks"] == 0        # e-stopped before the first chunk ever landed
dds.damped is True
```

The same loop with a 0.25 s inference never trips, which localizes it precisely: the bug is
"first call is slow", and a cold server is exactly that.

**Fix.** Don't arm starvation detection until the first chunk has been spliced (e.g. gate on
`self.stats["chunks"] > 0`), or give the first inference an explicit grace window. Warming the
server is a mitigation, not a fix — any request that stalls past 1 s while the plan is empty
does this.

---

## 2. RTC guidance never covers the splice slot  **[Critical]**

This is the one that quietly makes RTC do nothing at the seam.

The client promises the server a delay of `d` ticks and, when the chunk lands, installs it at
slot `d` — `start = p["d"]` ([loop.py:322](third_party/openpi/ego2g1/deploy/loop.py#L322)).
So **slot `d` is the first action of the new chunk that ever executes**, and it is the slot that
must be continuous with what the robot just did.

The server builds the guidance mask with
`start = min(d, execution_horizon)` ([rtc.py:71](third_party/openpi/ego2g1/serve/rtc.py#L71)),
`execution_horizon = 10` by default. The mask is 1.0 on `[0, start)`, decays over
`[start, execution_horizon)`, and is **0.0 from `execution_horizon` onward**.

When `d >= execution_horizon` the decay region is empty and the mask is `ones(10) + zeros(40)`.
The slot the client splices at — slot `d` ≥ 10 — has weight **exactly 0**. It is generated
freely. RTC has constrained only slots the client throws away.

This is not an edge case, it is the default:

```
serve  execution_horizon = 10       (serve/__main__.py:49)
deploy initial_d         = 12       (deploy/__main__.py:61)

weights[8:16] = [1. 1. 0. 0. 0. 0. 0. 0.]     weight at splice slot d=12 -> 0.0
```

and the adaptive budget only makes it worse, for every realistic latency:

```
latency 300 ms -> d=11    guidance weight at splice slot = 0.0
latency 350 ms -> d=13    guidance weight at splice slot = 0.0
latency 400 ms -> d=14    guidance weight at splice slot = 0.0
latency 500 ms -> d=18    guidance weight at splice slot = 0.0
```

Driving the real loop at 400 ms latency, every replan sent `d=12` against a server whose
`execution_horizon` is 10.

Slots `[execution_horizon, d)` are also unconstrained even though the robot genuinely executed the
old chunk's actions for those instants — so the new chunk isn't required to agree with them either.

The root cause is structural: **`execution_horizon` does not appear anywhere in `deploy/`.**
The client picks `d` with no knowledge of the server's overlap window, even though
`PolicyClient` already receives it in the handshake (`self.rtc["execution_horizon"]`,
[client.py:116](third_party/openpi/ego2g1/deploy/client.py#L116)) and never reads it.

The invariant the paper needs is `d < execution_horizon`. Nothing enforces or even checks it.

**Fix.** Pick one and make it load-bearing:
- server: derive the mask with `end = max(execution_horizon, d + margin)`; or
- client: clamp `d <= client.rtc["execution_horizon"] - 1` and raise if the budget wants more.

Either way, add an assertion — silent clamping is what hid this.

---

## 3. `DelayBudget.d` is unbounded  **[High]**

`d = max(1, ceil(q95 * headroom * fps))` with no ceiling
([client.py:64](third_party/openpi/ego2g1/deploy/client.py#L64)). A 2 s inference (GPU
contention, another process, a swap) yields `d = 69` against `H = 50`. Then:

- `ChunkQueue.replace(actions, anchor, d)` clips the index to `H` → the chunk is installed with
  **zero usable actions** (`remaining() == 0`, `pop() is None`);
- `trigger = H - d - replan_margin` collapses to `<= 0`, so `_should_infer` fires immediately and
  the loop spins inferring chunks it can never use;
- the trajectory drains and finding #1's starvation trip is the only thing that stops it.

Confirmed:

```python
b = DelayBudget(fps=30, initial=12)
for _ in range(10): b.observe(2.0)
b.d                      # 69  >= H
q.replace(chunk, anchor, b.d)
q.remaining()            # 0
```

**Fix.** Cap `d` at `min(execution_horizon - 1, H - replan_margin - 1)` and log loudly when the
cap binds — a budget that wants more than the horizon means the server is too slow to drive this
robot, and it should say so rather than degrade into a spin.

---

## 4. The observation state's hand block leads its eef block by ~3 ticks  **[High]**

`kinematics.state(arm_q, self._last_hand)` builds the 30-dim observation: 18 eef dims from the
joints measured **now**, 12 hand dims from `self._last_hand`.

But `_last_hand` is set inside `_top_up` ([loop.py:222](third_party/openpi/ego2g1/deploy/loop.py#L222)),
which pops actions *ahead of the wall clock* to keep `lookahead_s` of joint trajectory queued.
`_last_hand` therefore holds the command for the **furthest-future** slot it popped — not the one
currently being emitted. The docstring's intent ("the last COMMAND we sent") is right; the value
is `lookahead_s` early.

Measured on the real loop with a chunk whose hand command ramps 0→1 across the 50 slots:

```
state_hand - emitted_hand:  mean +0.0648   (1 slot = 0.0200)
=> the state's hand block leads reality by ~3.2 slots  (= lookahead_s 0.10 s @ 30 Hz)
```

Training pairs the pose and the hand command at the *same* tick. Here the two halves of the state
vector are ~107 ms apart, and the skew is largest exactly when the hand is moving fastest — during
a grasp. That is a systematic OOD in 12 of the 30 state dims.

**Fix.** Use the command that is actually being emitted at the observation instant:
`self.traj_hand.eval(now)`, split per hand (it already exists and is wall-clock indexed).

---

## 5. Camera staleness is never checked  **[High]**

`HeadCamera.age()` exists ([camera.py:83](third_party/openpi/ego2g1/deploy/camera.py#L83)) and is
called by nothing outside `connect()`. The loop's only image check is
`if image is None: watchdog.trip(...)` ([loop.py:245](third_party/openpi/ego2g1/deploy/loop.py#L245)),
and `read()` returns the last frame forever once one has arrived.

If the ZMQ stream from the G1's `image_server` dies mid-episode — board reboot, Wi-Fi drop, the
`_pump` thread's `except: continue` swallowing a persistent error — the policy keeps being fed a
**frozen frame** while the arm keeps moving on its predictions. `SafetyLimits` has no image-age
field at all.

**Fix.** Add `max_image_age` to `SafetyLimits` and a `watchdog.check_image_age(self.cam.age())`
next to the existing `check_state_age`. The principle the module docstring already states for
lowstate — "If we cannot SEE the robot, we must not command it" — applies verbatim to the camera.

---

## 6. `serve --record` crashes on startup  **[Medium]**

```python
meta = policy.metadata                                    # line 74, fine
if args.record:
    policy = _openpi_policy.PolicyRecorder(policy, ...)   # line 90
websocket_policy_server.WebsocketPolicyServer(
    policy=policy, ..., metadata=policy.metadata)         # line 94  <-- recorder
```

`PolicyRecorder` defines no `metadata` property, and neither does `BasePolicy`:

```
AttributeError: 'PolicyRecorder' object has no attribute 'metadata'
```

Stock `scripts/serve_policy.py` gets this right — it saves `policy_metadata = policy.metadata`
*before* wrapping — so this is a regression introduced by the rewrite. `meta` is already in scope;
`metadata=meta` is the whole fix.

---

## 7. Server `--rtc False` + async client ⇒ a hard seam every chunk  **[Medium]**

`select_sampler` returns `PLAIN` whenever `rtc_enabled` is False, even with a prefix in hand
([rtc.py:109](third_party/openpi/ego2g1/serve/rtc.py#L109)). The returned chunk has had *no*
guidance applied.

The client is unaware. It still sends `d = self.budget.d` and still splices at `start = p["d"]`,
discarding slots `[0, d)` of a chunk that was never asked to agree with anything the robot did.
Slot `d` is unrelated to the current motion — a step, smoothed only by the joint clamp.

`PolicyClient` reads `enabled` out of the handshake metadata (`self.rtc["enabled"]`) and the loop
never looks at it.

**Fix.** If the server advertises `rtc.enabled == False`, the client must run blocking (or send
`d = 0` and accept the pause). Same root cause as #2: the client's `d` and the server's sampler
are decoupled when they are two halves of one contract.

---

## 8. The RTC prefix's zero-padding is not provably weightless  **[Medium]**

`ChunkQueue.rtc_prefix` zero-pads the leftover out to `(H, 30)` and asserts in its docstring:
*"The padding is never read: prefix_weights() is 0 past the overlap."*
([chunk.py:140](third_party/openpi/ego2g1/deploy/chunk.py#L140))

That holds only while `len(leftover) >= execution_horizon`. It is not enforced anywhere. If the
control thread falls behind (large `m`), rows *inside* the weighted window are pure zeros:

```python
q.replace(actions, anchor, 0)            # H = 50
prefix = q.rtc_prefix(anchor_new, 45)    # 45 ticks elapsed
prefix[5:]                               # all zeros — padding
prefix_weights(3, 10, 50)[5:10]          # non-zero!  the model is guided toward the padding
```

And a zero vec9 is not a pose: `rot6d_to_mat(zeros(6))` returns the **zero matrix** (det = 0), not
a rotation. So the guidance target in those rows is meaningless, not merely stale.

The LeRobot reference handles this explicitly —
`if execution_horizon > prev_chunk_left_over.shape[1]: execution_horizon = prev_chunk_left_over.shape[1]`
(`policy_adapter/rtc/modeling_rtc.py:197`). This port dropped that clamp.

**Fix.** Send the true leftover length alongside the prefix and let the server shrink
`execution_horizon` to it, exactly as the reference does; or assert the invariant client-side.

---

## 9. Guided RTC guides the two *untrained* padding dims  **[Medium]**

`action_dim = 32`, `action_dim_actual = 30` — the model has two padding dims. `compute_loss`
slices the loss to `action_dim_actual` ([model.py:169](third_party/openpi/ego2g1/model.py#L169)),
so the model's output on dims 30:32 is **never trained** and is arbitrary.

`sample_actions_guided` forms its guidance error across the full model width
([model.py:334](third_party/openpi/ego2g1/model.py#L334)):

```python
err = (prefix_actions - x_1) * w          # (b, ah, 32)
correction = vjp_fn(err)[0]               # J^T err — mixes all 32 dims
```

`prefix_actions` is zero on the pad dims (from `PadStatesAndActions`), `x_1` is garbage there, so
`err[..., 30:]` is a nonzero, meaningless residual — and `vjp_fn` back-propagates it through the
action expert, **contaminating the correction on the 30 real dims**.

Harmless for `use_vjp=False` (the identity-Jacobian A/B) and for the pinned sampler, which is
probably why it survived.

**Fix.** Zero the error on the padding dims: `err = err.at[..., action_dim_actual:].set(0.0)`,
mirroring what `compute_loss` already does.

---

## 10–12. Lower-severity

**10. `check.replay` never applies its clamp.** It constructs `clamp = _safety.Clamp(...)` and
calls `clamp.reset(q0)`, then emits `q = traj.eval(t); d.send_arm(q)` — the clamp is never
invoked, and `max_step` is a no-op argument
([check.py:229](third_party/openpi/ego2g1/deploy/check.py#L229)). This is the rung that drives the
**real arm**; the 3 s ramp still protects the approach, but a corrupt frame inside the recorded
episode goes straight to the wire with no rate limit.

**11. `LoopConfig.startup_ramp_s` is dead config.** Declared and documented ("Ramp to the first
chunk's starting posture over this long, instead of snapping to it"), referenced exactly once —
its own declaration. The first chunk's first knot is pushed at `now + 1/fps`, so the arm *does*
snap toward it, rate-limited only by the joint clamp. Implement it or delete it; right now it
reads as a safety feature that is silently absent.

**12. `G1DDS._msg` is mutated without a lock.** `send_arm` (arm-emitter thread) and `damp()`
(whichever thread trips the watchdog) both mutate the shared `LowCmd_` and recompute its CRC
([dds.py:254](third_party/openpi/ego2g1/deploy/dds.py#L254)). Interleaved, one can publish a
message whose CRC was computed over a different field set — the firmware silently drops it.
`damp()` publishes 5× so it survives in practice, but a lock around `_msg` would make it airtight.

Also minor: `check.listen`'s docstring claims "No publishers, nothing commanded", but `connect()`
does initialize the `rt/lowcmd` and Brainco publishers (it never writes, so it is safe — but that
claim is the whole point of the rung).

---

## 13. Splicing after a hold lurches — the clamp bounds magnitude, not rate  **[High]**

Two budgets are easy to conflate here, so first, plainly:

- **runway** = slots of the *old* chunk still queued when inference fires. Bounds how long the
  GPU may take before the robot runs out of plan.
- **`execution_horizon`** = the width of the RTC guidance mask over the *new* chunk (finding #2).
  Unrelated to running out of plan.

The runway is self-regulating: `trigger = H - d - replan_margin`, so as `d` grows the loop fires
earlier and the plan remaining at trigger is always ≈ `d + replan_margin + lookahead`. The slack is
therefore ≈ `replan_margin + lookahead` ≈ 11 ticks (~0.37 s) *regardless of d*. Good design. It only
collapses when `d + replan_margin >= H` — finding #3.

But when an inference *does* outlast the runway, the recovery is wrong. Three things happen:

1. The emitter runs off the end of the plan and **holds the last knot** — the arm freezes stiff.
2. `elapsed > d`, so the splice skips to `start = min(elapsed, H-1)`. You get `H - elapsed` usable
   actions — and if `elapsed >= H`, **exactly one**, then an immediate replan (a spin).
3. **The resume is a lurch.** `_maybe_splice` calls `traj_arm.replace_after(now, [])`, which keeps
   knots `<= now` — including the *stale* last knot from before the freeze. `_top_up` then pushes the
   new chunk's first knot at `now + dt`. The resulting segment spans `[t_stale, now+dt]`, but the
   emitter evaluates at `now` — already ~92% of the way along it. The interpolation alpha is near 1
   the instant the knot is pushed:

```
emit before splice   : 0.0000
emit 2 ms later      : 0.1392
JUMP in one 2 ms emit: 0.1392 rad  ->  70 rad/s instantaneous
intended per-emit step: 0.0090 rad (4.5 rad/s)   -> 15x the intended rate limit
```

`Clamp` bounds the step *magnitude* (0.15 rad per 30 Hz knot) but not its *rate*, because the
segment it is meant to be spread across started while the arm was frozen. Measured on the full loop
with a 1.2 s inference: **0.1466 rad between consecutive 200 Hz emits**, where the design intends
~0.0225.

This is exactly the failure `dds.py`'s own header warns about — *"a step change in q_target is a
torque spike proportional to the jump. The 30 Hz action stream must be interpolated up to 500 Hz
before it gets here; that is TrajectoryBuffer's job."* After a hold, TrajectoryBuffer is not doing
that job.

(If the freeze lasts past 1.0 s, finding #1's starvation watchdog e-stops first — so an overrun is a
race between a lurch and a damp.)

**Fix.** On splice, re-seed the buffer at the currently-emitted configuration rather than leaving a
stale knot to interpolate from: `traj.replace_after(now, [])` → `traj.seed(now, traj.eval(now))`, so
the new segment spans `[now, now+dt]` and the clamp's 0.15 rad is spread across a full 33 ms as
intended. Same for `traj_hand`.

---

## What is correct (traced, not assumed)

Worth stating, because most of it is subtle and it all holds up:

- **The chunk index arithmetic.** `t_k = anchor_time + (k+1)/fps`, splice at `start = d` with
  `anchor_time = t_request`, prefix aligned by wall clock as `m = (now - anchor_time) * fps`, and
  `new slot i ↔ old slot m+i`. I traced all three against each other and against the not-in-flight
  branch; they are mutually consistent, and the `in_flight` distinction at
  [loop.py:309](third_party/openpi/ego2g1/deploy/loop.py#L309) (don't skip `elapsed` slots when
  nothing was executing) is a real bug that was correctly avoided.
- **The guided-RTC sign convention.** With openpi's `x_t = t·noise + (1-t)·x_1`, `v = noise - x_1`:
  `clean(x) = x - t·v` recovers `x_1` exactly, and since `dt < 0`, `v_guided = v - gw·correction`
  moves `x_1` *toward* the target. Correct, and correctly distinguished from PI's flipped-τ
  reference, which would have inverted it.
- **`se3.reanchor_chunk`.** `δ' = (T_new⁻¹ T_old) δ` preserves the absolute target; hand dims pass
  through as absolute. Right.
- **The prefix riding the training input chain** (`serve/policy.py:174-176`) rather than reusing the
  previous chunk's model-space tensor. This is the correct — and non-obvious — way to build the
  guidance target: it lands each row at its *destination* slot's normalization constants.
- **`TrajectoryBuffer`.** Holds past the last knot rather than extrapolating; drops out-of-order
  knots instead of rewinding. (`replace_after` keeping the segment the emitter is currently inside is
  right *while a chunk is running* — it is only wrong after a hold, which is finding #13.)
- **The e-stop publishes damping rather than stopping.** The firmware holds the last setpoint
  forever, so "stop publishing" is not a stop. `damp()` gets this right and latches.
- **`kinematics`** reuses `data_extraction/sim/g1.py` rather than reimplementing the IK, with
  separate `MjData` for FK and IK, and `tracking_error` reads the backend *after* `solve_tick`
  has written the solution back into it (it does).

---

## Suggested order of work

1. #1, #2, #13 — none is optional before hardware. #1 stops the first run from happening at all;
   #2 means RTC is decorative; #13 is a torque spike into the real arm.
2. #3 and #7 — both are the same missing client↔server contract as #2; fix them together.
3. #4 and #5 — silent model-quality / safety gaps that will present as "the policy is bad".
4. #6, #8, #9 — correctness cleanups.
5. #10–12 — hygiene, but #10 touches the real arm.
