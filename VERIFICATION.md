# `ego2g1/{deploy,serve}` — findings for independent verification

**Subject:** `third_party/openpi/ego2g1/deploy/` and `third_party/openpi/ego2g1/serve/` (both
untracked/new), traced through `ego2g1/{model,transforms,data_config,config,common,chunk_math}.py`
and openpi's `Policy` / websocket layer.

**Repository state:** openpi fork at `813e653`; `src/openpi/` is **unmodified** vs. upstream
merge-base `15a9616` (verified: `git diff --name-only 15a9616 HEAD` lists nothing under `src/`).

**Baseline:** the package's own 48 tests (`ego2g1/tests/test_{deploy,deploy_loop,serve,common}.py`)
**all pass**. None of the findings below is caught by them. See §Blind-spot for why.

## Reproducing

```bash
cd third_party/openpi
../../.venv/bin/python -m pytest ../../audit_repro/test_findings.py -q -s
# -> 21 passed
```

`audit_repro/test_findings.py` contains one or more tests per finding. **Each test asserts the bug**
— a PASS means the defect is present. After a correct fix, the corresponding test must FAIL. All
outputs quoted below are verbatim from that run.

## Confidence key

| | |
|---|---|
| **Observed** | reproduced at runtime; the quoted output is the observation |
| **Analytic** | proven from code structure + a reachability argument; the runtime effect is not directly measured |

I flag the analytic ones explicitly and state what I did *not* verify. No hardware was involved: all
runtime evidence uses the package's own fakes (`FakeDDS`, `FakeCamera`, `FakePolicy` from
`ego2g1/tests/test_deploy_loop.py`) driving the **real** `DeployLoop`, `TrajectoryBuffer`, `Clamp`,
`Watchdog`, `ChunkQueue`, and `Kinematics` (real MuJoCo + mink IK).

---

# Findings

## F1 — Cold-start e-stop: the robot is damped during its first inference — **CRITICAL, Observed**

**Claim.** `DeployLoop` trips the starvation watchdog and damps the arm before the first action
chunk ever arrives, whenever the first inference takes longer than `max_starvation` (1.0 s).

**Mechanism.**
- `loop.py:100-105` (`start`) seeds the trajectory with a **single** knot, so
  `traj_arm.runway(now) == 0.0` from the first control tick.
- `loop.py:187` calls `watchdog.check_starvation(self.traj_arm.runway(now), now)` on **every** tick.
- `safety.py:126-143` (`check_starvation`) is duration-based with
  `SafetyLimits.max_starvation = 1.0` (`safety.py:42`). It has **no** notion of "no plan exists yet";
  its signature has no `armed`/first-chunk parameter.

So the starvation timer is already running while the first inference is in flight. A cold policy
server JIT-compiles π0.5 on its first request — tens of seconds, not milliseconds.

**Evidence.** `test_F1_live_loop_estops_on_a_slow_first_inference`, real `DeployLoop`, first
inference 1.3 s (far shorter than a real cold start):

```
F1: tripped=True reason='no trajectory for 1.0s — planner is dead' chunks=0 damped=True
```

The same loop with a 0.25 s inference never trips (`ego2g1/tests/test_deploy_loop.py` passes), which
localizes the fault to "the first call is slow" — i.e. exactly a cold server.

**What would falsify it.** Evidence that the policy server is always warm before `deploy` starts, or
that `serve` performs a warmup inference. Neither exists: `serve/__main__.py` calls
`serve_forever()` immediately after `create_policy` (`serve/__main__.py:93-95`), with no warmup.

---

## F2 — The RTC guidance mask is zero at the slot the client splices at — **CRITICAL, Observed**

**Claim.** For every realistic inference latency, the guidance weight at the *only* slot that matters
— the first slot of the new chunk that actually executes — is exactly `0.0`. RTC therefore provides
no continuity constraint at the seam it exists to smooth.

**Mechanism.**
- Client: on splice, `start = p["d"]` (`loop.py:322`). Slots `[0, d)` of the new chunk are
  **discarded**; slot `d` is the first one executed.
- Server: `prefix_weights` sets `start = min(d, execution_horizon)` (`rtc.py:71`) and the mask is
  `0.0` for all `i >= execution_horizon` (`rtc.py:96`).
- Therefore when `d >= execution_horizon`, the decay region is empty, the mask is
  `ones(execution_horizon) + zeros(H - execution_horizon)`, and `w[d] == 0`.
- Defaults ship in violation: `serve/__main__.py:49` `execution_horizon = 10`;
  `deploy/__main__.py:61` `initial_d = 12`.
- `DelayBudget` only makes it worse: `d = ceil(q95 · 1.15 · 30)` (`client.py:64`).

**Evidence.**

```
F2: weights[8:16] = [1. 1. 0. 0. 0. 0. 0. 0.]   w[splice slot d=12] = 0.0
F2: latency 300 ms -> d=11  w[splice]=0.0
F2: latency 350 ms -> d=13  w[splice]=0.0
F2: latency 400 ms -> d=14  w[splice]=0.0
F2: latency 500 ms -> d=18  w[splice]=0.0
F2: d values actually sent = [12, 12, 12]  (server execution_horizon=10)
```

The last line is the **live loop** (`test_F2_live_loop_sends_d_past_the_servers_horizon`) at 400 ms
latency: every replan sent `d=12` to a server whose window is 10.

**Structural root cause.** `grep -rn execution_horizon ego2g1/deploy/` returns **nothing**. The
client picks `d` with no knowledge of the server's window, although `PolicyClient` *receives* it in
the handshake (`client.py:116`, `self.rtc = dict(cfg["rtc"])`) and never reads it. The invariant the
algorithm requires — `d < execution_horizon` — is nowhere asserted, only silently clamped, which is
what concealed it.

**Note on severity.** Slots `[0, execution_horizon)` *are* pinned, and a flow model produces smooth
chunks, so slot `d` is probably not wildly discontinuous — it is merely **unconstrained**. I am
claiming the guidance is ineffective at the seam, not that the arm necessarily jumps. Quantifying the
residual discontinuity requires a real checkpoint; I did not do that.

---

## F3 — `DelayBudget.d` is unbounded; a slow inference installs an empty chunk — **HIGH, Observed**

**Claim.** `d` has no ceiling. When `d >= H`, `ChunkQueue.replace` clips the consumption index to
`H`, so the chunk lands with zero usable actions and the loop spins.

**Mechanism.** `client.py:62-64` — `self._d = max(1, int(np.ceil(q * self.headroom * self.fps)))`.
No cap. Then `chunk.py:93` — `self._index = int(np.clip(d, 0, self.horizon))`. And
`loop.py:234` — `trigger = self.H - self.budget.d - self.cfg.replan_margin`, which goes `<= 0`.

**Evidence.** `test_F3_unbounded_delay_budget_empties_the_chunk`, ten observations of a 2 s inference:

```
F3: d=69 (H=50)
```
then `q.replace(chunk, anchor, 69)` → `q.remaining() == 0`, `q.pop() is None`, and
`H - d - replan_margin = 50 - 69 - 8 = -27 <= 0`.

**Reachability.** A 2 s inference is not exotic — GPU contention, a second process, or memory
pressure. The loop has no other guard.

---

## F4 — The observation state's hand block is ~3 ticks in the future — **HIGH, Observed**

**Claim.** The 12 hand dims of the 30-dim state sent to the policy are the command for
`now + lookahead_s`, while the 18 eef dims are the pose measured at `now`. Training pairs both at the
same tick.

**Mechanism.**
- `loop.py:243` — `state = self.kin.state(arm_q, self._last_hand)`; `arm_q` is read at `now`
  (`loop.py:241`).
- `self._last_hand` is assigned in `_top_up` (`loop.py:222`), which pops actions **ahead of the wall
  clock** to keep `lookahead_s` of joint trajectory queued (`loop.py:197`). It therefore holds the
  command for the *furthest-future* slot popped, not the one currently emitted.
- `kinematics.py:95-110` (`state`) documents the intent correctly — "hand_cmds is the LAST COMMAND we
  sent" — but the value supplied is early.

**Evidence.** `test_F4_state_hand_block_is_in_the_future`. Real loop, `lookahead_s = 0.10` (3 ticks
at 30 Hz), driven by a chunk whose hand command ramps 0→1 across the 50 slots (a closing grasp), so
one slot = 0.02:

```
F4: state_hand - emitted_hand: mean +0.0517
    => the state's hand block leads reality by ~2.6 slots (lookahead_s=0.10 @ 30 Hz == 3 ticks)
```

(The measured lead varies 2.6–3.2 slots run to run with scheduler jitter; the sign and the ~`lookahead_s`
magnitude are stable.) Ground truth for comparison is `traj_hand.eval(now)` — the command the emitter
is actually sending — which is available and unused.

**Impact claim.** This is a systematic train/serve distribution shift in 12 of 30 state dims, worst
exactly when the hand is moving fastest. I am **not** claiming a measured degradation in policy
performance; that requires a checkpoint and a rollout.

---

## F5 — Camera staleness is never checked — **HIGH, Observed (static)**

**Claim.** If the camera stream dies mid-episode, the policy is fed a frozen frame indefinitely while
the arm keeps moving. Nothing detects it.

**Mechanism.**
- `camera.py:83-85` defines `HeadCamera.age()`. Nothing outside `HeadCamera.connect` calls it:
  `.age()` does not appear in `loop.py`.
- The loop's **only** image check is `if image is None: self.watchdog.trip(...)` (`loop.py:245-248`),
  and `read()` returns the last frame forever once one has arrived (`camera.py:78-81`).
- `SafetyLimits` (`safety.py:31-47`) has no image/frame/camera field.
- `camera.py:66-69` — `_pump` swallows every exception silently (`except Exception: time.sleep(0.01);
  continue`), so a persistent read failure is invisible.

**Evidence.** `test_F5_camera_age_has_no_consumer` asserts all four facts.

**Caveat.** Verified as an *absent guard*, not by killing a live camera. The absence is unambiguous.

---

## F6 — `serve --record` crashes on startup — **MEDIUM, Observed**

**Claim.** `python -m ego2g1.serve --record` raises `AttributeError` before serving.

**Mechanism.** `serve/__main__.py`:
- line 74: `meta = policy.metadata` (fine)
- line 90: `policy = _openpi_policy.PolicyRecorder(policy, "policy_records")`
- line 94: `metadata=policy.metadata` — `policy` is now the **recorder**.

`PolicyRecorder` (`src/openpi/policies/policy.py:113-135`) defines no `metadata` property, and
neither does `BasePolicy` (`packages/openpi-client/src/openpi_client/base_policy.py`).

**Evidence.** `test_F6_policy_recorder_has_no_metadata` constructs a `PolicyRecorder` around a policy
that *does* have `.metadata` and asserts `AttributeError` on access. It also asserts the source
ordering (`index("PolicyRecorder(policy") < index("metadata=policy.metadata")`).

**This is a regression.** Stock `scripts/serve_policy.py` does it correctly:
`policy_metadata = policy.metadata` (line 101) **before** the wrap (line 105), then passes
`metadata=policy_metadata` (line 115). The test asserts that ordering too.

---

## F7 — Server `--rtc False` + async client ⇒ unguided chunk spliced at slot `d` — **MEDIUM, Analytic**

**Claim.** If the server is run with RTC disabled, it returns a chunk generated with no guidance, but
the client still discards slots `[0, d)` and begins at slot `d`. The client never learns this.

**Mechanism.** `rtc.py:107-111` — `select_sampler` returns `PLAIN` whenever `rtc_enabled` is False,
even with a prefix present. Meanwhile `loop.py:259` still sets `d = self.budget.d`, and `loop.py:322`
still splices at `start = p["d"]`. `PolicyClient` receives `rtc.enabled` (`client.py:116`) and the
loop never reads it.

**Evidence.** `test_F7_loop_ignores_the_servers_rtc_enabled_flag` asserts `select_sampler(...
rtc_enabled=False) is PLAIN`, that the loop still sends `budget.d` and splices at `p["d"]`, and that
`client.rtc` appears nowhere in `loop.py`.

**Analytic, not observed.** I did not measure the resulting seam discontinuity — it would require
running the real server with `--rtc False`. The *coupling gap* is certain; its magnitude is not
quantified. The joint clamp (`safety.py:62-81`) would convert a step into a slew, bounding the harm.

---

## F8 — The RTC prefix's zero padding can carry non-zero guidance weight — **MEDIUM, Observed**

**Claim.** `ChunkQueue.rtc_prefix` zero-pads to `(H, 30)` and documents that "the padding is never
read: prefix_weights() is 0 past the overlap" (`chunk.py:126`). That holds only while
`len(leftover) >= execution_horizon`, which nothing enforces.

**Mechanism.** `chunk.py:140-142` pads with zeros; the number of real rows is **never communicated**
to the server (`grep n_real` in `chunk.py` and `rtc.py` → nothing). A zero vec9 is not a pose:
`rot6d_to_mat(zeros(6))` returns the **zero matrix** (det 0), by `chunk_math.py:28-35` with its
`1e-12` guard.

**Evidence.** `test_F8_padding_is_weighted_when_the_leftover_is_short`. Control loop 45 ticks behind
(`m=45`, `H=50` ⇒ 5 real rows), `d=3`, `execution_horizon=10`:

```
F8: n_real=5, but weights[5:10] = [0.316 0.189 0.099 0.041 0.01 ] (non-zero on padding)
```

**Reachability.** Requires the control thread to fall behind such that `H - m < execution_horizon`.
Under nominal timing `m ≈ 27`, so `n_real ≈ 23 > 10` and the invariant holds by luck. It is broken
by the degraded regime of F13.

**Corroboration.** The LeRobot reference this was ported from clamps exactly this:
`unitree-deploy-main/policy_adapter/rtc/modeling_rtc.py:197` —
`if execution_horizon > prev_chunk_left_over.shape[1]: execution_horizon = prev_chunk_left_over.shape[1]`.
The port dropped it.

---

## F9 — Guided RTC applies its guidance error to two *untrained* padding dims — **MEDIUM, Analytic**

**Claim.** `sample_actions_guided` forms its guidance error across the full 32-dim model action
space, but the model is never trained on dims 30:32, so the error there is meaningless — and the VJP
back-propagates it into the 30 real dims.

**Mechanism.**
- `config.py:41-42` — `action_dim = 32`, `action_dim_actual = 30`.
- `model.py:168-169` (`compute_loss`) — `loss = loss[..., : self.action_dim_actual]`. Dims 30:32
  therefore receive **no gradient during training**; the model's output there is unconstrained.
- `model.py:334` — `err = (prefix_actions - x_1) * w` — shape `(b, ah, 32)`. `prefix_actions` is zero
  on the pad dims (from `PadStatesAndActions`); `x_1[..., 30:]` is arbitrary; so `err[..., 30:]` is a
  non-zero, meaningless residual.
- `model.py:335` — `correction = vjp_fn(err)[0]` — `Jᵀ` couples all 32 output dims back onto `x_t`
  across all dims. `action_dim_actual` appears nowhere in the method.

**Evidence.** `test_F9_guidance_error_spans_the_padding_dims` asserts each of the above from source.

**Analytic, not observed.** The *mechanism* (untrained dims → non-zero error → `Jᵀ` coupling) is
certain from the code. The **magnitude** of the contamination on the real dims is **not measured** —
it depends on the trained network's Jacobian and could be small. A verifier could settle this by
running `sample_actions_guided` twice on a real checkpoint with dims 30:32 of `prefix_actions`
perturbed, and comparing the first 30 output dims. I did not have a checkpoint.

Does not affect `use_vjp=False` (correction = err, so the pad error stays in the pad dims) nor the
pinned sampler.

---

## F10 — `check.replay` builds a safety `Clamp` and never applies it — **LOW, Observed (static)**

**Claim.** The `replay` bring-up rung — which drives the **real arm** — constructs a rate-limiting
clamp, resets it, and never invokes it. `max_step` is a no-op parameter.

**Mechanism.** `check.py:229-230` — `clamp = _safety.Clamp(_safety.SafetyLimits(max_joint_step=max_step))`,
`clamp.reset(q0)`. The emit loop (`check.py:244-251`) does `q = traj.eval(t); d.send_arm(q)` — no
clamp call.

**Evidence.** `test_F10_replay_clamp_is_dead_code` asserts the construction, the reset, the absence
of any `clamp(...)` invocation, that `max_step` is still a parameter, and that `d.send_arm(q)` is in
the body.

**Mitigating.** The 3 s ramp (`check.py:236`) is interpolated, so the *approach* is safe. The gap is
that a corrupt or non-finite frame inside the recorded episode reaches the wire unlimited.

---

## F11 — `LoopConfig.startup_ramp_s` is dead config — **LOW, Observed (static)**

**Claim.** A documented safety feature ("Ramp to the first chunk's starting posture over this long,
instead of snapping to it", `loop.py:57-59`) is never implemented.

**Evidence.** `test_F11_startup_ramp_is_never_read` — `inspect.getsource(_loop).count("startup_ramp_s") == 1`,
i.e. the declaration only. The first chunk's first knot is pushed at `now + 1/fps` (`loop.py:206`
with `_anchor_time = now`, `loop.py:319`), so the arm slews to it at the clamp limit rather than
ramping.

---

## F12 — `G1DDS._msg` is mutated and CRC'd from multiple threads without a lock — **LOW, Analytic**

**Claim.** `send_arm` (arm-emitter thread, 500 Hz) and `damp()` (whichever thread trips the watchdog)
both mutate the shared `LowCmd_` object and recompute its CRC with no mutual exclusion. Interleaved,
one can publish a message whose CRC was computed over a different field set; the firmware silently
drops such a message.

**Mechanism.** `dds.py:249-255` (`send_arm`) and `dds.py:284-293` (`damp`) both do
`self._msg.motor_cmd[i].* = ...; self._msg.crc = self._crc.Crc(self._msg); self._pub.Write(self._msg)`.
`self._lock` (`dds.py:83`) guards only `_lowstate`/`_hand_state`, not `_msg`.

**Evidence.** `test_F12_lowcmd_message_has_no_lock` asserts both methods CRC+Write the same `_msg`
and that neither takes a lock.

**Analytic; low severity.** `damp()` writes 5× (`dds.py:292-294`), so a single dropped frame is
survivable, and `_estopped` latches `send_arm` off (`dds.py:238-239`). I have **not** observed a
corrupted CRC. This is a hardening item, not a demonstrated failure.

---

## F13 — Splicing after a hold delivers a full clamp-step in one emit period — **HIGH, Observed**

**Claim.** When an inference outlasts the remaining plan, the emitter holds its last knot (arm
freezes) and, when the chunk lands, the joint command **steps** — the `Clamp` bounds the step's
*magnitude* but not its *rate*, because the interpolation segment it is spread over starts in the
past.

**Mechanism.**
- On splice, `loop.py:338-339` calls `traj_arm.replace_after(now, [])`, which keeps all knots
  `<= now` (`trajectory.py:65-67`) — **including the stale pre-freeze knot**.
- `_top_up` then pushes the new chunk's first knot at `≈ now + dt` (`loop.py:206, 218`), clamped to
  ≤ `max_joint_step` = 0.15 rad from `clamp.reset(q_now)` (`loop.py:344-346`, `safety.py:73`).
- The resulting segment spans `[t_stale, now+dt]`, but `eval` is called at `≈ now`
  (`trajectory.py:88-93`), i.e. with interpolation `alpha ≈ (now - t_stale)/(now + dt - t_stale)`,
  which is ~0.92 after a 400 ms hold. The emitter lands almost entirely on the new knot **in a single
  emit period**.

**Evidence 1** (isolated, `test_F13_splice_after_hold_steps_the_emitted_stream`) — 400 ms hold, one
0.15 rad clamp step, 500 Hz emitter:

```
F13: emit before=0.0000 after=0.1392  jump=0.1392 rad in one 2 ms emit (70 rad/s);
     intended 0.0090 rad -> 15x
```

**Evidence 2** (live loop, `test_F13_live_loop_lurches_when_inference_overruns`) — first inference
0.2 s, then 1.2 s (overruns the ~0.77 s runway), 200 Hz emitter:

```
F13: late splices=2  max joint step between 200 Hz emits = 0.1458 rad (intended 0.0225)
```

This is precisely what `dds.py:16-18` warns against: *"A step change in q_target is a torque spike
proportional to the jump. The 30 Hz action stream must be interpolated up to 500 Hz before it gets
here; that is TrajectoryBuffer's job."* After a hold, it is not doing that job.

**Interaction with F1.** If the freeze exceeds `max_starvation` (1.0 s), F1's watchdog e-stops first.
An overrun is a race between a lurch and a damp.

---

## F14 — The pinned (train-time RTC) sampler can pin zero padding as clean ground truth — **HIGH, Analytic**

**Claim.** `sample_actions_rtc` freezes slots `< d` at the values in `prefix_actions`. When
`d > n_real`, those rows are zero padding, and they are pinned as **clean, committed actions at
t=0** — conditioning the entire chunk on a target that is not a pose.

**Mechanism.** `model.py:207-209`:
```python
slot_is_prefix = jnp.arange(self.action_horizon) < d
x_init = jnp.where(slot_is_prefix[None, :, None], prefix_actions, noise)
```
`d` is not capped to the real prefix length — the server has no way to know it (see F8).

**Evidence.** `test_F14_pinned_sampler_pins_padding_when_d_exceeds_the_real_prefix` asserts the code
shape and demonstrates reachability: with the loop 42 ticks into a 50-slot chunk,

```
F14: n_real=8, d=12 -> rows 8..11 are ZERO PADDING, pinned as clean t=0 ground truth
```

**Analytic, and currently latent.** `rtc_training` defaults to `False` (`config.py:69`), so **no
shipped checkpoint takes this path today**. It is a live bug the moment an RTC-trained checkpoint is
served. Unlike F8 (which degrades softly — padding merely gets a weight), this path *asserts* the
padding as truth.

---

# Things I checked and found **correct**

Stated so the verifier can bound the review, and because several are subtle enough that a reader
might assume they are broken.

1. **Chunk index arithmetic.** `t_k = anchor_time + (k+1)/fps` (`loop.py:206`), splice at
   `start = d` with `anchor_time = t_request` (`loop.py:322, 331`), prefix aligned by wall clock as
   `m = (now - anchor_time)·fps` (`loop.py:257`). All three are mutually consistent, and the
   `in_flight` branch (`loop.py:309-319`) correctly avoids skipping `elapsed` slots when nothing was
   executing.
2. **Guided-RTC sign convention.** With openpi's `x_t = t·noise + (1-t)·x_1` and `v = noise - x_1`,
   `clean(x) = x - t·v` recovers `x_1` exactly, and since `dt < 0`, `v_guided = v - gw·correction`
   moves `x_1` *toward* the target. Correct — and correctly distinguished from PI's flipped-τ
   reference, which would have inverted the sign (`model.py:272-276`).
3. **`se3.reanchor_chunk`** (`common/se3.py:55-84`). `δ' = (T_new⁻¹ T_old) δ` preserves the absolute
   target; hand dims pass through as absolute. Correct.
4. **Routing the RTC prefix through the training input chain** (`serve/policy.py:174-176`) rather
   than reusing the previous chunk's model-space tensor. This is the correct — and non-obvious —
   construction: it lands each row at its *destination* slot's normalization constants.
5. **`TrajectoryBuffer`** holds past the last knot rather than extrapolating, and drops out-of-order
   knots rather than rewinding (`trajectory.py:42-56, 75-93`). Sound. (F13 is about `replace_after`
   *after a hold*, not about these.)
6. **The e-stop publishes damping rather than stopping** (`dds.py:273-294`), which is right: the
   firmware holds the last setpoint forever, so silence is not a stop. It latches correctly.
7. **`Kinematics`** uses separate `MjData` for FK and IK, and `tracking_error` reads the IK backend
   *after* `solve_tick` writes the solution back into it (`sim/g1.py:258-259` — `self.backend.set_qpos(q)`).
   Correct ordering.
8. **Camera eye keys.** `cam_{left,right}_high` matches the vendor's `ImageClientCamera.async_read`
   (`unitree-deploy-main/.../imageclient.py:695-696`), and BGR→RGB flip is right.
9. **The runway is self-regulating.** `trigger = H - d - replan_margin` means the plan remaining at
   trigger is always `≈ d + replan_margin + lookahead`, so slack ≈ 11 ticks regardless of `d`. This is
   good design and is *not* the source of F3 (which is about `d + replan_margin >= H`).

---

# Blind spot: why the existing 48 tests pass

`ego2g1/tests/test_deploy_loop.py::_run` fixes `initial_d=8` and `latency=0.25`. That is the **one
region of the parameter space where none of these fire**:

- `d=8 < execution_horizon=10` ⇒ F2 cannot appear (the shipped default is `d=12`).
- `latency=0.25 < max_starvation=1.0` ⇒ F1 cannot appear.
- latency well inside the runway ⇒ F13 cannot appear.
- `FakePolicy` **ignores `prev_chunk` entirely** and returns the same smooth drift chunk regardless
  (`test_deploy_loop.py:93-101`), so `test_commanded_joints_stay_continuous_across_chunk_seams` cannot
  detect a guidance failure — the fake is continuous by construction.

Any fix should be accompanied by parameterization over `d`, latency, and `n_real`.

---

# Structural note (per the "both trees self-contained" constraint)

Making **both** `ego2g1/` and `data_extraction/` self-contained requires `ego2g1/` to carry its own
copy of the MuJoCo G1 model, the IK, and the hand constants — today it reaches out via
`sys.path.insert("../..")` (`deploy/kinematics.py:28-43`, `eval_replay/scene.py:12-15`,
`eval_replay/viewer.py:106`). That means duplicating:

- `data_extraction/sim/g1.py` (G1Backend + DualArmIK) and `common/frames.py`;
- `data_extraction/hand/constants.py` (+ `fk_tables.py`) for `eval_replay`;
- `data_extraction/assets/unitree_g1/` (**38 MB**) and `revo2/` (**22 MB**).

Two consequences the verifier should weigh:

1. **This is a deliberate trade.** The duplication is the *price* of independence, and the code
   already accepts that trade once — `chunk_math.py:1-13` is an admitted byte-pinned copy of
   `data_extraction/loader/`, policed by `data_extraction/tests/test_loader_equivalence.py`. Extending
   the same pattern to the sim is consistent, but it doubles the surface that can drift.
2. **The drift is safety-critical, so it must be tested, not just asserted.** `kinematics.py:1-4`
   states the requirement plainly: *"the deployment kinematics MUST be the same kinematics that
   generated the training labels."* A copy without an equivalence test converts a compile-time
   guarantee into a silent-divergence risk. The guard should hash the MJCF + meshes and pin
   `ARM_JOINTS`, `EE_SITES`, and the IK cost/limit configuration across both trees — the same role
   `test_loader_equivalence.py` plays for the loader math.

The alternative (one copy, `data_extraction` importing `ego2g1.sim`) removes the drift risk entirely
but violates the stated constraint. Flagging the trade explicitly rather than deciding it.

Separately, `deploy/camera.py:42-43` imports from the **untracked** `unitree-deploy-main/` vendored
tree at the repo root, which is not part of either package and is not in git. That is an independent
self-containment break, and it is small: the dependency is one ZMQ SUB socket (`tcp://host:55555`,
`RCVHWM=1`) yielding a JPEG that is `cv2.imdecode`d to a `(480, 1280, 3)` BGR stereo pair and split at
`width//2` (`unitree-deploy-main/.../imageclient.py:358-362, 695-696`). ~45 lines to inline.
