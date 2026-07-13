# Fixes for the deploy/serve audit, and a structure to stop them recurring

Companion to [DEPLOY_SERVE_AUDIT.md](DEPLOY_SERVE_AUDIT.md). Part 1 fixes the 14 findings.
Part 2 argues that most of them share two root causes that are *structural*, and proposes a layout
that makes them unrepresentable.

---

# Part 0 — one more finding, surfaced while designing the fix

## 14. The pinned (train-time RTC) sampler can pin zero-padding as ground truth  **[High]**

`sample_actions_rtc` freezes slots `< d` at the values in `prefix_actions`
([model.py:207](third_party/openpi/ego2g1/model.py#L207)):

```python
slot_is_prefix = jnp.arange(self.action_horizon) < d
x_init = jnp.where(slot_is_prefix[None, :, None], prefix_actions, noise)
```

But `ChunkQueue.rtc_prefix` zero-pads the prefix to `(H, 30)`, and only the first `H - m` rows are
real. If `d > H - m` — reachable exactly when the control loop has fallen behind, i.e. finding #13's
regime — rows `H-m .. d-1` are **zero padding**, and they get pinned as clean, committed actions at
`t=0`. A zero vec9 is not a pose: `rot6d_to_mat(zeros(6))` is the zero matrix.

The guided path degrades softly here (finding #8: the padding merely gets a non-zero *weight*). The
pinned path does not degrade at all — it *asserts* the padding as truth and integrates the rest of
the chunk conditioned on it.

Same root cause as #2/#8, same fix: the server must know how many prefix rows are real. It currently
has no way to know, because the client never tells it.

---

# Part 1 — the fixes

## 1.1 The RTC contract (fixes #2, #3, #7, #8, #14)

Four findings, one root cause: **the guidance contract is split across two processes and neither
validates it.** The client picks `d`; the server picks `execution_horizon`; the client knows how many
prefix rows are real and never says so; the server clamps `d` into its own window and calls it a day.

The fix is not to add validation. It is to make the mask a **function of what the client sends**, so
there is no cross-process invariant left to violate.

### The reframing

`execution_horizon` is an *absolute* slot index, which is why it collides with `d`. Replace it with
`soft_window` — the number of slots of soft agreement **past `d`**. Then:

```
start = min(d, n_real)                    # committed: these WILL execute -> weight 1
end   = min(start + soft_window, n_real)  # soft agreement, decaying to 0
                                          # beyond `end`: free
```

Three invariants now hold **by construction**, with nothing to check:

| | |
|---|---|
| `start <= end <= n_real <= H` | padding can never carry weight — kills #8 and #14 |
| slot `d` is the *first* slot of the decay | the splice slot always has ~max weight, never 0 — kills #2 |
| the mask depends only on `(d, n_real, soft_window)` | client and server can compute the *same* array — kills #7 |

`d` no longer needs to be smaller than anything. The clamp that silently broke RTC disappears
because there is nothing left to clamp.

### `ego2g1/core/rtc.py` — the contract, importable by both sides

```python
"""The RTC guidance mask. Lives in core because it is a CONTRACT: the server
applies it, and the client must be able to reproduce it exactly to know what it
is promising. Pure numpy — the robot PC imports this without JAX."""

import dataclasses, enum, math
import numpy as np


class AttentionSchedule(str, enum.Enum):
    ZEROS = "zeros"; ONES = "ones"; LINEAR = "linear"; EXP = "exp"


@dataclasses.dataclass(frozen=True)
class PrefixSpec:
    """Everything the mask depends on that the CLIENT owns."""
    d: int          # ticks of the OLD chunk that will execute during inference
    n_real: int     # rows of prev_chunk that are real actions, not zero padding
    horizon: int

    def __post_init__(self):
        if not 0 <= self.d:                    raise ValueError(f"d={self.d} < 0")
        if not 0 < self.n_real <= self.horizon:
            raise ValueError(f"n_real={self.n_real} not in (0, {self.horizon}]")


def prefix_weights(spec: PrefixSpec, soft_window: int,
                   schedule: AttentionSchedule = AttentionSchedule.EXP) -> np.ndarray:
    """(horizon,) soft mask over the NEW chunk's slots.

      i < start            -> 1.0   committed; the robot WILL execute these, so the
                                    new chunk has no choice but to agree
      start <= i < end     -> decay; increasingly free
      i >= end             -> 0.0   generated from scratch

    start is the splice slot (`d`, capped at the real prefix length), so the slot the
    client actually begins executing is always the FIRST slot of the decay — never
    past the end of the mask. That is the whole point.
    """
    H = int(spec.horizon)
    start = min(int(spec.d), int(spec.n_real), H)
    end   = min(start + int(soft_window), int(spec.n_real), H)

    if schedule is AttentionSchedule.ZEROS:
        w = np.zeros(H, np.float32); w[:start] = 1.0; return w
    if schedule is AttentionSchedule.ONES:
        w = np.ones(H, np.float32); w[end:] = 0.0; return w

    n_decay = max(end - start, 0)
    decay = np.linspace(1.0, 0.0, n_decay + 2, dtype=np.float32)[1:-1] if n_decay else np.zeros(0, np.float32)
    if schedule is AttentionSchedule.EXP:
        decay = decay * np.expm1(decay) / (math.e - 1.0)
    return np.concatenate([np.ones(start, np.float32), decay, np.zeros(H - end, np.float32)])
```

### Client side

`ChunkQueue.rtc_prefix` must report how much of what it returns is real:

```python
def rtc_prefix(self, anchor_new: dict, start: int) -> tuple[np.ndarray, int] | tuple[None, int]:
    """-> ((H, 30) zero-padded, n_real) or (None, 0).

    n_real is the number of leading rows that are ACTUAL leftover actions. The
    server needs it: rows past it are zero padding, and a zero vec9 is not a pose
    (rot6d_to_mat(zeros) is the ZERO matrix). Nothing downstream may weight them,
    and nothing may pin them.
    """
    with self._lock:
        if self._actions is None or self._anchor is None:
            return None, 0
        s = int(np.clip(start, 0, self.horizon))
        leftover, anchor_old = self._actions[s:].copy(), dict(self._anchor)
    if len(leftover) == 0:
        return None, 0
    rehomed = se3.reanchor_chunk(leftover, anchor_old, anchor_new)
    out = np.zeros((self.horizon, layout.DIM), np.float32)
    out[: len(rehomed)] = rehomed
    return out, len(rehomed)
```

`DelayBudget` gets the cap that #3 needs — and it is a *loop-viability* cap, unrelated to the mask:

```python
def __init__(self, fps, *, initial=12, quantile=0.95, window=50, headroom=1.15, max_d: int = 16):
    ...
    self.max_d = int(max_d)
    self.capped = False

def observe(self, latency_s: float) -> None:
    with self._lock:
        self._samples.append(latency_s)
        if len(self._samples) > self._window: self._samples.pop(0)
        if len(self._samples) >= 5:
            q = float(np.quantile(self._samples, self.quantile))
            want = max(1, int(np.ceil(q * self.headroom * self.fps)))
            self._d, self.capped = min(want, self.max_d), want > self.max_d
```

and the loop supplies `max_d = min(H - replan_margin - 1, H // 2)` and refuses to keep pretending
when the cap binds:

```python
# DeployLoop.__init__
if self.budget.max_d >= self.H - self.cfg.replan_margin:
    raise ValueError(f"max_d={self.budget.max_d} leaves no chunk to execute (H={self.H})")

# in _run_infer, after budget.observe(latency)
if self.budget.capped:
    logger.error("inference latency exceeds the delay budget cap (d wants > %d ticks). "
                 "The server is too slow to drive this robot asynchronously — "
                 "use --blocking or a faster checkpoint.", self.budget.max_d)
```

### Server side

`Ego2G1Policy.infer` reads `n_real`, validates the prefix *before* it enters the transform stack,
and caps `d` for the pinned sampler too:

```python
prev_chunk = inputs.pop("prev_chunk", None)
d          = int(inputs.pop("d", 0) or 0)
n_real     = int(inputs.pop("n_real", 0) or 0)

if prev_chunk is not None:
    prev_chunk = np.asarray(prev_chunk, dtype=np.float32)
    if prev_chunk.shape != (self._action_horizon, self._action_dim_actual):
        raise ValueError(f"prev_chunk {prev_chunk.shape} != "
                         f"({self._action_horizon}, {self._action_dim_actual})")
    if not 0 < n_real <= self._action_horizon:
        raise ValueError(f"prev_chunk supplied without a valid n_real ({n_real})")
    if not np.all(np.isfinite(prev_chunk[:n_real])):
        raise ValueError("prev_chunk has non-finite rows")

spec = _rtc.PrefixSpec(d=d, n_real=n_real, horizon=self._action_horizon)
...
if sampler is _rtc.Sampler.GUIDED:
    weights = _rtc.prefix_weights(spec, self._rtc.soft_window, self._rtc.prefix_attention_schedule)
    actions = self._sample_guided(sample_rng, observation, prefix, jnp.asarray(weights), ...)
else:  # PINNED
    d_eff = min(spec.d, spec.n_real)          # FIX #14: never pin zero padding
    actions = self._sample_pinned(sample_rng, observation, prefix,
                                  jnp.asarray(d_eff, jnp.int32), num_steps=self._rtc.num_steps, ...)
```

`out["rtc"]` should echo back what was actually applied, so the client can assert:

```python
out["rtc"] = {"sampler": sampler.value, "d": d, "n_real": n_real,
              "soft_window": self._rtc.soft_window}
```

### #7: server RTC off + async client

Currently the server silently returns an unguided chunk and the client still splices it at slot `d`.
`select_sampler` should not be the only thing that knows. Make the loop refuse the combination it
cannot handle:

```python
# DeployLoop.__init__
if not client.rtc.get("enabled", True) and not cfg.blocking:
    raise ValueError(
        "the server has RTC disabled but this client is async: it would splice an "
        "unguided chunk at slot d and step the arm at every seam. "
        "Run the server with --rtc, or the client with --blocking."
    )
```

`PolicyClient` already receives `rtc.enabled` in the handshake and never looks at it. That is the
whole bug.

---

## 1.2 Cold-start e-stop (fixes #1)

Two independent changes; do both.

**(a) Don't arm starvation until there is a plan to starve.** The watchdog currently counts from
tick 0, when the loop has *by construction* not planned anything yet.

```python
# safety.py
def check_starvation(self, runway: float, now: float, *, armed: bool = True) -> None:
    """`armed` is False until the first chunk lands. Before that an empty
    trajectory is not a fault — it is the normal state of a loop waiting for its
    first inference, which on a cold server is a JIT compile, not a control fault."""
    if not armed or runway > 0.0:
        self._starving_since = None
        return
    if self._starving_since is None:
        self._starving_since = now
    elif now - self._starving_since > self.limits.max_starvation:
        self.trip(f"no trajectory for {now - self._starving_since:.1f}s — planner is dead")

# loop.py, _control
self.watchdog.check_starvation(self.traj_arm.runway(now), now, armed=self.stats["chunks"] > 0)
```

**(b) Bound the first inference explicitly**, so a hung server is still caught — just by the right
gate, with a sane timeout:

```python
# SafetyLimits
max_inference_s: float = 30.0   # generous: covers a cold JIT. A HUNG server still trips.

# loop.py, _control
with self._infer_lock:
    t_req = self._infer_started_at
if t_req is not None and now - t_req > self.limits.max_inference_s:
    self.watchdog.trip(f"inference has not returned in {now - t_req:.0f}s — server is hung")
```

**(c) Warm the server so the cliff never exists.** This is the real fix; the two above are the
safety net. Add to `serve/__main__.py`, before `serve_forever()`:

```python
def _warmup(policy) -> None:
    """Compile both sampler paths before the first client connects. Otherwise the
    robot's FIRST inference pays a 30-60 s JIT while its trajectory buffer is empty."""
    cfg = policy.metadata["ego2g1"]
    H, D = cfg["action_horizon"], cfg["action_dim"]
    obs = {"observation/image": np.zeros((224, 224, 3), np.uint8),
           "observation/state": np.zeros(D, np.float32),
           "prompt": "warmup"}
    t0 = time.monotonic(); policy.infer(dict(obs))
    logging.info("warmup: plain sampler compiled in %.1f s", time.monotonic() - t0)
    t0 = time.monotonic()
    policy.infer({**obs, "prev_chunk": np.zeros((H, D), np.float32), "d": 1, "n_real": H})
    logging.info("warmup: %s sampler compiled in %.1f s",
                 "pinned" if cfg["rtc_training"] else "guided", time.monotonic() - t0)
```

---

## 1.3 The splice-after-hold lurch (fixes #13)

`replace_after(now, [])` leaves the *stale* pre-freeze knot in the buffer, so the next pushed knot
forms a segment `[t_stale, now+dt]` that the emitter is already ~92 % of the way along. The clamp's
0.15 rad then lands in a single 2 ms emit (measured: 70 rad/s, 15× the intended rate limit).

Give `TrajectoryBuffer` an operation that expresses the actual intent — *"the plan restarts here"*:

```python
def reanchor(self, t: float) -> None:
    """Drop the future and pin a knot at exactly `t` holding the value the emitter
    is producing right now.

    replace_after() alone is not enough: it keeps the last knot BEFORE `t`, which
    after a hold can be hundreds of ms stale. The next pushed knot then spans
    [t_stale, t+dt] and the emitter, evaluating at `t`, lands almost entirely on it
    — delivering a full clamp-step in one emit period. Pinning a knot at `t` makes
    the next segment exactly one control period long, which is what the rate limit
    assumes.
    """
    with self._lock:
        if not self._t:
            return
        q = self._eval_locked(float(t))
        cut = bisect.bisect_right(self._t, float(t))
        self._t, self._q = self._t[:cut], self._q[:cut]
        if self._t and self._t[-1] == float(t):
            self._q[-1] = q
        else:
            self._t.append(float(t)); self._q.append(q)
```

(`eval` is refactored to `_eval_locked` + a locking wrapper.) Then in `_maybe_splice`:

```python
-        self.traj_arm.replace_after(now, [])
-        self.traj_hand.replace_after(now, [])
+        self.traj_arm.reanchor(now)
+        self.traj_hand.reanchor(now)
```

`replace_after` can then be deleted — `reanchor` is the only caller's actual need, and the `knots`
argument was never used with a non-empty list.

Regression test:

```python
def test_splice_after_a_hold_does_not_step_the_emitted_stream():
    tb = TrajectoryBuffer(2); tb.seed(100.0, np.zeros(2)); tb.push(100.1, np.zeros(2))
    now = 100.5                                  # held for 400 ms
    before = tb.eval(now)
    tb.reanchor(now)
    tb.push(now + 1/30, tb.eval(now) + 0.15)     # a full clamp step
    after = tb.eval(now + 0.002)                 # the next 500 Hz emit
    assert abs(after - before).max() < 0.15 / (500/30) * 1.5
```

---

## 1.4 The state's hand block is 3 ticks in the future (fixes #4)

`_last_hand` is set from the *furthest-future* slot `_top_up` popped, which is `lookahead_s` ahead of
the wall clock. Read the command that is actually being emitted instead — `traj_hand` is already
wall-clock indexed, so this is a lookup, not new bookkeeping:

```python
# loop.py
def _hand_command_at(self, now: float) -> dict:
    """The hand command the robot is being given AT `now`.

    NOT `_top_up`'s last popped action: that runs `lookahead_s` ahead of the wall
    clock, so it would put the state's 12 hand dims ~3 ticks in the future while
    its 18 eef dims are the pose measured now. Training pairs both at the same tick,
    and the skew is worst exactly when the hand is moving fastest — mid-grasp.
    """
    v = self.traj_hand.eval(now)
    if v is None:
        return {h: np.zeros(layout.HAND_DIM, np.float32) for h in layout.HANDS}
    n = layout.HAND_DIM
    return {h: np.asarray(v[i*n:(i+1)*n], np.float32) for i, h in enumerate(layout.HANDS)}

# _maybe_infer
-        state = self.kin.state(arm_q, self._last_hand)
+        state = self.kin.state(arm_q, self._hand_command_at(now))
```

`self._last_hand` and its assignment in `_top_up` are then dead — delete both.

---

## 1.5 Camera staleness (fixes #5)

`HeadCamera.age()` exists and nothing calls it. The rule the module already states for lowstate —
*"If we cannot SEE the robot, we must not command it"* — applies verbatim to the camera:

```python
# SafetyLimits
max_image_age: float = 0.5   # the head cam streams at ~30 Hz; 0.5 s is 15 dropped frames

# Watchdog
def check_image_age(self, age: float) -> None:
    self._strike("image", age > self.limits.max_image_age,
                 f"camera frame stale for {age:.2f}s — the policy would run on a frozen image")

# loop.py, _control (every tick, not just at inference)
self.watchdog.check_image_age(self.cam.age())

# loop.py, _maybe_infer — belt and braces: never infer on a stale frame
image = self.cam.read()
if image is None or self.cam.age() > self.limits.max_image_age:
    self.watchdog.trip("no fresh camera frame")
    return
```

Also fix the silent-failure path that lets this happen unnoticed — `HeadCamera._pump` swallows every
exception:

```python
 except Exception:
-    time.sleep(0.01)
-    continue
+    self._errors += 1
+    if self._errors % 100 == 1:
+        logger.exception("camera read failed (%d consecutive)", self._errors)
+    time.sleep(0.01)
+    continue
```

---

## 1.6 The small ones

**#6 — `serve --record` crashes.** `meta` is already in scope one line above:

```python
-    ).serve_forever()  # metadata=policy.metadata  <- policy is the recorder by now
+    websocket_policy_server.WebsocketPolicyServer(
+        policy=policy, host=args.host, port=args.port, metadata=meta,
+    ).serve_forever()
```

**#9 — guided RTC guides the untrained pad dims.** The loss is masked to `action_dim_actual`, so the
model's output on dims 30:32 is arbitrary; the guidance error there is meaningless and the VJP
smears it back into the real dims. Mask it, exactly as `compute_loss` does:

```python
# model.py, sample_actions_guided
dim_mask = jnp.arange(self.action_dim) < (self.action_dim_actual or self.action_dim)
w = weights[None, :, None] * dim_mask[None, None, :]   # (1, ah, ad)
```

One line, and it makes the existing `err = (prefix_actions - x_1) * w` correct with no other change.

**#10 — `check.replay` never applies its clamp.** Rate-limit the knots as they are built, which is
where the clamp belongs (it is a per-30 Hz-knot limit, not a per-emit one):

```python
 traj.seed(now, q0)
-traj.push(now + ramp_s, arm[0])
+traj.push(now + ramp_s, clamp(arm[0], ramp_s))
 for k in range(1, len(arm)):
-    traj.push(now + ramp_s + k / fps, arm[k])
+    if not np.all(np.isfinite(arm[k])):
+        raise ValueError(f"episode frame {k} has non-finite joints")
+    traj.push(now + ramp_s + k / fps, clamp(arm[k], 1.0 / fps))
     htraj.push(now + ramp_s + k / fps, hand_cmds[k])
```

**#11 — `startup_ramp_s` is dead.** Implement it; it is three lines, and the first move of an episode
is the one where the arm is furthest from the policy's expected start:

```python
# _maybe_splice, the not-in_flight branch
 start = 0
-self._anchor_time = now
+# First chunk of the episode: begin executing `startup_ramp_s` from now, so the
+# emitter interpolates from the MEASURED pose to slot 0 over the ramp instead of
+# slewing there at the clamp limit.
+ramp = self.cfg.startup_ramp_s if self.stats["chunks"] == 0 else 0.0
+self._anchor_time = now + ramp
```

This works because the buffer already holds a knot at `now` (the seed) and `_top_up` will push slot
0 at `now + ramp + dt` — one long, linearly interpolated segment. Everything downstream
(`m = (now - anchor_time) * fps`, the trigger, `runway`) stays correct, and `runway > lookahead`
during the ramp means `_top_up` will not pop, so no spurious replan fires. If you would rather not
carry the feature, **delete the field** — a documented safety knob that does nothing is worse than
no knob.

**#12 — `G1DDS._msg` has no lock.** `send_arm` (emitter thread) and `damp()` (whichever thread trips)
both mutate the shared `LowCmd_` and recompute its CRC:

```python
self._cmd_lock = threading.Lock()   # in __init__

# send_arm and damp: wrap the mutate + CRC + Write
with self._cmd_lock:
    for k, i in enumerate(ARM_IDX_FLAT): self._msg.motor_cmd[i].q = float(arm_q14[k])
    for k, i in enumerate(WAIST_IDX):    self._msg.motor_cmd[i].q = float(waist[k])
    self._msg.crc = self._crc.Crc(self._msg)
    self._pub.Write(self._msg)
```

**`check.listen` claims "No publishers".** Make it true rather than fixing the docstring — the whole
value of that rung is proving we are not commanding:

```python
def connect(self, *, timeout: float = 5.0, read_only: bool = False) -> None:
    ...
    if not read_only:
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_); self._pub.Init()
```

---

## 1.7 Tests that would have caught all of this

The existing 48 pass because they test the loop with `initial_d=8` and a 0.25 s policy — i.e. inside
the one regime where none of the bugs fire. Add:

| Test | Catches |
|---|---|
| `prefix_weights(...)[spec.d] > 0.5` for `d` in `0..H`, `n_real` in `1..H` | #2 |
| `prefix_weights(...)[n_real:] == 0` for all `d, soft_window` | #8 |
| loop with a 5 s first inference reaches `chunks == 1` without tripping | #1 |
| loop with a 2 s steady-state inference logs the cap and does not spin | #3 |
| `TrajectoryBuffer`: max per-emit step after a 400 ms hold ≤ clamp/emit-rate | #13 |
| state's hand block == `traj_hand.eval(now)` at inference time | #4 |
| camera frozen for 1 s ⇒ watchdog trips | #5 |
| `serve --record` constructs a server | #6 |
| guided sampling with garbage in dims 30:32 of the prefix ⇒ identical actions | #9 |

The parameterization is the point: every one of these bugs lives at a boundary the fixtures never
crossed.

---

# Part 2 — making `ego2g1/` self-contained

Constraint: **everything lives under `third_party/openpi/ego2g1/`.** Nothing may reach outside it —
not `data_extraction/`, not the vendored `unitree-deploy-main/`.

Today it reaches outside in exactly three places:

| Reaches out to | From | How |
|---|---|---|
| `data_extraction.sim.g1` (G1Backend, DualArmIK, MODEL_XML) | `deploy/kinematics.py` | `sys.path.insert("../..")` |
| `data_extraction.{sim.g1, common.frames, hand.constants}` + assets | `eval_replay/{scene,viewer}.py` | `--data-extraction-path` on `sys.path` |
| `unitree_deploy.robot_devices.cameras.imageclient` | `deploy/camera.py` | an untracked vendored zip at the repo root |

Plus a fourth, subtler one: `ego2g1` is **not in openpi's wheel packages**, so `import ego2g1` only
works if your CWD happens to be the openpi root. Every entrypoint's docstring says "run from the
openpi root" — that instruction *is* the bug.

## 2.1 The dependency arrow is backwards

`ego2g1` lives inside `third_party/openpi/`, i.e. inside a vendored dependency of the outer repo. And
it imports *out* into its own parent (`sys.path.insert("../..")` to reach `data_extraction`). A
dependency reaching up into the application that vendors it is the wrong direction, and it is the
direct cause of the path hacks, the `--data-extraction-path` flag, and the triplicated vec9/rot6d
convention (`data_extraction/common/rot6d.py`, `ego2g1/chunk_math.py`, `ego2g1/common/se3.py`) held
together by a byte-equivalence test.

**Flip it.** Make `ego2g1/` the canonical owner of the shared kinematics, assets, and math, and have
`data_extraction/` import *from* `ego2g1`. That is the natural direction (the app imports its
vendored library), it makes `ego2g1` self-contained exactly as required, and it leaves **one copy**
of the 60 MB of MJCF/meshes rather than two.

## 2.2 Target layout

```
third_party/openpi/ego2g1/          <-- self-contained. imports nothing outside this tree.
│
├── core/                  # PURE NUMPY. no mujoco, no jax, no openpi, no DDS.
│   ├── layout.py          #   THE 30-dim contract: slices, ARM_JOINTS, HAND_MOTOR_ORDER
│   ├── se3.py             #   THE vec9/rot6d definition + se3_inv/compose/reanchor/pose_of
│   ├── chunk.py           #   RelativeChunkActions, BoundaryAware*, make_delta_timestamps
│   └── rtc.py             #   PrefixSpec + prefix_weights   <- the contract from §1.1
│
├── sim/                   # + mujoco, mink, qpsolvers.  STILL no jax/openpi.
│   ├── g1.py              #   G1Backend + DualArmIK          (moved from data_extraction/sim/)
│   ├── g1_hands.py        #   (moved)
│   ├── placement.py       #   (moved)
│   └── hand.py            #   revo2 constants + FK tables    (moved from data_extraction/hand/)
│
├── assets/                # canonical, single copy (~60 MB, moved not copied)
│   ├── unitree_g1/        #   scene_fixed_base.xml + meshes
│   └── revo2/
│
├── deploy/                # ROBOT PC:  core + sim + unitree_sdk2py + zmq + websockets.  NO JAX.
│   ├── loop.py  queue.py  trajectory.py  safety.py      (chunk.py -> queue.py; see below)
│   ├── client.py  kinematics.py  dds.py  camera.py
│   └── check.py  __main__.py
│
├── serve/                 # GPU BOX: + jax + openpi        (rtc.py moves to core/)
├── eval_replay/           # + jax + openpi + sim
├── train.py  model.py  config.py  transforms.py  data_config.py  dataset.py  norm.py  stamp.py
└── tests/
```

and in the outer repo:

```python
# data_extraction/sim/g1.py  ->  deleted
# data_extraction/common/{rot6d,frames}.py  ->  deleted
# data_extraction/hand/constants.py  ->  deleted
# data_extraction/assets/  ->  deleted

from ego2g1.core import layout, se3        # the one vec9/rot6d definition
from ego2g1.sim import g1                  # the one G1 model + IK
from ego2g1.sim import hand                # the one revo2 constant set
```

### What collapses

- **`chunk_math.py` and `common/` disappear**, absorbed into `core/`. So does
  `data_extraction/tests/test_loader_equivalence.py` — it exists only to police a duplication that no
  longer exists.
- **`_import_sim()`, `data_extraction_path=`, `--data-extraction-path`, `repo="../.."` all
  disappear.** `from ego2g1.sim import g1`.
- **`unitree-deploy-main/` disappears** (see §2.4).
- `deploy/chunk.py` → `deploy/queue.py`, because `core/chunk.py` now owns the word "chunk" (the
  action-chunk math) and `ChunkQueue` is a queue.

## 2.3 The one-line enabler

`ego2g1` is invisible to the installed package. Add to `third_party/openpi/pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/openpi", "ego2g1"]
```

`uv pip install -e third_party/openpi` then makes `import ego2g1` work from **any** CWD — which is
what lets `data_extraction` import it, and what lets every entrypoint drop its "run from the openpi
root" caveat.

Importantly this does **not** drag JAX into `data_extraction`: `ego2g1/__init__.py` is a bare
docstring, so `import ego2g1.sim` pulls mujoco and nothing else. That discipline already protects
`deploy/`; it is now load-bearing for two consumers instead of one, so **make it a test**:

```python
# tests/test_import_isolation.py
def test_deploy_does_not_import_jax():
    out = subprocess.run([sys.executable, "-c",
        "import ego2g1.deploy.loop, ego2g1.deploy.kinematics, ego2g1.deploy.dds, sys; "
        "assert 'jax' not in sys.modules, sorted(m for m in sys.modules if 'jax' in m); "
        "assert 'openpi' not in sys.modules"], capture_output=True)
    assert out.returncode == 0, out.stderr.decode()

def test_core_is_pure_numpy():
    out = subprocess.run([sys.executable, "-c",
        "import ego2g1.core.layout, ego2g1.core.se3, ego2g1.core.rtc, sys; "
        "assert not {'jax','mujoco','openpi','torch'} & set(sys.modules)"], capture_output=True)
    assert out.returncode == 0, out.stderr.decode()
```

The layered dependency rule, stated once and enforced:

```
core     ->  numpy
sim      ->  core + mujoco, mink, qpsolvers
deploy   ->  core + sim + unitree_sdk2py, pyzmq, websockets, opencv     (NEVER jax/openpi)
serve    ->  core + openpi, jax
train    ->  core + openpi, jax
```

## 2.4 Drop the vendored `unitree-deploy-main/`

`deploy/camera.py` imports exactly one class from that 90-file untracked tree. That class is a ZMQ
SUB socket that receives a JPEG and splits it. The whole contract is:

- `REQ tcp://<host>:60000` → returns the camera config (or just hardcode it — you already know it);
- `SUB tcp://<host>:55555`, `RCVHWM=1`, no topic filter → raw JPEG bytes;
- `cv2.imdecode` → BGR `(480, 1280, 3)`; the head camera is `binocular: true`, so split at `width//2`
  into left/right `(480, 640, 3)`.

That is the entire dependency. Reimplement it in `deploy/camera.py` and delete the vendored tree:

```python
class HeadCamera:
    """G1 head stereo camera: a JPEG-over-ZMQ SUB stream from image_server on the
    robot's onboard board. We take one eye.

    Self-contained on purpose — the vendor's ImageClient drags in a 90-file package
    to do what forty lines of pyzmq do, and it swallowed camera failures silently.
    """

    def __init__(self, *, host="192.168.123.164", eye="left", port=55555,
                 stereo_width=1280, flip_bgr=True):
        if eye not in ("left", "right"):
            raise ValueError(f"eye must be left|right, got {eye}")
        self.host, self.eye, self.port = host, eye, port
        self.half = stereo_width // 2
        self.flip_bgr = flip_bgr
        self._frame, self._t = None, 0.0
        self._errors = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def connect(self, *, timeout: float = 10.0) -> None:
        import zmq
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVHWM, 1)          # only ever the newest frame
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.connect(f"tcp://{self.host}:{self.port}")
        self._thread = threading.Thread(target=self._pump, name="camera", daemon=True)
        self._thread.start()
        t0 = time.monotonic()
        while self.age() > timeout:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"no frames from image_server at {self.host}:{self.port}. Is it running?\n"
                    f"  ssh unitree@{self.host} && conda activate tv && "
                    f"cd ~/image_server && python image_server.py")
            time.sleep(0.05)

    def _pump(self) -> None:
        import cv2, zmq
        while not self._stop.is_set():
            try:
                jpg = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.002)
                continue
            except Exception:
                self._errors += 1
                if self._errors % 100 == 1:
                    logger.exception("camera recv failed (%d consecutive)", self._errors)
                time.sleep(0.01)
                continue
            bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                self._errors += 1
                continue
            self._errors = 0
            img = bgr[:, : self.half] if self.eye == "left" else bgr[:, self.half :]
            if self.flip_bgr:
                img = img[..., ::-1]                  # BGR -> RGB
            with self._lock:
                self._frame = np.ascontiguousarray(img, dtype=np.uint8)
                self._t = time.monotonic()
```

`read()` / `age()` / `close()` are unchanged. `age()` now finally has a consumer — the watchdog from
§1.5.

## 2.5 Migration order

Each step leaves the tree working.

1. **`pyproject.toml`: add `ego2g1` to the wheel packages** (§2.3), reinstall. `import ego2g1` now
   works from any CWD. *Nothing else changes yet* — this is the enabler.
2. **Create `ego2g1/core/`.** Merge `chunk_math.py` + `common/layout.py` + `common/se3.py` +
   `data_extraction/common/{rot6d,frames}.py` into `core/{layout,se3,chunk}.py`. Point
   `data_extraction` at them. Delete the byte-equivalence test — it is now tautological.
3. **Move the sim.** `data_extraction/{sim,hand,assets}` → `ego2g1/{sim,sim/hand.py,assets}`. Delete
   `_import_sim`, `data_extraction_path`, `--data-extraction-path`, `repo="../.."`. The `fk`/`ik`
   rungs of `check.py` lose an argument each.
4. **Rewrite `camera.py`** (§2.4). Delete `unitree-deploy-main/`.
5. **Add `tests/test_import_isolation.py`** (§2.3). This is the guard rail that keeps `deploy/` free
   of JAX now that the tree is one package.
6. **Then apply Part 1.** The RTC fix wants `core/rtc.py` to exist; doing it first means writing the
   contract twice.

## 2.6 Why this closes bugs, not just files

`core/` is not tidiness. **Findings #2, #7, #8, and #14 are each a fact the client and the server had
to agree on, with no shared place to put it** — the guidance mask, the prefix length, the 30-dim
layout, the vec9 convention. The client can't import the server's `rtc.py` (it would pull JAX), so
the contract got re-derived on each side and drifted.

A pure-numpy `core/` that *both* sides import is the structural form of the §1.1 fix: the client can
compute the exact mask the server will apply, and assert on it. Without it you are back to two
implementations of one contract, which is precisely how `execution_horizon=10` and `initial_d=12`
came to ship together.

## 2.7 Housekeeping

- `deploy/requirements.txt` loses `unitree_deploy` and gains nothing — `pyzmq` and `opencv-python`
  were already there for the vendor's transport.
- Root clutter → `.gitignore`: `episode_38_dashboard.html`, `MUJOCO_LOG.TXT`, `.DS_Store`.
- `OPENPI_EDITS.md` documents `src/openpi` edits that **no longer exist** — I diffed the fork against
  its upstream merge-base and `src/` is untouched. Rename it to what it actually describes now: the
  `Pi0` subclass and the gemma monkeypatch.
- `ego2g1/README.md` should state the layered dependency rule from §2.3. It is the one invariant that
  keeps the robot PC installable.
