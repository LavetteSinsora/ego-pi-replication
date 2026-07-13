"""Reproductions for DEPLOY_SERVE_AUDIT.md.

Every test here ASSERTS THE BUG. A PASS means the defect is present.
After a fix lands, the corresponding test should FAIL — that is the point.

Run:
    cd third_party/openpi
    ../../.venv/bin/python -m pytest ../../audit_repro/test_findings.py -q -s

Findings 7, 9, 12, 14 are static/analytic (marked `analytic`): they assert the code
shape that produces the defect, not the defect's runtime effect. Everything else is
observed at runtime.
"""

import inspect
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
OPENPI = REPO / "third_party/openpi"
sys.path.insert(0, str(OPENPI))
sys.path.insert(0, str(OPENPI / "ego2g1/tests"))

from ego2g1.common import layout, se3                       # noqa: E402
from ego2g1.deploy import chunk as _chunk                   # noqa: E402
from ego2g1.deploy import loop as _loop                     # noqa: E402
from ego2g1.deploy import safety as _safety                 # noqa: E402
from ego2g1.deploy.client import DelayBudget                # noqa: E402
from ego2g1.deploy.trajectory import TrajectoryBuffer       # noqa: E402
from ego2g1.serve import rtc as _rtc                        # noqa: E402

H_DEFAULT = 50
ANCHOR = {h: np.eye(4) for h in layout.HANDS}


@pytest.fixture(scope="session")
def kin():
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    from ego2g1.deploy.kinematics import Kinematics
    return Kinematics(REPO)


def _arm0():
    import pandas as pd
    f = sorted((REPO / "lerobot_datasets/ego2g1/put_bottle_in_box").glob("data/*/*.parquet"))[0]
    return np.stack(pd.read_parquet(f)["arm_qpos"].to_numpy())[0]


# =============================================================================
# F1 [CRITICAL] cold-start e-stop: the watchdog damps during the first inference
# =============================================================================

def test_F1_starvation_timer_runs_before_the_first_chunk_exists():
    """loop.start() seeds ONE knot -> runway == 0 from tick 0. check_starvation is
    called every control tick with no notion of 'we have not planned yet'."""
    tripped = []
    limits = _safety.SafetyLimits()
    assert limits.max_starvation == 1.0                      # safety.py:42
    wd = _safety.Watchdog(limits, on_trip=lambda: tripped.append(1))
    t = 100.0
    for _ in range(200):                                     # ~2 s of control ticks
        wd.check_starvation(0.0, t)                          # runway == 0, as at startup
        t += 1 / 90.0
    assert wd.tripped and "planner is dead" in wd.reason
    # check_starvation has no 'armed'/'first chunk' parameter at all:
    assert "armed" not in inspect.signature(_safety.Watchdog.check_starvation).parameters


def test_F1_live_loop_estops_on_a_slow_first_inference(kin):
    """End-to-end against the repo's own fakes. 1.3 s is FAR shorter than a real
    cold-server JIT compile of pi0.5."""
    from test_deploy_loop import FakeCamera, FakeDDS, FakePolicy
    dds, policy = FakeDDS(_arm0()), FakePolicy(latency_s=1.3)
    lp = _loop.DeployLoop(
        _loop.LoopConfig(task="t", fps=30, arm_hz=200.0, hand_hz=100.0),
        dds=dds, camera=FakeCamera(), kinematics=kin, client=policy,
        budget=DelayBudget(30, initial=8),
        limits=_safety.SafetyLimits(max_tracking_error_m=0.25),
    )
    lp.start(); time.sleep(1.6); lp.stop()
    print(f"\n  F1: tripped={lp.watchdog.tripped!r} reason={lp.watchdog.reason!r} "
          f"chunks={lp.stats['chunks']} damped={dds.damped}")
    assert lp.watchdog.tripped
    assert lp.stats["chunks"] == 0        # damped before the first chunk ever landed
    assert dds.damped


# =============================================================================
# F2 [CRITICAL] the RTC guidance mask is zero at the slot the client splices at
# =============================================================================

def test_F2_shipped_defaults_put_d_past_the_execution_horizon():
    eh = 10   # serve/__main__.py:49   execution_horizon
    d0 = 12   # deploy/__main__.py:61  initial_d
    assert d0 > eh
    w = _rtc.prefix_weights(d0, eh, H_DEFAULT)
    print(f"\n  F2: weights[8:16] = {w[8:16].round(3)}   w[splice slot d={d0}] = {w[d0]}")
    assert w[d0] == 0.0                       # loop.py:322 -> start = p["d"] = 12
    assert np.all(w[eh:] == 0.0)


@pytest.mark.parametrize("latency_ms", [300, 350, 400, 500])
def test_F2_every_realistic_latency_lands_d_past_the_horizon(latency_ms):
    eh = 10
    b = DelayBudget(fps=30, initial=12)
    for _ in range(10):
        b.observe(latency_ms / 1000.0)
    w = _rtc.prefix_weights(b.d, eh, H_DEFAULT)
    print(f"\n  F2: latency {latency_ms} ms -> d={b.d}  w[splice]={w[min(b.d, H_DEFAULT-1)]}")
    assert b.d > eh
    assert w[min(b.d, H_DEFAULT - 1)] == 0.0


def test_F2_live_loop_sends_d_past_the_servers_horizon(kin):
    from test_deploy_loop import FakeCamera, FakeDDS, FakePolicy
    eh = _rtc.RTCConfig().execution_horizon
    policy = FakePolicy(latency_s=0.40)
    lp = _loop.DeployLoop(
        _loop.LoopConfig(task="t", fps=30, arm_hz=200.0, hand_hz=100.0),
        dds=FakeDDS(_arm0()), camera=FakeCamera(), kinematics=kin, client=policy,
        budget=DelayBudget(30, initial=12),
        limits=_safety.SafetyLimits(max_tracking_error_m=0.25),
    )
    lp.start(); time.sleep(4.0); lp.stop()
    ds = [c["d"] for c in policy.calls if c["has_prefix"]]
    print(f"\n  F2: d values actually sent = {ds}  (server execution_horizon={eh})")
    assert ds and all(d > eh for d in ds)
    for d in ds:
        assert _rtc.prefix_weights(d, eh, policy.action_horizon)[d] == 0.0


# =============================================================================
# F3 [HIGH] DelayBudget.d is unbounded; d >= H installs an empty chunk
# =============================================================================

def test_F3_unbounded_delay_budget_empties_the_chunk():
    b = DelayBudget(fps=30, initial=12)
    for _ in range(10):
        b.observe(2.0)                        # a 2 s inference
    print(f"\n  F3: d={b.d} (H={H_DEFAULT})")
    assert b.d >= H_DEFAULT                   # client.py:64 has no cap
    q = _chunk.ChunkQueue(H_DEFAULT, 30)
    q.replace(np.zeros((H_DEFAULT, layout.DIM), np.float32), ANCHOR, b.d)
    assert q.remaining() == 0 and q.pop() is None
    assert H_DEFAULT - b.d - 8 <= 0           # loop.py:234 trigger collapses -> spin


# =============================================================================
# F4 [HIGH] the state's hand block leads its eef block by ~lookahead_s
# =============================================================================

def test_F4_state_hand_block_is_in_the_future(kin):
    from test_deploy_loop import FakeCamera, FakeDDS

    class RampPolicy:
        action_horizon, action_dim, fps = 50, 30, 30
        rtc_training, rtc, hands = False, {}, ("left", "right")
        def __init__(self): self.calls = []
        def infer(self, image, state, prompt, *, prev_chunk=None, d=0):
            time.sleep(0.2)
            self.calls.append(1)
            a = np.zeros((50, layout.DIM), np.float32)
            ramp = np.linspace(0.0, 1.0, 50, dtype=np.float32)   # a closing grasp
            for h in layout.HANDS:
                T = np.tile(np.eye(4), (50, 1, 1))
                T[:, 0, 3] = 0.002 * np.arange(1, 51)
                a[:, layout.EEF[h]] = se3.se3_to_vec9(T)
                a[:, layout.HAND[h]] = ramp[:, None]
            return {"actions": a, "client_latency_s": 0.2,
                    "rtc": {"sampler": "guided" if prev_chunk is not None else "plain", "d": d}}

    policy = RampPolicy()
    lp = _loop.DeployLoop(
        _loop.LoopConfig(task="t", fps=30, arm_hz=200.0, hand_hz=100.0, lookahead_s=0.10),
        dds=FakeDDS(_arm0()), camera=FakeCamera(), kinematics=kin, client=policy,
        budget=DelayBudget(30, initial=8),
        limits=_safety.SafetyLimits(max_tracking_error_m=0.25),
    )
    lp.start(); time.sleep(2.5)
    leads = []
    for _ in range(40):
        now = time.monotonic()
        emitted = lp.traj_hand.eval(now)                     # what the hand IS commanded now
        if emitted is not None:
            sent = np.concatenate([lp._last_hand[h] for h in layout.HANDS])   # what we SEND
            leads.append(float(sent[0] - emitted[0]))
        time.sleep(0.02)
    lp.stop()
    leads = np.array(leads)
    slots = leads.mean() * 50                                # ramp spans 1.0 over 50 slots
    print(f"\n  F4: state_hand - emitted_hand: mean {leads.mean():+.4f} "
          f"=> the state's hand block leads reality by ~{slots:.1f} slots "
          f"(lookahead_s=0.10 @ 30 Hz == 3 ticks)")
    assert leads.mean() > 0 and slots > 1.5


# =============================================================================
# F5 [HIGH] camera staleness is never checked
# =============================================================================

def test_F5_camera_age_has_no_consumer():
    from ego2g1.deploy import camera as _camera
    assert hasattr(_camera.HeadCamera, "age")                # exists...
    assert ".age()" not in inspect.getsource(_loop)          # ...and nothing calls it
    assert not any(k in f for f in _safety.SafetyLimits.__dataclass_fields__
                   for k in ("image", "frame", "cam"))       # no image-age limit exists
    # the loop's ONLY image check:
    assert "if image is None:" in inspect.getsource(_loop.DeployLoop._maybe_infer)


# =============================================================================
# F6 [MEDIUM] serve --record raises AttributeError at startup
# =============================================================================

def test_F6_policy_recorder_has_no_metadata():
    from openpi.policies import policy as P
    assert not hasattr(P.PolicyRecorder, "metadata")
    assert not hasattr(P.BasePolicy, "metadata")

    class Fake(P.BasePolicy):
        def infer(self, obs): return {}
        @property
        def metadata(self): return {"ego2g1": {}}

    rec = P.PolicyRecorder(Fake(), "/tmp/_audit_records")
    with pytest.raises(AttributeError, match="metadata"):
        rec.metadata                                          # serve/__main__.py:94

    src = (OPENPI / "ego2g1/serve/__main__.py").read_text()
    assert src.index("PolicyRecorder(policy") < src.index("metadata=policy.metadata")
    # stock openpi does it correctly, so this is a regression:
    stock = (OPENPI / "scripts/serve_policy.py").read_text()
    assert stock.index("policy_metadata = policy.metadata") < stock.index("PolicyRecorder(policy")


# =============================================================================
# F7 [MEDIUM] analytic — server --rtc False + async client: the loop never checks
# =============================================================================

@pytest.mark.analytic
def test_F7_loop_ignores_the_servers_rtc_enabled_flag():
    assert _rtc.select_sampler(rtc_training=False, has_prefix=True,
                               rtc_enabled=False) is _rtc.Sampler.PLAIN   # unguided chunk
    src = inspect.getsource(_loop)
    assert "self.budget.d" in src                     # ...yet the client still sends d>0
    assert "start = p[\"d\"]" in src                  # ...and still splices at slot d

    # PolicyClient DOES receive rtc.enabled in the handshake...
    from ego2g1.deploy import client as _client
    assert 'self.rtc = dict(cfg["rtc"])' in inspect.getsource(_client.PolicyClient.__init__)
    # ...and the loop NEVER reads it. The only 'rtc' the loop touches are its own
    # queue.rtc_prefix() and the sampler NAME echoed back for a log line.
    uses = [ln.strip() for ln in src.splitlines()
            if "rtc" in ln and not ln.strip().startswith("#")]
    assert all(("rtc_prefix" in u) or ('out.get("rtc"' in u) or ('"sampler"' in u)
               for u in uses), uses
    assert "client.rtc" not in src and "self.client.rtc" not in src


# =============================================================================
# F8 [MEDIUM] the RTC prefix's zero padding can carry non-zero guidance weight
# =============================================================================

def test_F8_padding_is_weighted_when_the_leftover_is_short():
    q = _chunk.ChunkQueue(H_DEFAULT, 30)
    q.replace(np.full((H_DEFAULT, layout.DIM), 0.5, np.float32), ANCHOR, 0)
    prefix = q.rtc_prefix(ANCHOR, 45)                 # loop fell behind: 45 ticks elapsed
    assert np.all(prefix[5:] == 0.0)                  # rows 5.. are PADDING (chunk.py:140)
    w = _rtc.prefix_weights(3, 10, H_DEFAULT)         # d=3, execution_horizon=10
    print(f"\n  F8: n_real=5, but weights[5:10] = {w[5:10].round(3)} (non-zero on padding)")
    assert np.any(w[5:10] > 0.0)
    # and a zero vec9 is not a pose at all:
    R = se3.rot6d_to_mat(np.zeros(6))
    assert np.allclose(R, 0.0) and abs(np.linalg.det(R)) < 1e-12
    # nothing anywhere reports how many prefix rows are real:
    assert "n_real" not in inspect.getsource(_chunk)
    assert "n_real" not in inspect.getsource(_rtc)


# =============================================================================
# F9 [MEDIUM] analytic — guided RTC guides the two UNTRAINED padding dims
# =============================================================================

@pytest.mark.analytic
def test_F9_guidance_error_spans_the_padding_dims():
    from ego2g1 import config as _config, model as _model
    cfg = _config.Ego2G1TrainConfig()
    assert (cfg.action_dim, cfg.action_dim_actual) == (32, 30)      # 2 pad dims
    loss_src = inspect.getsource(_model.Ego2G1Pi0.compute_loss)
    assert "loss[..., : self.action_dim_actual]" in loss_src        # dims 30:32 NEVER trained
    g = inspect.getsource(_model.Ego2G1Pi0.sample_actions_guided)
    assert "err = (prefix_actions - x_1) * w" in g                  # err spans all 32 dims
    assert "correction = vjp_fn(err)[0]" in g                       # J^T mixes them into 0:30
    assert "action_dim_actual" not in g                             # no mask anywhere


# =============================================================================
# F10 [LOW] check.replay constructs a Clamp and never applies it
# =============================================================================

def test_F10_replay_clamp_is_dead_code():
    from ego2g1.deploy import check
    src = inspect.getsource(check.replay)
    assert "clamp = _safety.Clamp" in src and "clamp.reset(q0)" in src
    assert "clamp(" not in src.replace("Clamp(", "")          # never invoked
    assert "max_step" in inspect.signature(check.replay).parameters   # ...so max_step is a no-op
    assert "d.send_arm(q)" in src                            # and this drives the REAL arm


# =============================================================================
# F11 [LOW] LoopConfig.startup_ramp_s is dead config
# =============================================================================

def test_F11_startup_ramp_is_never_read():
    assert _loop.LoopConfig.startup_ramp_s == 2.0
    assert inspect.getsource(_loop).count("startup_ramp_s") == 1   # the declaration only


# =============================================================================
# F12 [LOW] analytic — G1DDS._msg mutated + CRC'd from multiple threads, no lock
# =============================================================================

@pytest.mark.analytic
def test_F12_lowcmd_message_has_no_lock():
    from ego2g1.deploy import dds as _dds
    send, damp = inspect.getsource(_dds.G1DDS.send_arm), inspect.getsource(_dds.G1DDS.damp)
    for s in (send, damp):
        assert "self._msg.crc = self._crc.Crc(self._msg)" in s
        assert "self._pub.Write(self._msg)" in s
        assert "_lock" not in s          # both mutate the SAME _msg and CRC it, unlocked
    # send_arm runs on the arm-emitter thread; damp() on whichever thread trips the watchdog
    assert "self.dds.damp" in inspect.getsource(_loop.DeployLoop.__init__)


# =============================================================================
# F13 [HIGH] splicing after a hold delivers a full clamp-step in ONE emit period
# =============================================================================

def test_F13_splice_after_hold_steps_the_emitted_stream():
    dt, emit_hz = 1 / 30.0, 500.0
    tb = TrajectoryBuffer(2)
    tb.seed(100.0, np.zeros(2))
    tb.push(100.1, np.zeros(2))
    now = 100.5                                   # emitter has HELD for 400 ms (overrun)

    q_before = tb.eval(now)
    tb.replace_after(now, [])                     # loop.py:338 — keeps the STALE 100.1 knot
    q_new = tb.eval(now) + 0.15                   # one clamp step (SafetyLimits.max_joint_step)
    tb.push(now + dt, q_new)                      # _top_up pushes the new chunk's first knot
    q_after = tb.eval(now + 1 / emit_hz)          # the very next 500 Hz emit

    jump = float(np.abs(q_after - q_before).max())
    budget = 0.15 / (emit_hz * dt)                # what one emit SHOULD move
    print(f"\n  F13: emit before={q_before[0]:.4f} after={q_after[0]:.4f}  "
          f"jump={jump:.4f} rad in one {1000/emit_hz:.0f} ms emit "
          f"({jump*emit_hz:.0f} rad/s); intended {budget:.4f} rad -> {jump/budget:.0f}x")
    assert jump > 5 * budget                      # the rate limit is not enforced


def test_F13_live_loop_lurches_when_inference_overruns(kin):
    from test_deploy_loop import FakeCamera, FakeDDS

    class StallPolicy:
        action_horizon, action_dim, fps = 50, 30, 30
        rtc_training, rtc, hands = False, {}, ("left", "right")
        def __init__(self): self.n = 0
        def infer(self, image, state, prompt, *, prev_chunk=None, d=0):
            lat = 0.2 if self.n == 0 else 1.2     # first fast (past F1), then overrun
            self.n += 1
            time.sleep(lat)
            a = np.zeros((50, layout.DIM), np.float32)
            for h in layout.HANDS:
                T = np.tile(np.eye(4), (50, 1, 1))
                T[:, 0, 3] = 0.004 * np.arange(1, 51)
                a[:, layout.EEF[h]] = se3.se3_to_vec9(T)
            return {"actions": a, "client_latency_s": lat,
                    "rtc": {"sampler": "plain", "d": d}}

    dds = FakeDDS(_arm0())
    lp = _loop.DeployLoop(
        _loop.LoopConfig(task="t", fps=30, arm_hz=200.0, hand_hz=100.0),
        dds=dds, camera=FakeCamera(), kinematics=kin, client=StallPolicy(),
        budget=DelayBudget(30, initial=12),
        limits=_safety.SafetyLimits(max_tracking_error_m=0.5),
    )
    lp.start(); time.sleep(4.0); lp.stop()

    q = np.stack([s[1] for s in dds.sent])
    step = float(np.abs(np.diff(q, axis=0)).max())
    budget = 0.15 / (200.0 / 30.0)                # 200 Hz emit, 0.15 rad per 30 Hz knot
    print(f"\n  F13: late splices={lp.stats['late']}  "
          f"max joint step between 200 Hz emits = {step:.4f} rad (intended {budget:.4f})")
    assert lp.stats["late"] > 0
    assert step > 2 * budget


# =============================================================================
# F14 [HIGH] analytic — the pinned sampler pins zero-padding as clean ground truth
# =============================================================================

@pytest.mark.analytic
def test_F14_pinned_sampler_pins_padding_when_d_exceeds_the_real_prefix():
    from ego2g1 import model as _model
    src = inspect.getsource(_model.Ego2G1Pi0.sample_actions_rtc)
    assert "slot_is_prefix = jnp.arange(self.action_horizon) < d" in src
    assert "x_init = jnp.where(slot_is_prefix[None, :, None], prefix_actions, noise)" in src
    assert "n_real" not in src               # d is NOT capped to the real prefix length

    # reachable: when the loop falls behind, n_real = H - m can be < d
    q = _chunk.ChunkQueue(H_DEFAULT, 30)
    q.replace(np.full((H_DEFAULT, layout.DIM), 0.5, np.float32), ANCHOR, 0)
    prefix = q.rtc_prefix(ANCHOR, 42)        # m=42 -> only 8 real rows
    n_real = int((np.abs(prefix).sum(axis=1) > 0).sum())
    assert n_real == 8
    d = 12                                   # the shipped initial_d
    print(f"\n  F14: n_real={n_real}, d={d} -> rows {n_real}..{d-1} are ZERO PADDING, "
          f"pinned as clean t=0 ground truth")
    assert d > n_real
