"""Step 0: sanity-check the patched Revo2 MJCF against the spec/URDF contract.

Verifies, for each hand:
  - model loads; 6 actuators in BrainCo motor order semantics
  - ctrl = 0      -> settles at fully-open pose (all actuated joints ~0 rad)
  - ctrl = max    -> settles at fully-closed pose (spec angles 59/90/81 deg)
  - distal couplings track (fingers 1.155x proximal, thumb 1.0x)
  - fingertip sites move sensibly (open->closed sweep)
Also renders open/half/closed poses to outputs/ if a GL backend is available.

Run: uv run --no-project --with mujoco,numpy python data_filtering/step0_sanity_check.py
"""

import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

# actuator order in the MJCF = [thumb rot (metacarpal), thumb flex (proximal),
# index, middle, ring, pinky] -- NOTE this differs from the BrainCo motor-ID
# order [thumb flex, thumb rot, index..pinky]; brainco_mapping.py handles that.
SPEC_MAX_RAD = np.array([1.57, 1.03, 1.41, 1.41, 1.41, 1.41])
FINGERS = ["thumb_metacarpal", "thumb_proximal", "index_proximal",
           "middle_proximal", "ring_proximal", "pinky_proximal"]


def settle(model, data, ctrl, seconds=3.0):
    # 3 s: the thumb-rotation servo is force-capped at 0.5 Nm against 0.5 damping,
    # so a full 90-deg sweep alone takes ~2 s
    import mujoco
    data.ctrl[:] = ctrl
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def check_hand(prefix: str) -> None:
    import mujoco

    path = HERE / "assets" / "revo2_mjcf" / f"xml_{prefix}" / f"brainco-{prefix}hand-v2-patched.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    print(f"\n=== {prefix} hand: {path.name} ===")
    print(f"nq={model.nq} nu={model.nu} ngeom={model.ngeom} neq={model.neq}")
    assert model.nu == 6 and model.neq == 5, "expected 6 actuators, 5 couplings"

    act_joints = [model.actuator(i).name for i in range(model.nu)]
    expected = [f"{prefix}_{f}_joint" for f in FINGERS]
    assert act_joints == expected, f"actuator order changed: {act_joints}"

    def qpos_of(joint_suffix):
        return data.qpos[model.joint(f"{prefix}_{joint_suffix}").qposadr[0]]

    # fully open
    mujoco.mj_resetData(model, data)
    settle(model, data, np.zeros(6))
    open_q = np.array([qpos_of(f + "_joint") for f in FINGERS])
    print("ctrl=0 (open)   qpos:", np.round(open_q, 4))
    assert np.all(np.abs(open_q) < 0.03), "open pose did not settle near 0"

    # fully closed, phase 1: contacts disabled -- a pure servo/coupling test.
    # (With contacts on, a full fist self-collides by design: fingers curl
    # ~174 deg total into the palm and the thumb jams against them, so the
    # servos stall exactly like the real hand's current limit would.)
    ctrl_max = model.actuator_ctrlrange[:, 1]
    assert np.allclose(ctrl_max, SPEC_MAX_RAD), f"ctrlrange != spec: {ctrl_max}"
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mujoco.mj_resetData(model, data)
    settle(model, data, ctrl_max)
    closed_q = np.array([qpos_of(f + "_joint") for f in FINGERS])
    print("ctrl=max, no contact (servo test) qpos:", np.round(closed_q, 4),
          "= deg", np.round(np.degrees(closed_q), 1))
    err = np.abs(closed_q - SPEC_MAX_RAD)
    print("  |qpos - target| max:", round(float(err.max()), 4), "rad")
    assert err.max() < 0.06, "servos did not reach spec angles in free space"

    # coupling ratios at closed pose
    for f, ratio in [("thumb", 1.0), ("index", 1.155), ("middle", 1.155),
                     ("ring", 1.155), ("pinky", 1.155)]:
        prox = qpos_of(f"{f}_proximal_joint")
        dist = qpos_of(f"{f}_distal_joint")
        ok = abs(dist - ratio * prox) < 0.05
        print(f"  {f:6s} distal/proximal = {dist/prox:.3f} (target {ratio})",
              "OK" if ok else "FAIL")
        assert ok

    # fingertip travel open->closed (right model has tip sites; the simplified
    # left vendor model has no tip bodies, so fall back to distal link frames)
    fingers5 = ["thumb", "index", "middle", "ring", "pinky"]
    if prefix == "right":
        ids = [model.site(f"{prefix}_{f}_tip").id for f in fingers5]
        xpos = data.site_xpos
    else:
        ids = [model.body(f"{prefix}_{f}_distal_link").id for f in fingers5]
        xpos = data.xpos
    closed_tips = xpos[ids].copy()
    mujoco.mj_resetData(model, data)
    settle(model, data, np.zeros(6))
    open_tips = xpos[ids].copy()
    travel = np.linalg.norm(closed_tips - open_tips, axis=1)
    print("  fingertip travel open->closed (cm):", np.round(travel * 100, 1))
    assert np.all(travel > 0.03), "fingertips barely moved; couplings broken?"

    # fully closed, phase 2: contacts back on -- informational: how far the
    # hand physically gets when everything is commanded shut at once.
    model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    mujoco.mj_resetData(model, data)
    settle(model, data, ctrl_max)
    stalled_q = np.array([qpos_of(f + "_joint") for f in FINGERS])
    print("ctrl=max, with contact (self-collision stall) deg:",
          np.round(np.degrees(stalled_q), 1), f"| ncon={data.ncon}")


def try_render(prefix: str) -> None:
    import mujoco

    path = HERE / "assets" / "revo2_mjcf" / f"xml_{prefix}" / f"brainco-{prefix}hand-v2-patched.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    try:
        renderer = mujoco.Renderer(model, height=480, width=480)
    except Exception as e:
        print(f"(render skipped: {e})")
        return
    frames = []
    for frac in (0.0, 0.5, 1.0):
        mujoco.mj_resetData(model, data)
        settle(model, data, frac * model.actuator_ctrlrange[:, 1])
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.distance, cam.elevation, cam.azimuth = 0.35, -25, 130
        cam.lookat[:] = [0.0, 0.0, 0.06]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    out = HERE / "outputs" / f"step0_{prefix}_open_half_closed.png"
    strip = np.concatenate(frames, axis=1)
    try:
        import PIL.Image
        PIL.Image.fromarray(strip).save(out)
        print("rendered:", out)
    except ImportError:
        np.save(out.with_suffix(".npy"), strip)
        print("PIL missing; saved raw array:", out.with_suffix(".npy"))
    renderer.close()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    for prefix in ("right", "left"):
        check_hand(prefix)
    for prefix in ("right", "left"):
        try_render(prefix)
    print("\nstep 0 sanity check: ALL PASSED")
