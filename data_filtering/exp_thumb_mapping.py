"""Experiment: choose the thumb-rotation mapping (direct-angle vs range-ratio).

Step-3 replay showed the direct-angle mapping drives the thumb ~0.4 rad into the
curled index for most of every episode. Hypothesis: Ability's thumb rotator
spans 120 deg vs BrainCo's 90 deg, so copying the angle 1:1 over-rotates the
thumb into the palm; rescaling the span ("range" mode) should preserve relative
opposition depth and relieve the interpenetration.

Part A: on a ~30-episode sample (right hand), replay both variants with contacts
        and compare blocked-tracking error and thumb-index contact.
Part B: for one bad episode, render video frame + both variants' kinematic poses
        into a PNG strip for eyeball judgment against the human hand.

Run: uv run --no-project --with mujoco,numpy,pandas,pyarrow,imageio,imageio-ffmpeg \\
       python data_filtering/exp_thumb_mapping.py
"""

import glob
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd

from brainco_mapping import cmd_to_mjcf_ctrl, convert_state38

HERE = Path(__file__).parent
COUPLE = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}
VARIANTS = ("direct", "span")


def load_right():
    m = mujoco.MjModel.from_xml_path(
        str(HERE / "assets/revo2_mjcf/xml_right/brainco-righthand-v2-patched.xml"))
    d = mujoco.MjData(m)
    act = np.array([m.joint(m.actuator(i).trnid[0]).qposadr[0] for i in range(m.nu)])
    # finger of each geom via its BODY name (left model's geoms are unnamed)
    finger_of = []
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        finger_of.append(next((f for f in COUPLE if f"_{f}_" in body), "palm"))
    return m, d, act, finger_of


def set_pose(m, d, act, cmd):
    d.qpos[act] = cmd_to_mjcf_ctrl(cmd)
    for f, r in COUPLE.items():
        d.qpos[m.joint(f"right_{f}_distal_joint").qposadr[0]] = \
            r * d.qpos[m.joint(f"right_{f}_proximal_joint").qposadr[0]]
    mujoco.mj_forward(m, d)


def replay(m, d, act, finger_of, state_cmd, action_cmd):
    mujoco.mj_resetData(m, d)
    set_pose(m, d, act, state_cmd[0])
    T = len(action_cmd)
    err = np.zeros((T, 6))
    thumb_index_steps = 0
    for t in range(T):
        d.ctrl[:] = cmd_to_mjcf_ctrl(action_cmd[t])
        for _ in range(10):
            mujoco.mj_step(m, d)
        err[t] = np.abs(d.qpos[act] - d.ctrl)
        for c in d.contact[: d.ncon]:
            pair = {finger_of[c.geom1], finger_of[c.geom2]}
            if pair == {"thumb", "index"} and c.dist < 0:
                thumb_index_steps += 1
    return err, thumb_index_steps


def part_a():
    files = sorted(glob.glob(str(HERE.parent / "data/data/chunk-*/episode_*.parquet")))[::17]
    m, d, act, finger_of = load_right()
    print(f"Part A: replaying {len(files)} sample episodes, right hand")
    rows = []
    for variant in VARIANTS:
        errs, ti, frames = [], 0, 0
        for f in files:
            df = pd.read_parquet(f, columns=["observation.state", "action"])
            state = np.stack(df["observation.state"].to_numpy())
            action = np.stack(df["action"].to_numpy())
            sc = convert_state38(state, variant)["right"]["cmd"]
            ac = convert_state38(action, variant)["right"]["cmd"]
            e, t_i = replay(m, d, act, finger_of, sc, ac)
            errs.append(e)
            ti += t_i
            frames += len(e)
        e = np.concatenate(errs)
        # MJCF actuator order: [thumb_rot, thumb_flex, index, middle, ring, pinky]
        rows.append({
            "variant": variant,
            "thumb_rot_err_mean": round(float(e[:, 0].mean()), 3),
            "thumb_rot_err_max": round(float(e[:, 0].max()), 3),
            "hard_frames_frac": round(float((e > 0.25).any(axis=1).mean()), 4),
            "thumb_index_contact_steps": ti,
            "frames": frames,
        })
    print(pd.DataFrame(rows).to_string(index=False))


def part_b(ep=0, n_frames=4):
    m, d, act, _ = load_right()
    renderer = mujoco.Renderer(m, height=360, width=360)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance, cam.elevation, cam.azimuth = 0.35, -25, 130
    cam.lookat[:] = [0.0, 0.0, 0.06]

    df = pd.read_parquet(
        HERE.parent / f"data/data/chunk-000/episode_{ep:06d}.parquet",
        columns=["observation.state"])
    state = np.stack(df["observation.state"].to_numpy())
    cmds = {v: convert_state38(state, v)["right"]["cmd"] for v in VARIANTS}

    # 3 frames of deepest closure (grasp holds) + 1 of least closure (open hand)
    score = cmds["direct"].sum(axis=1)
    picks, min_gap = [], 20
    for t in np.argsort(-score):
        if all(abs(t - p) >= min_gap for p in picks):
            picks.append(int(t))
        if len(picks) == n_frames - 1:
            break
    picks.append(int(np.argmin(score)))
    picks.sort()

    video = HERE.parent / f"data/videos/chunk-000/observation.images.cam_high/episode_{ep:06d}.mp4"
    reader = imageio.get_reader(str(video))
    rows = []
    for t in picks:
        vframe = reader.get_data(t)
        import PIL.Image
        vframe = np.asarray(PIL.Image.fromarray(vframe).resize(
            (round(vframe.shape[1] * 360 / vframe.shape[0]), 360)))
        tiles = [vframe]
        for v in VARIANTS:
            set_pose(m, d, act, cmds[v][t])
            renderer.update_scene(d, camera=cam)
            tiles.append(renderer.render().copy())
        rows.append(np.concatenate(tiles, axis=1))
    reader.close()
    strip = np.concatenate(rows, axis=0)
    out = HERE / "outputs" / f"exp_thumb_ep{ep}_video_direct_range.png"
    import PIL.Image
    PIL.Image.fromarray(strip).save(out)
    print(f"Part B: frames {picks} -> {out}")


if __name__ == "__main__":
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    part_a()
    part_b()
