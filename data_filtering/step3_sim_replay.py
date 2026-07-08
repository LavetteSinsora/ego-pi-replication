"""Step 3: dynamic replay of the converted commands in MuJoCo.

Exactly the scheme discussed: initialize the hand at the episode's first state,
then at every dataset tick (20 fps) command action[t] (= state[t+1]) through the
position servos and integrate physics for 50 ms. Because the servos, force
limits, couplings, and self-collision are modeled, two failure modes surface
that the analytic step-2 check cannot see:

  - persistent tracking error: the commanded pose is blocked (finger pressed
    into another finger/palm) or the servo cannot keep up;
  - self-collision events: which link pairs touch, how often, how hard.

Contact itself is not automatically "bad" (a full fist legitimately touches the
palm; in reality there is usually an object/cloth in between) -- the primary
infeasibility signal is LARGE STEADY tracking error; contacts are the diagnosis.

Run: uv run --no-project --with mujoco,numpy,pandas python data_filtering/step3_sim_replay.py
"""

import glob
import json
import re
import time
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from brainco_mapping import BRAINCO_MOTOR_NAMES, cmd_to_mjcf_ctrl

HERE = Path(__file__).parent
FPS = 20.0
ERR_SOFT = 0.10   # rad, ~5.7 deg: noticeable lag
ERR_HARD = 0.25   # rad, ~14 deg: command effectively unreachable
COUPLE = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}
FINGER_OF = ["thumb", "index", "middle", "ring", "pinky", "palm"]


def load_hand(prefix: str):
    path = HERE / "assets/revo2_mjcf" / f"xml_{prefix}" / f"brainco-{prefix}hand-v2-patched.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    act_adr = np.array([model.joint(model.actuator(i).trnid[0]).qposadr[0]
                        for i in range(model.nu)])
    # distal joints driven by the equality couplings, keyed by finger name
    dist_adr = {f: model.joint(f"{prefix}_{f}_distal_joint").qposadr[0] for f in COUPLE}
    prox_adr = {f: model.joint(f"{prefix}_{'thumb_proximal' if f == 'thumb' else f + '_proximal'}_joint").qposadr[0]
                for f in COUPLE}
    # classify by BODY name -- the left vendor model's geoms are all unnamed
    geom_finger = []
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        finger = next((f for f in FINGER_OF[:5] if f"_{f}_" in body), "palm")
        geom_finger.append(finger)
    return model, data, act_adr, dist_adr, prox_adr, geom_finger


def replay_episode(model, data, act_adr, dist_adr, prox_adr, geom_finger,
                   state_cmd, action_cmd, n_sub):
    mujoco.mj_resetData(model, data)
    init_ctrl = cmd_to_mjcf_ctrl(state_cmd[0])
    data.qpos[act_adr] = init_ctrl
    for f, ratio in COUPLE.items():
        data.qpos[dist_adr[f]] = ratio * data.qpos[prox_adr[f]]
    mujoco.mj_forward(model, data)

    T = len(action_cmd)
    err = np.zeros((T, len(act_adr)))
    contact_frames = 0
    pair_counts = {}
    max_penetration = 0.0
    for t in range(T):
        data.ctrl[:] = cmd_to_mjcf_ctrl(action_cmd[t])
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        err[t] = np.abs(data.qpos[act_adr] - data.ctrl)
        if data.ncon:
            contact_frames += 1
            for c in data.contact[: data.ncon]:
                pen = -c.dist
                if pen <= 0:
                    continue
                pair = tuple(sorted((geom_finger[c.geom1], geom_finger[c.geom2])))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                max_penetration = max(max_penetration, pen)
    return err, contact_frames, pair_counts, max_penetration


def main():
    files = sorted(glob.glob(str(HERE / "outputs/converted/episode_*.npz")))
    assert files, "run step1_convert.py first"

    hands = {p: load_hand(p) for p in ("left", "right")}
    n_sub = round(1 / FPS / hands["left"][0].opt.timestep)

    rows, flags = [], {}
    pair_totals = {}
    t0 = time.time()
    for i, fpath in enumerate(files):
        ep = int(re.search(r"episode_(\d+)", fpath).group(1))
        z = np.load(fpath)
        rec = {"episode": ep, "frames": len(z["left_state_cmd"])}
        for hand in ("left", "right"):
            model, data, act_adr, dist_adr, prox_adr, geom_finger = hands[hand]
            err, ncf, pairs, pen = replay_episode(
                model, data, act_adr, dist_adr, prox_adr, geom_finger,
                z[f"{hand}_state_cmd"], z[f"{hand}_action_cmd"], n_sub)
            # err is in MJCF actuator order; report per BrainCo motor via ctrl mapping
            rec[f"{hand}_err_mean"] = float(err.mean())
            rec[f"{hand}_err_max"] = float(err.max())
            rec[f"{hand}_soft_frames"] = int((err > ERR_SOFT).any(axis=1).sum())
            rec[f"{hand}_hard_frames"] = int((err > ERR_HARD).any(axis=1).sum())
            rec[f"{hand}_contact_frames"] = ncf
            rec[f"{hand}_max_penetration_mm"] = round(pen * 1000, 2)
            flags[f"ep{ep:06d}_{hand}_hard"] = (err > ERR_HARD).any(axis=1)
            for k, v in pairs.items():
                pair_totals[k] = pair_totals.get(k, 0) + v
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} episodes, {time.time()-t0:.0f}s elapsed")

    df = pd.DataFrame(rows).sort_values("episode")
    df.to_csv(HERE / "outputs" / "step3_per_episode.csv", index=False)
    np.savez_compressed(HERE / "outputs" / "step3_flags.npz", **flags)

    tot = df["frames"].sum()
    summary = {
        "frames": int(tot),
        "soft_err_frames": {h: int(df[f"{h}_soft_frames"].sum()) for h in ("left", "right")},
        "hard_err_frames": {h: int(df[f"{h}_hard_frames"].sum()) for h in ("left", "right")},
        "contact_frames": {h: int(df[f"{h}_contact_frames"].sum()) for h in ("left", "right")},
        "max_penetration_mm": {h: float(df[f"{h}_max_penetration_mm"].max()) for h in ("left", "right")},
        "contact_pair_step_counts": {f"{a}-{b}": v for (a, b), v in
                                     sorted(pair_totals.items(), key=lambda kv: -kv[1])},
    }
    (HERE / "outputs" / "step3_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nreplayed {len(files)} episodes / {tot} frames per hand "
          f"({time.time()-t0:.0f}s total)")
    for h in ("left", "right"):
        print(f"  {h:5s}: err>{ERR_SOFT} rad frames {summary['soft_err_frames'][h]} "
              f"({summary['soft_err_frames'][h]/tot:.2%}), "
              f"err>{ERR_HARD} rad frames {summary['hard_err_frames'][h]} "
              f"({summary['hard_err_frames'][h]/tot:.2%}), "
              f"contact frames {summary['contact_frames'][h]} "
              f"({summary['contact_frames'][h]/tot:.2%})")
    print("  contact pairs (physics-step counts):")
    for k, v in list(summary["contact_pair_step_counts"].items())[:8]:
        print(f"    {k:16s} {v}")
    print("\n  top-10 episodes by hard tracking error frames:")
    df["hard"] = df["left_hard_frames"] + df["right_hard_frames"]
    for _, r in df.nlargest(10, "hard").iterrows():
        print(f"    ep {int(r.episode):4d}  hard L={int(r.left_hard_frames)} "
              f"R={int(r.right_hard_frames)}  err_max L={r.left_err_max:.2f} "
              f"R={r.right_err_max:.2f} rad")


if __name__ == "__main__":
    main()
