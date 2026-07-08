"""Step 2: analytic velocity-feasibility check over the converted commands.

Two independent detectors over every frame of every episode (both hands):

1. Velocity feasibility: the per-frame command delta, expressed in rad/s at the
   dataset's 20 fps, must stay within the Revo 2 servo velocity limits from the
   official URDF. A frame that demands more is physically untrackable: the real
   hand would lag the label, so state[t+1] (= action[t]) would not be reached.

2. Freeze-then-jump signature: >= FREEZE_MIN consecutive frames where the whole
   hand's command is numerically frozen, followed by a > JUMP_THR (fraction of
   range) single-frame jump. That is the classic hand-tracking-lost-and-
   reacquired artifact (the source ARKit/Pico-style confidences were dropped in
   dataset conversion, so this heuristic is the stand-in).

Run: uv run --no-project --with numpy,pandas python data_filtering/step2_velocity_check.py
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from brainco_mapping import BRAINCO_MOTOR_NAMES, BRAINCO_RANGE_RAD, BRAINCO_VEL_LIMIT_RAD_S

HERE = Path(__file__).parent
FPS = 20.0
FREEZE_MIN = 10      # frames (0.5 s) of exactly-frozen pose
FREEZE_EPS = 1e-6    # rad; retargeted values are float, exact repeats mean upstream froze
JUMP_THR = 0.15      # fraction of full range in a single frame


def check_episode(cmd: np.ndarray):
    """cmd: (T, 6) normalized commands for one hand. Returns per-frame flag arrays."""
    dq_rad = np.diff(cmd, axis=0) * BRAINCO_RANGE_RAD          # (T-1, 6) rad/frame
    vel = np.abs(dq_rad) * FPS                                  # rad/s
    vel_viol = np.zeros(len(cmd), dtype=bool)
    vel_viol[1:] = (vel > BRAINCO_VEL_LIMIT_RAD_S).any(axis=1)  # flag the arrival frame

    frozen_step = (np.abs(dq_rad) < FREEZE_EPS).all(axis=1)     # (T-1,)
    jump = (np.abs(np.diff(cmd, axis=0)) > JUMP_THR).any(axis=1)
    freeze_jump = np.zeros(len(cmd), dtype=bool)
    run = 0
    for t, fz in enumerate(frozen_step):
        if fz:
            run += 1
        else:
            if run >= FREEZE_MIN and jump[t]:
                freeze_jump[t - run: t + 2] = True              # freeze + the jump frame
            run = 0
    return vel_viol, freeze_jump, vel


def main():
    files = sorted(glob.glob(str(HERE / "outputs/converted/episode_*.npz")))
    assert files, "run step1_convert.py first"

    rows = []
    flags_out = {}
    worst = np.zeros(6)
    for f in files:
        ep = int(re.search(r"episode_(\d+)", f).group(1))
        z = np.load(f)
        rec = {"episode": ep}
        for hand in ("left", "right"):
            cmd = z[f"{hand}_state_cmd"]
            vel_viol, freeze_jump, vel = check_episode(cmd)
            worst = np.maximum(worst, vel.max(axis=0))
            rec[f"{hand}_vel_viol_frames"] = int(vel_viol.sum())
            rec[f"{hand}_freeze_jump_frames"] = int(freeze_jump.sum())
            rec[f"{hand}_max_vel_frac"] = float((vel / BRAINCO_VEL_LIMIT_RAD_S).max())
            flags_out[f"ep{ep:06d}_{hand}_vel"] = vel_viol
            flags_out[f"ep{ep:06d}_{hand}_freezejump"] = freeze_jump
        rec["frames"] = len(z["left_state_cmd"])
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("episode")
    np.savez_compressed(HERE / "outputs" / "step2_flags.npz", **flags_out)
    df.to_csv(HERE / "outputs" / "step2_per_episode.csv", index=False)

    tot = df["frames"].sum()
    summary = {
        "frames": int(tot),
        "vel_violation_frames": {h: int(df[f"{h}_vel_viol_frames"].sum()) for h in ("left", "right")},
        "freeze_jump_frames": {h: int(df[f"{h}_freeze_jump_frames"].sum()) for h in ("left", "right")},
        "worst_vel_rad_s": dict(zip(BRAINCO_MOTOR_NAMES, worst.round(3).tolist())),
        "vel_limit_rad_s": dict(zip(BRAINCO_MOTOR_NAMES, BRAINCO_VEL_LIMIT_RAD_S.tolist())),
    }
    (HERE / "outputs" / "step2_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"checked {len(files)} episodes / {tot} frames")
    for h in ("left", "right"):
        v, fj = summary["vel_violation_frames"][h], summary["freeze_jump_frames"][h]
        print(f"  {h:5s}: velocity-infeasible frames {v} ({v/tot:.2%}), "
              f"freeze->jump frames {fj} ({fj/tot:.2%})")
    print("  worst observed velocity vs limit (rad/s):")
    for n in BRAINCO_MOTOR_NAMES:
        print(f"    {n:10s} {summary['worst_vel_rad_s'][n]:7.3f} / {summary['vel_limit_rad_s'][n]}")
    print("\n  top-10 episodes by demanded velocity (fraction of limit):")
    df["max_vel_frac"] = df[["left_max_vel_frac", "right_max_vel_frac"]].max(axis=1)
    for _, r in df.nlargest(10, "max_vel_frac").iterrows():
        print(f"    ep {int(r.episode):4d}  max {r.max_vel_frac:5.2f}x limit, "
              f"viol frames L={int(r.left_vel_viol_frames)} R={int(r.right_vel_viol_frames)}")


if __name__ == "__main__":
    main()
