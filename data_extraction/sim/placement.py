"""Episode placement: one rigid transform S carrying the human wrist data
into the robot's world, plus the per-side wrist->flange alignment B.

Migrated verbatim from wrist_replay/replay.py (first_valid_tick,
compute_placement, refine_placement, calibrate_alignment). Weights and the
reach margin are keyword args; callers pass cfg values
(reach_margin_m, reach_w, anchor_w).
"""

import sys

import numpy as np

from ..common import frames


def first_valid_tick(grid):
    ok = grid.l_valid & grid.r_valid
    if not ok.any():
        sys.exit("no control tick has both wrists valid - cannot place robot")
    return int(np.argmax(ok))


def compute_placement(grid, k0, f_left, f_right):
    """Rigid transform S (yaw + translation) that carries the human wrist
    data into the robot's world: human t0 wrist midpoint -> nominal flange
    midpoint, human L->R direction -> robot L->R direction, human mean wrist
    height -> nominal flange height."""
    h_l, h_r = grid.l_pos[k0], grid.r_pos[k0]
    d_h = (h_r - h_l)[:2]
    d_f = (f_right - f_left)[:2]
    if np.linalg.norm(d_h) < 0.05:
        print("  WARNING: wrists nearly coincident at t0; yaw from L->R "
              "direction is unreliable, using identity yaw")
        theta = 0.0
    else:
        theta = (np.arctan2(d_f[1], d_f[0]) - np.arctan2(d_h[1], d_h[0]))
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    m_h = (h_l + h_r) / 2.0
    m_f = (f_left + f_right) / 2.0
    t = m_f - Rz @ m_h
    S = frames.se3_from_rot(Rz, t)
    print(f"  placement S: yaw={np.degrees(theta):.1f} deg, t={np.round(t, 3)}")
    return S


def refine_placement(S0, grid, backend, margin=0.03, reach_w=50.0, anchor_w=0.2):
    """Refine the heuristic placement over (dx, dy, dz, dyaw).

    Two-term objective:
    - reach violation (weight `reach_w`, dominant): squared overshoot of
      every tick's target beyond each arm's reach sphere - feasibility over
      the whole episode. Weighted heavily so the anchor term can never buy
      comfort at the price of unreachable targets (residual overshoot must
      stay well under `margin`).
    - ready-pose anchor (weight `anchor_w`): squared distance of the t0
      wrists from the ready-pose flange positions (upper arm vertical,
      forearm horizontal, hands in front of the chest). This keeps the
      solution in the natural manipulation zone instead of drifting to
      wherever the reach spheres happen to fit (e.g. down at the pelvis).

    The base stays upright by construction (yaw-only), and inter-hand
    geometry (e.g. one hand a little higher) is preserved exactly - a rigid
    transform moves both hands together."""
    from scipy.optimize import minimize

    sh = {s: backend.shoulder_anchor(s) for s in ("left", "right")}
    rr = {s: backend.max_reach(s) - margin for s in ("left", "right")}
    pts = {"left": grid.l_pos, "right": grid.r_pos}
    t0p = {"left": grid.l_pos[0], "right": grid.r_pos[0]}
    backend.reset_nominal()
    ready = {s: backend.flange_pose(s)[:3, 3] for s in ("left", "right")}
    mid = np.concatenate([grid.l_pos[:1], grid.r_pos[:1]]).mean(axis=0)
    mid = (S0 @ np.append(mid, 1.0))[:3]

    def build(p):
        dx, dy, dz, dyaw = p
        c, s_ = np.cos(dyaw), np.sin(dyaw)
        Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        return frames.se3_from_rot(Rz, mid - Rz @ mid + np.array([dx, dy, dz])) @ S0

    def total(S):
        v = 0.0
        for s in ("left", "right"):
            p = pts[s] @ S[:3, :3].T + S[:3, 3]
            d = np.linalg.norm(p - sh[s], axis=1)
            v += reach_w * float((np.maximum(0.0, d - rr[s]) ** 2).sum())
            a = (S @ np.append(t0p[s], 1.0))[:3]
            v += anchor_w * float(((a - ready[s]) ** 2).sum())
        return v

    res = minimize(lambda p: total(build(p)), np.zeros(4), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 1500})
    S = build(res.x)
    for s in ("left", "right"):
        p = pts[s] @ S[:3, :3].T + S[:3, 3]
        d = np.linalg.norm(p - sh[s], axis=1)
        n_out = int((d > rr[s]).sum())
        a = (S @ np.append(t0p[s], 1.0))[:3]
        print(f"  reach check {s}: {n_out}/{len(d)} ticks beyond reach "
              f"(worst overshoot {max(0.0, float(d.max() - rr[s])) * 100:.1f} cm); "
              f"t0 wrist {np.round(a, 3)} vs ready flange {np.round(ready[s], 3)}")
    print(f"  placement refine: shift {np.round(res.x[:3], 3)} m, "
          f"yaw {np.degrees(res.x[3]):.1f} deg")
    return S


def calibrate_alignment(grid, k0, backend):
    """B per side so the flange target orientation at t0 equals the robot's
    nominal flange orientation exactly: B = R_h(t0)^T @ R_flange_nominal."""
    out = {}
    for side in ("left", "right"):
        R_h = frames.mat_from_quat(
            (grid.l_quat if side == "left" else grid.r_quat)[k0])
        R_f = backend.flange_pose(side)[:3, :3]
        out[side] = frames.se3_from_rot(R_h.T @ R_f)
    return out
