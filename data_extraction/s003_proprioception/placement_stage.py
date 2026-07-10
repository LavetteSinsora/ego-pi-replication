"""s003_placement: one rigid transform S per source episode.

Wraps the migrated two-stage placement (t0 heuristic -> whole-episode
Nelder-Mead over x,y,z,yaw minimizing reach violation + ready-pose anchor).
S is shared by all sub-episodes of the recording; actions are provably
invariant to it, proprioception is not.
"""

import numpy as np


def run_episode(cfg, ep_path):
    from ..sim.g1 import G1Backend
    from ..sim.placement import compute_placement, first_valid_tick, refine_placement
    from .grid_util import load_grid

    if cfg.placement_scope != "episode":
        raise NotImplementedError(f"placement_scope={cfg.placement_scope} (task scope is a follow-up)")

    grid, _ = load_grid(cfg, ep_path.stem)
    backend = G1Backend()
    k0 = first_valid_tick(grid)
    S0 = compute_placement(grid, k0,
                           backend.flange_pose("left")[:3, 3],
                           backend.flange_pose("right")[:3, 3])
    S = refine_placement(S0, grid, backend, margin=cfg.reach_margin_m,
                         reach_w=cfg.reach_w, anchor_w=cfg.anchor_w)

    # reach stats for the manifest (refine_placement prints them; recompute
    # so they are machine-readable for s004/dashboard)
    reach = {}
    for side, pre in (("left", "l"), ("right", "r")):
        pts = getattr(grid, f"{pre}_pos") @ S[:3, :3].T + S[:3, 3]
        d = np.linalg.norm(pts - backend.shoulder_anchor(side), axis=1)
        rr = backend.max_reach(side) - cfg.reach_margin_m
        reach[side] = {"n_out": int((d > rr).sum()),
                       "worst_overshoot_cm": max(0.0, float(d.max() - rr)) * 100.0,
                       "n_ticks": int(len(d))}
    return {"S": S}, {"k0": int(k0), "reach": reach}
