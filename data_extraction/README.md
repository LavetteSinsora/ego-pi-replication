# data_extraction

Raw Pico egocentric recordings → LeRobot training dataset for π0/openpi, plus
a verification dashboard. Fully self-contained: everything needed (SE3 utils,
G1 sim + IK, placement, BrainCo Revo2 retargeting, assets) was migrated into
this package — no imports from `wrist_replay/` or `pico2usable/`.

Design record: [SPEC.md](SPEC.md) (interfaces, schemas, frame conventions).
All tunables: [config.py](config.py) (single frozen dataclass; per-stage
dependency-closure hashes drive caching).

## What one training datapoint is

- input: egocentric image `o_t` + proprioception `s_t` (per hand: flange pose
  in the pelvis frame, 3+6D rot, + 6 BrainCo motor positions → 30 dims)
- output: `H=50` actions, each the flange pose at tick `t+k` **relative to the
  flange pose at tick t** (the pose simultaneous with `o_t` — what FK returns
  at capture time), plus the 6 hand commands → (50, 30)

The dataset stores **absolute per-tick poses**; relative chunks are computed
in the data loader (`loader/relative_actions.py`), so `H` can change without
regenerating data. Stored poses are flange-frame (`G(t) = pelvis⁻¹·S·T_wrist·B`
with one global `B`), so deployment composes `T_target = T_anchor_FK · Δ_k`
with no extra alignment on the robot.

## Run

```bash
# all stages, all episodes (s005 needs lerobot; everything else runs without)
.venv/bin/python -m data_extraction.run_pipeline --jobs 4

# subsets / overrides
.venv/bin/python -m data_extraction.run_pipeline --through s004 --limit 3
.venv/bin/python -m data_extraction.run_pipeline --stages s004 --force
.venv/bin/python -m data_extraction.run_pipeline --set action_horizon=25 --set repo_id=ego-pi/test
```

Stages: `s001` uniform 30 Hz grid → `s003_placement` per-episode rigid `S` →
`b_calib` global wrist→flange `B` → `s003_state` IK pass (proprio + error
signals) → `s002_01` canonical flange poses (+ selftest/continuity gates) →
`s002_02` BrainCo commands → `s004` filters + sub-episode split → `s005`
LeRobot dataset + `extraction_meta.json` sidecar.

Intermediates land in `data_extraction_work/<episode>/<stage>.npz`; a stage
re-runs only when its config fields (dependency closure) or inputs changed.

## Filters (s004)

Per-tick bad masks, OR'd; interior bad runs ≤ `bridge_max_ticks` (default 9
ticks = 0.3 s) are bridged instead of splitting — those ticks stay in the
sub-episode but never anchor a datapoint (`anchor_bad`, enforced by the
loader). The rest splits into runs ≥ H+1 ticks; the run containing the
recording's last tick is `episode_real_end` (only there may action chunks pad
by repeating — "hold pose"; elsewhere padding would lie, and pi0 ignores
`action_is_pad`, so the loader's boundary wrapper drops those datapoints).

interpolation gap · camera staleness · wrist velocity · IK tracking error ·
hand blocked (sustained servo error under static command) · non-pinch
self-collision · fingertip retarget residual (thumb/index/middle only —
ring/pinky miss ~20–30 mm systematically: morphology, not tracking).

## Dashboard

```bash
# one page with an episode dropdown (index.html + per-episode reports + batch)
.venv/bin/python -m data_extraction.dashboard --site -o dashboard_site

# single episode / batch table only
.venv/bin/python -m data_extraction.dashboard episode_1 -o ep1.html
.venv/bin/python -m data_extraction.dashboard --batch

# interactive 3D (macOS needs mjpython)
.venv/bin/mjpython -m data_extraction.dashboard.viewer episode_1
```

Reads the final LeRobot dataset when it exists (`--source auto`), else the
work dir. Replays the **stored** data through the deployment-shaped chunked
loop — measured-anchor mode re-anchors on achieved FK exactly like a real
rollout — on a G1 with the BrainCo hands mounted on both flanges, alongside
the dynamic Revo2 hand replay, the human video, error curves for both anchor
modes, and the filter timeline.

## Training side (openpi fork)

`loader/` is the canonical numpy implementation; `third_party/openpi/src/
openpi/training/egopi.py` mirrors it for the training env (config
`pi0_egopi`; `DataConfig.custom_delta_timestamps` + `dataset_root` +
`boundary_aware` + `expected_config_hash` are honored by
`create_torch_dataset`). Equivalence is pinned by:

```bash
.venv/bin/python -m data_extraction.tests.test_loader_equivalence
```

which checks decoded loader actions against direct stage-npz computation and
the boundary indexing (needs lerobot + a written dataset).

## Environment

Repo venv (`.venv`, uv-managed): numpy scipy h5py mujoco mink pillow imageio
pyarrow. s005 + the equivalence test additionally need the openpi-pinned
lerobot (pulls torch):

```bash
uv pip install --python .venv/bin/python "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
```
