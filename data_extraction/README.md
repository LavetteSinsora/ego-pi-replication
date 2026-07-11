# Data extraction condensed README
Turns Pico-collected egocentric recordings into a LeRobot dataset for VLA training. 

## Run the pipeline
In [config.py](config.py), specify:
  - `episode_dir`
  - `output_root`: root directory where outputs will land
  - `repo_id`: dataset/subdirectory's name 
  - `control_hz`: robot's control frequency (generated action label is at this frequency)
  - `task_prompt`, `action_horizon`

Then, run:
```bash
.venv/bin/python -m data_extraction.run_pipeline --jobs 4
```
Flags:
- `--jobs N` — episodes processed in parallel
- `--through s004` / `--stages s001,s004` — run a stage subset
- `--set key=value` (repeatable) — overrides certain field in config
- `--config file.json` - overrides entire config

Output will land in `cfg.output_root/cfg.repo_id`

## Inspect one converted episode

```bash
.venv/bin/python -m data_extraction.dashboard [episode_name]
```
Inspect converted `episode_name.hdf5` via the dashboard `episode_name_dashboard.html`.

## Data schema

**Consumes**: one raw HDF5 per demonstration (collected via ego_collect from Pico). Fields used:

- `camera/images_left_jpeg` + `camera/timestamps_ns` — headset JPEG frames
  (960×540)
- `left/right_hand_pose (N,26,7)` + `left/right_hand_active` +
  `timestamps_ns` — 26 OpenXR hand joints per side, each
  `[x y z qx qy qz qw]` (quaternion scalar-last), in **Pico's world frame**
  (OpenXR-style, y-up). Wrist = joint 1.
- `body_pose (N,24,7)` is recorded but **not consumed**.

Both streams are resampled onto a uniform control grid at `cfg.control_hz`
(default 30 Hz) (poses
lerp/slerp-interpolated; images matched to the nearest frame, never
interpolated).

**Produces**: a LeRobot dataset at `cfg.output_root/cfg.repo_id`
(fps = cfg.control_hz). We filter noisy ticks, so each episode can be splitted into multiple LeRobot episodes.

Per frame:

1. `image` — nearest camera frame for the tick
2. `state (30,)` — per hand (2 × 15):
   - flange pose in the G1 base frame: position (3) + 6D rotation (6)
   - BrainCo hand motor values: thumb flexion (1) + finger rotation (5), in [0, 1]
3. action (eef): 
  - `pose.left/right (9,)` — absolute flange pose in G1 base frame 
  - `hand.left/right (6,)` - absolute BrainCo hand motor values
  - Note that hand pose is stored as absolute pose. During training, `loader/relative_actions.py` converts *absolute pose* to *pose relative to the chunk-start tick*, while BrainCo command stays absolute
4. `arm_qpos (14,)` — G1 joint angle (2 x 7) producing current flange pose, for diagnostic

Hand motor values are the BrainCo Revo2's native command: normalized
position in **[0,1]** (0 = open, 1 = closed), order
`[thumb_flex, thumb_rot, index, middle, ring, pinky]`

Sidecar `extraction_meta.json` records
  - data extraction config (dict) + `config_hash` (asserted during training so train uses expected data confgi)
  - The real episode index each LeRobot episode comes from (since real episode can be splitted after filtering)
