# data_filtering — Ability→BrainCo conversion + feasibility filtering

Converts the finger channels of the local EgoDex→Ability-Hand LeRobot dataset
(`data/`, 38-dim state/action; finger blocks at dims 9:19 left / 28:38 right)
into **BrainCo Revo 2** motor commands, and screens every frame for physical
executability on that hand. Code entry points, in pipeline order:

| script | what it does |
|---|---|
| `patch_vendor_mjcf.py` | regenerates corrected MJCF from the pristine vendor XMLs |
| `step0_sanity_check.py` | servo/coupling/range sanity of the patched models (+renders) |
| `step1_convert.py` | converts all episodes → `outputs/converted/*.npz` + saturation stats |
| `step2_velocity_check.py` | analytic velocity-feasibility + freeze→jump artifact scan |
| `step3_sim_replay.py` | MuJoCo dynamic replay: tracking error + self-collision report |
| `step4_visual_check.py` | side-by-side mp4: left render \| human video \| right render |
| `exp_thumb_mapping.py` | the experiment that selected the "span" mapping mode |
| `brainco_mapping.py` | the mapping itself (importable; single source of truth) |

Run any script with `uv run --no-project --with mujoco,numpy,pandas,pyarrow,imageio,imageio-ffmpeg,pillow python data_filtering/<script>.py`
(the repo venv is broken; see repo notes. Rendering needs `MUJOCO_GL=egl`, set automatically.)

## The BrainCo Revo 2 command contract (verified from vendor docs/code)

- 6 motors, ID order `[thumb_flex, thumb_rot, index, middle, ring, pinky]`.
- Command = normalized motor position, `0–1000` int in the SDK, `[0,1]` float via
  Unitree's DDS bridge (`rt/brainco/{left,right}/cmd`, `unitree_go MotorCmds_.q`,
  100 Hz, `dq`=speed, recommend 1.0). 0 = fully open, 1 = fully closed.
- Firmware maps the command **linearly onto joint angle**: `angle = min + cmd·(max−min)`,
  factory `min=0`, `max = [59°, 90°, 81°, 81°, 81°, 81°]` (Modbus reg 1070 doc;
  unit-mode reg 937 can switch the whole protocol to physical 0.1° units).
  Caution: min/max registers are user-writable and reset at power-off.
- Passive distal segments follow mechanically: fingers ×1.155, thumb ×1.0
  (from the official `revo2_description` URDF mimic tags).
- URDF velocity limits: thumb_flex 2.5303, thumb_rot 2.6175, fingers 2.2685 rad/s.

## Vendor model assets & patches

`assets/revo2_mjcf/` is BrainCo's official MuJoCo release (Aliyun OSS
`Revo2_xml.zip`; GitHub mirror: `BrainCoTech/brainco-description`), plus the two
official URDFs. The vendor MJCF has real bugs; `patch_vendor_mjcf.py` produces
`brainco-*hand-v2-patched.xml` (the only files the pipeline loads):

1. **Distal couplings broken twice**: MuJoCo equality semantics are
   `joint1 = polycoef(joint2)` but the vendor put proximal as joint1 (inverting
   the ratio — measured distal/proximal = 1/1.155), and additionally swapped the
   index/thumb multipliers. Fixed to URDF ground truth.
2. **Right-hand joint ranges** were 87/60/84° (Unitree's URDF values); left had
   the spec's 90/59/81°. Both now use the spec values, which match the firmware's
   factory min/max registers. Distal ranges get ~4% headroom above the coupled
   max — a range limit exactly at the coupled max fights the equality constraint
   and stalls the finger ~2° short.
3. **Right-hand fingertip bodies had free, unactuated hinge joints** (they'd
   dangle and fake self-collisions); removed — real tips are rigid.

Left-model quirks to remember: it is a simplified variant (no tip/touch bodies,
no sites, all geoms unnamed — classify contacts by *body* name), and is not a
perfect mirror of the right (middle/ring distal bodies carry a small extra
pre-rotation). Fingertip-level collision on the left is therefore coarser.

## The mapping (why "span" mode)

Source values are Ability Hand URDF joint radians; only 6 of the 10 dims per
hand are independent (PIP = 1.05851325·MCP + 0.72349796 exactly — data-verified).
BrainCo order requires swapping the two thumb dims, and Ability's thumb rotator
is negative-going (sign flip).

Two candidate value maps were tested (`exp_thumb_mapping.py`):

- **direct** (angle copied 1:1, clamped): REJECTED. Ability spans are ~1.74–2.09
  rad vs BrainCo's 1.03–1.57, so mid-range human poses render as a clenched fist:
  31–43% thumb-flex saturation, thumb pressed ~0.4 rad into the curled index for
  most of every episode (4644 interpenetration steps in a 30-episode sample), and
  the side-by-side against the human video shows a fist where the human pinches.
- **span** (fraction of source span → fraction of target range): SELECTED.
  `cmd = q / [1.74, 2.0944, 1.74, 1.74, 1.74, 1.74]` (thumb rot uses the
  data-verified 2.0944 span — the retargeter exceeded PSYONIC's URDF bound of
  1.74; all others are the PSYONIC URDF limits). This is also exactly BrainCo's
  own command semantics ("percentage of travel"). Result: ~0% saturation, zero
  thumb-index interpenetration in the sample, open/pinch poses visually match
  the video. Residual known gap: pinches show near-contact rather than contact;
  fine-tune the thumb channels against real hardware when available.

Deployment note: a policy trained on these commands outputs exactly what the
Unitree DDS bridge expects (`q ∈ [0,1]` per motor, absolute) — no runtime
conversion beyond clamping.

## Outputs (`outputs/`)

- `converted/episode_XXXXXX.npz`: `{left,right}_{state,action}_cmd` (T×6 float32,
  [0,1]) + `_saturated` masks.
- `step1_summary.json`, `step1_per_episode.csv` — saturation stats.
- `step2_flags.npz` (per-frame velocity/freeze-jump flags), `step2_summary.json`, csv.
- `step3_flags.npz` (per-frame hard-tracking-error flags), `step3_summary.json`, csv.
- `visual/ep_XXXXXX_compare.mp4` — step-4 strips (red top bar on a hand tile =
  that frame had a saturated command).
