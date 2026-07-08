"""Ability Hand joint angles (dataset) -> BrainCo Revo 2 motor commands.

Dataset side (see data/meta/modality.json and the repo memory notes):
  state/action are 38-dim; finger blocks are dims 9:19 (left) and 28:38 (right),
  Ability Hand URDF joint angles in radians, ordered
    [thumb_rot, thumb_flex, idx_MCP, idx_PIP, mid_MCP, mid_PIP,
     ring_MCP, ring_PIP, pky_MCP, pky_PIP]
  The 4 PIPs are exact mimics (PIP = 1.05851325*MCP + 0.72349796, residual ~2e-7
  in the data), so there are 6 independent DOFs -- same topology as the Revo 2.
  Ability thumb_rot is NEGATIVE-going: 0 = thumb lateral (open), -2.09 rad = full
  opposition; BrainCo thumb rotation is positive 0..1.57 rad, hence the sign flip.

BrainCo side (verified from the vendor protocol docs, see data_filtering/README.md):
  6 motors, ID order [thumb_flex, thumb_rot, index, middle, ring, pinky].
  Command is a normalized motor position; firmware maps it LINEARLY onto the
  joint-angle range:  angle = min + cmd * (max - min), factory min=0 and
  max = [59, 90, 81, 81, 81, 81] deg. 0 = fully open, 1 = fully closed.
  Unitree's DDS bridge (rt/brainco/{l,r}/cmd) takes exactly this [0,1] value.

Conversion policy: direct angle preservation with clamping ("option A"): the
BrainCo joint is commanded to the same absolute angle the Ability joint held,
saturating at the Revo 2's smaller ranges. Clamp masks are returned so the
filtering pipeline can quantify saturation instead of silently accepting it.
"""

import numpy as np

# dataset layout
STATE_DIM = 38
FINGERS_LEFT = slice(9, 19)
FINGERS_RIGHT = slice(28, 38)
PIP_MIMIC_COEF = (1.05851325, 0.72349796)  # PIP = a*MCP + b

# within a 10-dim Ability finger block: indices of the 6 independent DOFs,
# ordered to match BrainCo motor IDs [thumb_flex, thumb_rot, idx, mid, ring, pky]
ABILITY_IDX_FOR_BRAINCO = np.array([1, 0, 2, 4, 6, 8])
THUMB_ROT_SIGN = -1.0  # Ability opposition is negative-going, BrainCo positive
# Ability Hand source spans per BrainCo motor, for the "span" mapping mode.
# PSYONIC URDF (ability-hand-api): thumb_q1 rotator [-1.74, 0], thumb_q2 flexor
# [0, 1.74], finger q1 MCP [0, 1.74]. Exception: the dataset's retargeting used a
# loosened thumb-rotator bound -- its minimum hits -2.0944 (-120 deg) exactly --
# so we take the data-verified 2.0944 span there.
ABILITY_SPAN_FOR_BRAINCO = np.array([1.74, 2.0944, 1.74, 1.74, 1.74, 1.74])

BRAINCO_MOTOR_NAMES = ["thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky"]
BRAINCO_RANGE_RAD = np.array([1.03, 1.57, 1.41, 1.41, 1.41, 1.41])
# from the official revo2_description URDF <limit velocity=...>
BRAINCO_VEL_LIMIT_RAD_S = np.array([2.5303, 2.6175, 2.2685, 2.2685, 2.2685, 2.2685])

# the patched MJCF's actuator order is [thumb_rot(metacarpal), thumb_flex(proximal),
# index, middle, ring, pinky]: ctrl slot i takes BrainCo motor BRAINCO_MOTOR_FOR_MJCF[i]
BRAINCO_MOTOR_FOR_MJCF = np.array([1, 0, 2, 3, 4, 5])


def ability10_to_brainco_rad(q10: np.ndarray, mode: str = "span") -> np.ndarray:
    """(..., 10) Ability finger block -> (..., 6) unclamped BrainCo joint angles [rad].

    mode:
      "direct" -- joint angles copied 1:1 (thumb sign-flipped). Rejected in the
                  mapping experiment: Ability's spans are ~1.7-2.1 rad vs BrainCo's
                  1.03-1.57, so mid-range human poses render as a clenched fist.
      "span"   -- fraction-of-range preserved: q / ability_span * brainco_range.
                  Matches BrainCo's own command semantics (percentage of travel).
    """
    q6 = np.asarray(q10)[..., ABILITY_IDX_FOR_BRAINCO].copy()
    q6[..., 1] *= THUMB_ROT_SIGN
    if mode == "span":
        q6 *= BRAINCO_RANGE_RAD / ABILITY_SPAN_FOR_BRAINCO
    else:
        assert mode == "direct", mode
    return q6


def brainco_rad_to_cmd(q6_rad: np.ndarray, clip: bool = True):
    """BrainCo joint angles [rad] -> normalized command in [0,1] + saturation mask.

    The mask marks values that fell outside [0, range] BEFORE clamping.
    """
    cmd = np.asarray(q6_rad) / BRAINCO_RANGE_RAD
    mask = (cmd < 0.0) | (cmd > 1.0)
    if clip:
        cmd = np.clip(cmd, 0.0, 1.0)
    return cmd, mask


def cmd_to_mjcf_ctrl(cmd6: np.ndarray) -> np.ndarray:
    """Normalized [0,1] command (BrainCo motor order) -> MJCF ctrl vector [rad]."""
    rad = np.asarray(cmd6) * BRAINCO_RANGE_RAD
    return rad[..., BRAINCO_MOTOR_FOR_MJCF]


def convert_state38(arr: np.ndarray, mode: str = "span"):
    """(T, 38) state/action array -> dict with per-hand (T, 6) commands and masks."""
    out = {}
    for hand, sl in (("left", FINGERS_LEFT), ("right", FINGERS_RIGHT)):
        rad = ability10_to_brainco_rad(arr[:, sl], mode)
        cmd, sat = brainco_rad_to_cmd(rad)
        out[hand] = {"rad_unclamped": rad, "cmd": cmd, "saturated": sat}
    return out
