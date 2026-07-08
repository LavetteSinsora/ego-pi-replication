"""Generate corrected MJCF files from the pristine BrainCo vendor XMLs.

The vendor MJCF (Revo2_xml.zip from BrainCo's Aliyun OSS, a.k.a. brainco-*hand-v2.xml)
has three issues relative to the official URDF (revo2_description) and the spec sheet
(thumb rot 0-90deg, thumb flex 0-59deg, fingers 0-81deg = firmware min/max register defaults):

1. Broken distal couplings, two bugs at once. (a) MuJoCo equality semantics are
   joint1 = polycoef(joint2), but the vendor wrote joint1=proximal, joint2=distal --
   inverted, so their "x1.155" actually yields distal = proximal/1.155 (verified by
   simulation). (b) The multipliers are also swapped between index and thumb. The URDF
   ground truth: thumb distal = 1.0 x proximal, all four fingers distal = 1.155 x
   proximal. We rewrite all five equality lines with joint1=distal, joint2=proximal.
2. Joint/ctrl ranges differ from the spec (87/60/84deg instead of 90/59/81deg). We align
   the sim to the spec ranges because those are what the firmware maps command 0..1000 onto.
3. The five fingertip bodies have free (unactuated, uncoupled, rangeless) hinge joints;
   on the real hand the tip is rigid w.r.t. the distal link. We drop those joints so the
   tips can't dangle and produce phantom self-collisions during replay.

Run: python3 data_filtering/patch_vendor_mjcf.py
Writes brainco-{left,right}hand-v2-patched.xml next to the originals.
"""

import re
from pathlib import Path

ASSETS = Path(__file__).parent / "assets" / "revo2_mjcf"

# (old, new, expected_count) applied per file; {p} = hand prefix "left"/"right"
COMMON_REPLACEMENTS = [
    # 1. rewrite couplings: joint1 must be the DRIVEN (distal) joint, and
    #    multipliers per URDF (fingers x1.155, thumb x1.0)
    ('<joint joint1="{p}_index_proximal_joint" joint2="{p}_index_distal_joint" polycoef="0 1. 0 0"/>',
     '<joint joint1="{p}_index_distal_joint" joint2="{p}_index_proximal_joint" polycoef="0 1.155 0 0"/>', 1),
    ('<joint joint1="{p}_middle_proximal_joint" joint2="{p}_middle_distal_joint" polycoef="0 1.155 0 0"/>',
     '<joint joint1="{p}_middle_distal_joint" joint2="{p}_middle_proximal_joint" polycoef="0 1.155 0 0"/>', 1),
    ('<joint joint1="{p}_ring_proximal_joint" joint2="{p}_ring_distal_joint" polycoef="0 1.155 0 0"/>',
     '<joint joint1="{p}_ring_distal_joint" joint2="{p}_ring_proximal_joint" polycoef="0 1.155 0 0"/>', 1),
    ('<joint joint1="{p}_pinky_proximal_joint" joint2="{p}_pinky_distal_joint" polycoef="0 1.155 0 0"/>',
     '<joint joint1="{p}_pinky_distal_joint" joint2="{p}_pinky_proximal_joint" polycoef="0 1.155 0 0"/>', 1),
    ('<joint joint1="{p}_thumb_proximal_joint" joint2="{p}_thumb_distal_joint" polycoef="0 1.155 0 0"/>',
     '<joint joint1="{p}_thumb_distal_joint" joint2="{p}_thumb_proximal_joint" polycoef="0 1.0 0 0"/>', 1),
    # 2. widen truncated ctrlranges to the full URDF/spec joint ranges
    ('ctrlrange="0 1.51"', 'ctrlrange="0 1.57"', 1),
    ('ctrlrange="0 1.04"', 'ctrlrange="0 1.03"', 1),
    ('ctrlrange="0 1.46"', 'ctrlrange="0 1.41"', 4),
]

# Actuated-joint ranges: only the RIGHT vendor file deviates from the URDF/spec
# (87/60/84 deg, matching Unitree's G1 URDF); the left file already has the spec
# values 90/59/81 deg. Align right to spec.
# Distal (coupled) joint ranges get ~4% headroom ABOVE the coupled maximum
# (1.155 x proximal max for fingers, 1.0 x for thumb): a range limit set exactly
# at the coupled max makes MuJoCo's limit constraint fight the equality
# constraint at full close and the finger stalls ~2 deg short. The equality
# drives these joints, so the range never actually binds.
RIGHT_ONLY_REPLACEMENTS = [
    ('range="0 1.5184"', 'range="0 1.57"', 1),    # thumb metacarpal (rotation), 90 deg
    ('name="right_thumb_proximal_joint" pos="0 0 0" axis="1 0 0" range="0 1.0472"',
     'name="right_thumb_proximal_joint" pos="0 0 0" axis="1 0 0" range="0 1.03"', 1),
    ('name="right_thumb_distal_joint" pos="0 0 0" axis="1 0 0" range="0 1.0472"',
     'name="right_thumb_distal_joint" pos="0 0 0" axis="1 0 0" range="0 1.08"', 1),
    ('range="0 1.4661"', 'range="0 1.41"', 4),    # finger proximal, 81 deg
    # finger distal range stays at the vendor's 1.693 (headroom over 1.155*1.41=1.629)
]

LEFT_ONLY_REPLACEMENTS = [
    ('name="left_thumb_distal_joint" pos="0 0 0" axis="-1 0 0" range="0 1.03"',
     'name="left_thumb_distal_joint" pos="0 0 0" axis="-1 0 0" range="0 1.08"', 1),
    ('range="0 1.63"', 'range="0 1.693"', 4),     # finger distal headroom
]

# 3. fingertip joints to remove (note: thumb tip joint has no "_joint" suffix)
TIP_JOINT_RE = r'\s*<joint name="{p}_(?:thumb_tip|index_tip_joint|middle_tip_joint|ring_tip_joint|pinky_tip_joint)"[^/]*/>'


def patch(prefix: str) -> Path:
    src = ASSETS / f"xml_{prefix}" / f"brainco-{prefix}hand-v2.xml"
    dst = src.with_name(src.stem + "-patched.xml")
    text = src.read_text()

    replacements = COMMON_REPLACEMENTS + (
        RIGHT_ONLY_REPLACEMENTS if prefix == "right" else LEFT_ONLY_REPLACEMENTS)
    for old, new, count in replacements:
        old, new = old.format(p=prefix), new.format(p=prefix)
        found = text.count(old)
        assert found == count, f"{src.name}: expected {count}x {old!r}, found {found}"
        text = text.replace(old, new)

    # the left vendor model is a simplified variant with no tip bodies at all
    expected_tips = 5 if prefix == "right" else 0
    text, n_tips = re.subn(TIP_JOINT_RE.format(p=prefix), "", text)
    assert n_tips == expected_tips, \
        f"{src.name}: expected {expected_tips} tip joints, removed {n_tips}"

    text = text.replace(
        f'<mujoco model="brainco-{prefix}hand-v2">',
        f'<mujoco model="brainco-{prefix}hand-v2-patched">', 1)
    dst.write_text(text)

    # "-free" variant for kinematic visualization: base gets the free joint the
    # vendor left commented out, so the wrist orientation data can drive it
    free_line = '<joint name="floating_base_joint" type="free"/>'
    commented = f"<!--      {free_line}-->"
    assert text.count(commented) == 1, f"{src.name}: freejoint comment not found"
    free_dst = src.with_name(src.stem + "-patched-free.xml")
    free_dst.write_text(text.replace(commented, free_line, 1))
    return dst


if __name__ == "__main__":
    for prefix in ("left", "right"):
        print("wrote", patch(prefix))
