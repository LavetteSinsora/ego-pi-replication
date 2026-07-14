"""Measure the PHYSICAL flange->Revo2-base rotation (R_mount) from the real robot,
and report the `revo2_mount_rpy_deg` to put in config.py.

R_mount is a mechanical constant (the hand is bolted on), so it cannot be read from
encoders — FK only gives you R_flange. You supply the one thing the robot can't tell
you: how the hand is ORIENTED, observed in a known arm pose.

Procedure
---------
1. Command the arm to a known configuration. Default here is ALL ARM JOINTS = 0
   (pass --arm-qpos to use another; 7 values for the chosen side, in
   sim.g1.ARM_JOINTS order).
2. Look at the real hand and report two directions **in the robot's pelvis frame**:
       +x = forward (out of the chest)   +y = robot's LEFT   +z = up
   --fingers : direction wrist -> middle knuckle (the way the fingers extend)
   --back    : the back-of-hand normal (pointing out of the BACK of the hand)
   Give them as axis words (+x, -y, ...) or as vectors ("0.7,0,-0.7"). They need not
   be exactly perpendicular — the tool orthonormalizes.
   NOTE: use the `=` form for values starting with '-', else argparse eats the dash:
       --back=-y        (not: --back -y)
3. The tool computes, using the same palm-frame construction as the pipeline
   (hand/fk_tables.py: x = wrist->middle-knuckle, z = back-of-hand normal, y = z*x):

       palm = R_flange · R_mount · G_r      =>      R_mount = R_flange^T · palm · G_r^T

   and prints R_mount as xyz-euler degrees -> `revo2_mount_rpy_deg`.
4. It also compares against the mount the CURRENT dataset implies (b_calib's
   mount_R_{side}) and reports how far the dataset's B is from the physically
   correct one. A large angle means deployed grasps will be rotated about the wrist.

Then: set revo2_mount_rpy_deg, set b_alignment="geometric", regenerate (B is baked
into the labels -> new config hash -> retrain).

    python -m data_extraction.calibrate_mount --side right --fingers +x --back +z
"""

import argparse
import pathlib

import numpy as np

AXES = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0],
        "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1]}


def parse_vec(s: str) -> np.ndarray:
    key = s.strip().lower()
    if key in AXES:
        v = np.array(AXES[key], float)
    else:
        v = np.array([float(x) for x in key.replace(" ", "").split(",")], float)
    if v.shape != (3,) or np.linalg.norm(v) < 1e-9:
        raise SystemExit(f"bad direction {s!r} — use +x/-y/... or 'a,b,c'")
    return v / np.linalg.norm(v)


def palm_from_observation(fingers, back) -> np.ndarray:
    """Same construction as hand/fk_tables.palm_frame_from_points: columns [x, y, z]
    with x = fingers, z = back-of-hand normal (orthogonalized), y = z x x."""
    x = fingers / np.linalg.norm(fingers)
    z = back - np.dot(back, x) * x  # orthogonalize against x
    n = np.linalg.norm(z)
    if n < 1e-6:
        raise SystemExit("--back is parallel to --fingers; they must differ")
    z /= n
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt


def geodesic_deg(A, B) -> float:
    c = (np.trace(A.T @ B) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    from scipy.spatial.transform import Rotation

    from .hand.fk_tables import load_tables
    from .sim.g1 import ARM_JOINTS, G1Backend

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--fingers", required=True, help="wrist->middle-knuckle dir, pelvis frame (+x/-y/... or a,b,c)")
    ap.add_argument("--back", required=True, help="back-of-hand normal, pelvis frame")
    ap.add_argument("--arm-qpos", nargs=7, type=float, default=None,
                    help="arm joints for THIS side in ARM_JOINTS order (default: all zeros)")
    args = ap.parse_args()

    be = G1Backend()
    be.data.qpos[:] = 0.0
    if args.arm_qpos is not None:
        be.data.qpos[be.arm_qpos_adr[args.side]] = np.asarray(args.arm_qpos, float)
    import mujoco
    mujoco.mj_forward(be.model, be.data)

    R_flange = be.world_to_base(be.flange_pose(args.side))[:3, :3]
    G_r = load_tables(args.side)["robot_palm"]
    palm_obs = palm_from_observation(parse_vec(args.fingers), parse_vec(args.back))

    R_mount = orthonormalize(R_flange.T @ palm_obs @ G_r.T)
    rpy = Rotation.from_matrix(R_mount).as_euler("xyz", degrees=True)

    print(f"\narm pose: {'all zeros' if args.arm_qpos is None else 'custom'} "
          f"({', '.join(ARM_JOINTS[args.side][:2])}, ...)")
    print(f"R_flange (pelvis frame), rows=axes:\n{np.round(R_flange, 3)}")
    print(f"\nobserved palm frame (columns x=fingers, y, z=back-of-hand):\n{np.round(palm_obs, 3)}")
    print(f"\n=> MEASURED R_mount (flange -> Revo2 base):\n{np.round(R_mount, 4)}")
    print(f"\n   revo2_mount_rpy_deg = ({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f})   # {args.side}")

    # how far is the CURRENT dataset's implied mount / B from this?
    bc = pathlib.Path(__file__).resolve().parent / "work" / "_global" / "b_calib.npz"
    if bc.exists():
        z = np.load(bc)
        mount_data = z[f"mount_R_{args.side}"]
        B_cur = z[f"B_{args.side}"][:3, :3]
        # recover R_wp from the stored calibration: mount_R = B^T R_wp G_r^T
        R_wp = B_cur @ mount_data @ G_r
        B_new = orthonormalize(R_wp @ G_r.T @ R_mount.T)  # geometric-mode B for the measured mount
        print(f"\n--- vs the CURRENT dataset (b_calib, mode=dataset_mean) ---")
        print(f"   dataset-implied mount vs measured mount : {geodesic_deg(mount_data, R_mount):6.1f} deg")
        print(f"   dataset B            vs corrected B     : {geodesic_deg(B_cur, B_new):6.1f} deg")
        print(f"   corrected B (geometric) rpy: "
              f"{np.round(Rotation.from_matrix(B_new).as_euler('xyz', degrees=True), 1)}")
        print("\n   A large angle here means the labels were built with the WRONG hand convention:\n"
              "   the arm would reach correctly but the hand comes out rotated about the wrist.\n"
              "   Fix: set revo2_mount_rpy_deg above, b_alignment='geometric', regenerate, retrain.")
    else:
        print(f"\n(no {bc} — skipping the comparison against the current dataset)")


if __name__ == "__main__":
    main()
