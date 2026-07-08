"""Step 4: visual gold-standard check -- BrainCo replica vs. the human video.

For chosen episodes, writes an mp4 laid out as
    [ left-hand render | egocentric human video | right-hand render ]
where each hand model is KINEMATICALLY posed at that frame's converted command
(the label, not the dynamics -- step 3 covers dynamics), and the hand BASE is
oriented by the dataset's wrist 6D rotation (dims 3:9 left / 22:28 right), so
the model turns the way the human wrist does. Watching the strip answers: when
the human's fingers curl/pinch/touch, does the commanded BrainCo pose match?

A red bar on top of a hand tile means >=1 of that hand's 6 commands was
saturated (clamped into [0,1]) at that frame.

Run: uv run --no-project --with mujoco,numpy,pandas,imageio,imageio-ffmpeg,pillow \\
       python data_filtering/step4_visual_check.py [--probe] [ep ...]
--probe renders a 4-frame PNG strip of ep 0 instead of mp4s (axis calibration).
Default episodes: 0, the worst step-2 (velocity) episode, and the two worst
step-3 (tracking-error) episodes, if those reports exist.
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd

from brainco_mapping import cmd_to_mjcf_ctrl, convert_state38

HERE = Path(__file__).parent
OUT = HERE / "outputs" / "visual"
TILE = 360  # render size; matches video height
COUPLE = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}
WRIST_SLICE = {"left": slice(0, 9), "right": slice(19, 28)}

# The wrist poses are in an OpenCV-style egocentric CAMERA frame (verified from
# the data: left hand x<0, right x>0, z ~ +0.4 m in front): x right, y down,
# z forward. Remap to a MuJoCo world where the camera looks along +y:
# world x = data x (right), world y = data z (forward), world z = -data y (up).
DATA_TO_WORLD = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, -1.0, 0.0]])


def rot6d_to_mat(v6: np.ndarray) -> np.ndarray:
    """Two orthonormal rotation-matrix columns -> full rotation matrix."""
    c1 = v6[:3] / np.linalg.norm(v6[:3])
    c2 = v6[3:] - np.dot(v6[3:], c1) * c1
    c2 /= np.linalg.norm(c2)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=1)


def pick_default_episodes():
    eps = [0]
    s2 = HERE / "outputs" / "step2_per_episode.csv"
    s3 = HERE / "outputs" / "step3_per_episode.csv"
    if s2.exists():
        df = pd.read_csv(s2)
        df["m"] = df[["left_max_vel_frac", "right_max_vel_frac"]].max(axis=1)
        eps.append(int(df.loc[df["m"].idxmax(), "episode"]))
    if s3.exists():
        df = pd.read_csv(s3)
        df["hard"] = df["left_hard_frames"] + df["right_hard_frames"]
        eps += df.nlargest(2, "hard")["episode"].astype(int).tolist()
    return sorted(set(eps))


class HandPoser:
    def __init__(self, prefix: str):
        path = HERE / "assets/revo2_mjcf" / f"xml_{prefix}" / f"brainco-{prefix}hand-v2-patched-free.xml"
        self.prefix = prefix
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.base_adr = self.model.joint("floating_base_joint").qposadr[0]
        self.act_adr = np.array([self.model.joint(self.model.actuator(i).trnid[0]).qposadr[0]
                                 for i in range(self.model.nu)])
        self.dist = {f: self.model.joint(f"{prefix}_{f}_distal_joint").qposadr[0] for f in COUPLE}
        self.prox = {f: self.model.joint(f"{prefix}_{f}_proximal_joint").qposadr[0] for f in COUPLE}
        self.renderer = mujoco.Renderer(self.model, height=TILE, width=TILE)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)
        # emulate the egocentric camera: it sits on -world-y looking along +y,
        # so screen right = world x = data x, screen down = -world z = data y
        # elevation must stay 0: the data rotation already encodes the real
        # camera's pitch, because the wrist pose is in the camera frame
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.4, 0, 90
        self.cam.lookat[:] = [0.0, 0.0, 0.0]

    def render_cmd(self, cmd6: np.ndarray, wrist9: np.ndarray | None = None,
                   pos_offset: np.ndarray | None = None) -> np.ndarray:
        if wrist9 is not None:
            mat_w = DATA_TO_WORLD @ rot6d_to_mat(wrist9[3:9])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat_w.flatten())
            self.data.qpos[self.base_adr + 3: self.base_adr + 7] = quat
            pos = wrist9[:3] - (pos_offset if pos_offset is not None else wrist9[:3])
            self.data.qpos[self.base_adr: self.base_adr + 3] = DATA_TO_WORLD @ pos
        else:
            self.data.qpos[self.base_adr: self.base_adr + 3] = 0.0
            self.data.qpos[self.base_adr + 3: self.base_adr + 7] = (1, 0, 0, 0)
        self.data.qpos[self.act_adr] = cmd_to_mjcf_ctrl(cmd6)
        for f, ratio in COUPLE.items():
            self.data.qpos[self.dist[f]] = ratio * self.data.qpos[self.prox[f]]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def load_episode(ep: int):
    df = pd.read_parquet(HERE.parent / f"data/data/chunk-000/episode_{ep:06d}.parquet",
                         columns=["observation.state"])
    state = np.stack(df["observation.state"].to_numpy())
    conv = convert_state38(state)
    return state, conv


def video_reader(ep: int):
    video = HERE.parent / f"data/videos/chunk-{ep // 1000:03d}/observation.images.cam_high/episode_{ep:06d}.mp4"
    return imageio.get_reader(str(video))


def resize_to_tile(frame: np.ndarray) -> np.ndarray:
    import PIL.Image
    if frame.shape[0] == TILE:
        return frame
    return np.asarray(PIL.Image.fromarray(frame).resize(
        (round(frame.shape[1] * TILE / frame.shape[0]), TILE)))


def compose_episode(ep: int, posers) -> Path:
    state, conv = load_episode(ep)
    mean_pos = {h: state[:, WRIST_SLICE[h]][:, :3].mean(axis=0) for h in ("left", "right")}
    reader = video_reader(ep)
    out_path = OUT / f"ep_{ep:06d}_compare.mp4"
    writer = imageio.get_writer(str(out_path), fps=20, codec="libx264",
                                quality=7, macro_block_size=1)
    T = len(state)
    for t, frame in enumerate(reader):
        if t >= T:
            break
        tiles = []
        for hand in ("left", "right"):
            tile = posers[hand].render_cmd(conv[hand]["cmd"][t],
                                           state[t, WRIST_SLICE[hand]],
                                           mean_pos[hand]).copy()
            if conv[hand]["saturated"][t].any():
                tile[:14, :] = (200, 30, 30)
            tiles.append(tile)
        writer.append_data(np.concatenate([tiles[0], resize_to_tile(frame), tiles[1]], axis=1))
    writer.close()
    reader.close()
    return out_path


def probe(ep: int = 0, frames=(6, 53, 181, 299)):
    """4-frame PNG strip [video | left | right] for axis-convention calibration."""
    posers = {p: HandPoser(p) for p in ("left", "right")}
    state, conv = load_episode(ep)
    mean_pos = {h: state[:, WRIST_SLICE[h]][:, :3].mean(axis=0) for h in ("left", "right")}
    reader = video_reader(ep)
    rows = []
    for t in frames:
        vframe = resize_to_tile(reader.get_data(t))
        tiles = [posers[h].render_cmd(conv[h]["cmd"][t], state[t, WRIST_SLICE[h]],
                                      mean_pos[h])
                 for h in ("left", "right")]
        rows.append(np.concatenate([tiles[0], vframe, tiles[1]], axis=1))
        print(f"t={t}: wrist L pos {np.round(state[t, 0:3], 3)}  "
              f"R pos {np.round(state[t, 19:22], 3)}")
    reader.close()
    import PIL.Image
    out = OUT / f"probe_ep{ep}.png"
    PIL.Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print("probe strip:", out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if a != "--probe"]
    if "--probe" in sys.argv[1:]:
        probe(int(args[0]) if args else 0)
        return
    episodes = [int(a) for a in args] or pick_default_episodes()
    posers = {p: HandPoser(p) for p in ("left", "right")}
    for ep in episodes:
        print("wrote", compose_episode(ep, posers))


if __name__ == "__main__":
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
