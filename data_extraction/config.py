"""Single source of truth for every pipeline-affecting variable.

All stages read from one frozen `PipelineConfig`. Each stage declares the
fields it depends on (STAGE_FIELDS); a stage's cache key is the hash of its
own fields plus every upstream stage's (STAGE_DEPS closure), so changing a
field re-runs exactly the stages it can affect and nothing else.

Derived values (fps, min_subepisode_ticks, state/action dims) are properties,
never independently settable - two knobs for one fact is a footgun.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    # ---- timeline / chunking (thread through everything) ----
    control_hz: float = 30.0          # control tick grid; also the LeRobot fps
    action_horizon: int = 50          # chunk length H (ticks per inference)
    hands: tuple[str, ...] = ("left", "right")
    rotation_repr: str = "6d"         # "6d" only for now (state+actions)

    # ---- s001 resampling ----
    max_gap_ms: float = 70.0          # bracketing-sample gap for a valid tick
    cam_stale_ms: float = 80.0        # max nearest-camera-frame age (filtered in s004)
    grid_anchor: str = "first_camera_frame"

    # ---- s002_01 eef label / frame convention ----
    pose_frame: str = "flange"        # "flange" (G(t), B baked in) | "wrist"
    # "dataset_mean": chordal mean of per-episode t0 calibrations - fixed,
    # global, self-consistent. "geometric" (palm-to-palm) needs the TRUE
    # flange->Revo2 mount rotation in revo2_mount_rpy_deg; with the default
    # identity guess it is ~120 deg off and IK cannot track - switch to it
    # only once the physical mount transform is measured.
    b_alignment: str = "dataset_mean"  # "dataset_mean" | "geometric" | "per_episode"
    revo2_mount_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)  # flange->Revo2 base
    max_tick_rotation_deg: float = 60.0   # continuity gate: max single-tick rotation

    # ---- s002_02 hand label ----
    fingertip_source: str = "pico"    # "pico" | "hamer" (stub)
    hand_calib: str = "per_episode"   # "per_episode" | "shared_recording"
    hand_calib_recording: str = ""    # episode path when hand_calib == shared_recording
    hand_align: str = "geometric"     # HandRetargeter align mode
    hand_rate_limit: bool = True

    # ---- s003 placement + proprioception ----
    placement_scope: str = "episode"  # "episode" | "task" (follow-up)
    reach_margin_m: float = 0.03
    reach_w: float = 50.0
    anchor_w: float = 0.2
    proprio_source: str = "ik_fk"     # "ik_fk" (sim FK, deployment-faithful) | "direct"
    state_content: str = "eef+hand"   # "eef" | "eef+hand" | "eef+hand+qpos"
    ik_iters: int = 5

    # ---- s004 filters + splitting ----
    vel_max_m_s: float = 1.5
    ang_vel_max_deg_s: float = 500.0
    ik_err_max_cm: float = 2.0
    ik_err_max_deg: float = 10.0
    hand_blocked_filter: bool = True
    hand_contact_filter: bool = True
    hand_residual_max_mm: float = 25.0
    hand_residual_min_run: int = 3    # residual must exceed threshold this many consecutive ticks
    # fingers whose retarget residual gates data quality. Ring/pinky are
    # excluded by default: the retargeter has no pinch terms for them and
    # their ~20-30 mm residual is systematic morphology mismatch, not a
    # tracking failure (measured on put_bottle_in_box ep1-3).
    hand_residual_fingers: tuple[str, ...] = ("thumb", "index", "middle")
    filter_hands_independently: bool = True
    allow_terminal_padding: bool = True
    # Interior bad runs of <= this many ticks are BRIDGED instead of splitting
    # the episode: the flagged ticks stay inside the sub-episode (their stored
    # poses may appear inside action windows) but are excluded as datapoint
    # anchors (anchor_bad mask, enforced by the loader's boundary indexing).
    # 0 disables bridging. 9 ticks = 0.3 s at 30 Hz; chosen because 8 of the
    # 11 interior gaps on put_bottle_in_box are <= 9 ticks.
    bridge_max_ticks: int = 9

    # ---- s005 output ----
    output_root: str = str(REPO_ROOT / "lerobot_datasets")
    repo_id: str = "ego2g1/put_bottle_in_box"
    video_codec: str = "libx264"
    image_size: tuple[int, int] | None = None   # None = native
    task_prompt: str = "put the bottle in the box"
    robot_type: str = "unitree_g1_brainco_revo2"

    # ---- loader / training ----
    use_quantile_norm: bool = True

    # ---- paths (not part of any cache key) ----
    episodes_dir: str = str(REPO_ROOT / "data" / "put_bottle_in_box")
    work_dir: str = str(REPO_ROOT / "data_extraction_work")

    # ------------------------------------------------------------- derived
    @property
    def fps(self) -> float:
        return self.control_hz

    @property
    def min_subepisode_ticks(self) -> int:
        return self.action_horizon + 1   # anchor + one full chunk

    @property
    def eef_dim(self) -> int:
        assert self.rotation_repr == "6d", "only 6d implemented"
        return 9                          # 3 trans + 6d rot

    @property
    def hand_dim(self) -> int:
        return 6                          # BrainCo Revo2 motors

    @property
    def state_dim(self) -> int:
        per_hand = {"eef": self.eef_dim,
                    "eef+hand": self.eef_dim + self.hand_dim,
                    "eef+hand+qpos": self.eef_dim + self.hand_dim + 7}[self.state_content]
        return per_hand * len(self.hands)

    @property
    def action_dim(self) -> int:
        return (self.eef_dim + self.hand_dim) * len(self.hands)

    # -------------------------------------------------------------- hashing
    def stage_hash(self, stage: str) -> str:
        fields = set()
        for s in _closure(stage):
            fields |= set(STAGE_FIELDS[s])
        payload = {f: getattr(self, f) for f in sorted(fields)}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        """Hash over every data-affecting field (sections A-F); stamped into
        the dataset and asserted by the training config + dashboard."""
        return self.stage_hash("s005")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["_derived"] = {"fps": self.fps,
                         "min_subepisode_ticks": self.min_subepisode_ticks,
                         "state_dim": self.state_dim,
                         "action_dim": self.action_dim,
                         "config_hash": self.config_hash}
        return d


# Fields each stage's *own* computation reads. Cache keys use the dependency
# closure, so downstream stages inherit upstream fields automatically.
STAGE_FIELDS = {
    "s001": ("control_hz", "max_gap_ms", "grid_anchor"),
    "s003_placement": ("placement_scope", "reach_margin_m", "reach_w", "anchor_w"),
    "b_calib": ("b_alignment", "revo2_mount_rpy_deg", "hands"),
    "s003_state": ("proprio_source", "ik_iters", "state_content", "pose_frame"),
    "s002_01": ("pose_frame", "rotation_repr", "max_tick_rotation_deg"),
    "s002_02": ("fingertip_source", "hand_calib", "hand_calib_recording",
                "hand_align", "hand_rate_limit"),
    "s004": ("action_horizon", "cam_stale_ms", "vel_max_m_s", "ang_vel_max_deg_s",
             "ik_err_max_cm", "ik_err_max_deg", "hand_blocked_filter",
             "hand_contact_filter", "hand_residual_max_mm", "hand_residual_min_run",
             "hand_residual_fingers", "filter_hands_independently",
             "allow_terminal_padding", "bridge_max_ticks"),
    "s005": ("repo_id", "video_codec", "image_size", "task_prompt", "robot_type",
             "use_quantile_norm"),
}

STAGE_DEPS = {
    "s001": (),
    "s003_placement": ("s001",),
    "b_calib": ("s003_placement",),
    "s003_state": ("b_calib",),
    "s002_01": ("b_calib",),
    "s002_02": ("s001",),
    "s004": ("s003_state", "s002_01", "s002_02"),
    "s005": ("s004",),
}

# execution order (global stages marked; the rest run per episode)
STAGE_ORDER = ["s001", "s003_placement", "b_calib", "s003_state",
               "s002_01", "s002_02", "s004", "s005"]
GLOBAL_STAGES = {"b_calib", "s005"}


def _closure(stage: str) -> set:
    out, todo = set(), [stage]
    while todo:
        s = todo.pop()
        if s in out:
            continue
        out.add(s)
        todo += list(STAGE_DEPS[s])
    return out


def load_config(path: str | None = None, overrides: dict | None = None) -> PipelineConfig:
    """Config from defaults, optionally updated by a JSON file then a dict of
    CLI overrides. Tuples survive the JSON round-trip."""
    values = {}
    if path:
        values.update(json.loads(Path(path).read_text()))
    if overrides:
        values.update(overrides)
    fields = {f.name: f for f in dataclasses.fields(PipelineConfig)}
    unknown = set(values) - set(fields)
    if unknown:
        raise SystemExit(f"unknown config fields: {sorted(unknown)}")
    for k, v in values.items():
        if isinstance(v, list):
            values[k] = tuple(v)
    return PipelineConfig(**values)
