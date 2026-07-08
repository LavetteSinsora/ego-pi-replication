"""Step 1: convert the whole dataset's finger channels to BrainCo commands.

For every episode parquet, converts state and action finger blocks (both hands)
to normalized BrainCo commands, writes them to outputs/converted/episode_XXXXXX.npz,
and accumulates saturation statistics (how often each motor's command had to be
clamped into [0,1], i.e. the human/Ability pose exceeded the Revo 2's range).

Run: uv run --no-project --with pyarrow,pandas,numpy python data_filtering/step1_convert.py
"""

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from brainco_mapping import BRAINCO_MOTOR_NAMES, BRAINCO_RANGE_RAD, convert_state38

HERE = Path(__file__).parent
OUT = HERE / "outputs" / "converted"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    files = sorted(glob.glob(str(HERE.parent / "data/data/chunk-*/episode_*.parquet")))
    assert files, "no episode parquets found under data/"

    n_frames = 0
    sat_count = {("state", h): np.zeros(6, dtype=np.int64) for h in ("left", "right")}
    sat_count.update({("action", h): np.zeros(6, dtype=np.int64) for h in ("left", "right")})
    # pooled histogram of unclamped commands, 1% bins over [-1, 3]
    hist = {h: np.zeros((6, 400), dtype=np.int64) for h in ("left", "right")}
    bin_edges = np.linspace(-1.0, 3.0, 401)
    per_episode = []

    for f in files:
        ep = int(re.search(r"episode_(\d+)", f).group(1))
        df = pd.read_parquet(f, columns=["observation.state", "action"])
        state = np.stack(df["observation.state"].to_numpy())
        action = np.stack(df["action"].to_numpy())
        n_frames += len(state)

        conv = {"state": convert_state38(state), "action": convert_state38(action)}
        save = {}
        ep_sat = {}
        for kind in ("state", "action"):
            for hand in ("left", "right"):
                c = conv[kind][hand]
                sat_count[(kind, hand)] += c["saturated"].sum(axis=0)
                save[f"{hand}_{kind}_cmd"] = c["cmd"].astype(np.float32)
                save[f"{hand}_{kind}_saturated"] = c["saturated"]
                if kind == "state":
                    unclamped = c["rad_unclamped"] / BRAINCO_RANGE_RAD
                    for m in range(6):
                        hist[hand][m] += np.histogram(unclamped[:, m], bins=bin_edges)[0]
                ep_sat[f"{hand}_{kind}"] = float(c["saturated"].any(axis=1).mean())
        np.savez_compressed(OUT / f"episode_{ep:06d}.npz", **save)
        per_episode.append({"episode": ep, "frames": len(state), **ep_sat})

    summary = {
        "episodes": len(files),
        "frames": n_frames,
        "saturation_fraction": {
            f"{kind}_{hand}": dict(zip(BRAINCO_MOTOR_NAMES,
                                       (sat_count[(kind, hand)] / n_frames).round(4).tolist()))
            for kind in ("state", "action") for hand in ("left", "right")
        },
        "unclamped_cmd_quantiles": {},
    }
    for hand in ("left", "right"):
        qs = {}
        for m, name in enumerate(BRAINCO_MOTOR_NAMES):
            c = np.cumsum(hist[hand][m]) / hist[hand][m].sum()
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            qs[name] = {q: round(float(np.interp(q, c, centers)), 3)
                        for q in (0.01, 0.5, 0.99)}
        summary["unclamped_cmd_quantiles"][hand] = qs

    (HERE / "outputs" / "step1_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(per_episode).to_csv(HERE / "outputs" / "step1_per_episode.csv", index=False)

    print(f"converted {len(files)} episodes / {n_frames} frames -> {OUT}")
    print("\nsaturation fraction of frames (state), per motor:")
    for hand in ("left", "right"):
        row = summary["saturation_fraction"][f"state_{hand}"]
        print(f"  {hand:5s}: " + "  ".join(f"{k}={v:.1%}" for k, v in row.items()))
    print("\nunclamped command quantiles (state, 1 = full close):")
    for hand in ("left", "right"):
        for name, q in summary["unclamped_cmd_quantiles"][hand].items():
            print(f"  {hand:5s} {name:10s} q01={q[0.01]:6.2f}  q50={q[0.5]:6.2f}  q99={q[0.99]:6.2f}")


if __name__ == "__main__":
    main()
