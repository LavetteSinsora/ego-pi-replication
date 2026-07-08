"""Interactive dashboard for the local LeRobot dataset in data/.

Single-page dashboard combining:
  * episode video browser (prev/next/auto-advance, like video_browser.py)
  * per-episode state/action time-series charts with a playhead synced to the
    video (click a chart to seek)
  * dataset-level views: episode-length histogram, per-dimension statistics
  * a joint-map reference explaining all 38 state/action dimensions
    (Ability Hand joint order, verified from the data itself)

Videos are mp4v-encoded, which browsers cannot decode; playback reuses
video_browser.py's ffmpeg transcode cache (~/.cache/video-browser).

Usage (pyarrow is the only non-stdlib dependency; ffmpeg/ffprobe on PATH):
    uv run --no-project --with pyarrow python scripts/data_dashboard.py
    uv run --no-project --with pyarrow python scripts/data_dashboard.py --root data --port 8124 --no-open
"""

import argparse
import http.server
import itertools
import json
import pathlib
import re
import shutil
import sys
import threading
import webbrowser

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import video_browser  # noqa: E402  (transcode cache for mp4v videos)

import pyarrow.parquet as pq  # noqa: E402

# Verified against every frame: PIP = MIMIC_M * MCP + MIMIC_B holds exactly,
# matching the PSYONIC Ability Hand URDF mimic joints. This pins the joint order.
MIMIC_M, MIMIC_B = 1.05851325, 0.72349796

FINGER_JOINTS = [
    ("thumb rotation", "actuated"),
    ("thumb flexion", "actuated"),
    ("index MCP", "actuated"),
    ("index PIP", "coupled"),
    ("middle MCP", "actuated"),
    ("middle PIP", "coupled"),
    ("ring MCP", "actuated"),
    ("ring PIP", "coupled"),
    ("pinky MCP", "actuated"),
    ("pinky PIP", "coupled"),
]
WRIST_COMPONENTS = [
    ("position x", "m"),
    ("position y", "m"),
    ("position z", "m"),
    ("rotation c1x", "6d"),
    ("rotation c1y", "6d"),
    ("rotation c1z", "6d"),
    ("rotation c2x", "6d"),
    ("rotation c2y", "6d"),
    ("rotation c2z", "6d"),
]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Dataset:
    def __init__(self, root: pathlib.Path):
        self.root = root
        meta = root / "meta"
        self.info = json.loads((meta / "info.json").read_text())
        self.modality = json.loads((meta / "modality.json").read_text())
        self.fps = self.info["fps"]
        self.state_dim = self.info["features"]["observation.state"]["shape"][0]
        self.video_key = next(
            k for k, v in self.info["features"].items() if v["dtype"] == "video"
        )
        self.episodes = sorted(
            read_jsonl(meta / "episodes.jsonl"), key=lambda e: e["episode_index"]
        )
        starts = list(
            itertools.accumulate([0] + [e["length"] for e in self.episodes[:-1]])
        )
        self.summary_bytes = json.dumps(
            self._build_summary(meta, starts), separators=(",", ":")
        ).encode()
        self._ep_cache: dict[int, bytes] = {}
        self._ep_lock = threading.Lock()

    def _aggregate_stats(self, meta: pathlib.Path) -> list[dict]:
        d = self.state_dim
        mins = [float("inf")] * d
        maxs = [float("-inf")] * d
        sums = [0.0] * d
        total = 0
        for line in read_jsonl(meta / "episodes_stats.jsonl"):
            st = line["stats"]["observation.state"]
            count = st["count"]
            n = count[0] if isinstance(count, list) else count
            total += n
            for i in range(d):
                mins[i] = min(mins[i], st["min"][i])
                maxs[i] = max(maxs[i], st["max"][i])
                sums[i] += st["mean"][i] * n
        return [
            {"min": mins[i], "max": maxs[i], "mean": sums[i] / total} for i in range(d)
        ]

    def _dim_meta(self) -> list[dict]:
        names = self.info["features"]["observation.state"]["names"][0]
        dims = [None] * self.state_dim
        for group, span in sorted(
            self.modality["state"].items(), key=lambda kv: kv[1]["start"]
        ):
            hand = "left" if "left" in group else "right"
            components = FINGER_JOINTS if "fingers" in group else WRIST_COMPONENTS
            for rel, comp in enumerate(components):
                i = span["start"] + rel
                if "fingers" in group:
                    label, kind, unit = f"{hand} {comp[0]}", comp[1], "rad"
                else:
                    label, kind = f"{hand} wrist {comp[0]}", "pose"
                    unit = "m" if comp[1] == "m" else ""
                dims[i] = {
                    "i": i,
                    "name": names[i],
                    "group": group,
                    "label": label,
                    "kind": kind,
                    "unit": unit,
                }
        return dims

    def _build_summary(self, meta: pathlib.Path, starts: list[int]) -> dict:
        dims = self._dim_meta()
        for dim, st in zip(dims, self._aggregate_stats(meta)):
            dim.update(st)
        video_info = self.info["features"][self.video_key]
        return {
            "robot_type": self.info["robot_type"],
            "codebase_version": self.info["codebase_version"],
            "fps": self.fps,
            "total_episodes": self.info["total_episodes"],
            "total_frames": self.info["total_frames"],
            "video": {
                "width": video_info["info"]["video.width"],
                "height": video_info["info"]["video.height"],
                "codec": video_info["info"]["video.codec"],
            },
            "tasks": [t["task"] for t in read_jsonl(meta / "tasks.jsonl")],
            "episodes": [
                {"i": e["episode_index"], "len": e["length"], "start": s}
                for e, s in zip(self.episodes, starts)
            ],
            "dims": dims,
            "coupling": {"m": MIMIC_M, "b": MIMIC_B},
        }

    def episode_path(self, i: int) -> pathlib.Path:
        return self.root / self.info["data_path"].format(
            episode_chunk=i // self.info["chunks_size"], episode_index=i
        )

    def video_path(self, i: int) -> pathlib.Path:
        return self.root / self.info["video_path"].format(
            episode_chunk=i // self.info["chunks_size"],
            video_key=self.video_key,
            episode_index=i,
        )

    def episode_json(self, i: int) -> bytes:
        with self._ep_lock:
            if i in self._ep_cache:
                return self._ep_cache[i]
        table = pq.read_table(
            self.episode_path(i), columns=["observation.state", "action", "timestamp"]
        )
        payload = {
            "i": i,
            "len": table.num_rows,
            "timestamps": [round(t, 4) for t in table["timestamp"].to_pylist()],
            "state": [
                [round(v, 5) for v in row]
                for row in table["observation.state"].to_pylist()
            ],
            "action": [
                [round(v, 5) for v in row] for row in table["action"].to_pylist()
            ],
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        with self._ep_lock:
            if len(self._ep_cache) > 16:
                self._ep_cache.pop(next(iter(self._ep_cache)))
            self._ep_cache[i] = body
        return body


class Handler(video_browser.Handler):
    ds: Dataset
    page: bytes

    def _serve(self, head_only: bool):
        if self.path == "/":
            self._send_bytes(self.page, "text/html; charset=utf-8", head_only)
            return
        if self.path == "/api/summary":
            self._send_bytes(self.ds.summary_bytes, "application/json", head_only)
            return
        m = re.fullmatch(r"/api/episode/(\d+)", self.path)
        if m:
            i = int(m.group(1))
            if i >= len(self.ds.episodes):
                self.send_error(404)
                return
            self._send_bytes(self.ds.episode_json(i), "application/json", head_only)
            return
        m = re.fullmatch(r"/video/(\d+)", self.path)
        if m:
            i = int(m.group(1))
            if i >= len(self.ds.episodes):
                self.send_error(404)
                return
            try:
                path = video_browser.playable_path(self.ds.video_path(i))
            except Exception:
                self.send_error(500, "transcode failed (is ffmpeg installed?)")
                return
            self._send_file(path, head_only)
            return
        self.send_error(404)

    def _send_bytes(self, body: bytes, ctype: str, head_only: bool):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "data",
        help="LeRobot dataset root (contains meta/, data/, videos/)",
    )
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        print("warning: ffmpeg/ffprobe not on PATH — video playback will fail")

    ds = Dataset(args.root.resolve())
    Handler.ds = ds
    Handler.page = PAGE.encode()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(ds.episodes)} episodes from {ds.root}")
    print(f"serving at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset dashboard</title>
<style>
:root {
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10); --hand: #eceae4;
  --s1: #2a78d6; --s2: #1baf7a; --s3: #eda100; --s4: #008300;
  --s5: #4a3aa7; --s6: #e34948; --s7: #e87ba4; --s8: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255,255,255,0.10); --hand: #262624;
    --s1: #3987e5; --s2: #199e70; --s3: #c98500; --s4: #008300;
    --s5: #9085e9; --s6: #e66767; --s7: #d55181; --s8: #d95926;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1500px; margin: 0 auto; padding: 18px 20px 48px; }
h1 { font-size: 20px; margin: 0; }
h2 { font-size: 16px; margin: 28px 0 10px; }
.sub { color: var(--ink2); margin: 4px 0 14px; }
.chip { display: inline-block; font-size: 12px; color: var(--ink2);
        border: 1px solid var(--border); border-radius: 20px; padding: 1px 10px;
        margin-left: 8px; vertical-align: 2px; }
.tiles { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
.tile { background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 9px 16px 8px; min-width: 108px; }
.tile .lbl { font-size: 12px; color: var(--ink2); }
.tile .val { font-size: 21px; font-weight: 600; }
.tile .val small { font-size: 12px; font-weight: 400; color: var(--muted); }
.card { background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 12px 14px; }
button, select { background: var(--surface); color: var(--ink);
  border: 1px solid var(--axis); border-radius: 6px; padding: 5px 12px;
  font: inherit; font-size: 13px; cursor: pointer; }
button:hover { border-color: var(--muted); }
button.active { background: var(--ink); color: var(--page); border-color: var(--ink); }
select { max-width: 230px; }
label { user-select: none; font-size: 13px; color: var(--ink2); }
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
            margin-bottom: 12px; }
#pos { min-width: 6.5em; text-align: center; font-variant-numeric: tabular-nums; }
.keys { color: var(--muted); font-size: 12px; }
.spacer { flex: 1; }
.grid { display: grid; grid-template-columns: minmax(380px, 1.15fr) minmax(430px, 1fr);
        gap: 14px; align-items: start; }
@media (max-width: 1120px) { .grid { grid-template-columns: 1fr; } }
video { width: 100%; aspect-ratio: 16 / 9; background: #000; border-radius: 8px; display: block; }
.epmeta { color: var(--ink2); font-size: 12.5px; margin-top: 8px;
          font-variant-numeric: tabular-nums; }
.chart-controls { display: flex; gap: 6px; align-items: center; flex-wrap: wrap;
                  margin-bottom: 10px; }
.chart-card { margin-bottom: 12px; }
.chart-head { display: flex; justify-content: space-between; align-items: baseline;
              margin-bottom: 2px; }
.chart-title { font-weight: 600; font-size: 13px; }
.chart-unit { color: var(--muted); font-size: 12px; }
.plot { position: relative; }
.plot svg { display: block; }
.plot text { font: 11px system-ui, sans-serif; font-variant-numeric: tabular-nums;
             fill: var(--muted); }
.playhead { position: absolute; width: 2px; background: var(--ink2); opacity: .85;
            pointer-events: none; border-radius: 1px; }
.tooltip { position: absolute; pointer-events: none; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px;
  font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.28); z-index: 6;
  white-space: nowrap; display: none; }
.tooltip .tt-head { color: var(--ink2); margin-bottom: 3px;
                    font-variant-numeric: tabular-nums; }
.tt-row { display: flex; align-items: center; gap: 6px; line-height: 1.6; }
.tt-row b { font-variant-numeric: tabular-nums; }
.tt-row .nm { color: var(--ink2); }
.tt-row .av { color: var(--muted); font-variant-numeric: tabular-nums; }
.key { display: inline-block; width: 16px; height: 3px; border-radius: 2px; flex: none; }
.legend { display: flex; gap: 4px 14px; flex-wrap: wrap; align-items: center;
          margin: 2px 0 10px; }
.legend .item { display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
                font-size: 12.5px; color: var(--ink2); padding: 1px 2px; }
.legend .item.off { opacity: .35; }
.note { color: var(--muted); font-size: 12px; margin: -4px 0 10px; }
.loading { opacity: .5; transition: opacity .15s; }
details { margin-top: 10px; }
summary { cursor: pointer; color: var(--ink2); font-size: 13px; }
.tblwrap { overflow: auto; max-height: 420px; margin-top: 8px;
           border: 1px solid var(--border); border-radius: 8px; }
table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
th, td { text-align: right; padding: 4px 10px; border-bottom: 1px solid var(--grid);
         font-variant-numeric: tabular-nums; white-space: nowrap; }
th { position: sticky; top: 0; background: var(--surface); color: var(--ink2);
     font-weight: 600; z-index: 2; }
th:first-child, td:first-child, th.l, td.l { text-align: left; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
@media (max-width: 1120px) { .two-col { grid-template-columns: 1fr; } }
.prov { color: var(--ink2); font-size: 13px; }
.prov a { color: var(--s1); }
.modal-backdrop { position: fixed; inset: 0; background: rgba(0,0,0,.55);
  display: none; z-index: 20; overflow: auto; padding: 4vh 16px; }
.modal { max-width: 900px; margin: 0 auto; }
.modal h3 { margin: 4px 0 10px; font-size: 16px; }
.modal-grid { display: grid; grid-template-columns: 330px 1fr; gap: 18px; }
@media (max-width: 820px) { .modal-grid { grid-template-columns: 1fr; } }
.jlist { font-size: 13px; }
.jlist .row { display: flex; align-items: center; gap: 8px; padding: 3px 0; }
.jlist .dims { color: var(--muted); font-variant-numeric: tabular-nums;
               min-width: 62px; }
.dot { width: 12px; height: 12px; border-radius: 50%; flex: none; }
.dot.hollow { background: var(--surface) !important; border: 2.5px solid; }
.jlist h4 { margin: 12px 0 4px; font-size: 13px; }
.closebtn { float: right; }
.hand-caption { color: var(--muted); font-size: 12px; margin-top: 6px; }
.formula { font-family: ui-monospace, monospace; font-size: 12.5px;
           background: var(--page); border-radius: 6px; padding: 2px 6px; }
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>Dataset dashboard<span class="chip" id="robotChip"></span></h1>
  <div class="sub" id="taskLine"></div>
  <div class="tiles" id="tiles"></div>
</header>

<section>
  <div class="controls">
    <button id="prev">&#9664; Prev</button>
    <span id="pos"></span>
    <button id="next">Next &#9654;</button>
    <select id="sel"></select>
    <label><input type="checkbox" id="auto"> auto-advance</label>
    <label><input type="checkbox" id="loop" checked> loop</label>
    <span class="keys">keys: &#8592; / &#8594;</span>
    <span class="spacer"></span>
    <button id="jointBtn">Joint map &mdash; what do the 38 dims mean?</button>
  </div>

  <div class="grid">
    <div class="card">
      <video id="v" controls autoplay muted></video>
      <div class="epmeta" id="frameReadout"></div>
      <div class="epmeta" id="epMeta"></div>
    </div>

    <div id="chartsPane">
      <div class="chart-controls">
        <button data-group="position" class="gbtn active">Wrist position</button>
        <button data-group="rotation" class="gbtn">Wrist rotation</button>
        <button data-group="fingers" class="gbtn">Fingers</button>
        <span class="spacer"></span>
        <label><input type="checkbox" id="ovl"> action overlay</label>
        <label id="pipLabel" style="display:none"><input type="checkbox" id="pip"> coupled PIP joints</label>
      </div>
      <div class="legend" id="legend"></div>
      <div class="note" id="groupNote"></div>
      <div class="card chart-card" id="chartL"></div>
      <div class="card chart-card" id="chartR"></div>
      <details id="epTableDetails">
        <summary>Episode data as table (current signal group)</summary>
        <div class="tblwrap" id="epTable"></div>
      </details>
    </div>
  </div>
</section>

<section>
  <h2>Dataset overview</h2>
  <div class="two-col">
    <div class="card">
      <div class="chart-head"><span class="chart-title">Episode length distribution</span>
        <span class="chart-unit" id="histSub"></span></div>
      <div class="plot" id="hist"></div>
    </div>
    <div class="card prov" id="prov">
      <div class="chart-title" style="margin-bottom:6px">Provenance</div>
      This dataset is derived from Apple&rsquo;s
      <a href="https://arxiv.org/abs/2505.11709" target="_blank">EgoDex</a>
      (egocentric Apple&nbsp;Vision&nbsp;Pro recordings with ARKit 25-joint hand
      tracking; <a href="https://github.com/apple/ml-egodex" target="_blank">apple/ml-egodex</a>),
      retargeted to a Unitree humanoid with PSYONIC Ability Hands
      (robot type <b id="provRobot"></b>).
      <ul style="margin:8px 0 0; padding-left:18px">
        <li>No real robot in the loop: <b>action &asymp; state at the next frame</b>
            (check the &ldquo;action overlay&rdquo; toggle above).</li>
        <li>Finger PIP joints are mechanical mimics of their MCP joint:
            <span class="formula" id="couplingFormula"></span> holds exactly on
            every frame &mdash; this is what pins down the joint order.</li>
        <li>Wrist orientation dims are two orthonormal columns of the rotation
            matrix (&ldquo;6D rotation&rdquo;), verified on every frame.</li>
      </ul>
    </div>
  </div>

  <details>
    <summary>Per-dimension statistics &mdash; all 38 state/action dims (dataset-wide)</summary>
    <div class="tblwrap"><table id="dimTable"></table></div>
    <div class="note" style="margin-top:6px">Statistics are for observation.state;
      action statistics are near-identical (action &asymp; next-frame state).</div>
  </details>
</section>

</div>

<div class="modal-backdrop" id="modalBack">
  <div class="card modal">
    <button class="closebtn" id="modalClose">Close &times;</button>
    <h3>Joint map &mdash; the 38 state/action dimensions</h3>
    <div class="modal-grid">
      <div>
        <svg viewBox="0 0 330 320" width="100%" aria-label="Left hand joint diagram">
          <line x1="114" y1="172" x2="114" y2="100" stroke-width="26" stroke-linecap="round" style="stroke:var(--hand)"/>
          <line x1="148" y1="172" x2="148" y2="78"  stroke-width="26" stroke-linecap="round" style="stroke:var(--hand)"/>
          <line x1="182" y1="172" x2="182" y2="68"  stroke-width="26" stroke-linecap="round" style="stroke:var(--hand)"/>
          <line x1="216" y1="172" x2="216" y2="82"  stroke-width="26" stroke-linecap="round" style="stroke:var(--hand)"/>
          <line x1="232" y1="246" x2="286" y2="192" stroke-width="26" stroke-linecap="round" style="stroke:var(--hand)"/>
          <rect x="101" y="166" width="128" height="114" rx="26" style="fill:var(--hand)"/>
          <rect x="135" y="270" width="60" height="40" rx="10" style="fill:var(--hand)"/>
          <text x="114" y="86" text-anchor="middle">pinky</text>
          <text x="148" y="64" text-anchor="middle">ring</text>
          <text x="182" y="54" text-anchor="middle">middle</text>
          <text x="216" y="68" text-anchor="middle">index</text>
          <text x="300" y="182" text-anchor="middle">thumb</text>
          <!-- PIP dots (hollow = coupled mimic) -->
          <circle cx="114" cy="132" r="7" style="fill:var(--surface);stroke:var(--s6);stroke-width:2.5"/>
          <circle cx="148" cy="122" r="7" style="fill:var(--surface);stroke:var(--s5);stroke-width:2.5"/>
          <circle cx="182" cy="116" r="7" style="fill:var(--surface);stroke:var(--s4);stroke-width:2.5"/>
          <circle cx="216" cy="124" r="7" style="fill:var(--surface);stroke:var(--s3);stroke-width:2.5"/>
          <!-- MCP dots (solid = actuated) -->
          <circle cx="114" cy="169" r="8" style="fill:var(--s6);stroke:var(--surface);stroke-width:2"/>
          <circle cx="148" cy="169" r="8" style="fill:var(--s5);stroke:var(--surface);stroke-width:2"/>
          <circle cx="182" cy="169" r="8" style="fill:var(--s4);stroke:var(--surface);stroke-width:2"/>
          <circle cx="216" cy="169" r="8" style="fill:var(--s3);stroke:var(--surface);stroke-width:2"/>
          <!-- thumb -->
          <circle cx="238" cy="242" r="8" style="fill:var(--s1);stroke:var(--surface);stroke-width:2"/>
          <circle cx="266" cy="213" r="8" style="fill:var(--s2);stroke:var(--surface);stroke-width:2"/>
          <!-- wrist -->
          <circle cx="165" cy="292" r="8" style="fill:var(--ink2);stroke:var(--surface);stroke-width:2"/>
          <!-- dim labels (left hand) -->
          <text x="126" y="136">18</text>
          <text x="160" y="126">16</text>
          <text x="194" y="120">14</text>
          <text x="228" y="128">12</text>
          <text x="103" y="158">17</text>
          <text x="137" y="158">15</text>
          <text x="171" y="158">13</text>
          <text x="205" y="158">11</text>
          <text x="216" y="252">9</text>
          <text x="278" y="225">10</text>
          <text x="177" y="297">0&ndash;8</text>
        </svg>
        <div class="hand-caption">Left hand shown, palm toward you; numbers are the
        left-hand dims. The right hand mirrors it at <b>+19</b>
        (e.g. index MCP = dim 11 left / dim 30 right). Solid dot = actuated joint,
        hollow dot = coupled mimic. Colors match the Fingers chart.</div>
      </div>
      <div class="jlist">
        <h4>Wrist pose &mdash; dims 0&ndash;8 (L) / 19&ndash;27 (R)</h4>
        <div class="row"><span class="dims">0&ndash;2 / 19&ndash;21</span>
          wrist position x, y, z in meters (world/ARKit frame)</div>
        <div class="row"><span class="dims">3&ndash;8 / 22&ndash;27</span>
          orientation as 6D rotation: the first two columns of the 3&times;3
          rotation matrix (third column = their cross product)</div>
        <h4>Fingers &mdash; dims 9&ndash;18 (L) / 28&ndash;37 (R), radians</h4>
        <div class="row"><span class="dot" style="background:var(--s1)"></span>
          <span class="dims">9 / 28</span> thumb rotation (actuated; range is negative)</div>
        <div class="row"><span class="dot" style="background:var(--s2)"></span>
          <span class="dims">10 / 29</span> thumb flexion (actuated)</div>
        <div class="row"><span class="dot" style="background:var(--s3)"></span>
          <span class="dims">11 / 30</span> index MCP flexion (actuated)</div>
        <div class="row"><span class="dot hollow" style="border-color:var(--s3)"></span>
          <span class="dims">12 / 31</span> index PIP (coupled mimic)</div>
        <div class="row"><span class="dot" style="background:var(--s4)"></span>
          <span class="dims">13 / 32</span> middle MCP flexion (actuated)</div>
        <div class="row"><span class="dot hollow" style="border-color:var(--s4)"></span>
          <span class="dims">14 / 33</span> middle PIP (coupled mimic)</div>
        <div class="row"><span class="dot" style="background:var(--s5)"></span>
          <span class="dims">15 / 34</span> ring MCP flexion (actuated)</div>
        <div class="row"><span class="dot hollow" style="border-color:var(--s5)"></span>
          <span class="dims">16 / 35</span> ring PIP (coupled mimic)</div>
        <div class="row"><span class="dot" style="background:var(--s6)"></span>
          <span class="dims">17 / 36</span> pinky MCP flexion (actuated)</div>
        <div class="row"><span class="dot hollow" style="border-color:var(--s6)"></span>
          <span class="dims">18 / 37</span> pinky PIP (coupled mimic)</div>
        <h4>Coupling (verified on every frame)</h4>
        <div><span class="formula">PIP = 1.05851 &middot; MCP + 0.72350</span>
          &mdash; matches the PSYONIC Ability Hand URDF mimic joints, which is
          what pins this joint ordering down.</div>
      </div>
    </div>
  </div>
</div>

<script>
"use strict";
const $ = id => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, styles) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs || {}) e.setAttribute(k, attrs[k]);
  for (const k in styles || {}) e.style[k] = styles[k];
  return e;
}
function fmt(x, d) { return (+x).toFixed(d === undefined ? 3 : d); }
function fmtInt(x) { return x.toLocaleString("en-US"); }

let SUM = null, EP = null, cur = 0, fetchTok = 0;
let group = "position", hiddenKeys = new Set();
const CHARTS = [];
const v = $("v"), sel = $("sel");

// ---- series specs per signal group -----------------------------------------
const GROUPS = {
  position: {
    unit: "m", base: { L: 0, R: 19 }, note: "",
    series: [
      { k: "px", name: "x", color: "var(--s1)", off: 0 },
      { k: "py", name: "y", color: "var(--s2)", off: 1 },
      { k: "pz", name: "z", color: "var(--s3)", off: 2 },
    ],
  },
  rotation: {
    unit: "6D rotation (two columns of R)", base: { L: 0, R: 19 },
    note: "Dims 3–8 per hand: first two columns of the wrist rotation matrix (orthonormal by construction).",
    series: [
      { k: "r0", name: "c1·x", color: "var(--s1)", off: 3 },
      { k: "r1", name: "c1·y", color: "var(--s2)", off: 4 },
      { k: "r2", name: "c1·z", color: "var(--s3)", off: 5 },
      { k: "r3", name: "c2·x", color: "var(--s4)", off: 6 },
      { k: "r4", name: "c2·y", color: "var(--s5)", off: 7 },
      { k: "r5", name: "c2·z", color: "var(--s6)", off: 8 },
    ],
  },
  fingers: {
    unit: "rad", base: { L: 9, R: 28 },
    note: "PIP joints are mechanical mimics of MCP (PIP = 1.059·MCP + 0.723) — enable “coupled PIP joints” to draw them (dashed).",
    series: [
      { k: "f0", name: "thumb rot", color: "var(--s1)", off: 0 },
      { k: "f1", name: "thumb flex", color: "var(--s2)", off: 1 },
      { k: "f2", name: "index", color: "var(--s3)", off: 2 },
      { k: "f3", name: "index PIP", color: "var(--s3)", off: 3, dash: "6 4", pip: true },
      { k: "f4", name: "middle", color: "var(--s4)", off: 4 },
      { k: "f5", name: "middle PIP", color: "var(--s4)", off: 5, dash: "6 4", pip: true },
      { k: "f6", name: "ring", color: "var(--s5)", off: 6 },
      { k: "f7", name: "ring PIP", color: "var(--s5)", off: 7, dash: "6 4", pip: true },
      { k: "f8", name: "pinky", color: "var(--s6)", off: 8 },
      { k: "f9", name: "pinky PIP", color: "var(--s6)", off: 9, dash: "6 4", pip: true },
    ],
  },
};
function activeSpecs() {
  const g = GROUPS[group];
  const showPIP = $("pip").checked;
  return g.series.filter(s => !s.pip || (group === "fingers" && showPIP));
}

// ---- axis ticks -------------------------------------------------------------
function niceTicks(lo, hi, n) {
  if (!(hi > lo)) { lo -= 1; hi += 1; }
  const span = hi - lo, raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  let step = 10 * mag;
  for (const m of [1, 2, 5, 10]) if (raw <= m * mag) { step = m * mag; break; }
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-6; t += step)
    ticks.push(+t.toFixed(10));
  return ticks;
}

// ---- line chart -------------------------------------------------------------
function renderLineChart(card, opts) {
  card.textContent = "";
  const head = document.createElement("div");
  head.className = "chart-head";
  const t1 = document.createElement("span"); t1.className = "chart-title";
  t1.textContent = opts.title;
  const t2 = document.createElement("span"); t2.className = "chart-unit";
  t2.textContent = opts.unit;
  head.append(t1, t2);
  const plot = document.createElement("div"); plot.className = "plot";
  card.append(head, plot);

  const W = Math.max(card.clientWidth - 28, 320), H = 185;
  const ml = 46, mr = 12, mt = 8, mb = 22;
  const pw = W - ml - mr, ph = H - mt - mb;
  const xs = opts.xs, n = xs.length, T = xs[n - 1] || 1;
  const X = t => ml + (t / T) * pw;

  let lo = Infinity, hi = -Infinity;
  for (const s of opts.series) {
    for (const arr of [s.values, s.action]) {
      if (!arr) continue;
      for (const y of arr) { if (y < lo) lo = y; if (y > hi) hi = y; }
    }
  }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.07 || 0.1;
  lo -= pad; hi += pad;
  const Y = y => mt + (1 - (y - lo) / (hi - lo)) * ph;

  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  for (const ty of niceTicks(lo, hi, 4)) {
    svg.append(svgEl("line", { x1: ml, x2: ml + pw, y1: Y(ty), y2: Y(ty), "stroke-width": 1 },
      { stroke: "var(--grid)" }));
    const txt = svgEl("text", { x: ml - 7, y: Y(ty) + 3.5, "text-anchor": "end" });
    txt.textContent = Math.abs(ty) < 1e-9 ? "0" : +ty.toFixed(6) + "";
    svg.append(txt);
  }
  for (const tx of niceTicks(0, T, 6)) {
    const txt = svgEl("text", { x: X(tx), y: H - 6, "text-anchor": "middle" });
    txt.textContent = tx + "s";
    svg.append(txt);
  }
  svg.append(svgEl("line", { x1: ml, x2: ml + pw, y1: mt + ph, y2: mt + ph, "stroke-width": 1 },
    { stroke: "var(--axis)" }));

  function pathFor(arr, extra, styles) {
    let d = "";
    for (let i = 0; i < n; i++)
      d += (i ? "L" : "M") + X(xs[i]).toFixed(1) + "," + Y(arr[i]).toFixed(1);
    const p = svgEl("path", Object.assign({
      d, fill: "none", "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }, extra), styles);
    return p;
  }
  for (const s of opts.series) {
    if (s.action) svg.append(pathFor(s.action,
      { "stroke-width": 1.5, "stroke-dasharray": "2 3", opacity: .65 }, { stroke: s.color }));
    svg.append(pathFor(s.values,
      s.dash ? { "stroke-dasharray": s.dash } : {}, { stroke: s.color }));
  }

  const cross = svgEl("line", { y1: mt, y2: mt + ph, "stroke-width": 1, visibility: "hidden" },
    { stroke: "var(--muted)" });
  svg.append(cross);
  const hit = svgEl("rect", { x: ml, y: mt, width: pw, height: ph, fill: "transparent" },
    { cursor: "crosshair" });
  svg.append(hit);

  const playhead = document.createElement("div");
  playhead.className = "playhead";
  playhead.style.top = mt + "px";
  playhead.style.height = ph + "px";
  playhead.style.left = ml + "px";
  const tip = document.createElement("div"); tip.className = "tooltip";
  plot.append(svg, playhead, tip);

  function frameAt(clientX) {
    const bb = svg.getBoundingClientRect();
    const t = ((clientX - bb.left - ml) / pw) * T;
    return Math.max(0, Math.min(n - 1, Math.round(t * SUM.fps)));
  }
  hit.addEventListener("pointermove", e => {
    const i = frameAt(e.clientX), sx = X(xs[i]);
    cross.setAttribute("x1", sx); cross.setAttribute("x2", sx);
    cross.setAttribute("visibility", "visible");
    tip.textContent = "";
    const h = document.createElement("div"); h.className = "tt-head";
    h.textContent = "frame " + i + " · " + fmt(xs[i], 2) + " s";
    tip.append(h);
    for (const s of opts.series) {
      const row = document.createElement("div"); row.className = "tt-row";
      const key = document.createElement("span"); key.className = "key";
      key.style.background = s.dash
        ? `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)`
        : s.color;
      const val = document.createElement("b"); val.textContent = fmt(s.values[i]);
      row.append(key, val);
      if (s.action) {
        const av = document.createElement("span"); av.className = "av";
        av.textContent = "a: " + fmt(s.action[i]);
        row.append(av);
      }
      const nm = document.createElement("span"); nm.className = "nm";
      nm.textContent = s.name;
      row.append(nm);
      tip.append(row);
    }
    tip.style.display = "block";
    const tw = tip.offsetWidth;
    tip.style.left = (sx + 14 + tw > W ? sx - 14 - tw : sx + 14) + "px";
    tip.style.top = mt + 4 + "px";
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    tip.style.display = "none";
  });
  hit.addEventListener("click", e => {
    v.currentTime = frameAt(e.clientX) / SUM.fps;
  });

  CHARTS.push({ playhead, toX: t => X(Math.min(t, T)) });
}

// ---- charts pane ------------------------------------------------------------
function renderCharts() {
  if (!EP) return;
  CHARTS.length = 0;
  const g = GROUPS[group];
  const specs = activeSpecs();
  const overlay = $("ovl").checked;
  for (const hand of ["L", "R"]) {
    const base = g.base[hand];
    const series = specs.filter(s => !hiddenKeys.has(s.k)).map(s => ({
      name: s.name, color: s.color, dash: s.dash,
      values: EP.state.map(r => r[base + s.off]),
      action: overlay ? EP.action.map(r => r[base + s.off]) : null,
    }));
    renderLineChart($("chart" + hand), {
      title: hand === "L" ? "Left hand" : "Right hand",
      unit: g.unit, xs: EP.timestamps, series,
    });
  }
  renderLegend(specs);
  $("groupNote").textContent = g.note;
  $("pipLabel").style.display = group === "fingers" ? "" : "none";
  syncPlayhead();
  if ($("epTableDetails").open) renderEpTable();
}

function renderLegend(specs) {
  const leg = $("legend");
  leg.textContent = "";
  for (const s of specs) {
    const item = document.createElement("span");
    item.className = "item" + (hiddenKeys.has(s.k) ? " off" : "");
    const key = document.createElement("span"); key.className = "key";
    key.style.background = s.dash
      ? `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)`
      : s.color;
    item.append(key, document.createTextNode(s.name));
    item.onclick = () => {
      hiddenKeys.has(s.k) ? hiddenKeys.delete(s.k) : hiddenKeys.add(s.k);
      renderCharts();
    };
    leg.append(item);
  }
}

function renderEpTable() {
  const g = GROUPS[group];
  const specs = activeSpecs().filter(s => !hiddenKeys.has(s.k));
  const wrap = $("epTable");
  wrap.textContent = "";
  const table = document.createElement("table");
  const thead = document.createElement("tr");
  for (const h of ["frame", "t (s)"]) {
    const th = document.createElement("th"); th.textContent = h; thead.append(th);
  }
  for (const hand of ["L", "R"]) for (const s of specs) {
    const th = document.createElement("th");
    th.textContent = hand + " · " + s.name;
    thead.append(th);
  }
  table.append(thead);
  const frag = document.createDocumentFragment();
  for (let i = 0; i < EP.len; i++) {
    const tr = document.createElement("tr");
    const c0 = document.createElement("td"); c0.textContent = i;
    const c1 = document.createElement("td"); c1.textContent = fmt(EP.timestamps[i], 2);
    tr.append(c0, c1);
    for (const hand of ["L", "R"]) for (const s of specs) {
      const td = document.createElement("td");
      td.textContent = fmt(EP.state[i][g.base[hand] + s.off], 4);
      tr.append(td);
    }
    frag.append(tr);
  }
  table.append(frag);
  wrap.append(table);
}

// ---- playhead / frame readout ----------------------------------------------
function syncPlayhead() {
  if (!EP) return;
  const t = v.currentTime;
  const i = Math.max(0, Math.min(EP.len - 1, Math.round(t * SUM.fps)));
  for (const c of CHARTS) c.playhead.style.left = c.toX(t).toFixed(1) + "px";
  const meta = SUM.episodes[cur];
  $("frameReadout").textContent =
    "frame " + i + " / " + (EP.len - 1) + " · t = " + fmt(t, 2) +
    " s · global index " + fmtInt(meta.start + i);
}
function rafLoop() {
  syncPlayhead();
  if (!v.paused && !v.ended) requestAnimationFrame(rafLoop);
}
v.addEventListener("play", rafLoop);
for (const ev of ["pause", "seeked", "timeupdate", "loadedmetadata"])
  v.addEventListener(ev, syncPlayhead);
v.addEventListener("ended", () => { if ($("auto").checked) show(cur + 1); });

// ---- episode switching -------------------------------------------------------
async function show(i) {
  const N = SUM.episodes.length;
  cur = ((i % N) + N) % N;
  v.src = "/video/" + cur;
  v.loop = $("loop").checked;
  sel.value = cur;
  $("pos").textContent = (cur + 1) + " / " + N;
  fetch("/video/" + ((cur + 1) % N), { method: "HEAD" });  // warm next transcode
  const tok = ++fetchTok;
  $("chartsPane").classList.add("loading");
  const ep = await (await fetch("/api/episode/" + cur)).json();
  if (tok !== fetchTok) return;
  EP = ep;
  $("chartsPane").classList.remove("loading");
  let sum = 0;
  for (let f = 0; f < EP.len; f++)
    for (let d = 0; d < EP.state[f].length; d++)
      sum += Math.abs(EP.action[f][d] - EP.state[f][d]);
  const meta = SUM.episodes[cur];
  $("epMeta").textContent =
    EP.len + " frames · " + fmt(EP.len / SUM.fps, 1) + " s · global index " +
    fmtInt(meta.start) + "–" + fmtInt(meta.start + EP.len - 1) +
    " · mean |action − state| = " + fmt(sum / (EP.len * EP.state[0].length), 4);
  renderCharts();
}

// ---- dataset-level charts -----------------------------------------------------
function renderHist() {
  const secs = SUM.episodes.map(e => e.len / SUM.fps);
  const lo = Math.floor(Math.min.apply(null, secs));
  const hi = Math.ceil(Math.max.apply(null, secs));
  const nb = hi - lo;
  const bins = new Array(nb).fill(0);
  for (const s of secs) bins[Math.min(nb - 1, Math.floor(s - lo))]++;
  const sorted = secs.slice().sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  $("histSub").textContent = "min " + fmt(sorted[0], 1) + " s · median " +
    fmt(med, 1) + " s · max " + fmt(sorted[sorted.length - 1], 1) + " s";

  const plot = $("hist");
  plot.textContent = "";
  const W = Math.max(plot.clientWidth, 320), H = 190;
  const ml = 40, mr = 8, mt = 10, mb = 24;
  const pw = W - ml - mr, ph = H - mt - mb;
  const ymax = Math.max.apply(null, bins);
  const yticks = niceTicks(0, ymax, 4).filter(t => Number.isInteger(t));
  const Y = c => mt + (1 - c / (yticks[yticks.length - 1] || ymax)) * ph;
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  for (const ty of yticks) {
    svg.append(svgEl("line", { x1: ml, x2: ml + pw, y1: Y(ty), y2: Y(ty), "stroke-width": 1 },
      { stroke: "var(--grid)" }));
    const txt = svgEl("text", { x: ml - 6, y: Y(ty) + 3.5, "text-anchor": "end" });
    txt.textContent = ty;
    svg.append(txt);
  }
  const tip = document.createElement("div"); tip.className = "tooltip";
  const slot = pw / nb;
  for (let b = 0; b < nb; b++) {
    const bw = Math.min(24, slot - 2);
    const x = ml + b * slot + (slot - bw) / 2;
    const y = Y(bins[b]), h = mt + ph - y;
    const r = Math.min(4, bw / 2, h);
    if (bins[b] > 0) {
      const d = `M${x},${mt + ph} L${x},${y + r} Q${x},${y} ${x + r},${y} ` +
        `L${x + bw - r},${y} Q${x + bw},${y} ${x + bw},${y + r} L${x + bw},${mt + ph} Z`;
      svg.append(svgEl("path", { d }, { fill: "var(--s1)" }));
    }
    const hit = svgEl("rect", { x: ml + b * slot, y: mt, width: slot, height: ph,
      fill: "transparent" });
    hit.addEventListener("pointermove", e => {
      tip.textContent = "";
      const bEl = document.createElement("b");
      bEl.textContent = bins[b] + " episode" + (bins[b] === 1 ? "" : "s");
      const nm = document.createElement("span"); nm.className = "nm";
      nm.textContent = " · " + (lo + b) + "–" + (lo + b + 1) + " s";
      tip.append(bEl, nm);
      tip.style.display = "block";
      const bb = svg.getBoundingClientRect();
      const px = e.clientX - bb.left;
      tip.style.left = Math.min(px + 12, W - tip.offsetWidth - 4) + "px";
      tip.style.top = (e.clientY - bb.top - 34) + "px";
    });
    hit.addEventListener("pointerleave", () => { tip.style.display = "none"; });
    svg.append(hit);
  }
  svg.append(svgEl("line", { x1: ml, x2: ml + pw, y1: mt + ph, y2: mt + ph,
    "stroke-width": 1 }, { stroke: "var(--axis)" }));
  for (let b = 0; b <= nb; b += (nb > 12 ? 2 : 1)) {
    const txt = svgEl("text", { x: ml + b * slot, y: H - 8, "text-anchor": "middle" });
    txt.textContent = (lo + b) + "s";
    svg.append(txt);
  }
  plot.append(svg, tip);
}

function renderDimTable() {
  const table = $("dimTable");
  table.textContent = "";
  const thead = document.createElement("tr");
  const cols = ["dim", "meaning", "raw name", "group", "unit", "min", "mean", "max"];
  cols.forEach((c, j) => {
    const th = document.createElement("th");
    if (j >= 1 && j <= 4) th.className = "l";
    th.textContent = c;
    thead.append(th);
  });
  table.append(thead);
  for (const d of SUM.dims) {
    const tr = document.createElement("tr");
    const cells = [d.i, d.label + (d.kind === "coupled" ? " (coupled)" : ""),
      d.name, d.group, d.unit || "–", fmt(d.min), fmt(d.mean), fmt(d.max)];
    cells.forEach((cval, j) => {
      const td = document.createElement("td");
      if (j >= 1 && j <= 4) td.className = "l";
      td.textContent = cval;
      tr.append(td);
    });
    table.append(tr);
  }
}

// ---- header tiles --------------------------------------------------------------
function renderHeader() {
  $("robotChip").textContent = SUM.robot_type + " · LeRobot " + SUM.codebase_version;
  $("taskLine").textContent = "Task: “" + SUM.tasks.join("”, “") + "”";
  $("provRobot").textContent = SUM.robot_type;
  $("couplingFormula").textContent =
    "PIP = " + SUM.coupling.m.toFixed(5) + " · MCP + " + SUM.coupling.b.toFixed(5);
  const mins = Math.round(SUM.total_frames / SUM.fps / 60);
  const tiles = [
    ["Episodes", fmtInt(SUM.total_episodes), ""],
    ["Frames", fmtInt(SUM.total_frames), ""],
    ["Rate", SUM.fps, "fps"],
    ["Total footage", Math.floor(mins / 60) + " h " + (mins % 60) + " m", ""],
    ["State / action dims", SUM.dims.length, ""],
    ["Video", SUM.video.width + "×" + SUM.video.height, SUM.video.codec],
  ];
  const box = $("tiles");
  for (const [lbl, val, small] of tiles) {
    const t = document.createElement("div"); t.className = "tile";
    const l = document.createElement("div"); l.className = "lbl"; l.textContent = lbl;
    const vv = document.createElement("div"); vv.className = "val";
    vv.textContent = val;
    if (small) {
      const sm = document.createElement("small"); sm.textContent = " " + small;
      vv.append(sm);
    }
    t.append(l, vv);
    box.append(t);
  }
}

// ---- wiring ---------------------------------------------------------------------
$("prev").onclick = () => show(cur - 1);
$("next").onclick = () => show(cur + 1);
sel.onchange = () => show(+sel.value);
$("loop").onchange = e => { v.loop = e.target.checked; };
$("ovl").onchange = renderCharts;
$("pip").onchange = renderCharts;
for (const b of document.querySelectorAll(".gbtn")) {
  b.onclick = () => {
    group = b.dataset.group;
    hiddenKeys.clear();
    for (const o of document.querySelectorAll(".gbtn")) o.classList.toggle("active", o === b);
    renderCharts();
  };
}
document.addEventListener("keydown", e => {
  if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  if (e.key === "ArrowLeft") show(cur - 1);
  if (e.key === "ArrowRight") show(cur + 1);
  if (e.key === "Escape") $("modalBack").style.display = "none";
});
$("jointBtn").onclick = () => { $("modalBack").style.display = "block"; };
$("modalClose").onclick = () => { $("modalBack").style.display = "none"; };
$("modalBack").addEventListener("click", e => {
  if (e.target === $("modalBack")) $("modalBack").style.display = "none";
});
$("epTableDetails").addEventListener("toggle", () => {
  if ($("epTableDetails").open) renderEpTable();
});
let resizeQueued = false;
new ResizeObserver(() => {
  if (resizeQueued) return;
  resizeQueued = true;
  requestAnimationFrame(() => { resizeQueued = false; renderCharts(); renderHist(); });
}).observe(document.querySelector(".wrap"));

(async function init() {
  SUM = await (await fetch("/api/summary")).json();
  renderHeader();
  for (const e of SUM.episodes)
    sel.add(new Option("episode_" + String(e.i).padStart(6, "0") + " · " +
      fmt(e.len / SUM.fps, 1) + "s", e.i));
  renderHist();
  renderDimTable();
  show(0);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
