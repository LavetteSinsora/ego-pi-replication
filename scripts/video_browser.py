"""Browse a directory of videos in the browser with next/prev controls.

VS Code's media preview cannot play mp4v (MPEG-4 Part 2) videos such as the
ones LeRobot datasets encoded with OpenCV's 'mp4v' fourcc, and it has no way
to step through a directory. This script serves a local player page instead:
videos whose codec a browser cannot decode are transcoded to H.264 with
ffmpeg on first view and cached under ~/.cache/video-browser.

Usage:
    python3 scripts/video_browser.py data/videos/chunk-000/observation.images.cam_high
    python3 scripts/video_browser.py data/videos --port 8123 --no-open

Then use the Prev/Next buttons or the left/right arrow keys. Requires only
the Python standard library plus ffmpeg/ffprobe on PATH.
"""

import argparse
import functools
import hashlib
import http.server
import json
import pathlib
import re
import subprocess
import threading
import webbrowser

BROWSER_SAFE_CODECS = {"h264", "vp8", "vp9", "av1"}
CACHE_DIR = pathlib.Path.home() / ".cache" / "video-browser"

_transcode_locks: dict[str, threading.Lock] = {}
_transcode_locks_guard = threading.Lock()

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Video Browser</title>
<style>
  body { margin: 0; background: #111; color: #ddd; font: 14px system-ui, sans-serif;
         display: flex; flex-direction: column; height: 100vh; }
  video { flex: 1; min-height: 0; width: 100%; background: #000; }
  #bar { display: flex; gap: 8px; align-items: center; padding: 8px 12px; flex-wrap: wrap; }
  button { background: #333; color: #ddd; border: 1px solid #555; border-radius: 4px;
           padding: 6px 16px; font-size: 14px; cursor: pointer; }
  button:hover { background: #444; }
  select { background: #222; color: #ddd; border: 1px solid #555; border-radius: 4px;
           padding: 5px; max-width: 45vw; }
  label { user-select: none; }
  #pos { min-width: 7em; text-align: center; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<video id="v" controls autoplay></video>
<div id="bar">
  <button id="prev">&#9664; Prev</button>
  <span id="pos"></span>
  <button id="next">Next &#9654;</button>
  <select id="sel"></select>
  <label><input type="checkbox" id="auto"> auto-advance</label>
  <label><input type="checkbox" id="loop" checked> loop</label>
  <span style="opacity:.6">keys: &#8592; / &#8594;</span>
</div>
<script>
const files = __FILES__;
const v = document.getElementById('v');
const sel = document.getElementById('sel');
files.forEach((f, i) => sel.add(new Option(f, i)));
let cur = 0;

function show(i) {
  cur = (i + files.length) % files.length;
  v.src = '/video/' + cur;
  v.loop = document.getElementById('loop').checked;
  sel.value = cur;
  document.getElementById('pos').textContent = (cur + 1) + ' / ' + files.length;
  document.title = files[cur];
  // Warm the transcode cache for the next video while this one plays.
  fetch('/video/' + ((cur + 1) % files.length), {method: 'HEAD'});
}

document.getElementById('prev').onclick = () => show(cur - 1);
document.getElementById('next').onclick = () => show(cur + 1);
sel.onchange = () => show(+sel.value);
document.getElementById('loop').onchange = e => { v.loop = e.target.checked; };
v.onended = () => { if (document.getElementById('auto').checked) show(cur + 1); };
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') show(cur - 1);
  if (e.key === 'ArrowRight') show(cur + 1);
});
show(0);
</script>
</body>
</html>
"""


def probe_codec(path: pathlib.Path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def playable_path(path: pathlib.Path) -> pathlib.Path:
    """Return a browser-playable file for `path`, transcoding to cache if needed."""
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()
    cached = CACHE_DIR / f"{key}.mp4"
    with _transcode_locks_guard:
        lock = _transcode_locks.setdefault(key, threading.Lock())
    with lock:
        if cached.exists():
            return cached
        if probe_codec(path) in BROWSER_SAFE_CODECS:
            return path
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".tmp.mp4")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast",
             "-c:a", "aac", "-movflags", "+faststart", str(tmp)],
            check=True,
        )
        tmp.rename(cached)
        return cached


class Handler(http.server.BaseHTTPRequestHandler):
    files: list[pathlib.Path] = []
    root: pathlib.Path

    def log_message(self, *args):
        pass

    def do_GET(self):
        self._serve(head_only=False)

    def do_HEAD(self):
        self._serve(head_only=True)

    def _serve(self, head_only: bool):
        if self.path == "/":
            names = [str(p.relative_to(self.root)) for p in self.files]
            body = PAGE.replace("__FILES__", json.dumps(names)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)
            return

        m = re.fullmatch(r"/video/(\d+)", self.path)
        if not m or int(m.group(1)) >= len(self.files):
            self.send_error(404)
            return
        try:
            path = playable_path(self.files[int(m.group(1))])
        except subprocess.CalledProcessError:
            self.send_error(500, "transcode failed")
            return
        self._send_file(path, head_only)

    def _send_file(self, path: pathlib.Path, head_only: bool):
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header or "")
        if m and (m.group(1) or m.group(2)):
            start = int(m.group(1)) if m.group(1) else size - int(m.group(2))
            end = min(int(m.group(2)), size - 1) if m.group(1) and m.group(2) else end
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        if head_only:
            return
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory", type=pathlib.Path, help="directory to scan for videos (recursive)")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    args = parser.parse_args()

    root = args.directory.resolve()
    files = sorted(p for ext in ("mp4", "avi", "mov", "mkv", "webm") for p in root.rglob(f"*.{ext}"))
    if not files:
        raise SystemExit(f"no videos found under {root}")

    handler = functools.partial(Handler)
    Handler.files = files
    Handler.root = root
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"{len(files)} videos from {root}")
    print(f"serving at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
