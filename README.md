# ego-pi-replication

Egocentric human recordings (Pico headset) → π₀.₅ fine-tuning for a
Unitree G1 + BrainCo Revo2 bimanual manipulation task
("put the bottle in the box"), built on a minimally patched fork of
[openpi](https://github.com/Physical-Intelligence/openpi).

## Layout

```
data_extraction/       Pico recordings -> LeRobot dataset pipeline + dashboard
                       (self-contained; see data_extraction/README.md + SPEC.md)
data/                  raw recordings + local datasets (git-ignored)
data_extraction_work/  pipeline intermediates + dashboard site (git-ignored)
scripts/               legacy LIBERO-OBJECT replication experiment scripts
                       (RunPod train/eval orchestration for the earlier project)
third_party/openpi/    openpi fork (git submodule, branch egopi-data);
                       will host the self-contained ego_* training folder
```

## Data pipeline

```bash
# raw hdf5 in data/put_bottle_in_box/ -> LeRobot dataset + sidecar
.venv/bin/python -m data_extraction.run_pipeline --jobs 4
# verification dashboard (serve it; fetch() is blocked on file://)
.venv/bin/python -m data_extraction.dashboard --site -o data_extraction_work/dashboard_site
python3 -m http.server -d data_extraction_work/dashboard_site 8123
```

One training datapoint: egocentric image + 30-dim state (per hand: flange
pose in pelvis frame as 3+6D, + 6 BrainCo motors) → H=50 actions, each the
flange pose at t+k relative to the pose at t, plus hand commands. The
dataset stores absolute per-tick poses; relative chunks are built in the
data loader, and boundary/anchor rules come from the extraction_meta.json
sidecar (see SPEC.md "Loader semantics").

## Training (in design)

Training code will live as a self-contained `ego_*` folder at the top level
of the openpi fork: its own train script/config importing openpi as a
library (π₀.₅ base, full fine-tune), no patches to openpi `src/`.

## Working with the submodule

The gitlink pins an exact fork commit. After committing in
`third_party/openpi`, stage and commit the pointer bump here. Set
`git config push.recurseSubmodules on-demand` once per clone so pushing
this repo pushes the fork first.
