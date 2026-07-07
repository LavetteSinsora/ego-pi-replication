# ego-pi-replication

Replication of π0.5 fine-tuning on LIBERO-OBJECT, built on a minimally patched
fork of [openpi](https://github.com/Physical-Intelligence/openpi). This repo
owns the experiment orchestration (RunPod training/eval scripts, analysis
tools, local dataset); the fork at `third_party/openpi` owns everything
openpi's machinery must resolve itself (the `action_dim_actual` loss-masking
patch, the LIBERO-OBJECT train configs, the evdev stub package, committed norm
stats, `uv.lock`).

## Layout

```
scripts/               experiment scripts (this repo)
  remote_setup.sh        one-time RunPod pod setup
  remote_preflight.sh    fail-fast environment checks
  remote_run.sh          full unattended train+eval experiment
  remote_env.sh          shared env, sourced by the above
  benchmark.py           LIBERO-OBJECT eval (10 tasks x N trials)
  benchmark_language.py  instruction-following probe
  extract_trainable.py   archive trainable weights from a checkpoint
  reconstruct_trainable.py  rebuild eval-ready params from base + npz
  merge_eval_shards.py   merge sharded eval results
  wandb_notify.py        alerts + artifact uploads
  video_browser.py       local dataset video viewer
data/                  local LeRobot dataset (git-ignored)
third_party/openpi/    openpi fork (git submodule, branch
                       libero_replication_modifications); venv lives here
```

## Setup

```bash
git clone --recurse-submodules https://github.com/LavetteSinsora/ego-pi-replication
cd ego-pi-replication
git config push.recurseSubmodules on-demand   # once per clone — see below
```

On a fresh RunPod pod (PyTorch/CUDA template), then:

```bash
export DATASET_REPO=<hf-user>/libero_object_summed_subsampling
bash scripts/remote_setup.sh
tmux new -s train
bash scripts/remote_run.sh
```

`remote_setup.sh` initializes submodules itself, so a plain `git clone` also
works. The Python environment is managed by uv *inside the fork*
(`cd third_party/openpi && uv sync`); there is deliberately no pyproject at
this level — scripts run with `third_party/openpi/.venv/bin/python`.

## Working with the submodule

The gitlink here pins an exact fork commit. After committing in
`third_party/openpi`, this repo shows `modified: third_party/openpi
(new commits)` — stage and commit that pointer bump like any change.

`push.recurseSubmodules on-demand` (set above) makes pushing this repo push
the fork first. Without it, pushing a pointer to an unpushed fork commit
breaks every fresh clone. The config is per-clone, so set it after each clone.

Rule of thumb for new code: if openpi's own machinery must resolve it
(configs in the registry, workspace packages, assets), it goes in the fork;
everything else goes in `scripts/` here.
