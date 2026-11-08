#!/usr/bin/env bash
# Reusable pi0.5 LoRA SFT runner for the openpi_sft configs.
#
# Pipeline:
#   [norm-stats] -> 100-step smoke (auto-deleted) -> background HF-push watcher
#   -> full training -> wait for watcher to finish uploading the final ckpt.
#
# openpi stays pristine; configs come from maniguard.openpi_sft via the tools/
# launchers (which register them into openpi's _CONFIGS_DICT). Run from the
# openpi venv (e.g. `uv run`).
#
# Defaults: only --config is strictly required. The experiment/run name, HF push
# repo and its visibility all default from the config's policy_metadata
# (default_exp / hf_repo / hf_private), so a fresh checkout can launch with just
# --config. CLI flags override.
#
# All run artifacts go under outputs/sft_runs/<exp>/ (gitignored):
#   checkpoints/  assets/(norm stats)  logs/   -- one self-contained folder per
#   run, identical regardless of where you launch from. The pi05_base warm-start
#   download is cached in outputs/openpi_cache (shared across runs).
#
# The watcher (hf_push_watcher.py) runs in parallel and uploads each checkpoint
# to HF the moment it finalizes -- training never blocks on uploads. After the
# run, hf_push.py can be invoked manually to backfill anything missed (it skips
# whatever is already complete on HF; same de-dup as the watcher).
#
# Usage:
#   tools/openpi_sft/run_sft.sh --config <name> [options]
#
# Required:
#   --config NAME      openpi config name (registered by maniguard.openpi_sft)
#
# Options (all optional -- sensible defaults from the config):
#   --exp NAME         experiment/run name (default: config policy_metadata.default_exp)
#   --steps N          num_train_steps override
#   --batch N          batch_size override
#   --num-workers N    dataloader worker count override (pure perf knob; lower it
#                      on hosts where spawning many workers is slow, e.g. a venv
#                      on a network filesystem)
#   --keep-period N    keep_period override (checkpoint cadence)
#   --norm-stats       force a fresh compute_norm_stats (refreshes the shared cache).
#                      DEFAULT (no flag): committed norm_stats/<config>/ if present,
#                      else the shared cache outputs/norm_stats/<config>/ from an
#                      earlier run, else compute ONCE into that cache -- every later
#                      run of the config reuses it (never trains unnormalized).
#   --no-smoke         skip the 100-step smoke test
#   --smoke-only       run only the smoke test, then exit
#   --resume           pass --resume to openpi (continue from last ckpt)
#   --overwrite        pass --overwrite to openpi (wipe ckpt dir)
#   --project NAME     wandb project to report to (default: the config's
#                      project_name). Use with --push-repo to keep a throwaway /
#                      shakedown run out of the real project and model repo.
#   --push-repo REPO   HF model repo to stream checkpoints to
#                      (default: config policy_metadata.hf_repo; empty disables push)
#   --no-push          disable the HF push watcher even if the config sets hf_repo
#   --push-private     force the push repo private (default: config hf_private)
#   --poll-interval N  watcher scan interval seconds (default 30)
#
# Env (export once in your shell rc; no need to prefix on the command line):
#   OPENPI_ROOT       openpi location (default: this repo root)
#   HF_TOKEN          required if pushing to HF
#   WANDB_API_KEY     required (training logs)
set -euo pipefail

# --- CPU thread caps (CRITICAL for multi-worker dataloading) ----------------
# openpi training is a single JAX process with ONE DataLoader feeding all GPUs
# via `num_workers` worker PROCESSES. Each worker decodes video + runs
# numpy/BLAS. Left uncapped, every worker's BLAS/OMP grabs ALL cores, so the
# workers oversubscribe the CPU and thrash -- a very slow dataloader init and
# periodic multi-minute stalls every `num_workers` steps (the prefetch buffer
# drains faster than the thrashing workers can refill). This bites hardest when
# several one-card runs share a host. Pinning one math thread per worker restores
# true N-way parallel decode. JAX/XLA is unaffected (it uses its own threadpool).
: "${OMP_NUM_THREADS:=1}"; export OMP_NUM_THREADS
: "${MKL_NUM_THREADS:=1}"; export MKL_NUM_THREADS
: "${OPENBLAS_NUM_THREADS:=1}"; export OPENBLAS_NUM_THREADS
: "${NUMEXPR_NUM_THREADS:=1}"; export NUMEXPR_NUM_THREADS

CONFIG=""; EXP=""; STEPS=""; BATCH=""; KEEP_PERIOD=""; PROJECT=""; NUM_WORKERS=""
NORM_STATS=0; SMOKE=1; SMOKE_ONLY=0; RESUME=0; OVERWRITE=0
PUSH_REPO=""; NO_PUSH=0; PUSH_PRIVATE=""; POLL_INTERVAL=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --exp) EXP="$2"; shift 2;;
    --project) PROJECT="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --keep-period) KEEP_PERIOD="$2"; shift 2;;
    --norm-stats) NORM_STATS=1; shift;;
    --no-smoke) SMOKE=0; shift;;
    --smoke-only) SMOKE_ONLY=1; shift;;
    --resume) RESUME=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --push-repo) PUSH_REPO="$2"; shift 2;;
    --no-push) NO_PUSH=1; shift;;
    --push-private) PUSH_PRIVATE=1; shift;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# OPENPI_ROOT / HF_TOKEN / WANDB_API_KEY are read straight from the environment
# (export them once in your shell rc). OPENPI_ROOT falls back to ../openpi.
if [[ -z "${OPENPI_ROOT:-}" ]]; then
  if [[ -f "$REPO_ROOT/scripts/train.py" ]]; then
    OPENPI_ROOT="$REPO_ROOT"                              # fork: openpi is this repo
  else
    OPENPI_ROOT="$(cd "$REPO_ROOT/.." && pwd)/openpi"     # sibling ../openpi fallback
  fi
fi
export OPENPI_ROOT

if [[ -z "$CONFIG" ]]; then
  echo "ERROR: --config is required" >&2
  exit 1
fi
if [[ ! -d "$OPENPI_ROOT" ]]; then
  echo "ERROR: OPENPI_ROOT does not exist: $OPENPI_ROOT" >&2
  echo "       set it in your shell rc (export OPENPI_ROOT=/path/to/openpi)." >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is unset." >&2
  echo "       export WANDB_API_KEY in your shell rc (~/.bashrc / ~/.zshrc)." >&2
  exit 1
fi

meta() { python "$HERE/_config_meta.py" "$CONFIG" "$1"; }

# --- resolve defaults from the config's policy_metadata --------------------
[[ -z "$EXP" ]] && EXP="$(meta default_exp)"
if [[ -z "$EXP" ]]; then
  echo "ERROR: no --exp and config has no policy_metadata.default_exp" >&2
  exit 1
fi
if [[ "$NO_PUSH" == "0" && -z "$PUSH_REPO" ]]; then
  PUSH_REPO="$(meta hf_repo)"
fi
[[ "$NO_PUSH" == "1" ]] && PUSH_REPO=""
# Visibility: --push-private forces private; otherwise follow config hf_private.
if [[ -z "$PUSH_PRIVATE" ]]; then
  [[ "$(meta hf_private)" == "true" ]] && PUSH_PRIVATE=1 || PUSH_PRIVATE=0
fi

if [[ -n "$PUSH_REPO" && -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: pushing to $PUSH_REPO but HF_TOKEN is unset." >&2
  echo "       export HF_TOKEN in your shell rc, or pass --no-push." >&2
  exit 1
fi

# --- run layout: one self-contained, gitignored folder per run -------------
RUN_DIR="$REPO_ROOT/outputs/sft_runs/$EXP"
CKPT_BASE="$RUN_DIR/checkpoints"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
# Norm-stats are a property of the (config, dataset) pair -- not of a run -- and
# are identical however often they are recomputed. So they live in ONE cache
# shared by every run of a config: computed once, reused by every later run
# (each scaling point / repeat would otherwise burn hours recomputing constants).
NORM_CACHE="$REPO_ROOT/outputs/norm_stats"
# pi05_base (and other GCS) warm-start downloads cached here, shared across runs.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/outputs/openpi_cache}"
mkdir -p "$OPENPI_DATA_HOME"
# HF caches: force into outputs/hf (gitignored, shared across runs). Forced (not
# :-) because a host shell rc may point these at a path that is not writable
# here. The training dataset is fetched
# into HF_LEROBOT_HOME. HF_TOKEN still comes from the environment, so overriding
# HF_HOME does not affect auth. Process-local only -- never touches your rc, so
# other projects on this box are unaffected.
export HF_LEROBOT_HOME="$REPO_ROOT/outputs/hf/lerobot"
export HF_HOME="$REPO_ROOT/outputs/hf/home"
export HF_DATASETS_CACHE="$REPO_ROOT/outputs/hf/datasets"
mkdir -p "$HF_LEROBOT_HOME" "$HF_HOME" "$HF_DATASETS_CACHE"

# openpi composes checkpoints as <checkpoint_base_dir>/<config_name>/<exp_name>/.
CKPT_DIR="$CKPT_BASE/$CONFIG/$EXP"

echo "[run_sft] config=$CONFIG exp=$EXP"
echo "[run_sft] run_dir=$RUN_DIR"
echo "[run_sft] openpi_root=$OPENPI_ROOT data_home=$OPENPI_DATA_HOME"
[[ -n "$PUSH_REPO" ]] && echo "[run_sft] push -> $PUSH_REPO (private=$PUSH_PRIVATE)" \
                      || echo "[run_sft] push disabled"

# --- preflight: cv2 must be importable -------------------------------------
# Nothing here calls OpenCV, but `opencv-python` AND `opencv-python-headless` are
# both in the dependency tree and install into the SAME site-packages/cv2/ dir --
# whichever lands last wins, and uv does not order them deterministically, so the
# same lockfile can resolve either way across `uv sync` runs. The full build links
# libGL, which a bare training container does not have. It does not fail at import
# in the parent: it fails inside the torch dataloader's SPAWNED workers, where the
# re-imported main module shadows stdlib `typing` with cv2/typing/ -- i.e. minutes
# into a run, as a confusing multiprocessing traceback. Catch it in one second here.
if ! python -c "import cv2" 2>/dev/null; then
  echo "ERROR: 'import cv2' fails in this venv, so the dataloader's spawned workers will die." >&2
  echo "       Cause: the full opencv-python build owns site-packages/cv2/ and needs libGL.so.1." >&2
  echo "       Fix (no root needed; nothing here uses OpenCV, so headless is equivalent):" >&2
  echo "         uv pip install --force-reinstall --no-deps opencv-python-headless" >&2
  echo "       Re-run this after every 'uv sync' that flips it back." >&2
  exit 1
fi

# --- dataset: ensure it is fully on local disk BEFORE anything reads it -----
# First run of a family in a workspace downloads it (patiently: the Hub
# rate-limits these many-small-file datasets hard, and a throttled multi-hour
# fetch is still a success -- see fetch_dataset.py); every later run finds the
# completion marker and starts instantly with zero Hub traffic. This is what
# rules out the mid-training stall-on-Hub-fetch failure mode.
echo "[run_sft] ensuring dataset is local ..."
python "$HERE/fetch_dataset.py" --config-name "$CONFIG" --dest-root "$HF_LEROBOT_HOME" 2>&1 | tee "$LOG_DIR/fetch_dataset.log"
FETCH_RC=${PIPESTATUS[0]}
if [[ "$FETCH_RC" != "0" ]]; then
  echo "ERROR: dataset fetch failed (rc=$FETCH_RC); not starting training." >&2
  exit 1
fi

# Norm-stats resolution (unless --norm-stats forces a fresh compute):
#   1. committed norm_stats/<config>/  -> use as-is (turnkey, no compute); else
#   2. cached outputs/norm_stats/<config>/ left by an earlier run -> reuse; else
#   3. compute ONCE into that cache, so every later run of this config reuses it.
# --norm-stats forces (3) even when 1/2 exist, refreshing the cache.
# This lets a family ship CONFIG-ONLY (no baked stats): the first run computes them
# and every run after that starts instantly. openpi's compute_norm_stats is
# deterministic over the full dataset, so a computed file equals a baked one.
# Presence is tested by finding an actual norm_stats.json, not just a directory --
# an empty dir would otherwise train silently unnormalized.
#
# Step (3) is skipped entirely when the config loads its statistics from ELSEWHERE
# -- openpi's DataConfigFactory resolves them as `assets.assets_dir or assets_dirs`,
# so a config with an explicit assets_dir (e.g. a fine-tune that reuses a base
# checkpoint's bundled stats) never reads a locally
# computed file. Computing one would be dead work at best; in practice it also
# runs a multiprocessing dataset pass that a bare container need not survive.
EXTERNAL_ASSETS="$(meta external_assets_dir)"
if [[ -n "$EXTERNAL_ASSETS" && "$NORM_STATS" != "1" ]]; then
  NORM_STATS=0
  ASSETS_BASE="$NORM_CACHE"
  echo "[run_sft] norm-stats come from the config's assets_dir ($EXTERNAL_ASSETS) -> skipping local compute"
elif [[ "$NORM_STATS" == "1" ]]; then
  ASSETS_BASE="$NORM_CACHE"
  echo "[run_sft] --norm-stats: recomputing into $ASSETS_BASE/$CONFIG (refreshes the cache)"
elif [[ -n "$(find "$REPO_ROOT/norm_stats/$CONFIG" -name norm_stats.json 2>/dev/null | head -1)" ]]; then
  ASSETS_BASE="$REPO_ROOT/norm_stats"
  echo "[run_sft] using committed norm-stats: $ASSETS_BASE/$CONFIG"
elif [[ -n "$(find "$NORM_CACHE/$CONFIG" -name norm_stats.json 2>/dev/null | head -1)" ]]; then
  ASSETS_BASE="$NORM_CACHE"
  echo "[run_sft] reusing cached norm-stats: $ASSETS_BASE/$CONFIG"
else
  ASSETS_BASE="$NORM_CACHE"
  echo "[run_sft] no norm-stats for $CONFIG -> computing once into $ASSETS_BASE/$CONFIG (later runs reuse it)"
  NORM_STATS=1
fi
mkdir -p "$ASSETS_BASE"

BASE_ARGS=( --assets-base-dir "$ASSETS_BASE" --checkpoint-base-dir "$CKPT_BASE" )
# Applies to the smoke run and the full run alike, so a throwaway run can be kept
# out of the wandb project the real runs report to.
[[ -n "$PROJECT" ]] && BASE_ARGS+=( --project-name "$PROJECT" )
# norm-stats writes into the SAME assets dir training reads (--assets-base-dir),
# else training silently runs unnormalized (openpi swallows missing norm stats).
# openpi's compute_norm_stats is tyro.cli(main) -> config name is the FLAG
# --config-name (not positional, unlike train.py).
CMD_NORM=( python "$HERE/compute_norm_stats.py" --config-name "$CONFIG" --assets-base-dir "$ASSETS_BASE" )
# Latest-run-wins locally too: leftover checkpoints from a previous run of
# this exp would otherwise make openpi refuse to start. --resume continues
# them; anything else replaces them.
if [[ "$RESUME" == "0" && "$OVERWRITE" == "0" && -d "$CKPT_DIR" ]]; then
  echo "[run_sft] existing checkpoints found for $EXP and no --resume -> latest-run-wins: overwriting"
  OVERWRITE=1
fi

TRAIN_ARGS=( "$CONFIG" --exp-name "$EXP" "${BASE_ARGS[@]}" )
[[ -n "$STEPS" ]] && TRAIN_ARGS+=( --num-train-steps "$STEPS" )
[[ -n "$BATCH" ]] && TRAIN_ARGS+=( --batch-size "$BATCH" )
[[ -n "$NUM_WORKERS" ]] && TRAIN_ARGS+=( --num-workers "$NUM_WORKERS" )
[[ -n "$KEEP_PERIOD" ]] && TRAIN_ARGS+=( --keep-period "$KEEP_PERIOD" )
[[ "$RESUME" == "1" ]] && TRAIN_ARGS+=( --resume )
[[ "$OVERWRITE" == "1" ]] && TRAIN_ARGS+=( --overwrite )

if [[ "$NORM_STATS" == "1" ]]; then
  echo "[run_sft] computing norm stats for $CONFIG ..."
  "${CMD_NORM[@]}" 2>&1 | tee "$LOG_DIR/normstats.log"
  # Fail loud if nothing landed where training will look (guards the silent
  # unnormalized-training failure mode).
  if [[ -z "$(find "$ASSETS_BASE/$CONFIG" -type f 2>/dev/null | head -1)" ]]; then
    echo "ERROR: norm stats not found under $ASSETS_BASE/$CONFIG after compute." >&2
    exit 1
  fi
fi

if [[ "$SMOKE" == "1" || "$SMOKE_ONLY" == "1" ]]; then
  echo "[run_sft] smoke test (100 steps) ..."
  SMOKE_EXP="${EXP}_smoke"
  # The smoke's job is to validate memory/throughput for the run that follows,
  # so it must honor the same --batch override the full run will use.
  SMOKE_ARGS=( "$CONFIG" --exp-name "$SMOKE_EXP" "${BASE_ARGS[@]}" )
  [[ -n "$BATCH" ]] && SMOKE_ARGS+=( --batch-size "$BATCH" )
  [[ -n "$NUM_WORKERS" ]] && SMOKE_ARGS+=( --num-workers "$NUM_WORKERS" )
  python "$HERE/train.py" "${SMOKE_ARGS[@]}" \
    --num-train-steps 100 --overwrite 2>&1 | tee "$LOG_DIR/smoke.log"
  echo "[run_sft] smoke OK; removing smoke ckpt dir"
  rm -rf "$CKPT_BASE/$CONFIG/$SMOKE_EXP" || true
  [[ "$SMOKE_ONLY" == "1" ]] && { echo "[run_sft] smoke-only done"; exit 0; }
fi

# --- background HF-push watcher --------------------------------------------
WATCHER_PID=""
cleanup() { [[ -n "$WATCHER_PID" ]] && kill "$WATCHER_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ -n "$PUSH_REPO" ]]; then
  WSTEPS="$STEPS"
  [[ -z "$WSTEPS" ]] && WSTEPS="$(meta num_train_steps)"
  WATCH_ARGS=( --ckpt-dir "$CKPT_DIR" --repo "$PUSH_REPO"
               --num-train-steps "$WSTEPS" --poll-interval "$POLL_INTERVAL" )
  [[ "$PUSH_PRIVATE" == "1" ]] && WATCH_ARGS+=( --private )
  # Latest-run-wins: a fresh (non --resume) run first clears the repo's old
  # checkpoint folders. Without this a re-run is SILENTLY dropped: its files
  # have the same names/sizes as the old run's, so the already-pushed check
  # would keep the stale weights. --resume continues the same logical run, so
  # its existing uploads are kept.
  [[ "$RESUME" == "0" ]] && WATCH_ARGS+=( --fresh )
  echo "[run_sft] starting HF-push watcher -> $PUSH_REPO (num_train_steps=$WSTEPS)"
  python "$HERE/hf_push_watcher.py" "${WATCH_ARGS[@]}" 2>&1 | tee "$LOG_DIR/watcher.log" &
  WATCHER_PID=$!
fi

echo "[run_sft] full training: $CONFIG / $EXP"
python "$HERE/train.py" "${TRAIN_ARGS[@]}" 2>&1 | tee "$LOG_DIR/train.log"
echo "[run_sft] training done."

if [[ -n "$WATCHER_PID" ]]; then
  echo "[run_sft] waiting for watcher to finish uploading final checkpoint ..."
  wait "$WATCHER_PID" || true
  WATCHER_PID=""
  echo "[run_sft] watcher finished."
fi
echo "[run_sft] done. ckpts: $CKPT_DIR"
[[ -n "$PUSH_REPO" ]] && cat <<EOF
[run_sft] to verify/backfill the HF push later:
  python $HERE/hf_push.py --ckpt-dir "$CKPT_DIR" \\
    --repo "$PUSH_REPO" --num-train-steps "${STEPS:-$(meta num_train_steps)}"$([[ "$PUSH_PRIVATE" == "1" ]] && echo " --private")
EOF
