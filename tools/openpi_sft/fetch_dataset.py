#!/usr/bin/env python3
"""Ensure a config's training dataset exists locally; download it patiently if not.

Called by ``run_sft.sh`` before norm-stats / training so the dataset is always
fully on local disk before the run starts -- a mid-training Hub fetch is exactly
the stall-every-few-steps failure mode this repo works hard to rule out.

Resolution (idempotent, one-time per family per workspace):
  1. ``<dest-root>/<repo_id>/.maniguard_fetch_complete`` marker present -> done,
     no Hub traffic at all (subsequent runs of the family start instantly).
  2. No marker -> ``snapshot_download`` into that directory. Already-complete
     files are skipped (resume), so a partially-downloaded dataset continues
     rather than restarting.
  3. On success (verified by ``meta/info.json``), write the marker.

RATE-LIMIT TOLERANCE: these datasets are ~13k small video files and the Hub
429-rate-limits that access pattern. Each attempt downloads at full speed for
as long as the Hub allows (authenticated via the ambient HF_TOKEN, which gets
friendlier limits); when throttled, the loop backs off briefly and RESUMES --
already-downloaded files are never re-fetched. The retry budget (default 48
attempts, backoff capped at 5 min) bounds pure waiting to a few hours on top of
the actual transfer time; if the Hub is still refusing after that, something is
genuinely wrong and failing loud beats silently waiting for days. The fetch is
one-time per family per workspace (completion marker).

``HF_HUB_OFFLINE=1`` is an internal testing shortcut only (datasets pre-staged
by hand); the canonical path for any real deployment is
this Hub download. Offline mode accepts a local copy iff its ``meta/info.json``
exists, else exits non-zero telling you to stage it.

Usage:
    python tools/openpi_sft/fetch_dataset.py --config-name <cfg> --dest-root $HF_LEROBOT_HOME
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import maniguard.openpi_sft  # noqa: E402,F401  -- registers the TrainConfigs

MARKER = ".maniguard_fetch_complete"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--dest-root", required=True, help="HF_LEROBOT_HOME (LeRobot resolves <root>/<repo_id>)")
    ap.add_argument("--max-attempts", type=int, default=48)
    args = ap.parse_args()

    from openpi.training.config import get_config

    cfg = get_config(args.config_name)
    repo_id = cfg.data.repo_id
    if not repo_id:
        sys.exit(f"ERROR: config {args.config_name} has no data.repo_id")
    dest = pathlib.Path(args.dest_root).expanduser() / repo_id
    marker = dest / MARKER
    info = dest / "meta" / "info.json"

    if marker.is_file():
        print(f"[fetch] {repo_id}: already complete ({marker.name} present); no Hub access.", flush=True)
        return

    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    if offline:
        if info.is_file():
            print(f"[fetch] {repo_id}: HF_HUB_OFFLINE=1 and local copy present; accepting as-is.", flush=True)
            marker.touch()
            return
        sys.exit(
            f"ERROR: HF_HUB_OFFLINE=1 but no local dataset at {dest} -- stage it "
            f"(zip/rsync) or unset HF_HUB_OFFLINE to allow downloading."
        )

    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {repo_id} -> {dest}", flush=True)
    for attempt in range(1, args.max_attempts + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(dest),
                max_workers=4,  # modest parallelism: fast enough, far less 429-prone
            )
            if not info.is_file():
                raise RuntimeError("snapshot completed but meta/info.json is missing")
            marker.touch()
            print(f"[fetch] {repo_id}: complete (attempt {attempt}).", flush=True)
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 -- every failure here is retry-worthy (throttle/transient)
            wait = min(30 * attempt, 300)  # 30s -> 5min cap; back off, then resume at full speed
            print(
                f"[fetch] attempt {attempt}/{args.max_attempts} failed "
                f"({type(e).__name__}: {str(e)[:120]}); resuming in {wait}s "
                f"(already-downloaded files are kept)",
                flush=True,
            )
            time.sleep(wait)
    sys.exit(f"ERROR: dataset {repo_id} still incomplete after {args.max_attempts} attempts.")


if __name__ == "__main__":
    main()
