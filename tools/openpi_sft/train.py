#!/usr/bin/env python3
"""Launcher: register ManiGuard SFT configs, then run openpi's ``scripts/train.py``.

This wrapper:
  1. puts the repo root on ``sys.path`` and imports ``maniguard.openpi_sft``,
     which registers the pi0.5 SFT TrainConfigs into openpi's ``_CONFIGS_DICT``;
  2. delegates to openpi's ``scripts/train.py`` via ``runpy`` with argv intact,
     so every openpi train.py flag works unchanged (the config registry is read
     at ``cli()`` call time, after our registration).

Usage (run with openpi's venv, e.g. ``uv run python``):
    uv run python tools/openpi_sft/train.py \
        pi05-base_datagen_jar_unsafe25_joint_2cam_lora \
        --exp-name=... [--num-train-steps=...] [--batch-size=...] [--overwrite]

``OPENPI_ROOT`` defaults to this repository root (openpi's ``scripts/`` live here).
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import maniguard.openpi_sft  # noqa: E402,F401  -- registers TrainConfigs on import

OPENPI_ROOT = os.environ.get("OPENPI_ROOT")
if not OPENPI_ROOT:
    # In the fork, openpi IS this repo (scripts/ live at the root); otherwise
    # fall back to a sibling ../openpi clone.
    OPENPI_ROOT = (
        str(REPO_ROOT)
        if (REPO_ROOT / "scripts" / "train.py").is_file()
        else str(REPO_ROOT.parent / "openpi")
    )
_train = os.path.join(OPENPI_ROOT, "scripts", "train.py")
if not os.path.isfile(_train):
    raise FileNotFoundError(
        f"openpi train.py not found at {_train}; set OPENPI_ROOT to your openpi clone."
    )

runpy.run_path(_train, run_name="__main__")
