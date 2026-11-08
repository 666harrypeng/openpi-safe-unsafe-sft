"""ManiGuard pi0.5 LoRA SFT task configs for the openpi-native trainer.

This package defines the task DataConfigs / policies / TrainConfigs and, at
import time, inserts the TrainConfigs into openpi's ``_CONFIGS_DICT`` via
:func:`train_configs.register`. openpi's own source is not modified — the
configs simply attach to its registry.

The launchers in ``tools/openpi_sft/`` import this package first (triggering the
registration below) and then hand off to openpi's own
``scripts/train.py`` / ``scripts/compute_norm_stats.py`` — so all the heavy
lifting (data loading, model, norm-stats, training loop) stays in openpi and we
only contribute the task-specific config.
"""

from maniguard.openpi_sft._augmax_patch import apply as _apply_augmax_guard
from maniguard.openpi_sft._lerobot_video_patch import apply as _apply_pyav_backend
from maniguard.openpi_sft.train_configs import register

# Neutralize the rare non-finite output of openpi's training-time image
# augmentation (augmax) before any training runs. See _augmax_patch for details.
_apply_augmax_guard()
# Fall back LeRobot video decode to PyAV where torchcodec's system FFmpeg is
# unavailable (no-op where torchcodec works). See _lerobot_video_patch.
_apply_pyav_backend()
register()
