"""pi0.5 LoRA SFT TrainConfigs for the jar_transport and stack_retrieve datagen datasets.

Three datasets per family, identical in size and task mix (40 episodes per base task):

* ``allsafe40``           every episode is safe.
* ``unsafe25``            25% of the episodes are unsafe successes; plain task prompt.
* ``unsafe25_safetytag``  the same episodes as ``unsafe25``; each frame's prompt is the task
                          instruction plus " Safety: safe" or " Safety: unsafe".

All six configs share one recipe: pi0.5 warm-started from ``pi05_base``, LoRA on both the
PaliGemma backbone (gemma_2b, r16) and the action expert (gemma_300m, r32), 2-camera input
(``image_left`` overview + wrist), absolute joint actions converted to per-step deltas.

Scale: one job owns all visible GPUs (pure data parallelism, ``fsdp_devices=1``);
``batch_size`` is the GLOBAL batch, set for 2 cards x 32 samples. Run length is 2 epochs of the
family's largest dataset (``steps = frames * 2 / batch``, rounded up to 50), so the three
datasets of a family share one schedule. ``peak_lr = 7e-5 * sqrt(batch / 256)``; warmup is ~3.4%
(Jar) / ~3.1% (Stack) of the run; ``save_interval = keep_period = ceil(steps / 4)`` (4 checkpoints
per run). If you change ``--batch``, recompute steps, LR and the checkpoint interval together.
"""

from __future__ import annotations

import openpi.training.optimizer as _optimizer
from openpi.models import pi0_config
from openpi.training import weight_loaders
from openpi.training.config import DataConfig, TrainConfig

from maniguard.openpi_sft.data_configs import Sim2CamLiberoDataConfig

_PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"
_ORG = "IDEAS-Lab-Northwestern"
_PROJECT = "maniguard-safe-unsafe"

_BATCH = 64  # global: 2 cards x 32
_PEAK_LR = 3.5e-5  # 7e-5 * sqrt(64 / 256)
_DECAY_LR = 3.5e-6  # peak / 10

# family -> (steps, warmup_steps, save/keep period)
#   jar:   largest dataset 944,815 frames   -> 944_815 * 2 / 64 = 29,525.5 -> 29,550
#   stack: largest dataset 2,656,420 frames -> 2_656_420 * 2 / 64 = 83,013.1 -> 83,050
_SCALE = {
    "jar": (29_550, 1_000, 7_388),
    "stack": (83_050, 2_600, 20_763),
}
_DATASETS = ("allsafe40", "unsafe25", "unsafe25-safetytag")


def _lora_model(**kw) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        **kw,
    )


def _config(fam: str, dataset: str) -> TrainConfig:
    steps, warmup, period = _SCALE[fam]
    slug = dataset.replace("-", "_")
    return TrainConfig(
        name=f"pi05-base_datagen_{fam}_{slug}_joint_2cam_lora",
        project_name=_PROJECT,
        policy_metadata={
            "hf_repo": f"{_ORG}/pi05-base-datagen-{fam}-{dataset}-joint-2cam-lora",
            "hf_private": False,
            "default_exp": f"datagen_{fam}_{slug}_joint_2cam",
        },
        model=_lora_model(dtype="bfloat16", discrete_state_input=True),
        data=Sim2CamLiberoDataConfig(
            repo_id=f"{_ORG}/datagen-{fam}-{dataset}-joint-5cam",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
            external_cam="left",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup,
            peak_lr=_PEAK_LR,
            decay_steps=steps,
            decay_lr=_DECAY_LR,
        ),
        num_train_steps=steps,
        batch_size=_BATCH,
        num_workers=48,  # dataloader worker processes; keep below the host's physical core count
        log_interval=100,
        fsdp_devices=1,
        save_interval=period,
        keep_period=period,
        freeze_filter=_lora_model().get_freeze_filter(),
        ema_decay=None,
    )


def _build_configs() -> list[TrainConfig]:
    return [_config(fam, ds) for fam in ("jar", "stack") for ds in _DATASETS]


def register() -> None:
    """Insert the TrainConfigs into openpi's ``_CONFIGS_DICT`` (idempotent)."""
    from openpi.training.config import _CONFIGS_DICT

    for cfg in _build_configs():
        # The cosine schedule must span exactly the run.
        if cfg.lr_schedule.decay_steps != cfg.num_train_steps:
            raise ValueError(f"{cfg.name}: decay_steps must equal num_train_steps")
        _CONFIGS_DICT[cfg.name] = cfg
