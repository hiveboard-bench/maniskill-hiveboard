# Defines pi0.5 training configurations.
# Created: 2026-09-16

import dataclasses
import pathlib

from typing_extensions import override

import flax.nnx as nnx

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

from baselines.pi05 import policy as fr3_policy

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
OPENPI_DIR = PROJECT_ROOT / "baselines" / "pi05" / "openpi"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines" / "pi05"
PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"
ACTION_HORIZON = 20


@dataclasses.dataclass(frozen=True)
class LeRobotFR3DataConfig(_config.DataConfigFactory):
    """Data pipeline for datasets written by baselines/pi05/convert.py."""

    # Create the configured object.
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = _transforms.Group(
            inputs=[fr3_policy.FR3Inputs(model_type=model_config.model_type), _transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask), fr3_policy.FR3Outputs()],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


# Create the pi0.5 model configuration.
def pi05_model(paligemma_variant, action_expert_variant):
    return pi0_config.Pi0Config(
        pi05=True,
        action_horizon=ACTION_HORIZON,
        discrete_state_input=False,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
    )


CONFIGS = {
    "pi05_fr3": dict(
        model=pi05_model("gemma_2b", "gemma_300m"),
        weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6),
        ema_decay=0.999,
        batch_size=32,
        num_train_steps=30_000,
    ),
    "pi05_fr3_lora": dict(
        model=pi05_model("gemma_2b_lora", "gemma_300m_lora"),
        weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE),
        freeze_filter=pi05_model("gemma_2b_lora", "gemma_300m_lora").get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        num_train_steps=30_000,
    ),
    "pi05_fr3_smoke": dict(
        model=pi05_model("dummy", "dummy"),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=2, peak_lr=5e-5, decay_steps=10, decay_lr=5e-6),
        ema_decay=None,
        batch_size=2,
        num_train_steps=10,
        log_interval=1,
        save_interval=10,
        num_workers=0,
        project_name="fr3-baselines-smoke",
    ),
}


# Assemble the selected pi0.5 model, data, and optimizer settings.
def make_config(
    name, repo_id, exp_name, *, num_train_steps=None, batch_size=None, wandb=True, overwrite=False, project=None,
    fsdp_devices=1, keep_period=None, freeze_vision=False, seed=42,
):
    settings = {"project_name": "fr3-baselines", "log_interval": 100, "save_interval": 1000, **CONFIGS[name], "seed": seed}
    if project is not None:
        settings["project_name"] = project
    if num_train_steps is not None:
        settings["num_train_steps"] = num_train_steps
        schedule = settings.get("lr_schedule", _optimizer.CosineDecaySchedule())
        warmup_steps = min(schedule.warmup_steps, max(1, num_train_steps // 10))
        settings["lr_schedule"] = dataclasses.replace(schedule, warmup_steps=warmup_steps, decay_steps=num_train_steps)
    if keep_period is not None:
        settings["keep_period"] = keep_period
    if freeze_vision:
        settings["freeze_filter"] = nnx.Any(settings.get("freeze_filter", nnx.Nothing), nnx_utils.PathRegex(".*PaliGemma/img.*"))
    if batch_size is not None:
        settings["batch_size"] = batch_size
    return _config.TrainConfig(
        name=name,
        exp_name=exp_name,
        data=LeRobotFR3DataConfig(repo_id=repo_id, base_config=_config.DataConfig(prompt_from_task=True)),
        assets_base_dir=str(OUTPUT_DIR / "assets"),
        checkpoint_base_dir=str(OUTPUT_DIR / "checkpoints"),
        wandb_enabled=wandb,
        overwrite=overwrite,
        fsdp_devices=fsdp_devices,
        **settings,
    )
