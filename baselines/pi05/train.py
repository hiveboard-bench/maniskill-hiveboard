# Trains a pi0.5 policy.
# Created: 2026-09-16

import argparse
import functools
import importlib.util

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as normalize
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.transforms as transforms

from baselines.pi05.config import CONFIGS, OPENPI_DIR, make_config


# Import one of openpi's scripts/*.py, which are not part of the installed openpi package.
def load_openpi_script(name):
    spec = importlib.util.spec_from_file_location(f"openpi_script_{name}", OPENPI_DIR / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RemoveStrings(transforms.DataTransformFn):
    """Drop string fields such as the prompt, which the statistics do not use (as in openpi's compute_norm_stats.py)."""

    # Process the input.
    def __call__(self, data):
        return {k: v for k, v in data.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


# Compute observation and action statistics for the FR3 dataset.
def compute_norm_stats(config, max_frames):
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.TransformedDataset(
        _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model),
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()],
    )
    num_frames = len(dataset) if max_frames is None else min(max_frames, len(dataset))
    num_batches = num_frames // config.batch_size
    loader = _data_loader.TorchDataLoader(
        dataset, local_batch_size=config.batch_size, num_workers=config.num_workers,
        shuffle=num_frames < len(dataset), num_batches=num_batches,
    )
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="norm stats"):
        for key, running in stats.items():
            running.update(np.asarray(batch[key]))
    output = config.assets_dirs / data_config.repo_id
    normalize.save(output, {key: running.get_statistics() for key, running in stats.items()})
    print(f"wrote norm stats to {output}")


# Initialize training state across devices.
def init_train_state_sharded(script, config, init_rng, mesh, *, resume):
    tx = script._optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    if resume:
        return train_state_shape, state_sharding
    partial_params = script._load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    frozen = set(train_state_shape.params.filter(config.freeze_filter).flat_state())
    cpu = jax.devices("cpu")[0]
    flat = traverse_util.flatten_dict(partial_params)
    for key in frozen & flat.keys():
        flat[key] = jax.device_put(flat[key], cpu).astype(jnp.bfloat16)
    partial_params = traverse_util.unflatten_dict(flat)
    partial_sharding = sharding.fsdp_sharding(partial_params, mesh)
    partial_params = jax.tree.map(jax.device_put, partial_params, partial_sharding)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=(replicated, partial_sharding),
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", choices=CONFIGS)
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset written by baselines/pi05/convert.py")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing checkpoint folder with this name")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--project", default=None, help="W&B project (default: the config's); WANDB_ENTITY selects the team")
    parser.add_argument("--fsdp-devices", type=int, default=1, help="GPUs to shard the model over; batch size must divide by it")
    parser.add_argument("--freeze-vision", action="store_true", help="also freeze the SigLIP image encoder (for 12 GB GPUs)")
    parser.add_argument("--keep-period", type=int, default=None, help="keep every checkpoint whose step is a multiple of this")
    parser.add_argument("--seed", type=int, default=42, help="training seed; run 3 seeds per method")
    parser.add_argument("--norm-frames", type=int, default=None, help="frames sampled for norm stats (default: all)")
    args = parser.parse_args()

    build = lambda: make_config(
        args.config, args.repo_id, args.exp_name, num_train_steps=args.steps, batch_size=args.batch_size,
        wandb=not args.no_wandb, overwrite=args.overwrite, project=args.project, fsdp_devices=args.fsdp_devices,
        keep_period=args.keep_period, freeze_vision=args.freeze_vision, seed=args.seed,
    )
    config = build()
    if not (config.assets_dirs / args.repo_id / "norm_stats.json").exists():
        compute_norm_stats(config, args.norm_frames)
        config = build()
    script = load_openpi_script("train")
    script.init_train_state = functools.partial(init_train_state_sharded, script)
    script.main(config)


if __name__ == "__main__":
    main()
