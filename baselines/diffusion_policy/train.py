# Trains the diffusion policy model.
# Created: 2026-09-14

import argparse
import os
import random
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from diffusers.optimization import get_cosine_schedule_with_warmup

from baselines.diffusion_policy.data import SequenceDataset, ensure_dataset, split_episodes
from baselines.diffusion_policy.evaluate import build_policy
from baselines.rollout import CONTROL_HZ, rollout
from baselines.diffusion_policy.model import EMAModel
from env.fr3_table_env import PROJECT_ROOT, set_domain_randomization

CONFIG_PATH = Path(__file__).with_name("config.yaml")
SMOKE_OVERRIDES = {
    "dataset.num_episodes": 4,
    "dataset.val_episodes": 1,
    "training.epochs": 2,
    "training.max_steps_per_epoch": 10,
    "training.num_workers": 2,
    "training.checkpoint_every": 1,
    "evaluation.every": 2,
    "evaluation.episodes": 1,
    "evaluation.video_episodes": 1,
    "evaluation.max_steps": 40,
    "wandb.project": "manip_fail-smoke",
}


# Set a nested configuration value from a dotted key.
def set_key(config, dotted, value):
    *parents, leaf = dotted.split(".")
    for key in parents:
        config = config[key]
    config[leaf] = value


# Read training settings and apply command line overrides.
def load_config(overrides, smoke):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if smoke:
        for key, value in SMOKE_OVERRIDES.items():
            set_key(config, key, value)
        for encoder in config["encoders"].values():
            encoder["batch_size"] = min(encoder["batch_size"], 8)
        config["wandb"]["project"] += "-smoke"
    for override in overrides:
        key, _, value = override.partition("=")
        value = yaml.safe_load(value)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        set_key(config, key, value)
    return config


# Move data to the selected device.
def to_device(batch, device):
    return {
        "images": {camera: frames.to(device, non_blocking=True) for camera, frames in batch["images"].items()},
        "state": batch["state"].to(device, non_blocking=True),
        "action": batch["action"].to(device, non_blocking=True),
    }


# Denoising loss on held-out episodes with fixed noise, so values are comparable across evaluations.
@torch.no_grad()
def validation_loss(policy, loader, device):
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(0)
        return float(np.mean([policy.compute_loss(to_device(batch, device)).item() for batch in loader]))


# Predict an action chunk from every frame of each episode and pair it with the demonstrated actions.
@torch.no_grad()
def open_loop(policy, dataset, device, batch_size=64):
    n_obs, n_act = dataset.n_obs_steps, dataset.n_action_steps
    episodes = []
    for start, end in dataset.episode_bounds:
        steps = np.arange(start, end)
        obs_index = np.clip(steps[:, None] + np.arange(-(n_obs - 1), 1), start, end - 1)
        act_index = np.clip(steps[:, None] + np.arange(n_act), start, end - 1)
        predicted = []
        for i in range(0, len(steps), batch_size):
            index = obs_index[i : i + batch_size]
            images = {
                camera: dataset.frames(camera, index.ravel()).reshape(len(index), n_obs, *dataset.frames(camera, [start]).shape[1:]).to(device)
                for camera in dataset.cameras
            }
            predicted.append(policy.predict_action(images, torch.from_numpy(dataset.state[index]).to(device)).cpu().numpy())
        episodes.append({"predicted": np.concatenate(predicted), "demo": dataset.action[act_index], "state": dataset.state[steps]})
    return episodes


# Measure prediction error against recorded demonstration actions.
def open_loop_metrics(episodes):
    predicted = np.concatenate([e["predicted"] for e in episodes])
    demo = np.concatenate([e["demo"] for e in episodes])
    state = np.concatenate([e["state"] for e in episodes])
    error = np.abs(predicted - demo)
    return {
        "val/joint_error_next_rad": float(error[:, 0, :7].mean()),
        "val/joint_error_chunk_rad": float(error[:, :, :7].mean()),
        "val/joint_error_chunk_end_rad": float(error[:, -1, :7].mean()),
        "val/gripper_error_m": float(error[:, :, 7].mean()),
        "val/hold_still_joint_error_chunk_rad": float(np.abs(state[:, None, :7] - demo[:, :, :7]).mean()),
    }


# Demonstrated vs predicted next joint targets over one held-out episode.
def trajectory_figure(episode, hz, title):
    time_axis = np.arange(len(episode["demo"])) / hz
    names = [f"joint {i + 1} (rad)" for i in range(7)] + ["gripper (m)"]
    figure, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    for i, axis in enumerate(axes.flat):
        axis.plot(time_axis, episode["demo"][:, 0, i], label="demonstrated", linewidth=2)
        axis.plot(time_axis, episode["predicted"][:, 0, i], label="predicted", linestyle="--")
        axis.set_title(names[i])
        axis.grid(alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel("time (s)")
    axes[0, 0].legend()
    figure.suptitle(title)
    figure.tight_layout()
    return figure


# Write model and optimizer state for resuming training.
def save_checkpoint(path, policy, ema, config, stats, epoch, metrics, optimizer=None, scheduler=None, best_success=None):
    state = dict(model=policy.state_dict(), ema_model=ema.averaged_model.state_dict(), config=config, stats=stats, epoch=epoch, metrics=metrics)
    if optimizer is not None:
        state |= dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), ema_step=ema.optimization_step, best_success=best_success)
    torch.save(state, path)


# Rollout, retried when SAPIEN fails to open the render device, which happens now and then on a busy GPU.
def evaluate_with_retries(*args, attempts=3, **kwargs):
    for attempt in range(1, attempts + 1):
        try:
            return rollout(*args, **kwargs)
        except RuntimeError as error:
            print(f"evaluation attempt {attempt}/{attempts} failed: {error}")
            torch.cuda.empty_cache()
            time.sleep(30)
    return None


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="config overrides as dotted.key=value")
    parser.add_argument("--smoke", action="store_true", help="tiny run that exercises data, training, evaluation, checkpoints and logging")
    args = parser.parse_args()
    config = load_config(args.overrides, args.smoke)
    encoder = config["encoder"]
    encoder_config = config["encoders"][encoder]
    train_config = config["training"]
    eval_config = config["evaluation"]
    set_domain_randomization(eval_config.get("domain_randomization", True))

    seed = train_config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

    data_config = config["dataset"]
    store = ensure_dataset(config["task"], data_config["record_dir"], data_config["hz"], data_config["cameras"])
    policy_config = config["policy"]
    train_episodes, val_episodes = split_episodes(store, data_config["num_episodes"], data_config["val_episodes"])
    window_args = (data_config["cameras"], policy_config["horizon"], policy_config["n_obs_steps"], policy_config["n_action_steps"])
    dataset = SequenceDataset(store, *window_args, train_episodes)
    val_dataset = SequenceDataset(store, *window_args, val_episodes) if val_episodes else None
    stats = dataset.stats()
    num_workers = min(train_config["num_workers"], os.cpu_count() or 1)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=encoder_config["batch_size"], shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=device.type == "cuda", persistent_workers=num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=encoder_config["batch_size"], num_workers=min(2, num_workers),
    ) if val_dataset else None
    steps_per_epoch = len(loader) if train_config["max_steps_per_epoch"] is None else min(len(loader), train_config["max_steps_per_epoch"])

    policy = build_policy(config, stats).to(device)
    ema = EMAModel(policy, power=train_config["ema_power"])
    encoder_params = list(policy.obs_encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in policy.parameters() if id(p) not in encoder_ids], "lr": train_config["lr"]},
            {"params": encoder_params, "lr": encoder_config["encoder_lr"]},
        ],
        betas=(0.95, 0.999), eps=1e-8, weight_decay=train_config["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, train_config["warmup_steps"], steps_per_epoch * train_config["epochs"])

    best_success = -1.0
    last_eval = {}
    global_step = 0
    start_epoch = 1
    if train_config.get("resume"):
        resume_path = PROJECT_ROOT / train_config["resume"]
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["model"])
        ema.averaged_model.load_state_dict(checkpoint["ema_model"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["epoch"] * steps_per_epoch
        last_eval = checkpoint["metrics"] or {}
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            ema.optimization_step = checkpoint["ema_step"]
            best_success = checkpoint["best_success"]
        else:
            for _ in range(global_step):
                scheduler.step()
            ema.optimization_step = global_step
            best_path = resume_path.parent / "best.ckpt"
            if best_path.exists():
                best_success = torch.load(best_path, map_location="cpu", weights_only=False)["metrics"]["success_rate"]
        print(f"resuming from {resume_path} at epoch {start_epoch} (best success so far {best_success:.2f})")
        name = resume_path.parent.name
        run_dir = resume_path.parent
    else:
        name = f"dp_{config['task']}_{encoder}_s{train_config['seed']}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = PROJECT_ROOT / config["output_dir"] / name
        run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    wandb_config = config["wandb"]
    run = wandb.init(
        project=wandb_config["project"], entity=wandb_config["entity"], mode=wandb_config["mode"], name=name,
        dir=run_dir, config=config, tags=["diffusion_policy", config["task"], encoder],
    )
    parameters = sum(p.numel() for p in policy.parameters())
    run.summary.update({
        "parameters": parameters, "train_windows": len(dataset), "train_episodes": len(train_episodes),
        "val_episodes": len(val_episodes),
    })
    print(f"{name}: {dataset.num_episodes} episodes, {len(dataset)} windows, {parameters / 1e6:.1f}M parameters, device {device}, amp {amp_dtype}")

    for epoch in range(start_epoch, train_config["epochs"] + 1):
        policy.train()
        epoch_start = time.time()
        losses = []
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = policy.compute_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            log_now = (global_step + 1) % 10 == 0
            if log_now:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf")).item()
            optimizer.step()
            scheduler.step()
            ema.step(policy)
            losses.append(loss.item())
            global_step += 1
            if log_now:
                run.log({
                    "train/loss": loss.item(), "train/grad_norm": grad_norm, "train/lr": scheduler.get_last_lr()[0],
                    "train/encoder_lr": scheduler.get_last_lr()[1], "train/epoch": epoch,
                }, step=global_step)

        epoch_metrics = {"train/epoch_loss": float(np.mean(losses)), "train/epoch_seconds": time.time() - epoch_start, "train/epoch": epoch}
        if device.type == "cuda":
            epoch_metrics["train/gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        memory = f", {epoch_metrics['train/gpu_memory_gb']:.1f} GB GPU" if device.type == "cuda" else ""
        print(f"epoch {epoch}: loss {epoch_metrics['train/epoch_loss']:.4f} ({epoch_metrics['train/epoch_seconds']:.0f} s, {len(losses)} steps{memory})")

        eval_metrics = {}
        result = None
        if eval_config["episodes"] > 0 and (epoch % eval_config["every"] == 0 or epoch == train_config["epochs"]):
            ema.averaged_model.eval()
            result = evaluate_with_retries(
                ema.averaged_model, config["task"], data_config["cameras"], data_config["hz"], eval_config["episodes"],
                eval_config["seed_offset"], eval_config["video_episodes"], device, eval_config["max_steps"],
                eval_config["num_envs"], sim_backend=eval_config.get("sim_backend", "gpu"), reuse_env=True,
            )
            if result is None:
                print(f"epoch {epoch}: evaluation skipped")
        if result is not None:
            metrics, videos, records = result
            last_eval = metrics
            eval_metrics = {f"eval/{key}": value for key, value in metrics.items()}
            for i, (video, record) in enumerate(zip(videos, records)):
                path = run_dir / f"eval_epoch{epoch:04d}_episode{i}.mp4"
                iio.imwrite(path, video, fps=CONTROL_HZ, codec="libx264")
                caption = f"epoch {epoch} | seed {record['seed']} | {record['outcome']} | {record['steps']} steps"
                eval_metrics[f"eval/video_{i}"] = wandb.Video(str(path), format="mp4", caption=caption)
            columns = ["epoch", *records[0].keys()]
            eval_metrics["eval/episodes"] = wandb.Table(columns=columns, data=[[epoch, *r.values()] for r in records])
            if val_loader is not None:
                eval_metrics["val/loss"] = validation_loss(ema.averaged_model, val_loader, device)
                episodes = open_loop(ema.averaged_model, val_dataset, device)
                eval_metrics |= open_loop_metrics(episodes)
                figure = trajectory_figure(episodes[0], data_config["hz"], f"held-out episode {val_episodes[0]}, epoch {epoch}")
                eval_metrics["val/trajectory"] = wandb.Image(figure)
                plt.close(figure)
            print(f"epoch {epoch}: success rate {metrics['success_rate']:.2f}")
            if metrics["success_rate"] > best_success:
                best_success = metrics["success_rate"]
                save_checkpoint(run_dir / "best.ckpt", policy, ema, config, stats, epoch, metrics)
        run.log(epoch_metrics | eval_metrics, step=global_step)

        if epoch % train_config["checkpoint_every"] == 0 or epoch == train_config["epochs"]:
            path = run_dir / f"epoch_{epoch:04d}.ckpt"
            save_checkpoint(path, policy, ema, config, stats, epoch, last_eval, optimizer, scheduler, best_success)
            (run_dir / "latest.ckpt").unlink(missing_ok=True)
            (run_dir / "latest.ckpt").hardlink_to(path)

    run.summary["best_success_rate"] = best_success
    run.finish()
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
