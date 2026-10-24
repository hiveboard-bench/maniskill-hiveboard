# Evaluates diffusion policy checkpoints.
# Created: 2026-09-14

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import torch

from baselines.diffusion_policy.model import DiffusionPolicy
from baselines.rollout import ACTION_DIM, CONTROL_HZ, STATE_DIM, rollout, task_overrides
from env.fr3_table_env import set_domain_randomization


# Load a diffusion policy checkpoint and its model settings.
def load_policy(checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    policy = build_policy(config, payload["stats"])
    policy.load_state_dict(payload["ema_model"])
    return policy.to(device).eval(), config


# Create the policy and observation adapter for evaluation.
def build_policy(config, stats):
    encoder = config["encoder"]
    return DiffusionPolicy(
        cameras=config["dataset"]["cameras"],
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        stats=stats,
        encoder=encoder,
        policy_config=config["policy"],
        encoder_config=config["encoders"][encoder],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=None, help="first reset seed; defaults to the config's evaluation seed")
    parser.add_argument("--videos", type=int, default=3, help="number of initial episodes to save as videos; set equal to --episodes to save all")
    parser.add_argument("--num-envs", type=int, default=None, help="episodes simulated in parallel; defaults to the config's")
    parser.add_argument("--record-dir", type=Path, help="write aligned rollout arrays, labels, and selected videos to a new sidecar directory")
    parser.add_argument("--base-camera-eye", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="override task base-camera eye position for this evaluation")
    parser.add_argument("--base-camera-target", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="override task base-camera look-at target; pair with --base-camera-eye")
    parser.add_argument("--camera-pose-randomization", action=argparse.BooleanOptionalAction, default=None,
                        help="override task camera-pose randomization; match the data collection setting when needed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="policy inference device; CPU can be useful when the GPU is occupied by training")
    parser.add_argument("--sim-backend", choices=("gpu", "cpu"), default="gpu",
                        help="physics backend; CPU backend supports one environment without GPU physics memory")
    args = parser.parse_args()
    if args.record_dir and args.record_dir.exists() and any(args.record_dir.iterdir()):
        parser.error(f"record directory must be empty: {args.record_dir}")
    if (args.base_camera_eye is None) != (args.base_camera_target is None):
        parser.error("--base-camera-eye and --base-camera-target must be supplied together")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available() else "cpu")
    policy, config = load_policy(args.checkpoint, device)
    set_domain_randomization(config["evaluation"].get("domain_randomization", True))
    seed_offset = config["evaluation"]["seed_offset"] if args.seed_offset is None else args.seed_offset
    num_envs = args.num_envs or config["evaluation"].get("num_envs", 16)
    checkpoint_stat = args.checkpoint.stat()
    camera_overrides = {}
    if args.base_camera_eye is not None:
        camera_overrides = {
            "cameras.base_camera.eye": args.base_camera_eye,
            "cameras.base_camera.target": args.base_camera_target,
        }
    if args.camera_pose_randomization is not None:
        camera_overrides["render_randomization.camera_pose"] = args.camera_pose_randomization
    with task_overrides(config["task"], camera_overrides):
        metrics, videos, _ = rollout(
            policy, config["task"], config["dataset"]["cameras"], config["dataset"]["hz"], args.episodes, seed_offset, args.videos, device,
            num_envs=num_envs,
            record_dir=args.record_dir,
            record_metadata={
                "policy_family": "diffusion_policy",
                "checkpoint": {"path": str(args.checkpoint.resolve()), "size_bytes": checkpoint_stat.st_size,
                               "mtime_ns": checkpoint_stat.st_mtime_ns},
                "task": config["task"], "encoder": config["encoder"], "policy": config["policy"],
                "camera_override": camera_overrides or None,
                "sim_backend": args.sim_backend,
                "domain_randomization": config["evaluation"].get("domain_randomization", True),
                "seed_offset": seed_offset, "episodes": args.episodes, "num_envs": num_envs,
            },
            sim_backend=args.sim_backend,
        )
    out_dir = args.record_dir or args.checkpoint.parent / f"eval_{args.checkpoint.stem}"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    video_dir = out_dir / "videos" if args.record_dir else out_dir
    video_dir.mkdir(exist_ok=True)
    for i, video in enumerate(videos):
        video_path = video_dir / f"episode_{i}.mp4"
        iio.imwrite(video_path, video, fps=CONTROL_HZ, codec="libx264")
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
