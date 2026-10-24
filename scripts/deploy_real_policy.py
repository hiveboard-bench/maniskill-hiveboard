# Runs a trained policy on the real robot.
# Created: 2026-10-16

import argparse
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import tasks
from mani_skill.utils.structs.pose import Pose
from tasks.generate_demos import CONTROL_MODE, real_pose

POLICY_HZ = 10


# Load the trained checkpoint and its camera settings.
def load_policy(checkpoint_path, official_repo, backbone_path, num_inference_steps, device):
    spec = importlib.util.spec_from_file_location("fr3_vision_backbone", backbone_path)
    vision = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = vision
    spec.loader.exec_module(vision)
    sys.path.insert(0, str(official_repo))
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    shape_meta, config = checkpoint["shape_meta"], checkpoint["config"]
    cameras = tuple(config.get("cameras") or [k for k, v in shape_meta["obs"].items() if v.get("type") == "rgb"])
    if tuple(shape_meta["action"]["shape"]) != (10,) or tuple(shape_meta["obs"]["robot_state"]["shape"]) != (22,):
        raise ValueError("expected the FR3 Cartesian contract: action [10], robot_state [22]")
    backbone = config.get("vision_backbone", "resnet18")
    height, width = shape_meta["obs"][cameras[0]]["shape"][1:]
    crop_ratio = float(config.get("crop_ratio", 0.0))
    crop = (round(height * crop_ratio), round(width * crop_ratio)) if 0.0 < crop_ratio < 1.0 else None
    encoder = MultiImageObsEncoder(
        shape_meta=shape_meta, rgb_model=vision.build_vision_backbone(backbone, "none"), resize_shape=None,
        crop_shape=crop, random_crop=False, use_group_norm=backbone == "resnet18",
        share_rgb_model=bool(config.get("share_rgb_encoder", False)), imagenet_norm=backbone == "resnet18",
    )
    scheduler_kwargs = dict(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
                            beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    if config.get("noise_scheduler", "ddpm") == "ddim":
        scheduler = DDIMScheduler(set_alpha_to_one=True, steps_offset=0, **scheduler_kwargs)
    else:
        scheduler = DDPMScheduler(**scheduler_kwargs)
    action_horizon = int(config["action_horizon"])
    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta, noise_scheduler=scheduler, obs_encoder=encoder, horizon=action_horizon,
        n_action_steps=min(8, action_horizon), n_obs_steps=int(config["obs_horizon"]),
        num_inference_steps=num_inference_steps or int(config["num_inference_steps"]), obs_as_global_cond=True,
        diffusion_step_embed_dim=128, down_dims=tuple(config["down_dims"]), kernel_size=5, n_groups=8,
        cond_predict_scale=True,
    )
    policy.load_state_dict(checkpoint["ema_policy" if "ema_policy" in checkpoint else "policy"])

    # Interpolate values linearly.
    def linear(center, scale):
        inverse = 1.0 / scale
        stats = {"min": center - scale, "max": center + scale, "mean": center, "std": scale}
        return SingleFieldLinearNormalizer.create_manual(scale=inverse, offset=-center * inverse, input_stats_dict=stats)

    normalizer = LinearNormalizer()
    normalizer["action"] = linear(checkpoint["action_mean"], checkpoint["action_std"])
    normalizer["robot_state"] = linear(checkpoint["state_mean"], checkpoint["state_std"])
    legacy = config.get("camera_normalization", "legacy_double") == "legacy_double"
    for camera in cameras:
        normalizer[camera] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.tensor([2.0 if legacy else 1.0]), offset=torch.tensor([-1.0 if legacy else 0.0]),
            input_stats_dict={"min": torch.tensor([0.0]), "max": torch.tensor([1.0]),
                              "mean": torch.tensor([0.5]), "std": torch.tensor([(1.0 / 12.0) ** 0.5])},
        )
    policy.set_normalizer(normalizer)
    return policy.to(device).eval(), cameras, (height, width), config


# Convert a 6D rotation to a wxyz quaternion.
def rot6d_to_wxyz(values):
    a1, a2 = values[:3], values[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    a2 = a2 - b1 @ a2 * b1
    b2 = a2 / max(np.linalg.norm(a2), 1e-8)
    matrix = np.stack((b1, b2, np.cross(b1, b2)), axis=1)
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return np.array([w, x, y, z])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True, help="official_diffusion_policy checkout")
    parser.add_argument("--vision-backbone", type=Path, required=True, help="the real repo's diffusion_policy/vision_backbone.py")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--board-xy", type=float, nargs=2, default=None, help="world xy of the board center")
    parser.add_argument("--module-yaw", type=float, default=None, help="breaker yaw in degrees")
    parser.add_argument("--board-yaw", type=float, default=0.0, help="board yaw about its center, degrees")
    parser.add_argument("--module-drop", type=float, default=0.0, help="lower the breaker module by this much, m")
    parser.add_argument("--wrist-height", type=float, default=None, help="wrist camera offset along the hand z axis, m")
    parser.add_argument("--wrist-roll", type=float, default=0.0, help="wrist camera turn about its optical axis, degrees")
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--steps-per-inference", type=int, default=6)
    parser.add_argument("--max-seconds", type=float, default=40.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/real_policy"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, cameras, (height, width), config = load_policy(
        args.checkpoint, args.official_repo, args.vision_backbone, args.num_inference_steps, device
    )
    obs_horizon = int(config["obs_horizon"])
    sim_cameras = {"front": "base_camera", "wrist": "wrist_camera"}
    board = tasks.ENV_CLASSES["circuit_breaker"].config["objects"]
    board.update(module="circuit_breaker", board_position_offset_xy=[[0.0, 0.0], [0.0, 0.0]],
                 board_yaw_range_degrees=[args.board_yaw, args.board_yaw])
    wrist = tasks.ENV_CLASSES["circuit_breaker"].config["cameras"]["wrist_camera"]
    if args.wrist_height is not None:
        wrist["position"] = [wrist["position"][0], wrist["position"][1], args.wrist_height]
    roll = Rotation.from_euler("x", args.wrist_roll, degrees=True)
    w, x, y, z = wrist["quaternion"]
    x, y, z, w = (Rotation.from_quat([x, y, z, w]) * roll).as_quat()
    wrist["quaternion"] = [w, x, y, z]
    if args.board_xy is not None:
        board["board_center_xy"] = list(args.board_xy)
    if args.module_yaw is not None:
        board["module_yaw_choices_degrees"] = [args.module_yaw]
    env = gym.make("CircuitBreakerFR3-v1", num_envs=1, obs_mode="rgb", control_mode=CONTROL_MODE, sim_backend="cpu")
    base = env.unwrapped
    arm = base.agent.controller.controllers["arm"]
    substeps = base.control_freq // POLICY_HZ
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        obs, _ = env.reset(seed=seed)
        if args.module_drop:
            pose = base.valve.pose
            base.valve.set_pose(Pose.create_from_pq(pose.p - torch.tensor([0.0, 0.0, args.module_drop]), pose.q))
            obs = base.get_obs()
        history = deque(maxlen=obs_horizon)
        plan, frames, log = deque(), [], []
        closed = base.agent.robot.get_qpos()[0, 7:9].sum().item() < 0.04
        for step in range(int(args.max_seconds * POLICY_HZ)):
            qpos = base.agent.robot.get_qpos()[0].cpu().numpy()
            qvel = base.agent.robot.get_qvel()[0].cpu().numpy()
            tcp = arm.ee_pose_at_base.raw_pose[0].cpu().numpy()
            state = np.concatenate([qpos[:7], qvel[:7], real_pose(tcp[None])[0], [qpos[7] + qpos[8]]])
            images = {
                camera: torch.from_numpy(cv2.resize(
                    obs["sensor_data"][sim_cameras[camera]]["rgb"][0].cpu().numpy(), (width, height),
                    interpolation=cv2.INTER_AREA,
                ).transpose(2, 0, 1).copy()).float() / 255.0
                for camera in cameras
            }
            history.append((images, torch.from_numpy(state).float()))
            while len(history) < obs_horizon:
                history.appendleft(history[0])
            if step % args.steps_per_inference == 0 or not plan:
                batch = {camera: torch.stack([h[0][camera] for h in history])[None].to(device) for camera in cameras}
                batch["robot_state"] = torch.stack([h[1] for h in history])[None].to(device)
                with torch.no_grad():
                    plan = deque(policy.predict_action(batch)["action"][0].cpu().numpy())
            action = plan.popleft()
            opening = float(np.clip(action[9] / 0.08, 0.0, 1.0))
            closed = True if opening < 0.35 else False if opening > 0.65 else closed
            command = np.concatenate([action[:3], rot6d_to_wxyz(action[3:9]), [0.0 if closed else 0.04]])
            log.append(dict(state=state.tolist(), action=action.tolist()))
            for _ in range(substeps):
                obs, _, terminated, truncated, info = env.step(torch.from_numpy(command).float()[None])
            frame = np.ascontiguousarray(np.concatenate(
                [obs["sensor_data"][sim_cameras[c]]["rgb"][0].cpu().numpy() for c in ("front", "wrist")], 1
            ))
            text = f"t={step / POLICY_HZ:.1f}s  breaker={float(base.valve.get_qpos()[0, 0]):+.2f} rad  gripper={'closed' if closed else 'open'}"
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(frame)
            if bool(info["success"].item()) or bool(terminated.any() or truncated.any()):
                break
        success = bool(info["success"].item())
        name = f"circuit_breaker_seed{seed}_{'success' if success else 'fail'}"
        iio.imwrite(args.out_dir / f"{name}.mp4", np.stack(frames), fps=POLICY_HZ, codec="libx264")
        (args.out_dir / f"{name}.json").write_text(json.dumps(log))
        results[seed] = dict(success=success, seconds=len(frames) / POLICY_HZ)
        print(seed, results[seed], flush=True)
    print(f"success {sum(r['success'] for r in results.values())}/{len(results)}; videos in {args.out_dir}")


if __name__ == "__main__":
    main()
