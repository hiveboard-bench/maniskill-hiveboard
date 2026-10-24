# Replays real robot trajectories in simulation.
# Created: 2026-10-16

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import h5py
import imageio.v3 as iio
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation

import tasks  # noqa: F401
from env.fr3_table_env import ROBOT_BASE_POSE, ROBOT_CONFIG
from mani_skill.utils.structs.pose import Pose
from tasks.generate_demos import CONTROL_MODE

WXYZ = [0, 1, 2, 6, 3, 4, 5]


# Run one real episode open loop; return sim TCP poses (xyzw), joints, gripper widths and camera frames.
def replay(env, episode, record=True):
    base = env.unwrapped
    joints = episode["observations/robot_states/joint_position"][:]
    width = episode["observations/robot_states/gripper_position"][:, 0]
    target = episode["actions/cartesian_position"][:][:, WXYZ]
    command = episode["actions/gripper_position"][:, 0] / 2
    env.reset(seed=0)
    base.agent.reset(torch.tensor(np.concatenate([joints[0], [width[0] / 2] * 2]), dtype=torch.float32)[None])
    state = episode["observations/robot_states/cartesian_position"][:]
    contact = (ROBOT_BASE_POSE * sapien.Pose(state[np.argmin(state[:, 2]), :3])).p[:2]
    shift = torch.tensor(contact, dtype=torch.float32) - (base.board_center_xy + base.module_cell_offsets)[0].cpu()
    for actor in (base.board, base.valve):
        actor.set_pose(Pose.create_from_pq(actor.pose.p + torch.cat([shift, torch.zeros(1)]), actor.pose.q))
    base.board_center_xy += shift.to(base.device)
    obs = base.get_obs()
    arm = base.agent.controller.controllers["arm"]
    poses, qpos, frames = [], [], []
    substeps = base.control_freq // json.loads(episode.attrs["robot_profile"])["control_freq"]
    for t in range(len(target)):
        pose = arm.ee_pose_at_base.raw_pose[0].cpu().numpy()
        poses.append(pose[[0, 1, 2, 4, 5, 6, 3]])
        qpos.append(base.agent.robot.get_qpos()[0].cpu().numpy())
        if record:
            frames.append([obs["sensor_data"][c]["rgb"][0].cpu().numpy() for c in ("base_camera", "wrist_camera")])
        action = torch.tensor(np.append(target[t], command[t]), dtype=torch.float32)[None]
        for _ in range(substeps):
            obs, *_ = env.step(action)
    return np.array(poses), np.array(qpos), frames


# Track the replayed trajectory.
def tracking(state, action):
    x = (action[:-1, :3] - state[:-1, :3]).ravel()
    y = (state[1:, :3] - state[:-1, :3]).ravel()
    return float(x @ y / (x @ x))


# Compute the tracking response.
def response(state, action):
    error = action[:-1, :3] - state[:-1, :3]
    distance = np.linalg.norm(error, axis=1)
    speed = ((state[1:, :3] - state[:-1, :3]) / 0.1 * error).sum(1) / np.maximum(distance, 1e-9)
    turn = (Rotation.from_quat(action[:-1, 3:]) * Rotation.from_quat(state[:-1, 3:]).inv()).as_rotvec()
    angle = np.linalg.norm(turn, axis=1)
    spin = (Rotation.from_quat(state[1:, 3:]) * Rotation.from_quat(state[:-1, 3:]).inv()).as_rotvec() / 0.1
    spin = (spin * turn).sum(1) / np.maximum(angle, 1e-9)
    return (distance * 1000, speed * 1000), (np.degrees(angle), np.degrees(spin))


# Compute error between two poses.
def pose_error(a, b):
    position = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) * 1000
    angle = np.degrees((Rotation.from_quat(a[:, 3:]) * Rotation.from_quat(b[:, 3:]).inv()).magnitude())
    return dict(position_rms_mm=float(np.sqrt(np.mean(position**2))), position_max_mm=float(position.max()),
                rotation_rms_deg=float(np.sqrt(np.mean(angle**2))))


# Build the frame label.
def label(image, text):
    image = np.ascontiguousarray(image)
    cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path, nargs="+", help="real .h5 files with their _front/_wrist MP4s alongside")
    parser.add_argument("--module", default="circuit_breaker", choices=("ball_valve", "circuit_breaker"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/replay_real"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("CircuitBreakerFR3-v1", num_envs=1, obs_mode="rgb", control_mode=CONTROL_MODE, sim_backend="cpu",
                   module=args.module)
    report, responses = {}, []
    for path in args.episodes:
        with h5py.File(path, "r") as episode:
            real = episode["observations/robot_states/cartesian_position"][:]
            real_joints = episode["observations/robot_states/joint_position"][:]
            real_width = episode["observations/robot_states/gripper_position"][:, 0]
            action = episode["actions/cartesian_position"][:]
            sim, sim_qpos, frames = replay(env, episode)
            model = dict(ROBOT_CONFIG)
            ROBOT_CONFIG.update(ee_deadband_meters=0.0, ee_deadband_degrees=0.0, ee_lag_seconds=1e-3,
                                ee_rotation_lag_seconds=1e-3, ee_max_speed=10.0, ee_max_angular_speed_degrees=1e4)
            ideal, _, _ = replay(env, episode, record=False)
            ROBOT_CONFIG.update(model)
            videos = {c: iio.imread(path.with_name(f"{path.stem}_{c}.mp4")) for c in ("front", "wrist")}
        free = slice(0, max(2, int(np.argmax(real[:, 2] < real[:, 2].min() + 0.01))))
        contact = slice(free.stop, len(real))
        responses.append((response(real[free], action[free]), response(sim[free], action[free])))
        report[path.stem] = dict(
            steps=len(real),
            free_steps=free.stop,
            lag_model=dict(free=pose_error(sim[free], real[free]), contact=pose_error(sim[contact], real[contact])),
            no_lag=dict(free=pose_error(ideal[free], real[free]), contact=pose_error(ideal[contact], real[contact])),
            alpha_free=dict(real=tracking(real[free], action[free]), sim=tracking(sim[free], action[free]),
                            no_lag=tracking(ideal[free], action[free])),
            joint_rms_rad=float(np.sqrt(np.mean((sim_qpos[:, :7] - real_joints) ** 2))),
            gripper_width_rms_mm=float(np.sqrt(np.mean((sim_qpos[:, 7:9].sum(1) - real_width) ** 2)) * 1000),
        )
        print(path.stem, json.dumps(report[path.stem], indent=1))
        grid = []
        for t, (sim_front, sim_wrist) in enumerate(frames):
            error = np.linalg.norm(sim[t, :3] - real[t, :3]) * 1000
            real_front = cv2.resize(videos["front"][t], sim_front.shape[1::-1])
            real_wrist = cv2.resize(videos["wrist"][t], sim_wrist.shape[1::-1])
            top = np.concatenate([label(real_front, f"real front  t={t / 10:.1f}s"),
                                  label(sim_front, f"sim front  TCP error {error:.0f} mm")], axis=1)
            bottom = np.concatenate([label(real_wrist, "real wrist"), label(sim_wrist, "sim wrist")], axis=1)
            grid.append(np.concatenate([top, bottom], axis=0))
        iio.imwrite(args.out_dir / f"{path.stem}.mp4", np.stack(grid), fps=10, codec="libx264")
    bins = {"translation mm": (0, 5, 10, 20, 40, 80), "rotation deg": (0, 4, 8, 15, 30, 60)}
    for axis, (name, edges) in enumerate(bins.items()):
        table = {}
        for low, high in zip(edges, edges[1:] + (np.inf,)):
            row = {}
            for source, index in (("real", 0), ("sim", 1)):
                error = np.concatenate([r[index][axis][0] for r in responses])
                speed = np.concatenate([r[index][axis][1] for r in responses])
                inside = (error >= low) & (error < high)
                row[source] = float(np.median(speed[inside])) if inside.any() else None
            table[f"{low}-{high}"] = row
        report[f"median speed toward command by {name} error"] = table
        print(name, json.dumps(table))
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out_dir}")
    env.close()


if __name__ == "__main__":
    main()
