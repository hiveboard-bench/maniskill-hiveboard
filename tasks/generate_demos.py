# Plans and records scripted FR3 demonstrations for the supported HiveBoard tasks.
# Created: 2026-09-14

import argparse
import json
import multiprocessing
import shutil
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v3 as iio
import numcodecs
import numpy as np
import sapien
import zarr
from scipy.spatial.transform import Rotation, Slerp

import copy
import threading
from concurrent.futures import ProcessPoolExecutor

import torch

from env.fr3_table_env import (
    CONFIG, DATA_CONFIG, FRONT_TABLE_TOP_Z, REFLEX_KEYS, ROBOT_BASE_POSE, ROBOT_CONFIG, LaggedEEPoseController,
    set_domain_randomization,
)
from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.utils.structs.pose import Pose
from tasks import TASKS
from tasks.showcase_video import ShowcaseVideo

_worker = threading.local()
_ik_lock = threading.Lock()


# Convert the current thread pose to a SAPIEN pose.
def _thread_sp(pose):
    raw = pose.raw_pose
    i = getattr(_worker, "idx", 0) if raw.shape[0] > 1 else 0
    raw = raw[i].cpu().numpy()
    return sapien.Pose(raw[:3], raw[3:])


Pose.sp = property(_thread_sp)


# Select the current thread row from a tensor.
def _thread_row(tensor):
    array = tensor.cpu().numpy()
    return array[getattr(_worker, "idx", 0)] if array.ndim == 2 else array

OPEN = 0.04
CLOSED = 0.0
TOP_DOWN = Rotation.from_quat([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])


# Limit vector magnitude without changing its direction.
def clip_vector(vector, limit):
    norm = np.linalg.norm(vector)
    return vector * min(1.0, limit / max(norm, 1e-9))


# Wrap an angle to the requested period.
def wrap(angle, period):
    return (angle + period / 2) % period - period / 2


# Return the rotation from a pose.
def rotation(pose):
    w, x, y, z = pose.q
    return Rotation.from_quat([x, y, z, w])


# Build a SAPIEN pose from position and orientation.
def make_pose(position, rot):
    x, y, z, w = rot.as_quat()
    return sapien.Pose(p=np.asarray(position, dtype=np.float32), q=[w, x, y, z])


# Return a top down grasp pose.
def top_down(yaw):
    return Rotation.from_euler("z", yaw) * TOP_DOWN


CONTROL_MODE = "pd_ee_target_pose"
RECORDING_NAME = f"scripted_{CONTROL_MODE}.h5"


# Configure absolute robot control.
def absolute_control(env):
    return isinstance(env.unwrapped.agent.controller.controllers["arm"], LaggedEEPoseController)


class CartesianPlanner:
    """Drive the TCP to world-frame waypoints: absolute targets in pd_ee_target_pose, closed-loop deltas in pd_ee_delta_pose."""

    max_step_meters = 0.04
    max_step_radians = 0.1
    bias_gain = 0.15
    bias_window_meters = 0.03
    bias_window_radians = 0.3
    max_pos_bias = 0.02
    max_rot_bias = np.deg2rad(12.0)
    torque_guard = 0.9
    torque_guard_steps = 2

    # Set up the state and resources used by this instance.
    def __init__(self, env, idx=0, step=None, info=None):
        self.env = env
        self.base = env.unwrapped
        self.idx = idx
        self.step = step if step is not None else (lambda action: env.step(action)[2:])
        arm = self.base.agent.controller.controllers["arm"]
        self.absolute = absolute_control(env)
        self.pos_scale = arm.config.pos_upper
        self.rot_scale = arm.config.rot_lower
        demo = self.base.config["demo"]
        self.max_speed = demo["max_speed"]
        self.contact_speed = demo["contact_speed"]
        self.max_angular_speed = demo["max_angular_speed"]
        self.max_acceleration = demo.get("max_acceleration_m_s2", 0.4)
        self.max_jerk = demo.get("max_jerk_m_s3", 4.0)
        self.max_angular_acceleration = demo.get("max_angular_acceleration_rad_s2", 1.2)
        self.max_angular_jerk = demo.get("max_angular_jerk_rad_s3", 12.0)
        self.smooth_trajectory = demo.get("smooth_trajectory", False)
        self.deadband_bias_enabled = demo.get("deadband_bias_enabled", True) and (
            ROBOT_CONFIG["ee_deadband_meters"] > 0 or ROBOT_CONFIG["ee_deadband_degrees"] > 0
        )
        self.max_pos_bias = demo.get("max_pos_bias_m", self.max_pos_bias)
        self.max_rot_bias = np.deg2rad(demo.get("max_rot_bias_degrees", np.rad2deg(self.max_rot_bias)))
        self.speed = self.max_speed
        self.dt = 1.0 / self.base.control_freq
        self.gripper = OPEN
        self.done = False
        self.steps = 0
        self.info = info if info is not None else slice_info(self.base.get_info(), idx)
        self.target = self.tcp()
        self.pos_bias = np.zeros(3)
        self.rot_bias = np.zeros(3)
        self.blocked = False
        self.strained = 0

    # Return the current tool center point pose.
    def tcp(self):
        return self.base.agent.tcp.pose.sp

    # The largest fraction of its force limit the arm drive asks of a joint (1 is the limit itself).
    def torque_saturated(self):
        arm = self.base.agent.controller.controllers["arm"]
        target = getattr(arm, "_target_qpos", None)
        if target is None:
            return 0.0
        robot = self.base.agent.robot
        q = robot.get_qpos()[self.idx, :7].cpu().numpy()
        qd = robot.get_qvel()[self.idx, :7].cpu().numpy()
        torque = np.asarray(arm.config.stiffness) * (target[self.idx].cpu().numpy() - q) - np.asarray(arm.config.damping) * qd
        return float((np.abs(torque) / np.asarray(ROBOT_CONFIG["arm_force_limits"])).max())

    # Return the observation after the latest planner step.
    def _stepped(self, terminated, truncated):
        self.steps += 1
        self.done = bool(terminated.any() or truncated.any())
        load = self.torque_saturated()
        self.strained = self.strained + 1 if load >= self.torque_guard else 0
        if self.strained >= self.torque_guard_steps:
            self.blocked = True
            self.target = self.tcp()
            self.pos_bias[:] = 0.0
            self.rot_bias[:] = 0.0

    # Take one step toward self.target; return True if already within tolerance.
    def servo(self, pos_tol=0.003, rot_tol=0.03, update_bias=True):
        root = rotation(self.base.agent.robot.pose.sp).inv()
        tcp = self.tcp()
        pos_err = root.apply(self.target.p - tcp.p)
        rot_err = (root * rotation(self.target) * rotation(tcp).inv() * root.inv()).as_rotvec()
        reached = np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol
        if self.absolute:
            if update_bias and self.deadband_bias_enabled and np.linalg.norm(pos_err) < self.bias_window_meters:
                self.pos_bias = clip_vector(self.pos_bias + self.bias_gain * pos_err, self.max_pos_bias)
            if update_bias and self.deadband_bias_enabled and np.linalg.norm(rot_err) < self.bias_window_radians:
                self.rot_bias = clip_vector(self.rot_bias + self.bias_gain * rot_err, self.max_rot_bias)
            target = self.base.agent.robot.pose.sp.inv() * self.target
            q = (Rotation.from_rotvec(self.rot_bias) * rotation(target)).as_quat()[[3, 0, 1, 2]]
            action = np.concatenate([target.p + self.pos_bias, q, [self.gripper]]).astype(np.float32)
            terminated, truncated, self.info = self.step(action)
            self._stepped(terminated, truncated)
            return reached
        pos_err *= min(1.0, self.max_step_meters / max(np.linalg.norm(pos_err), 1e-9))
        rot_err *= min(1.0, self.max_step_radians / max(np.linalg.norm(rot_err), 1e-9))
        euler = Rotation.from_rotvec(rot_err).as_euler("XYZ")
        arm = np.concatenate([pos_err / self.pos_scale, euler / self.rot_scale])
        terminated, truncated, self.info = self.step(np.append(arm, self.gripper).astype(np.float32))
        self._stepped(terminated, truncated)
        return reached

    # Slide the target to pose at self.speed and max_angular_speed, then servo up to max_steps to converge.
    def move_to(self, pose, max_steps=80, pos_tol=0.003, rot_tol=0.03, until=None):
        self.blocked = False
        start = self.target
        distance = np.linalg.norm(pose.p - start.p)
        angle = (rotation(pose) * rotation(start).inv()).magnitude()
        if self.smooth_trajectory:
            start_pos_bias = self.pos_bias.copy()
            start_rot_bias = self.rot_bias.copy()
            n = max(
                self._segment_steps(distance, self.speed, self.max_acceleration, self.max_jerk),
                self._segment_steps(
                    angle, self.max_angular_speed, self.max_angular_acceleration, self.max_angular_jerk
                ),
            )
        else:
            n = int(np.ceil(max(distance / (self.speed * self.dt), angle / (self.max_angular_speed * self.dt), 1)))
        slerp = Slerp([0, 1], Rotation.concatenate([rotation(start), rotation(pose)]))
        for i in range(1, n + 1):
            if self.done or self.blocked:
                return False
            if until is not None and until():
                return True
            if self.smooth_trajectory:
                phase = i / n
                progress = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
                self.target = make_pose(start.p + (pose.p - start.p) * progress, slerp(progress))
                self.pos_bias[:] = start_pos_bias * (1 - progress)
                self.rot_bias[:] = start_rot_bias * (1 - progress)
                self.servo(update_bias=False)
            else:
                self.target = make_pose(start.p + (pose.p - start.p) * i / n, slerp(i / n))
                self.servo()
        if self.smooth_trajectory:
            self.pos_bias[:] = 0.0
            self.rot_bias[:] = 0.0
        if self.blocked:
            return False
        self.target = pose
        for _ in range(max_steps):
            if self.done or self.blocked:
                return False
            if until is not None and until():
                return True
            if self.servo(pos_tol, rot_tol):
                return True
        return False

    # Command the finger width and wait for the delayed, rate-limited fingers to travel, plus steps to settle.
    def set_gripper(self, width, steps=10):
        travel = abs(width - self.gripper) / (ROBOT_CONFIG["gripper_max_speed"] * self.dt)
        delay = ROBOT_CONFIG["gripper_delay_seconds"] / self.dt
        self.gripper = width
        self.idle(int(np.ceil(travel + delay)) + steps)

    # Hold the last target pose, since a zero delta lets the compliant arm drift under contact.
    def idle(self, steps=None, until_success=False):
        while not self.done and (steps is None or steps > 0):
            if until_success and self.info["success"].any():
                break
            self.servo()
            steps = None if steps is None else steps - 1
        return self.info

    # Choose enough samples for a quintic segment to respect speed, acceleration, and jerk limits.
    def _segment_steps(self, distance, speed, acceleration, jerk):
        if distance <= 1e-9:
            return 4
        duration = max(
            1.875 * distance / speed,
            np.sqrt(5.773503 * distance / acceleration),
            np.cbrt(60.0 * distance / jerk),
        )
        return max(4, int(np.ceil(duration / self.dt)))


# Kinematics with the CPU IK model, which GPU simulation does not build.
def cpu_kinematics(planner):
    kinematics = planner.base.agent.controller.controllers["arm"].kinematics
    if not hasattr(kinematics, "pmodel"):
        if not hasattr(kinematics, "_cpu_copy"):
            cpu = copy.copy(kinematics)
            Kinematics._setup_cpu(cpu)
            kinematics._cpu_copy = cpu
        return kinematics._cpu_copy
    return kinematics


# Smallest distance to the arm joint limits of an IK solution for a world TCP pose, or -inf if IK fails.
def joint_margin(planner, pose):
    return path_margin(planner, [pose])


# Generate intermediate poses between the start and target.
def interpolated(start, end, samples=6):
    turning = Slerp([0, 1], Rotation.concatenate([rotation(start), rotation(end)]))
    return [make_pose(np.array(start.p) + (np.array(end.p) - start.p) * f, turning(f))
            for f in np.linspace(1 / samples, 1, samples)]


# Build the inverse kinematics chain.
def ik_chain(planner, poses):
    kinematics = cpu_kinematics(planner)
    q = planner.base.agent.robot.get_qpos()[planner.idx, :7].cpu().numpy()
    for pose in poses:
        target = planner.base.agent.robot.pose.sp.inv() * pose
        with _ik_lock:
            solution, ok, _ = kinematics.pmodel.compute_inverse_kinematics(
                kinematics.end_link_idx, target, initial_qpos=q, active_qmask=[1] * 7, max_iterations=200
            )
        if not ok:
            return None
        q = solution[:7]
    return q


# Smallest joint-limit margin along a chain of world TCP poses, or -inf if IK fails.
def path_margin(planner, poses):
    kinematics = cpu_kinematics(planner)
    q = planner.base.agent.robot.get_qpos()[planner.idx, :7].cpu().numpy()
    limits = planner.base.agent.robot.get_qlimits()[planner.idx, :7].cpu().numpy()
    margin = np.inf
    for pose in poses:
        target = planner.base.agent.robot.pose.sp.inv() * pose
        with _ik_lock:
            solution, ok, _ = kinematics.pmodel.compute_inverse_kinematics(
                kinematics.end_link_idx, target, initial_qpos=q, active_qmask=[1] * 7, max_iterations=200
            )
        if not ok or (margin < np.inf and np.abs(solution[:7] - q).max() > np.pi / 2):
            return -np.inf
        q = solution[:7]
        margin = min(margin, np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
    return margin


# Hook the ball valve lever with a closed gripper and pull it through a quarter turn.
def solve_ball_valve(planner, fault=None):
    env = planner.base
    if fault == "never_contacted":
        planner.set_gripper(CLOSED)
        return planner.idle()

    cfg = env.config["demo"]
    pivot = env.valve.pose.sp.p
    radius = cfg["pull_radius"]
    height = pivot[2] + cfg["pull_height"]
    hand = top_down(0.0)
    yaw = np.deg2rad(float(env.module_yaw_degrees[planner.idx].item()))
    radial = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    tangent = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    planner.set_gripper(CLOSED)
    planner.move_to(make_pose(pivot + radial * radius + tangent * -0.035 + [0, 0, cfg["approach_height"]], hand))
    planner.speed = cfg["contact_speed"]
    planner.move_to(make_pose(pivot + radial * radius + tangent * -0.025 + [0, 0, cfg["pull_height"]], hand))
    planner.move_to(make_pose(pivot + radial * radius + [0, 0, cfg["pull_height"]], hand), pos_tol=0.002)

    end = env.config["valve"]["target_angle"]
    if fault == "stopped_short":
        end *= 0.5
    angles = np.linspace(0.0, end, cfg["pull_steps"])[1:]
    if fault == "reopened":
        angles = angles[: len(angles) // 2]
    goal = env.config["valve"]["target_angle"]
    turned = lambda: abs(float(env.valve.get_qpos()[planner.idx, 0]) - goal) <= np.deg2rad(
        env.config["success"]["angle_tolerance_degrees"]) / 2
    for angle in angles:
        if planner.done:
            break
        point = pivot + radial * (radius * np.cos(angle)) + tangent * (-radius * np.sin(angle))
        point[2] += cfg["pull_height"]
        planner.move_to(make_pose(point, hand), max_steps=3, pos_tol=0.004)
        if fault is None and (turned() or planner.blocked or planner.info["success"].any()):
            break

    if fault == "reopened":
        planner.set_gripper(OPEN)
    if fault in {"stopped_short", "wrong_direction", "reopened"}:
        return planner.idle()
    planner.move_to(make_pose(np.array(planner.tcp().p) + [0, 0, cfg["approach_height"] - cfg["pull_height"]], hand))
    planner.move_to(make_pose(pivot - radial * (radius + 0.06) + [0, 0, cfg["approach_height"]], hand))
    return planner.idle(until_success=True)


# Sweep the breaker toggle from one stop to the other with a closed gripper.
def solve_circuit_breaker(planner, fault=None):
    env = planner.base
    if fault == "never_contacted":
        planner.set_gripper(CLOSED)
        return planner.idle()

    cfg = env.config["demo"]
    pivot = env.valve.pose.sp.p
    yaw = np.deg2rad(float(env.module_yaw_degrees[planner.idx].item()))
    tangent = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    hand = rotation(planner.tcp())
    start = -cfg["breaker_start"]
    end = cfg["breaker_sweep"]
    if fault in {"stopped_short", "reopened"}:
        end = -0.02

    planner.gripper = CLOSED
    planner.move_to(make_pose(pivot + tangent * start + [0, 0, cfg["approach_height"]], hand), max_steps=10, pos_tol=0.01)
    planner.speed = cfg["contact_speed"]
    planner.move_to(make_pose(pivot + tangent * start + [0, 0, cfg["breaker_height"]], hand), max_steps=60, pos_tol=0.003)
    target_angle = env.config["breaker"]["target_angle"] - np.deg2rad(env.config["success"]["angle_tolerance_degrees"])
    flipped = (lambda: float(env.valve.get_qpos()[planner.idx, 0]) >= target_angle) if fault is None else None
    planner.speed = cfg["breaker_sweep_speed"]
    sweep_end = pivot + tangent * end + [0, 0, cfg["breaker_height"] - cfg["breaker_sweep_drop"]]
    planner.move_to(make_pose(sweep_end, hand), max_steps=40, pos_tol=0.004, until=flipped)

    if fault == "reopened":
        planner.set_gripper(OPEN)
    if fault in {"stopped_short", "wrong_direction", "reopened"}:
        return planner.idle()
    planner.pos_bias[:] = 0.0
    planner.rot_bias[:] = 0.0
    planner.target = make_pose(planner.tcp().p, hand)
    planner.speed = cfg["max_speed"]
    lift = cfg["breaker_lift_height"] - cfg["breaker_height"]
    planner.move_to(make_pose(planner.tcp().p + [0, 0, lift], hand), max_steps=5, pos_tol=0.02)
    return planner.idle(until_success=True)


# Turn a gate valve wheel in reachable segments with regrasping.
def solve_gate_valve(planner, fault=None):
    env = planner.base
    cfg = env.config["demo"]
    if fault == "never_contacted":
        return planner.idle()
    per_grasp = np.deg2rad(cfg["turn_per_grasp_degrees"])
    target = env.mechanism_config["target_angle"]
    radius = env.wheel_radius if cfg["grasp"] == "rim" else 0.0
    period = np.pi / 2 if cfg["grasp"] == "bolt" else np.pi
    approaches = int(round(2 * np.pi / period)) if radius == 0 else 8
    for segment in range(1 if fault == "stopped_short" else cfg["max_segments"]):
        if planner.done:
            break
        angle = float(env.valve.get_qpos()[planner.idx, 0])
        remaining = target - angle
        if remaining <= 0.02 and fault is None:
            break
        turn = min(per_grasp, 1.25 * remaining + 0.2)
        if fault == "unwound" and segment:
            turn = -per_grasp
        center = env.valve_lever.pose.sp.p
        height = env.wheel_height + cfg["grasp_tcp_height"]

        def poses(index, fraction=1.0):
            phi = index * (period if radius == 0 else 2 * np.pi / approaches)
            offset = np.array([radius * np.cos(phi), radius * np.sin(phi), 0.0])
            hand = top_down(phi + np.deg2rad(cfg["grasp_yaw_offset_degrees"]))
            start = make_pose([center[0] + offset[0], center[1] + offset[1], height], hand)
            spin = Rotation.from_euler("z", turn * fraction)
            end_p = np.array([center[0], center[1], height]) + spin.apply(offset)
            return start, make_pose(end_p, spin * hand)

        checks = max(2, int(np.ceil(abs(turn) / np.deg2rad(15.0))))
        index = max(range(approaches), key=lambda i: path_margin(
            planner, [poses(i)[0]] + [poses(i, k / checks)[1] for k in range(1, checks + 1)]))
        grasp_pose, turn_pose = poses(index)
        arc = 1 if radius == 0 else max(1, int(abs(turn) / np.deg2rad(cfg["arc_step_degrees"])))
        above = np.array(grasp_pose.p) + [0, 0, cfg["approach_height"]]
        planner.set_gripper(OPEN, steps=4)
        planner.move_to(make_pose(above, rotation(grasp_pose)))
        planner.speed = planner.contact_speed
        planner.move_to(grasp_pose, pos_tol=0.002)
        planner.set_gripper(cfg["grasp_width"], steps=cfg["grasp_settle_steps"])
        done_turning = (lambda: float(env.valve.get_qpos()[planner.idx, 0]) >= target) if fault is None else None
        for step in range(1, arc + 1):
            planner.move_to(poses(index, step / arc)[1], max_steps=160 // arc + 20, pos_tol=0.004, rot_tol=0.02,
                            until=done_turning)
            if planner.blocked or (done_turning is not None and done_turning()):
                break
        planner.set_gripper(OPEN, steps=cfg["grasp_settle_steps"])
        planner.speed = planner.max_speed
        planner.move_to(make_pose(np.array(planner.tcp().p) + [0, 0, cfg["approach_height"]], rotation(planner.tcp())))
        if fault == "unwound" and segment:
            break
    return planner.idle(until_success=fault is None)


# Open the lid and press the button with the robot gripper.
def solve_button_cover(planner, fault=None):
    env = planner.base
    cfg = env.config["demo"]
    if fault == "never_contacted":
        return planner.idle()
    root = env.valve.pose.sp
    lid = env.config["objects"]["parts"]["lid"]
    lid_root = root * sapien.Pose(q=Rotation.from_euler("z", lid.get("yaw_degrees", 0.0), degrees=True).as_quat()[[3, 0, 1, 2]])
    local = lambda x, y, z: lid_root * sapien.Pose(p=[x, y, z])
    front_y = lid["hinge_y"] + lid["plate_y"] + lid["half_y"]
    plate_z = lid["hinge_z"] + lid["plate_z"]
    grip = local(0.0, front_y - 0.008, plate_z).p
    hinge = local(0.0, lid["hinge_y"], lid["hinge_z"]).p
    axis = rotation(lid_root).apply([1.0, 0.0, 0.0])
    button = env.button.pose.sp.p
    above = np.array([button[0], button[1], env.button_height + cfg["press_tcp_height"]])

    # Joint-limit margin of the lid arc and of the press, each a continuous wrist motion of its own.
    def wrist_margin(hand):
        spins = [Rotation.from_rotvec(axis * np.deg2rad(cfg["lid_open_degrees"]) * k / 4) for k in range(5)]
        lid_path = [make_pose(grip + [0, 0, 0.07], hand)] + [
            make_pose(hinge + spin.apply(grip - hinge), spin * hand) for spin in spins]
        press = [make_pose(above, hand), make_pose(above - [0, 0, cfg["press_depth_travel"]], hand)]
        return min(path_margin(planner, lid_path), path_margin(planner, press))

    yaw = rotation(root).as_euler("xyz")[2]
    hand = max((top_down(yaw), top_down(yaw + np.pi)), key=wrist_margin)

    planner.set_gripper(CLOSED, steps=5)
    if fault != "lid_closed":
        if cfg["lid_open"] == "drag":
            start = local(0.0, front_y + 0.035, plate_z).p
            planner.move_to(make_pose(start + [0, 0, 0.06], hand))
            planner.move_to(make_pose(start, hand), pos_tol=0.004)
            planner.speed = planner.contact_speed
            end = local(0.0, lid["hinge_y"] - cfg["lid_push_past"], plate_z + cfg["lid_push_lift"]).p
            opened = lambda: float(env.valve.get_qpos()[planner.idx, 1]) >= np.deg2rad(cfg["lid_open_degrees"])
            planner.move_to(make_pose(end, hand), max_steps=140, pos_tol=0.006, until=opened)
            planner.speed = planner.max_speed
        else:
            planner.set_gripper(OPEN, steps=4)
            planner.move_to(make_pose(grip + [0, 0, 0.07], hand))
            planner.speed = planner.contact_speed
            planner.move_to(make_pose(grip, hand), pos_tol=0.002)
            planner.set_gripper(cfg.get("lid_grip_width", 0.003), steps=6)
            for step in range(1, cfg["lid_arc_steps"] + 1):
                spin = Rotation.from_rotvec(axis * np.deg2rad(cfg["lid_open_degrees"]) * step / cfg["lid_arc_steps"])
                planner.move_to(make_pose(hinge + spin.apply(grip - hinge), spin * hand), max_steps=60, pos_tol=0.005)
                if env.valve.get_qpos()[planner.idx, 1] >= np.deg2rad(env.config["demo"]["lid_detent_trigger_degrees"]):
                    # Latch between arc segments, once the wrist has settled at its current pose.
                    engage = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                    engage[planner.idx] = True
                    env._engage_lid_detent(engage)
                    break
            planner.idle(10)
            planner.set_gripper(OPEN, steps=4)
            planner.target = planner.tcp()
            planner.pos_bias[:] = 0.0
            planner.rot_bias[:] = 0.0
            planner.speed = planner.max_speed
            planner.move_to(make_pose(planner.tcp().p + [0, 0, 0.05], rotation(planner.tcp())))
            planner.set_gripper(CLOSED, steps=4)

    if fault == "stopped_short":
        return planner.idle()
    press_hands = (top_down(yaw), top_down(yaw + np.pi))
    press_hand = max(press_hands, key=lambda h: path_margin(planner, [
        make_pose(above + [0, 0, 0.05], h), make_pose(above - [0, 0, cfg["press_depth_travel"]], h)
    ]))
    # Retreat before changing wrist orientation to keep the IK path clear of the open lid.
    planner.move_to(make_pose(above + [0, 0, 0.12], rotation(planner.tcp())))
    planner.move_to(make_pose(above + [0, 0, 0.05], press_hand))
    planner.move_to(make_pose(above, press_hand), pos_tol=0.004)
    planner.speed = planner.contact_speed
    planner.move_to(make_pose(above - [0, 0, cfg["press_depth_travel"]], press_hand), max_steps=120, pos_tol=0.002)
    planner.speed = planner.max_speed
    if fault is not None:
        return planner.idle()
    return planner.idle(cfg["hold_press_steps"], until_success=True)


# Pick up the key, insert it into the lock, and turn the barrel.
def solve_key_lock(planner, fault=None):
    env = planner.base
    cfg = env.config["demo"]
    if fault == "never_grasped":
        return planner.idle()
    key = env.key.pose.sp
    grip = (key * sapien.Pose(p=[0, 0, cfg["grasp_z"]])).p
    grasp = top_down(wrap(rotation(key).as_euler("xyz")[2], np.pi))
    planner.set_gripper(OPEN, steps=4)
    planner.move_to(make_pose(grip + [0, 0, cfg["approach_height"]], grasp))
    planner.speed = planner.contact_speed
    planner.move_to(make_pose(grip, grasp), pos_tol=0.002)
    planner.set_gripper(cfg["grasp_width"], steps=cfg["settle_steps"])
    planner.speed = planner.max_speed
    planner.move_to(make_pose(grip + [0, 0, cfg["lift_height"]], grasp))
    if fault == "dropped":
        planner.set_gripper(OPEN)
        return planner.idle()

    barrel = env.barrel.pose.sp
    slot_yaw = rotation(barrel).as_euler("xyz")[2]
    tip_z = env.keyway_height + (0.0 if fault == "not_inserted" else -cfg["insert_depth"])

    # Return the key tip pose.
    def key_tip():
        key_now = env.key.pose.sp
        return key_now, np.array(key_now.p)

    # Correct the hand from where it is so the key tip lands on the keyway, keeping the wrist pose reachable.
    def place_key(target_tip_z, max_steps, align_yaw):
        key_now, tip = key_tip()
        tcp = planner.tcp()
        yaw_error = slot_yaw - rotation(key_now).as_euler("xyz")[2]
        turn = Rotation.from_euler("z", wrap(yaw_error, np.pi)) if align_yaw else Rotation.identity()
        offset = turn.apply(tcp.p - tip)
        target = np.array([barrel.p[0] + offset[0], barrel.p[1] + offset[1], target_tip_z + offset[2]])
        planner.move_to(make_pose(target, turn * rotation(tcp)), max_steps=max_steps, pos_tol=0.002, rot_tol=0.02)
        return np.linalg.norm(key_tip()[1][:2] - barrel.p[:2])

    seated = env.keyway_height - env.config["success"]["insert_depth"]
    for _ in range(cfg["insert_tries"]):
        planner.speed = planner.max_speed
        for attempt in range(cfg["align_tries"]):
            if place_key(env.keyway_height + cfg["insert_clearance"], 140, attempt == 0) <= cfg["align_tolerance"]:
                break
            if planner.done:
                break
        planner.speed = planner.contact_speed
        place_key(tip_z, 160, False)
        if planner.done or float(env.key.pose.sp.p[2]) <= seated or fault == "not_inserted":
            break
    if fault == "not_inserted":
        planner.move_to(make_pose(planner.tcp().p + [0.06, 0.0, 0.02], rotation(planner.tcp())))
        planner.set_gripper(OPEN)
        return planner.idle()

    if fault != "not_turned":
        tcp = planner.tcp()
        target = np.deg2rad(env.config["success"]["turn_degrees"])
        for step in range(1, cfg["turn_steps"] + 1):
            if float(env.valve.get_qpos()[planner.idx, 0]) >= target:
                break
            part = Rotation.from_euler("z", np.deg2rad(cfg["turn_degrees"]) * step / cfg["turn_steps"])
            centre = np.array([barrel.p[0], barrel.p[1], tcp.p[2]])
            planner.move_to(make_pose(centre + part.apply(tcp.p - centre), part * rotation(tcp)),
                            max_steps=60, pos_tol=0.004, rot_tol=0.02)
    planner.set_gripper(OPEN, steps=cfg["settle_steps"])
    planner.speed = planner.max_speed
    planner.move_to(make_pose(planner.tcp().p + [0, 0, 0.08], rotation(planner.tcp())))
    return planner.idle(until_success=fault is None)


# Move the gripper along a bounded rotation sweep.
def sweep(planner, pose_at, angle, speed, acceleration, settle=15, until=None, near_end=None):
    planner.blocked = False
    peak = min(speed, np.sqrt(2 * acceleration * angle / np.pi))
    ramp = np.pi * peak / (2 * acceleration) if peak > 0 else 0.0
    cruise = max(0.0, (angle - peak * ramp) / peak) if peak > 0 else 0.0
    total = 2 * ramp + cruise
    ramped = lambda t: 0.5 * peak * (t - ramp / np.pi * np.sin(np.pi * t / ramp))
    steps = max(1, int(np.ceil(total / planner.dt)))
    for i in range(1, steps + 1):
        t = i * total / steps
        if near_end is not None and t >= total - near_end[0]:
            near_end[1]()
            near_end = None
        if t < ramp:
            s = ramped(t)
        elif t < ramp + cruise:
            s = 0.5 * peak * ramp + peak * (t - ramp)
        else:
            s = angle - ramped(total - t)
        planner.target = pose_at(min(s, angle))
        planner.servo()
        if planner.done or planner.blocked:
            return False
        if until is not None and until():
            planner.target = planner.tcp()
            break
    for _ in range(settle):
        if planner.servo(pos_tol=0.001, rot_tol=0.01) or planner.done:
            break
    return not planner.done


class ThreadTools:
    """Frames and moves shared by the threaded tasks (nuts, light bulb).

    The thread frame is the module frame as the part carries it: z along its axis, the origin the point that sits on
    the module origin once the part is seated; on_axis(height, turn) is that frame on the screw, `height` above the seat.
    """

    # Set up the state and resources used by this instance.
    def __init__(self, planner):
        self.planner = planner
        self.env = env = planner.base
        self.cfg = env.config["demo"]
        self.thread = env.config["thread"]
        self.idx = planner.idx
        self.valve = env.valve.pose.sp
        self.seat_inv = env.seat_pose_inv.sp
        self.symmetry = np.deg2rad(self.thread["grip_symmetry_degrees"])
        joint7 = env.agent.robot.get_qlimits()[self.idx, 6].cpu().numpy()
        self.j7_low, self.j7_high = float(joint7[0]) + 0.1, float(joint7[1]) - 0.1

    # Build a frame along the part axis.
    def axis_frame(self, pose):
        return pose * self.seat_inv

    # Project a pose onto the part axis.
    def on_axis(self, height, turn):
        return self.valve * sapien.Pose(p=[0, 0, height], q=Rotation.from_euler("z", turn).as_quat()[[3, 0, 1, 2]])

    # Return the part rotation.
    def part_turn(self, pose):
        return rotation(self.valve.inv() * self.axis_frame(pose)).as_euler("xyz")[2]

    # Height above the seat and turn of the part on the screw: the height the thread holds it at for its turn,.
    def threaded_state(self):
        return float(self.env.helix_height()[self.idx]), self.part_turn(self.env.threaded_nut.pose.sp)

    # Return the current wrist pose relative to the threaded part.
    def wrist(self):
        return float(self.env.agent.robot.get_qpos()[self.idx, 6])

    # Return the hand rotation.
    @staticmethod
    def hand_turn(hand):
        return Rotation.from_quat(np.r_[hand.q[1:], hand.q[0]]).as_euler("xyz")[2]

    # Align the gripper with the part before turning.
    @staticmethod
    def square(hand, height_offset, centre=True):
        held = Rotation.from_quat(np.r_[hand.q[1:], hand.q[0]])
        untilt = Rotation.align_vectors([[0.0, 0.0, -1.0]], [held.apply([0.0, 0.0, 1.0])])[0]
        p = [0.0, 0.0, height_offset] if centre else list(hand.p)
        return sapien.Pose(p=p, q=(untilt * held).as_quat()[[3, 0, 1, 2]])

    # Command the fingers and wait only until they stop: on the part, or at the commanded width.
    def grip(self, width, most_steps=20):
        planner, idx = self.planner, self.idx
        planner.gripper = width
        last = self.env.agent.robot.get_qpos()[idx, 7:9].cpu().numpy()
        for step in range(most_steps):
            planner.servo()
            now = self.env.agent.robot.get_qpos()[idx, 7:9].cpu().numpy()
            if step >= 2 and np.abs(now - last).max() < 2e-4:
                break
            last = now
        if width == CLOSED:
            planner.target = planner.tcp()
            planner.servo()

    # Lower and align the part until its thread engages.
    def engage(self, top_clearance, free_turn=False):
        planner, env, cfg = self.planner, self.env, self.cfg
        turn = self.part_turn(env.part.pose.sp)
        if free_turn:
            grasp = planner.tcp().inv() * self.axis_frame(env.part.pose.sp)
            outward = self.valve.p[:2] - planner.base.agent.robot.pose.sp.p[:2]
            outward = outward / np.linalg.norm(outward)
            top = self.thread["travel"] + top_clearance

            start = planner.tcp()

            # Score a candidate pose by reachability and joint margin.
            def score(t):
                hand = self.on_axis(top, t) * grasp.inv()
                pointing = rotation(hand).apply([0, 0, 1])[:2]
                turning = Slerp([0, 1], Rotation.concatenate([rotation(start), rotation(hand)]))
                path = [make_pose(np.array(start.p) + (np.array(hand.p) - start.p) * f, turning(f))
                        for f in np.linspace(0.125, 1, 8)]
                return path_margin(planner, path) > 0.1, float(pointing @ outward)

            turn = max(turn + np.deg2rad(np.arange(0, 360, 15)), key=score)

        # Find a hand pose for the target.
        def hand_for(height):
            loose = env.part.pose.sp
            grasp = planner.tcp().inv() * self.axis_frame(loose)
            twist = turn - self.part_turn(loose) if free_turn else 0.0
            return self.on_axis(height, self.part_turn(loose) + twist) * grasp.inv()

        top = self.thread["travel"]
        planner.move_to(hand_for(top + top_clearance))
        planner.speed = planner.contact_speed
        planner.move_to(hand_for(top + cfg["hover"]), pos_tol=0.0015, rot_tol=0.02)
        engaged = lambda: bool(env.thread_engaged[self.idx])
        planner.move_to(hand_for(top - 0.003), max_steps=40, pos_tol=0.001, until=engaged)
        planner.speed = planner.max_speed
        return engaged()

    # Turn the part downward.
    def turn_down(self, grip_height, centre=True):
        planner, cfg, thread, sym = self.planner, self.cfg, self.thread, self.symmetry
        most = np.deg2rad(cfg["turn_per_grasp_degrees"])
        speed, acceleration = cfg["spin_speed"], cfg["spin_acceleration"]
        up = lambda pose: sapien.Pose(p=[0, 0, cfg["regrasp_lift"]]) * pose
        square_turn = self.hand_turn(sapien.Pose(q=top_down(np.deg2rad(thread["flat_normal_degrees"])).as_quat()[[3, 0, 1, 2]]))

        def held():
            height, turn0 = self.threaded_state()
            return height, turn0, self.square(self.on_axis(height, turn0).inv() * planner.tcp(), grip_height, centre)

        def screw(height, turn0, hand):
            return lambda a: self.on_axis(height - thread["pitch"] * a / (2 * np.pi), turn0 - a) * hand

        def squaring(hand):
            return (square_turn - self.hand_turn(hand) + sym / 2) % sym - sym / 2

        while not planner.done and not planner.info["success"].any():
            height, turn0, hand = held()
            spin = screw(height, turn0, hand)
            remaining = height / thread["pitch"] * 2 * np.pi + np.deg2rad(20)
            turn = min(most, self.j7_high - self.wrist(), remaining)
            while turn > np.deg2rad(30) and path_margin(planner, [spin(a) for a in np.linspace(0, turn, 6)]) < 0.05:
                turn -= np.deg2rad(30)
            turn = max(turn, np.deg2rad(30))
            home = lambda: float(planner.info["thread_depth"].reshape(-1)[0]) >= thread["travel"] - 0.0001
            sweep(planner, spin, turn, speed, acceleration, settle=2, until=home)
            if home():
                planner.idle(cfg["settle_steps"] + planner.base.config["success"]["hold_steps"], until_success=True)
            if planner.info["success"].any():
                break
            if home():
                continue
            self.grip(cfg["regrasp_open_width"])
            height, turn0, hand = held()
            hand = sapien.Pose(p=[0, 0, grip_height], q=hand.q)
            offset = squaring(hand)
            planner.move_to(up(self.on_axis(height, turn0) * hand), max_steps=20, pos_tol=0.003)
            air = lambda a: up(self.on_axis(height, turn0 + a) * hand)
            steps = int(round(2 * np.pi / sym))
            options = [n for n in range(1, steps + 1) if self.wrist() - (n * sym + offset) >= self.j7_low] or [1]
            enough = [n for n in options if self.j7_high - (self.wrist() - (n * sym + offset)) >= most]
            for n in enough[:1] + sorted(set(options) - set(enough[:1]), reverse=True):
                back = n * sym + offset
                if path_margin(planner, [air(a) for a in np.linspace(0, back, max(2, int(back / 0.5) + 1))]) >= 0.05:
                    break
            lead = (ROBOT_CONFIG["gripper_delay_seconds"], lambda: setattr(planner, "gripper", CLOSED))
            sweep(planner, air, back, speed, acceleration, settle=2, near_end=None if cfg["regrasp_lift"] else lead)
            planner.move_to(self.on_axis(height, turn0 + back) * hand, max_steps=30, pos_tol=0.0015, rot_tol=0.02)
            self.grip(CLOSED)

        planner.idle(cfg["settle_steps"], until_success=True)
        planner.set_gripper(OPEN, steps=4)
        planner.move_to(make_pose(np.array(planner.tcp().p) + [0, 0, cfg["approach_height"]], rotation(planner.tcp())))
        return planner.idle(until_success=True)


# Pick up, align, and turn a nut down its screw.
def solve_thread(planner, fault=None):
    tools = ThreadTools(planner)
    env, cfg, valve = tools.env, tools.cfg, tools.valve
    flat = np.deg2rad(tools.thread["flat_normal_degrees"])

    loose = env.part.pose.sp
    yaw = rotation(tools.axis_frame(loose)).as_euler("xyz")[2] + flat
    grip = np.array([loose.p[0], loose.p[1], FRONT_TABLE_TOP_Z + cfg["grasp_height"]])
    to_board = valve.p[:2] - grip[:2]
    to_board_yaw = np.arctan2(to_board[1], to_board[0])

    # Score a candidate pick pose.
    def pick_score(k):
        hand = top_down(yaw + k * np.pi / 3)
        reach = path_margin(planner, [make_pose(grip + [0, 0, cfg["approach_height"]], hand), make_pose(grip, hand)])
        return reach > 0.05, abs(np.sin(yaw + k * np.pi / 3 - to_board_yaw)), reach

    hand = top_down(yaw + max(range(6), key=pick_score) * np.pi / 3)
    planner.set_gripper(cfg["pick_open_width"], steps=4)
    planner.move_to(make_pose(grip + [0, 0, cfg["approach_height"]], hand))
    planner.speed = planner.contact_speed
    planner.move_to(make_pose(grip, hand), pos_tol=0.002)
    planner.set_gripper(CLOSED, steps=cfg["settle_steps"])
    planner.speed = planner.max_speed
    planner.move_to(make_pose(grip + [0, 0, cfg["approach_height"]], hand))
    if not tools.engage(cfg["carry_clearance"]):
        return planner.idle(until_success=True)

    height, turn0 = tools.threaded_state()
    grip_height = (tools.on_axis(height, turn0).inv() * planner.tcp()).p[2]
    square_turn = tools.hand_turn(sapien.Pose(q=top_down(flat).as_quat()[[3, 0, 1, 2]]))
    tools.grip(cfg["regrasp_open_width"])
    hand = tools.square(tools.on_axis(height, turn0).inv() * planner.tcp(), grip_height)
    square_up = (square_turn - tools.hand_turn(hand) + np.pi / 6) % (np.pi / 3) - np.pi / 6
    hand = sapien.Pose(q=Rotation.from_euler("z", square_up).as_quat()[[3, 0, 1, 2]]) * hand
    planner.move_to(tools.on_axis(height, turn0) * hand, max_steps=20, pos_tol=0.0015, rot_tol=0.02)
    tools.grip(CLOSED)
    return tools.turn_down(grip_height)


# Stand a loose threaded part upright before engaging it.
def solve_lying_thread(planner, fault=None):
    tools = ThreadTools(planner)
    env, cfg = tools.env, tools.cfg
    grip = sapien.Pose(p=[0, 0, cfg["grip_height"]])
    lift = np.array([0, 0, cfg["approach_height"]])

    frame = tools.axis_frame(env.part.pose.sp)
    centre = np.array((frame * grip).p)
    axis = rotation(frame).apply([0, 0, 1])
    across = np.arctan2(axis[1], axis[0]) + np.pi / 2
    hand = max((top_down(across + k * np.pi) for k in range(2)),
               key=lambda h: path_margin(planner, [make_pose(centre + lift, h), make_pose(centre, h)]))
    planner.set_gripper(cfg["grip_open_width"], steps=4)
    planner.move_to(make_pose(centre + lift, hand))
    planner.speed = planner.contact_speed
    planner.move_to(make_pose(centre, hand), pos_tol=0.002)
    planner.set_gripper(CLOSED, steps=cfg["settle_steps"])
    planner.speed = planner.max_speed
    planner.move_to(make_pose(centre + lift, hand))
    above = np.array(tools.on_axis(tools.thread["travel"] + cfg["carry_clearance"] + cfg["grip_height"], 0.0).p)
    outward = above[:2] - planner.base.agent.robot.pose.sp.p[:2]
    outward = outward / np.linalg.norm(outward)
    in_hand = rotation(planner.tcp()).inv().apply(rotation(tools.axis_frame(env.part.pose.sp)).apply([0, 0, 1]))

    # Stand the part upright.
    def stand_up(yaw, sign):
        level = top_down(yaw) * Rotation.from_euler("y", sign * np.pi / 2)
        return level, level.apply(in_hand)[2] > 0.5, float(level.apply([0, 0, 1])[:2] @ outward)

    # Score a candidate pose by reachability and joint margin.
    def score(choice):
        yaw, sign = choice
        level, upright, pointing = stand_up(yaw, sign)
        carry = interpolated(planner.tcp(), make_pose(above, top_down(yaw)))
        reach = path_margin(planner, carry + interpolated(carry[-1], make_pose(above, level)))
        return upright, pointing > 0.3, reach

    yaw, sign = max(((y, s) for y in np.deg2rad(np.arange(0, 360, 15)) for s in (1, -1)), key=score)
    planner.move_to(make_pose(above, top_down(yaw)))
    planner.move_to(make_pose(above, stand_up(yaw, sign)[0]), max_steps=80)
    if not tools.engage(cfg["carry_clearance"], free_turn=True):
        return planner.idle(until_success=True)

    planner.set_gripper(cfg["grip_open_width"], steps=2)
    planner.move_to(sapien.Pose(p=lift + [0, 0, cfg["grip_height"]]) * planner.tcp())

    height, turn0 = tools.threaded_state()
    on_screw = tools.on_axis(height, turn0)

    # Move to a grasp pose.
    def grasp_at(yaw):
        return on_screw * sapien.Pose(p=grip.p, q=top_down(yaw).as_quat()[[3, 0, 1, 2]])

    # Return the wrist position after the motion.
    def wrist_after(yaw):
        path = interpolated(planner.tcp(), sapien.Pose(p=lift) * grasp_at(yaw)) + [grasp_at(yaw)]
        q = ik_chain(planner, path)
        if q is None or path_margin(planner, path) < 0.3:
            return np.inf
        return q[6]

    yaw = min(np.deg2rad(np.arange(0, 360, 10)), key=wrist_after)
    planner.move_to(sapien.Pose(p=lift) * grasp_at(yaw))
    planner.speed = planner.contact_speed
    planner.move_to(grasp_at(yaw), pos_tol=0.0015, rot_tol=0.02)
    planner.speed = planner.max_speed
    tools.grip(CLOSED)
    return tools.turn_down(cfg["grip_height"], centre=False)


# Grasp the drawer, align its pins, and slide it into the guides.
def solve_drawer(planner, fault=None):
    env = planner.base
    cfg = env.config["demo"]
    idx = planner.idx
    part = env.part.pose.sp
    body = env.part_body_frame
    handle = part * sapien.Pose(p=(body @ np.r_[cfg["handle"], 1.0])[:3])
    across = rotation(part).apply(body[:3, :3] @ [0.0, 0.0, 1.0])
    closing = TOP_DOWN.apply([0.0, 1.0, 0.0])
    yaw = np.arctan2(across[1], across[0]) - np.arctan2(closing[1], closing[0])
    module = env.valve.pose.sp
    seat = Pose.create(env.part_seated_pose).sp
    pin = np.asarray(cfg["pin_axis"])

    # The drawer's pose with its pins moved by an offset from the seat and tilted about them.
    def drawer_at(waypoint):
        *offset, tilt = waypoint
        turn = Rotation.from_euler("y", np.deg2rad(tilt)).as_quat()[[3, 0, 1, 2]]
        return module * sapien.Pose(p=pin + offset) * sapien.Pose(q=turn) * sapien.Pose(p=-pin) * seat

    # The hand poses that lower the drawer along the guide path for a grasp.
    def carried(hand_in_part):
        return [drawer_at(waypoint) * hand_in_part for waypoint in cfg["guide_path"]]

    # Waypoints from start to end, close enough for the hand to trace the tilt about the pins.
    def along(start, end):
        start, end = np.asarray(start), np.asarray(end)
        steps = max(1, int(np.ceil(max(np.linalg.norm(end[:3] - start[:3]) / cfg["guide_step"],
                                       abs(end[3] - start[3]) / cfg["guide_step_degrees"]))))
        return [start + (end - start) * k / steps for k in range(1, steps + 1)]

    # Score a candidate pose by reachability and joint margin.
    def score(turn):
        grasp = make_pose(handle.p, top_down(yaw + turn))
        path = carried(part.inv() * grasp)
        return min(path_margin(planner, [make_pose(handle.p + [0, 0, cfg["approach_height"]], rotation(grasp)), grasp]),
                   path_margin(planner, [path[0]] + path))

    grasp = make_pose(handle.p, top_down(yaw + max((0.0, np.pi), key=score)))
    planner.set_gripper(cfg["pick_open_width"], steps=4)
    planner.move_to(make_pose(handle.p + [0, 0, cfg["approach_height"]], rotation(grasp)))
    planner.speed = planner.contact_speed
    planner.move_to(grasp, pos_tol=0.002)
    planner.set_gripper(CLOSED, steps=cfg["settle_steps"])
    if fault == "never_grasped":
        planner.set_gripper(OPEN)
        return planner.idle()
    planner.speed = planner.max_speed
    planner.move_to(make_pose(planner.tcp().p + [0, 0, cfg["approach_height"]], rotation(planner.tcp())))
    above = carried(env.part.pose.sp.inv() * planner.tcp())[0]
    planner.move_to(make_pose(above.p + [0, 0, cfg["approach_height"] - cfg["guide_path"][0][2]], rotation(above)))
    planner.speed = cfg["guide_speed"]

    # Adjust the grasp toward the target placement pose.
    def aim(goal):
        for _ in range(cfg["aim_passes"]):
            miss = np.array(goal.p) - env.part.pose.sp.p
            planner.move_to(make_pose(planner.target.p + miss, rotation(planner.target)), max_steps=15,
                            pos_tol=0.0005, rot_tol=0.01)

    path = cfg["guide_path"]
    planner.move_to(drawer_at(path[0]) * (env.part.pose.sp.inv() * planner.tcp()), max_steps=40, pos_tol=0.001,
                    rot_tol=0.01)
    aim(drawer_at(path[0]))
    for start, end in zip(path, path[1:]):
        for waypoint in along(start, end):
            planner.move_to(drawer_at(waypoint) * (env.part.pose.sp.inv() * planner.tcp()), max_steps=4,
                            pos_tol=0.001, rot_tol=0.01)
        aim(drawer_at(end))
        if planner.done:
            break
    planner.set_gripper(cfg["pick_open_width"], steps=cfg["settle_steps"])
    planner.speed = planner.max_speed
    planner.move_to(make_pose(planner.tcp().p + [0, 0, cfg["approach_height"]], rotation(planner.tcp())))
    return planner.idle(until_success=fault is None)


# Lift the free spring eye and place it over the target peg.
def solve_shock_absorber(planner, fault=None):
    env = planner.base
    cfg = env.config["demo"]
    if fault == "never_grasped":
        return planner.idle()
    idx = planner.idx
    pivot = env.shock_segments[0].pose.sp
    offset = np.asarray(env.free_eye_offset, dtype=np.float64)
    turn = lambda r: sapien.Pose(q=r.as_quat()[[3, 0, 1, 2]])

    # The free end's link pose swung by phi, lifted by theta and the spring stretched by extension.
    def link(phi, theta, extension):
        return pivot * turn(Rotation.from_euler("z", phi) * Rotation.from_euler("y", -theta)) * sapien.Pose(p=[extension, 0, 0])

    # Return the current pose of the free spring eye.
    def eye(phi, theta, extension):
        return np.array((link(phi, theta, extension) * sapien.Pose(p=offset)).p)

    peg = np.array(env.peg.pose.sp.p)
    to_peg = pivot.inv() * sapien.Pose(p=peg)
    peg_phi = np.arctan2(to_peg.p[1], to_peg.p[0])
    peg_top = env.pivot_height + env.config["shock_absorber"]["peg_top"]
    eye_top = peg_top + cfg["eye_half_height"] + cfg["peg_clearance"]

    # Spring extension that puts the eye over the peg's distance from the clamped eye.
    def stretch_to(phi, theta):
        extension = 0.0
        for _ in range(3):
            reach = np.linalg.norm((eye(phi, theta, extension) - pivot.p)[:2])
            extension += (np.linalg.norm(to_peg.p[:2]) - reach) / np.cos(theta)
        return extension

    theta_up = 0.0
    for _ in range(3):
        theta_up += (eye_top - eye(0.0, theta_up, 0.0)[2]) / np.linalg.norm(offset[[0, 2]])

    free = eye(0.0, 0.0, 0.0)
    axis = rotation(pivot).apply([1.0, 0.0, 0.0])
    across = np.arctan2(axis[1], axis[0]) + np.pi / 2
    grip = free + [0, 0, cfg["grip_height"]]
    lift = np.array([0, 0, cfg["approach_height"]])

    # Hand poses along the whole move for a grasp (the hand in the free end's frame) and a swing (to the peg's.
    def plan(held, swing):
        up = [link(0.0, theta_up * f, 0.0) * held for f in (0.5, 1.0)]
        steps = max(2, int(np.ceil(abs(swing) / np.deg2rad(cfg["swing_step_degrees"]))))
        around = [link(swing * f, theta_up, stretch_to(swing * f, theta_up) * f) * held for f in np.linspace(0, 1, steps + 1)[1:]]
        down = [link(swing, theta_up * f, stretch_to(swing, theta_up * f)) * held for f in (0.5, 0.0)]
        return up, around, down

    swings = [wrap(peg_phi, 2 * np.pi)]
    swings.append(swings[0] - np.sign(swings[0]) * 2 * np.pi)
    options = [(top_down(across + k * np.pi), swing) for k in range(2) for swing in swings]

    def margin(option):
        hand, swing = option
        up, around, down = plan(link(0.0, 0.0, 0.0).inv() * make_pose(grip, hand), swing)
        approach = [make_pose(grip + lift, hand), make_pose(grip, hand)]
        return path_margin(planner, approach + up + around + down) > 0.05, -abs(swing)

    hand, swing = max(options, key=margin)
    planner.set_gripper(cfg["open_width"], steps=4)
    planner.move_to(make_pose(grip + lift, hand))
    planner.speed = planner.contact_speed
    planner.move_to(make_pose(grip, hand), pos_tol=0.002)
    planner.set_gripper(CLOSED, steps=cfg["settle_steps"])

    up, around, down = plan(env.shock.pose.sp.inv() * planner.tcp(), swing)
    for pose in up:
        planner.move_to(pose, max_steps=40, pos_tol=0.002, rot_tol=0.03)
    planner.speed = planner.max_speed
    for pose in around:
        planner.move_to(pose, max_steps=10, pos_tol=0.004, rot_tol=0.05)
    planner.speed = planner.contact_speed
    planner.move_to(around[-1], max_steps=40, pos_tol=0.001, rot_tol=0.02)
    for _ in range(cfg["aim_passes"]):
        miss = peg[:2] - env.free_eye()[idx, :2].cpu().numpy()
        if np.linalg.norm(miss) <= cfg["aim_tolerance"]:
            break
        planner.move_to(make_pose(planner.target.p + np.r_[miss, 0.0], rotation(planner.target)), max_steps=25,
                        pos_tol=0.0005, rot_tol=0.01)
    if fault == "not_inserted":
        planner.set_gripper(OPEN)
        return planner.idle()
    shift = np.array(planner.target.p) - np.array(around[-1].p)
    for pose in down:
        planner.move_to(sapien.Pose(p=shift) * pose, max_steps=40, pos_tol=0.0015, rot_tol=0.02)
    planner.set_gripper(cfg["open_width"], steps=cfg["settle_steps"])
    planner.target = planner.tcp()
    planner.speed = planner.max_speed
    planner.move_to(make_pose(planner.tcp().p + lift, rotation(planner.tcp())))
    return planner.idle(until_success=True)


SOLVERS = {
    "shock_absorber": solve_shock_absorber,
    "drawer": solve_drawer,
    "light_bulb": solve_lying_thread,
    "peg_insertion": solve_thread,
    "thread_m8": solve_thread,
    "thread_m30": solve_thread,
    "ball_valve": solve_ball_valve,
    "ball_valve_ring": solve_ball_valve,
    "circuit_breaker": solve_circuit_breaker,
    "gate_valve_small": solve_gate_valve,
    "gate_valve_large": solve_gate_valve,
    "button_cover": solve_button_cover,
    "key_lock": solve_key_lock,
}

# Per-env view of a batched info dict, keeping one-element tensors so solver code can call .item().
def slice_info(info, idx):
    return {key: value[idx : idx + 1] if torch.is_tensor(value) and value.dim() > 0 else value
            for key, value in info.items()}


class BatchStepper:
    """Steps a batched env once every running solver thread has submitted its env's action."""

    # Set up the state and resources used by this instance.
    def __init__(self, env, on_step=None):
        self.env = env
        self.base = env.unwrapped
        self.num_envs = self.base.num_envs
        self.on_step = on_step
        self.actions = np.zeros((self.num_envs, env.action_space.shape[-1]), dtype=np.float32)
        self.absolute = absolute_control(env)
        lock = threading.Lock()
        self.submitted_cv = threading.Condition(lock)
        self.stepped_cv = threading.Condition(lock)
        self.running = set()
        self.submitted = set()
        self.count = 0
        self.result = None

    # Advance one planner step.
    def step_fn(self, idx):
        # Advance the simulation and update the active recording.
        def step(action):
            with self.submitted_cv:
                self.actions[idx] = action
                self.submitted.add(idx)
                count = self.count
                if self.running <= self.submitted:
                    self.submitted_cv.notify()
                self.stepped_cv.wait_for(lambda: self.count > count)
                terminated, truncated, info = self.result
            return terminated[idx : idx + 1], truncated[idx : idx + 1], slice_info(info, idx)

        return step

    # Run solve(idx) for every env in its own thread, stepping the batch until they all return.
    def run(self, solve):
        errors = {}

        # Run one solver in its assigned environment row.
        def work(idx):
            _worker.idx = idx
            try:
                solve(idx)
            except BaseException as error: 
                errors[idx] = error
            finally:
                with self.submitted_cv:
                    self.running.discard(idx)
                    self.submitted.discard(idx)
                    self.submitted_cv.notify()

        threads = [threading.Thread(target=work, args=(idx,), daemon=True) for idx in range(self.num_envs)]
        self.running = set(range(self.num_envs))
        for thread in threads:
            thread.start()
        while True:
            with self.submitted_cv:
                self.submitted_cv.wait_for(lambda: self.running <= self.submitted)
                if not self.running:
                    break
                stepped = set(self.submitted)
                obs, _, terminated, truncated, info = self.env.step(torch.from_numpy(self.actions))
                info = {key: value.cpu() if torch.is_tensor(value) else value for key, value in info.items()}
                self.result = (terminated.cpu(), truncated.cpu(), info)
                if self.on_step is not None:
                    self.on_step(obs, info, self.actions, stepped)
                if not self.absolute:
                    self.actions[[i for i in range(self.num_envs) if i not in self.running]] = 0.0
                self.submitted = set()
                self.count += 1
                self.stepped_cv.notify_all()
        for thread in threads:
            thread.join()
        if errors:
            raise next(iter(errors.values()))


# Flatten a batched observation into {"agent/qpos": array, ...}, keeping only what the exporter reads.
def to_numpy_obs(obs, prefix=""):
    flat = {}
    for key, value in obs.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= to_numpy_obs(value, f"{path}/")
        elif path.startswith(("agent/", "extra/")) or path.endswith("/rgb"):
            flat[path] = value.cpu().numpy()
    return flat


class Recorder:
    """Write episodes in the raw layout the exporter reads: traj_<id>/obs, actions, success."""

    # Set up the state and resources used by this instance.
    def __init__(self, path, env_id, obs_mode, sim_backend, max_episode_steps, image_stride=1):
        self.path = path
        self.image_stride = image_stride
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")
        self.json = {
            "env_info": {
                "env_id": env_id,
                "max_episode_steps": max_episode_steps,
                "env_kwargs": {"obs_mode": obs_mode, "control_mode": CONTROL_MODE, "sim_backend": sim_backend},
                "domain_randomization": CONFIG["domain_randomization"],
            },
            "source_type": "motionplanning",
            "source_desc": f"Scripted Cartesian waypoints in {CONTROL_MODE}",
            "episodes": [],
        }
        self.episodes = []

    # Initialize the episode buffers and output metadata.
    def start(self, obs, seeds):
        flat = to_numpy_obs(obs)
        self.seeds = seeds
        self.buffers = [{"obs": [{k: v[j] for k, v in flat.items()}], "actions": [], "success": []}
                        for j in range(len(seeds))]

    # Append one step of observations and actions to the recording.
    def add(self, obs, info, actions, stepped):
        flat = to_numpy_obs(obs)
        for j in stepped:
            if j < len(self.buffers):
                buffer = self.buffers[j]
                keep_images = len(buffer["obs"]) % self.image_stride == 0
                buffer["obs"].append({key: value[j] for key, value in flat.items()
                                      if keep_images or not key.endswith("/rgb")})
                buffer["actions"].append(actions[j].copy())
                buffer["success"].append(bool(info["success"][j]))

    # Write the completed episode and its camera recordings.
    def save(self, j, prompt, success=True, metadata=None):
        buffer = self.buffers[j]
        group = self.file.create_group(f"traj_{len(self.episodes)}", track_order=True)
        group.attrs["image_stride"] = self.image_stride
        for key in buffer["obs"][0]:
            data = np.stack([step[key] for step in buffer["obs"] if key in step])
            chunks = (min(16, len(data)), *data.shape[1:]) if key.endswith("/rgb") else True
            group.create_dataset(f"obs/{key}", data=data, compression="gzip", compression_opts=5, chunks=chunks)
        group.create_dataset("actions", data=np.stack(buffer["actions"]))
        group.create_dataset("success", data=np.array(buffer["success"]))
        episode = {
            "episode_id": len(self.episodes),
            "episode_seed": int(self.seeds[j]),
            "control_mode": CONTROL_MODE,
            "elapsed_steps": len(buffer["actions"]),
            "success": bool(success),
            "prompt": prompt,
        }
        if metadata:
            episode.update(metadata)
        self.episodes.append(episode)

    # Compose recorded camera frames for the demonstration video.
    def frames(self, j, cameras):
        return np.stack([np.concatenate([step[f"sensor_data/{c}/rgb"] for c in cameras], axis=1)
                         for step in self.buffers[j]["obs"] if f"sensor_data/{cameras[0]}/rgb" in step])

    # Flush the recorder and close its output files.
    def close(self):
        self.json["episodes"] = self.episodes
        self.path.with_suffix(".json").write_text(json.dumps(self.json, indent=2))
        self.file.close()


# Yield recorded episodes resampled to config["hz"] with absolute actions, one observation per action.
def resample_episodes(h5_path, control_hz, config):
    hz = config["hz"]
    if control_hz % hz:
        raise ValueError(f"export hz {hz} must divide the {control_hz} Hz control rate")
    stride = control_hz // hz
    episodes = {e["episode_id"]: e for e in json.loads(h5_path.with_suffix(".json").read_text())["episodes"]}
    base_inv = rotation(ROBOT_BASE_POSE).inv()
    with h5py.File(h5_path, "r") as src:
        for name, traj in src.items():
            episode = episodes[int(name.split("_")[1])]
            obs = traj["obs"]
            idx = np.arange(0, len(traj["actions"]) + 1, stride)
            qpos = obs["agent/qpos"][()][idx]
            tcp = obs["extra/tcp_pose"][()][idx]
            position = base_inv.apply(tcp[:, :3] - ROBOT_BASE_POSE.p)
            quat = (base_inv * Rotation.from_quat(tcp[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
            tcp_base = np.concatenate([position, quat], axis=1).astype(np.float32)
            actions = traj["actions"][()][idx[:-1]]
            data = dict(
                timestamp=(idx[:-1] / control_hz).astype(np.float32),
                qpos=qpos[:-1],
                qvel=obs["agent/qvel"][()][idx[:-1]],
                tcp_pose=tcp_base[:-1],
                action_gripper=actions[:, -1:],
                success=traj["success"][()][idx[1:] - 1],
            )
            for key in ("valve_angle", "valve_velocity", "module_angle", "module_velocity"):
                obs_key = f"extra/{key}"
                if obs_key in obs:
                    data[key] = obs[obs_key][()][idx[:-1]]
            if "ee_pose" in config["actions"]:
                data["action_ee_pose"] = tcp_base[1:]
            if "joint_position" in config["actions"]:
                data["action_joint_position"] = qpos[1:, :7]
            if "ee_target" in config["actions"]:
                data["action_ee_target"] = actions[:, :7]
            images = {}
            for camera in config["cameras"]:
                key = f"sensor_data/{camera}/rgb"
                if key not in obs:
                    raise KeyError(f"{camera} images were not recorded; use an obs mode that includes rgb")
                image_stride = int(traj.attrs.get("image_stride", 1))
                if stride % image_stride:
                    raise ValueError(f"{hz} Hz needs images every {stride} steps but they were kept every {image_stride}")
                images[camera] = obs[key][()][idx[:-1] // image_stride]
            meta = dict(prompt=episode.get("prompt", ""), episode_seed=int(episode["episode_seed"]))
            yield name, meta, data, images


# Write episodes.h5 with low-dimensional data and one MP4 per camera per episode under videos/.
def write_hdf5(out_dir, episodes, hz):
    with h5py.File(out_dir / "episodes.h5", "w") as f:
        f.attrs.update(hz=hz, pose_frame="robot base, quaternion wxyz")
        for name, meta, data, images in episodes:
            group = f.create_group(name)
            group.attrs.update(meta)
            for key, value in data.items():
                group[key] = value
            for camera, frames in images.items():
                video = Path("videos") / camera / f"{name}.mp4"
                (out_dir / video).parent.mkdir(parents=True, exist_ok=True)
                iio.imwrite(out_dir / video, frames, fps=hz, codec="libx264")
                group.attrs[f"{camera}_video"] = str(video)


# Write recorded observations and actions to a Zarr dataset.
def write_zarr(out_dir, episodes, hz):
    root = zarr.open_group(out_dir / "dataset.zarr", mode="w", zarr_format=2)
    images_codec = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    arrays, ends, prompts, seeds, names = {}, [], [], [], []
    for name, meta, data, images in episodes:
        for key, value in {**data, **images}.items():
            if key not in arrays:
                chunks = (1, *value.shape[1:]) if key in images else (1024, *value.shape[1:])
                arrays[key] = root.create_array(
                    f"data/{key}", shape=(0, *value.shape[1:]), chunks=chunks, dtype=value.dtype,
                    compressors=images_codec if key in images else "auto",
                )
            arrays[key].append(value)
        ends.append((ends[-1] if ends else 0) + len(data["timestamp"]))
        prompts.append(meta["prompt"])
        seeds.append(meta["episode_seed"])
        names.append(name)
    root.create_array("meta/episode_ends", data=np.array(ends, dtype=np.int64))
    root.attrs.update(hz=hz, pose_frame="robot base, quaternion wxyz", prompts=prompts, episode_seeds=seeds, episode_names=names)


# Convert a pose to the real robot frame.
def real_pose(pose_wxyz):
    pose = np.asarray(pose_wxyz, dtype=np.float64)
    quaternion = pose[:, [4, 5, 6, 3]]
    matrix = Rotation.from_quat(quaternion).as_matrix()
    trace = np.trace(matrix, axis1=1, axis2=2)
    axis = np.argmax(np.diagonal(matrix, axis1=1, axis2=2), axis=1)
    lead = np.where(trace > 0, quaternion[:, 3], quaternion[np.arange(len(pose)), axis])
    quaternion = quaternion * np.where(lead < 0, -1.0, 1.0)[:, None]
    return np.concatenate([pose[:, :3], quaternion], axis=1)


# Export recorded episodes in the real robot data layout.
def write_oopsie(out_dir, episodes, hz):
    profile = dict(policy_name="scripted", robot_name="franka_research_3", gripper_name="franka_hand", is_biarm=False,
                   uses_mobile_base=False, control_freq=hz, camera_names=["front", "wrist"],
                   robot_state_keys=["joint_position", "joint_velocity", "cartesian_position", "gripper_position"],
                   robot_state_orientation_representation="quat", action_space=["cartesian_position", "gripper_position"],
                   orientation_representation="quat", controller="pd_ee_target_pose")
    for name, meta, data, images in episodes:
        stem = f"sim_{meta['episode_seed']:06d}"
        with h5py.File(out_dir / f"{stem}.h5", "w") as f:
            f.attrs.update(episode_id=stem, lab_id="sim", language_instruction=meta["prompt"], operator_name="scripted",
                           robot_profile=json.dumps(profile), schema="oopsiedata_format_v1", timestamp=0.0)
            states = f.create_group("observations/robot_states")
            states["joint_position"] = data["qpos"][:, :7].astype(np.float64)
            states["joint_velocity"] = data["qvel"][:, :7].astype(np.float64)
            states["cartesian_position"] = real_pose(data["tcp_pose"])
            states["gripper_position"] = data["qpos"][:, 7:9].sum(1, keepdims=True).astype(np.float64)
            f["actions/cartesian_position"] = real_pose(data["action_ee_target"])
            f["actions/gripper_position"] = 2 * data["action_gripper"].astype(np.float64)
            annotation = f.create_group("episode_annotations/scripted")
            annotation.attrs.update(source="scripted", success=1.0, taxonomy=json.dumps({"outcome": "success"}))
            for camera, real_name in (("base_camera", "front"), ("wrist_camera", "wrist")):
                video = f"{stem}_{real_name}.mp4"
                iio.imwrite(out_dir / video, images[camera], fps=hz, codec="libx264")
                f[f"observations/video_paths/{real_name}"] = video


# Export the recording in config["format"] to <recording dir>/export_<format>_<hz>hz/.
def export_dataset(h5_path, control_hz, config):
    writers = {"hdf5": write_hdf5, "zarr": write_zarr, "oopsie": write_oopsie}
    if config["format"] not in writers:
        raise ValueError(f"export format must be one of {sorted(writers)}")
    if config["format"] == "oopsie" and "ee_target" not in config["actions"]:
        raise ValueError("the oopsie format needs ee_target in export.actions")
    out_dir = h5_path.parent / f"export_{config['format']}_{config['hz']}hz"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    writers[config["format"]](out_dir, resample_episodes(h5_path, control_hz, config), config["hz"])
    return out_dir


# Parse command line options and run the requested task workflow.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("-n", "--num-traj", type=int, default=DATA_CONFIG["recording"]["num_episodes"])
    parser.add_argument("--record-dir", default="demos")
    parser.add_argument("--obs-mode", default=DATA_CONFIG["recording"]["obs_mode"])
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--showcase-video", action="store_true",
                        help="also write a showcase MP4 per episode (exterior view beside the real front and wrist "
                             "images) under <record-dir>/<env>/showcase, timeouts included and flagged")
    parser.add_argument("--image-hz", type=int, default=None,
                        help="keep camera images at this rate (must divide the control rate); exports above it are skipped")
    parser.add_argument("--viewer", action="store_true", help="show the demos live in the SAPIEN viewer")
    parser.add_argument("--num-envs", type=int, default=DATA_CONFIG["recording"]["num_envs"],
                        help="robots simulated in parallel, each on its own seed")
    parser.add_argument("--seed", type=int, default=0, help="first episode seed")
    parser.add_argument("--max-attempts", type=int, default=DATA_CONFIG["recording"]["max_attempts"])
    parser.add_argument("--sim-backend", default=DATA_CONFIG["recording"]["sim_backend"], choices=["gpu", "cpu"])
    parser.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction,
                        default=CONFIG["domain_randomization"],
                        help="robot and camera latencies plus visual variation (config/config.yaml domain_randomization)")
    args = parser.parse_args()

    set_domain_randomization(args.domain_randomization)
    env_id = TASKS[args.task]
    num_envs = 1 if args.viewer else max(1, args.num_envs)
    env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode=args.obs_mode,
        control_mode=CONTROL_MODE,
        render_mode="human" if args.viewer else "sensors",
        sim_backend=args.sim_backend,
        showcase_camera=args.showcase_video,
    )
    out_dir = Path(args.record_dir) / env_id
    showcase = ShowcaseVideo(env, args.task, out_dir / "showcase") if args.showcase_video else None
    control_hz = env.unwrapped.control_freq
    image_stride = control_hz // args.image_hz if args.image_hz else 1
    if args.image_hz and control_hz % args.image_hz:
        parser.error(f"--image-hz must divide the {control_hz} Hz control rate")
    recorder = Recorder(out_dir / RECORDING_NAME, env_id, args.obs_mode, args.sim_backend,
                        env.unwrapped.config["episode"]["max_episode_steps"], image_stride)
    cameras = list(env.unwrapped.config["cameras"])
    solver = SOLVERS[args.task]

    # Handle a completed simulation step.
    def on_step(obs, info, actions, stepped):
        recorder.add(obs, info, actions, stepped)
        if showcase is not None:
            showcase.capture(obs, sorted(stepped))
        if args.viewer:
            env.render()

    stepper = BatchStepper(env, on_step)
    saved = attempts = 0
    for first in range(0, args.max_attempts, num_envs):
        if saved >= args.num_traj:
            break
        seeds = list(range(args.seed + first, args.seed + min(first + num_envs, args.max_attempts)))
        attempts += len(seeds)
        obs, _ = env.reset(seed=seeds + [seeds[-1]] * (num_envs - len(seeds)))
        recorder.start(obs, seeds)
        if showcase is not None:
            showcase.start(obs, range(len(seeds)))
        outcomes = {}
        info = env.unwrapped.get_info()

        # Dispatch to the solver for the selected task.
        def solve(idx):
            if idx < len(seeds):
                planner = CartesianPlanner(env, idx, stepper.step_fn(idx), slice_info(info, idx))
                outcomes[idx] = solver(planner)

        stepper.run(solve)
        for idx, result in sorted(outcomes.items()):
            reflexes = [key for key in REFLEX_KEYS if bool(result[key].item())]
            success = bool(result["success"].item()) and not reflexes
            steps = len(recorder.buffers[idx]["actions"])
            prompt = env.unwrapped.prompts[idx]
            if success and saved < args.num_traj:
                if args.save_video:
                    video_path = out_dir / f"{saved}.mp4"
                    iio.imwrite(video_path, recorder.frames(idx, cameras), fps=control_hz // image_stride, codec="libx264")
                recorder.save(idx, prompt)
                saved += 1
            reflex = f" reflex={','.join(reflexes)}" if reflexes else ""
            print(f"seed={seeds[idx]} success={success}{reflex} steps={steps} prompt={prompt!r}")
            if showcase is not None:
                path = showcase.save(idx, seeds[idx], result)
                print(f"  showcase: {path if path else 'dropped, the robot was stuck at the timeout'}")
    print(f"saved {saved}/{attempts} successful trajectories to {out_dir}")
    recorder.close()
    env.close()
    configs = []
    for key in ("export", "real_export"):
        if not (DATA_CONFIG[key]["enabled"] and saved):
            continue
        if (control_hz // DATA_CONFIG[key]["hz"]) % image_stride:
            print(f"skipped the {DATA_CONFIG[key]['hz']} Hz {key}: images were kept at {control_hz // image_stride} Hz")
            continue
        configs.append(DATA_CONFIG[key])
    with ProcessPoolExecutor(max(1, len(configs)), mp_context=multiprocessing.get_context("spawn")) as pool:
        for out in pool.map(export_dataset, [recorder.path] * len(configs), [control_hz] * len(configs), configs):
            print(f"exported {out}")


if __name__ == "__main__":
    main()
