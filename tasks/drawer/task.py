# Simulates grasping a loose drawer, aligning its pins with the guides, and sliding it into place.
# Created: 2026-10-23

import numpy as np
import sapien
import torch

from env.fr3_table_env import load_task_config
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from tasks.hiveboard_assembly import HiveboardAssemblyFR3Env
from tasks.hiveboard_thread import quaternion_yaw_tilt

CONFIG = load_task_config(__file__)


@register_env("DrawerFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class DrawerFR3Env(HiveboardAssemblyFR3Env):
    """Stages: drawer grasped (1), aligned with the opening, its pins in the guides (2), fully inserted (3)."""

    config = CONFIG
    failure_keys = ("timeout",)

    # Create the loose drawer and its fixed guide structure.
    def _build_module(self):
        super()._build_module()
        for part in self.part._objs:
            body = part.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            body.set_sleep_threshold(0.0)
        self.seat_pose_inv = Pose.create(self.part_seated_pose.inv())
        self.ever_aligned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_seated = torch.zeros_like(self.ever_aligned)
        self.last_part_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.part_still = torch.zeros_like(self.ever_aligned)

    # Reset the drawer pose and alignment history for a new episode.
    def _initialize_module(self, env_idx: torch.Tensor):
        super()._initialize_module(env_idx)
        self.ever_aligned[env_idx] = False
        self.ever_seated[env_idx] = False
        self.part_still[env_idx] = False
        self.last_part_pose[env_idx] = self.part.pose.raw_pose[env_idx]

    # Where the drawer's pin axis is relative to its seat, in the module frame.
    def pin_offset(self):
        pin = torch.as_tensor(self.config["demo"]["pin_axis"], dtype=torch.float32, device=self.device)
        pin = pin.expand(self.num_envs, 3)
        offset = self.valve.pose.inv() * self.part.pose * self.seat_pose_inv
        return (offset * Pose.create_from_pq(p=pin)).p - pin

    # The drawer's offset from its seated pose, in the module frame: position, turn about z, tilt.
    def drawer_offset(self):
        offset = self.valve.pose.inv() * self.part.pose * self.seat_pose_inv
        turn, tilt = quaternion_yaw_tilt(offset.q)
        return offset.p, turn.abs(), tilt

    # Track pin alignment, insertion depth, and whether the drawer is seated.
    def _task_state(self):
        state = super()._task_state()
        tol = self.config["success"]
        p, turn, tilt = self.drawer_offset()
        pin = self.pin_offset()
        aligned = (turn <= np.deg2rad(tol["aligned_degrees"])) & (pin[:, 1].abs() <= tol["aligned_offset"]) & (
            pin[:, 2] <= tol["aligned_height"])
        seated = (torch.linalg.norm(p, axis=1) <= tol["seated_distance"]) & (
            turn <= np.deg2rad(tol["seated_degrees"])) & (tilt <= np.deg2rad(tol["seated_degrees"]))
        state.update(
            is_aligned=aligned,
            is_seated=seated,
            insert_height=p[:, 2],
            reached_goal=seated,
            is_static=self.part_still,
            hold_ok=seated & self.part_still,
        )
        return state

    # Update alignment and seating flags from the drawer motion.
    def _after_control_step(self):
        super()._after_control_step()
        tol = self.config["success"]
        pose = self.part.pose.raw_pose
        speed = torch.linalg.norm(pose[:, :3] - self.last_part_pose[:, :3], axis=1) * self.control_freq
        cos_half = (pose[:, 3:] * self.last_part_pose[:, 3:]).sum(dim=1).abs().clamp(max=1.0)
        spin = 2 * torch.arccos(cos_half) * self.control_freq
        self.part_still = (speed <= tol["linear_velocity"]) & (spin <= tol["part_angular_velocity"])
        self.last_part_pose = pose.clone()
        state = self._task_state()
        self.ever_aligned |= state["is_aligned"]
        self.ever_seated |= state["is_seated"]

    # Report the task failure conditions from the current state.
    def _failures(self, state, success, timeout):
        stage = self.ever_grasped.int() + (self.ever_grasped & self.ever_aligned).int() + (
            self.ever_grasped & self.ever_aligned & self.ever_seated).int()
        return dict(stage_reached=stage)
