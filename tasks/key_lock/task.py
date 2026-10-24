# Simulates grasping a key, inserting it into a lock, and turning it a quarter turn.
# Created: 2026-10-16

from typing import Any

import numpy as np
import sapien
import torch

from env.fr3_table_env import FRONT_TABLE_TOP_Z, load_task_config, rgba
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from tasks.hiveboard import ASSET_ROOT, HiveboardFR3Env

CONFIG = load_task_config(__file__)


@register_env("KeyLockFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class KeyLockFR3Env(HiveboardFR3Env):
    """Subgoals: key grasped, key inserted in the keyway, lock turned; success once turned and released."""

    config = CONFIG
    upright_module = True
    failure_keys = ("never_grasped", "dropped", "not_inserted", "not_turned", "timeout")

    # Load the joint and geometry settings for this task module.
    @property
    def mechanism_config(self):
        return self.config["key_lock"]

    # Create the module bodies, joints, and collision shapes in the HiveBoard scene.
    def _build_module(self):
        cfg = self.config["key_lock"]
        parts = self.config["objects"]["parts"]
        asset_dir = ASSET_ROOT / "key_lock"
        contact = sapien.physx.PhysxMaterial(cfg["contact_friction"], cfg["contact_friction"], 0.0)
        shell = sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["lock_color"]))
        key_material = sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["key_color"]))
        scale = [parts["mesh_scale"]] * 3

        builder = self.scene.create_articulation_builder()
        builder.set_name("key_lock")
        base = builder.create_link_builder()
        base.set_name("key_lock_base")
        body = parts["body"]
        plug = parts["plug"]
        base.add_visual_from_file(str(asset_dir / "key-lock_Key_Base_pivot_Corpo1_Mesh.obj"), scale=scale, material=shell,
                                  pose=sapien.Pose(p=[0.0, 0.0, body["height"]], q=[0.0, 0.0, 1.0, 0.0]))
        for axis in (0, 1):
            for sign in (-1, 1):
                outer = body["half_x"] if axis == 0 else body["half_y"]
                wall = (outer - plug["clearance"]) / 2
                half = [body["half_x"], body["half_y"], body["height"] / 2]
                half[axis] = wall
                offset = [plug["center_x"], 0.0, body["height"] / 2]
                offset[axis] += sign * (plug["clearance"] + wall)
                base.add_box_collision(pose=sapien.Pose(p=offset), half_size=half, material=contact)

        barrel = builder.create_link_builder(base)
        barrel.set_name("key_lock_barrel")
        barrel.set_joint_name("key_lock_joint")
        axis_q = [np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0]
        barrel.set_joint_properties(
            "revolute", [[0.0, cfg["turn_limit"]]],
            sapien.Pose(p=[plug["center_x"], 0.0, body["height"]], q=axis_q), sapien.Pose(q=axis_q),
            friction=cfg["barrel_friction"], damping=cfg["barrel_damping"],
        )
        keyway = parts["keyway"]
        for axis in (0, 1):
            for sign in (-1, 1):
                half = [keyway["half_x"] + keyway["wall"], keyway["half_y"] + keyway["wall"], keyway["depth"] / 2]
                half[axis] = keyway["wall"] / 2
                offset = [0.0, 0.0, -keyway["depth"] / 2]
                offset[axis] = sign * ((keyway["half_x"] if axis == 0 else keyway["half_y"]) + keyway["wall"] / 2)
                barrel.add_box_collision(pose=sapien.Pose(p=offset), half_size=half, material=contact)
        barrel.add_box_collision(pose=sapien.Pose(p=[0, 0, -keyway["depth"] - keyway["wall"] / 2]), material=contact,
                                 half_size=[keyway["half_x"] + keyway["wall"], keyway["half_y"] + keyway["wall"],
                                            keyway["wall"] / 2])
        barrel.set_mass_and_inertia(parts["barrel_mass"], sapien.Pose(), [1e-5] * 3)

        center = self.config["objects"]["board_center_xy"]
        builder.set_initial_pose(sapien.Pose(p=[center[0], center[1], FRONT_TABLE_TOP_Z]))
        self.valve = builder.build(fix_root_link=True)
        self.barrel = self.valve.links_map["key_lock_barrel"]
        self.valve.joints_map["key_lock_joint"].set_friction(cfg["barrel_friction"])
        self.valve_lever = self.barrel
        self.pivot_height = FRONT_TABLE_TOP_Z + self.config["objects"]["board_surface_height"]
        self.keyway_height = self.pivot_height + body["height"]

        key = parts["key"]
        key_builder = self.scene.create_actor_builder()
        key_builder.add_visual_from_file(str(asset_dir / "key-lock_Key_pivot_Corpo1_Mesh.obj"), scale=scale,
                                         pose=sapien.Pose(p=key["mesh_offset"], q=[np.sqrt(0.5), np.sqrt(0.5), 0, 0]),
                                         material=key_material)
        key_builder.add_box_collision(pose=sapien.Pose(p=key["tip_center"]), material=contact, half_size=key["tip_half"])
        key_builder.add_cylinder_collision(
            pose=sapien.Pose(p=key["shaft_center"], q=[np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0]),
            radius=key["shaft_radius"], half_length=key["shaft_half_length"], material=contact,
        )
        key_builder.add_box_collision(pose=sapien.Pose(p=key["bow_center"]), material=contact, half_size=key["bow_half"])
        key_builder.initial_pose = sapien.Pose(p=[0.2, 0.3, FRONT_TABLE_TOP_Z + 0.05])
        self.key = key_builder.build(name="key")
        self.key_length = key["length"]
        self.task_objects = ((self.key, key["bow_half"][1]),)

    # Reset the key position and lock barrel angle.
    def _initialize_module(self, env_idx: torch.Tensor):
        qpos = self.valve.get_qpos()
        qpos[env_idx] = 0.0
        self.valve.set_qpos(qpos)
        qvel = self.valve.get_qvel()
        qvel[env_idx] = 0
        self.valve.set_qvel(qvel)
        with torch.device(self.device):
            b = len(env_idx)
            rand = self.config["randomization"]
            board = self.board_center_xy[env_idx].cpu()
            near, far = rand["min_board_distance"], rand["max_board_distance"]

            # Place the key upright beside the board for pickup.
            def beside_board(xy, idx):
                distance = torch.linalg.norm(xy - board[idx], axis=1)
                return (distance >= near) & (distance <= far)

            xy = self._sample_workspace_xy(env_idx, valid=beside_board)
            xyz = torch.full((b, 3), FRONT_TABLE_TOP_Z + 0.0005)
            xyz[:, :2] = xy
            if rand["key_yaw_aligned"]:
                yaw = torch.deg2rad(
                    self.module_yaw_degrees[env_idx].cpu()
                    + self.config["objects"].get("module_yaw_offset_degrees", 0.0)
                )
            else:
                yaw = self._uniform(env_idx, -np.pi, np.pi)
            half = yaw / 2
            zeros = torch.zeros_like(half)
            q = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=1)
            self.key.set_pose(Pose.create_from_pq(p=xyz, q=q.to(self.device)))
            self.key.set_linear_velocity(torch.zeros((b, 3)))
            self.key.set_angular_velocity(torch.zeros((b, 3)))

    # Track key grasp, insertion depth, and barrel rotation.
    def _task_state(self):
        tol = self.config["success"]
        angle = self.valve.get_qpos()[:, 0]
        velocity = self.valve.get_qvel()[:, 0]
        rot = rotation_conversions.quaternion_to_matrix(self.key.pose.q)
        axis = rot[:, :, 2]
        tip = self.key.pose.p
        lock_xy = self.barrel.pose.p[:, :2]
        inserted = (
            (torch.linalg.norm(tip[:, :2] - lock_xy, axis=1) <= tol["insert_radius"])
            & (tip[:, 2] <= self.keyway_height - tol["insert_depth"])
            & (axis[:, 2] >= np.cos(np.deg2rad(tol["upright_degrees"])))
        )
        turned = angle >= np.deg2rad(tol["turn_degrees"])
        grasped = self.agent.is_grasping(self.key)
        released = ~grasped & (self.agent.finger_contact_force(self.key) < tol["release_contact_force"])
        return dict(
            is_grasped=grasped,
            is_static=(torch.abs(velocity) <= tol["angular_velocity"])
            & self.key.is_static(lin_thresh=tol["linear_velocity"], ang_thresh=tol["key_angular_velocity"]),
            is_contacted=self.agent.finger_contact_force(self.key) >= tol["contact_force"],
            is_released=released,
            is_inserted=inserted,
            is_turned=turned,
            lock_angle=angle,
            key_tip_height=tip[:, 2],
            key_to_lock_dist=torch.linalg.norm(tip[:, :2] - lock_xy, axis=1),
            off_table=self.key.pose.p[:, 2] < FRONT_TABLE_TOP_Z - 0.05,
            reached_goal=turned & inserted,
            hold_ok=turned & inserted & released,
        )

    # Add the module and task objects to the shared robot scene.
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.ever_inserted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_turned = torch.zeros_like(self.ever_inserted)

    # Reset task state and object poses at the start of each episode.
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.ever_inserted[env_idx] = False
        self.ever_turned[env_idx] = False

    # Update insertion and turn progress after the physics step.
    def _after_control_step(self):
        super()._after_control_step()
        state = self._task_state()
        self.ever_inserted |= state["is_inserted"]
        self.ever_turned |= state["is_turned"]

    # Report the task failure conditions from the current state.
    def _failures(self, state, success, timeout):
        return dict(
            stage=self.ever_grasped.int() + self.ever_inserted.int() + self.ever_turned.int(),
            subgoal_inserted=self.ever_inserted.clone(),
            never_grasped=~self.ever_grasped,
            dropped=self.ever_grasped & ~self.ever_inserted & ~state["is_grasped"] & state["is_static"] & timeout,
            not_inserted=self.ever_grasped & ~self.ever_inserted & timeout,
            not_turned=self.ever_inserted & ~self.ever_turned & timeout,
        )

    # Add task specific state fields to the observation.
    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                key_pose=self.key.pose.raw_pose,
                lock_pose=self.barrel.pose.raw_pose,
                lock_angle=self.valve.get_qpos(),
                tcp_to_key_pos=self.key.pose.p - self.agent.tcp.pose.p,
                key_to_lock_pos=self.barrel.pose.p - self.key.pose.p,
            )
        return obs

    # Score progress toward the task goal at the current step.
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp = self.agent.tcp.pose.p
        reach = 1 - torch.tanh(5 * torch.linalg.norm(self.key.pose.p - tcp, axis=1))
        reward = reach + info["is_grasped"].float()
        approach = 1 - torch.tanh(5 * info["key_to_lock_dist"])
        held = info["is_grasped"]
        reward[held] = (2 + approach)[held]
        reward[info["is_inserted"]] = 4 + 2 * torch.clamp(
            info["lock_angle"][info["is_inserted"]] / np.deg2rad(self.config["success"]["turn_degrees"]), 0, 1
        )
        reward[info["success"]] = 7
        return reward

    # Scale the dense reward to the task reward range.
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 7


# Hamilton product of wxyz quaternions, batched.
def quaternion_multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)
