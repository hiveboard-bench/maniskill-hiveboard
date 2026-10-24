# Simulates opening a hinged cover before pressing the button beneath it.
# Created: 2026-10-16

from typing import Any

import numpy as np
import sapien
import torch

from env.fr3_table_env import FRONT_TABLE_TOP_Z, load_task_config, rgba
from mani_skill.utils.registration import register_env
from tasks.hiveboard import ASSET_ROOT, HiveboardFR3Env
from tasks.hiveboard_assembly import to_pose, urdf_transform

CONFIG = load_task_config(__file__)
LID_BOARD_COLLISION_BIT = 28


@register_env("ButtonCoverFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class ButtonCoverFR3Env(HiveboardFR3Env):
    """Subgoals: lid open, then button held down; success once the button stays actuated."""

    config = CONFIG
    upright_module = True
    failure_keys = ("never_contacted", "lid_closed", "stopped_short", "timeout")

    # Load the joint and geometry settings for this task module.
    @property
    def mechanism_config(self):
        return self.config["button_cover"]

    # Build the hinged cover and spring loaded button from the URDF geometry.
    def _build_module(self):
        cfg = self.config["button_cover"]
        urdf = cfg["urdf"]
        parts = self.config["objects"]["parts"]
        asset_dir = ASSET_ROOT / "button_cover"
        contact = sapien.physx.PhysxMaterial(cfg["contact_friction"], cfg["contact_friction"], 0.0)
        shell = sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["body_color"]))
        button_material = sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["button_color"]))
        lid_material = sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["cover_color"]))
        to_module = urdf_transform(urdf["urdf_to_module"])
        x_to_z = sapien.Pose(q=[np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0])

        builder = self.scene.create_articulation_builder()
        builder.set_name("button_cover")
        base = builder.create_link_builder()
        base.set_name("button_cover_base")
        body_pose = to_pose(to_module @ urdf_transform(urdf["body_visual"]))
        base.add_visual_from_file(str(asset_dir / "button-cover_tn__corpo1_dC_Corpo1_Mesh.obj"), pose=body_pose,
                                  material=shell)
        base.add_multiple_convex_collisions_from_file(
            str(asset_dir / "button-cover_tn__corpo1_dC_Corpo1_Mesh_coacd.obj"), pose=body_pose, material=contact)

        button = builder.create_link_builder(base)
        button.set_name("button_cover_button")
        button.set_joint_name("button_cover_button_joint")
        button.set_joint_properties(
            "prismatic", [[cfg["press_depth"], 0.0]], to_pose(to_module), sapien.Pose(),
            friction=cfg["button_friction"], damping=cfg["button_damping"],
        )
        button.add_visual_from_file(str(asset_dir / "button-cover_tn__botao1_dC_Corpo1_Mesh.obj"),
                                    pose=to_pose(urdf_transform(urdf["button_visual"])), material=button_material)
        for radius, bottom, top in parts["button"]["discs"]:
            button.add_cylinder_collision(pose=sapien.Pose(p=[(bottom + top) / 2, 0, 0]), radius=radius,
                                          half_length=(top - bottom) / 2, material=contact)
        button.set_mass_and_inertia(parts["button"]["mass"], sapien.Pose(p=[0.009, 0, 0]), [1e-5] * 3)

        lid = builder.create_link_builder(base)
        lid.set_name("button_cover_lid")
        lid.set_joint_name("button_cover_lid_joint")
        closed = urdf_transform({"xyz": urdf["lid_joint_xyz"], "rpy": [0.0, 0.0, urdf["lid_closed_angle"]]})
        lid.set_joint_properties(
            "revolute", [[0.0, min(urdf["lid_open_limit"] - urdf["lid_closed_angle"],
                                   np.deg2rad(urdf["lid_wall_stop_degrees"]))]],
            to_pose(to_module @ closed) * x_to_z, x_to_z,
            friction=cfg["lid_friction"], damping=cfg["lid_damping"],
        )
        lid_pose = to_pose(urdf_transform(urdf["lid_visual"]))
        lid.add_visual_from_file(str(asset_dir / "button-cover_tn__Tampa1_dC_Corpo1_Mesh.obj"), pose=lid_pose,
                                 material=lid_material)
        lid.add_multiple_convex_collisions_from_file(
            str(asset_dir / "button-cover_tn__Tampa1_dC_Corpo1_Mesh_coacd.obj"), pose=lid_pose, material=contact)
        lid.set_mass_and_inertia(parts["lid"]["mass"], sapien.Pose(p=urdf["lid_center_of_mass"]),
                                 urdf["lid_inertia"])

        center = self.config["objects"]["board_center_xy"]
        builder.set_initial_pose(sapien.Pose(p=[center[0], center[1], FRONT_TABLE_TOP_Z]))
        self.valve = builder.build(fix_root_link=True)
        self.button_drive = self.valve.joints_map["button_cover_button_joint"]
        self.button_drive.set_drive_properties(
            stiffness=cfg["button_stiffness"], damping=cfg["button_damping"], force_limit=100.0
        )
        self.button_drive.set_friction(cfg["button_friction"])
        self.lid_drive = self.valve.joints_map["button_cover_lid_joint"]
        self.lid_drive.set_friction(cfg["lid_friction"])
        self.button = self.valve.links_map["button_cover_button"]
        self.lid = self.valve.links_map["button_cover_lid"]
        self.lid.set_collision_group_bit(2, LID_BOARD_COLLISION_BIT, 1)
        self.board.set_collision_group_bit(2, LID_BOARD_COLLISION_BIT, 1)
        self.valve_lever = self.button
        self.pivot_height = FRONT_TABLE_TOP_Z + self.config["objects"]["board_surface_height"]
        self.button_height = self.pivot_height + parts["button"]["z"]

    # Reset the module parts and joints to their starting poses.
    def _initialize_module(self, env_idx: torch.Tensor):
        self.button_drive.set_drive_target(0.0)
        # Drive properties belong to individual PhysX joints, so reset only these environments.
        for idx in env_idx.tolist():
            self.lid_drive._objs[idx].set_drive_properties(0.0, 0.0, 0.0)
        self.lid_drive.set_drive_target(0.0)
        qpos = self.valve.get_qpos()
        qpos[env_idx] = 0.0
        self.valve.set_qpos(qpos)
        qvel = self.valve.get_qvel()
        qvel[env_idx] = 0
        self.valve.set_qvel(qvel)

    # Track whether the lid is open and the button has been pressed.
    def _task_state(self):
        cfg = self.config["button_cover"]
        tol = self.config["success"]
        qpos = self.valve.get_qpos()
        press, lid_angle = -qpos[:, 0], qpos[:, 1]
        velocity = self.valve.get_qvel()
        lid_open = lid_angle >= np.deg2rad(tol["open_angle_degrees"])
        pressed = press >= -cfg["press_depth"] * tol["press_fraction"]
        contact = lambda link: torch.maximum(
            torch.linalg.norm(self.scene.get_pairwise_contact_forces(self.agent.hand, link), axis=1),
            self.agent.finger_contact_force(link),
        )
        return dict(
            is_grasped=self.agent.is_grasping(self.lid),
            is_static=torch.linalg.norm(velocity, axis=1) <= tol["joint_velocity"],
            is_contacted=(contact(self.lid) >= tol["contact_force"]) | (contact(self.button) >= tol["contact_force"]),
            button_contact=contact(self.button),
            lid_contact=contact(self.lid),
            press_depth=press,
            lid_angle=lid_angle,
            is_lid_open=lid_open,
            is_pressed=pressed,
            off_table=torch.zeros_like(lid_open),
            reached_goal=lid_open & pressed,
            hold_ok=lid_open & pressed,
        )

    # Update the button and cover state after contact with the gripper.
    def _after_control_step(self):
        super()._after_control_step()
        state = self._task_state()
        cfg = self.config["button_cover"]
        engage = (state["lid_angle"] >= np.deg2rad(cfg["detent_engage_degrees"])) & ~self.lid_detent_engaged
        self._engage_lid_detent(engage)
        self.ever_lid_open |= state["is_lid_open"]
        self.ever_pressed |= state["is_pressed"]
        self.ever_pressed_closed |= state["is_pressed"] & ~state["is_lid_open"]

    # Hold the lid after it reaches the open side of the hinge; each environment latches independently.
    def _engage_lid_detent(self, engage: torch.Tensor):
        cfg = self.config["button_cover"]
        if not engage.any():
            return
        targets = self.lid_drive.drive_target.clone()
        targets[engage] = self.valve.get_qpos()[engage, 1]
        for idx in torch.nonzero(engage).flatten().tolist():
            self.lid_drive._objs[idx].set_drive_properties(
                cfg["detent_stiffness"], cfg["detent_damping"], cfg["detent_force_limit"]
            )
        self.lid_drive.set_drive_target(targets)
        self.lid_detent_engaged |= engage

    # Add the module and task objects to the shared robot scene.
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.ever_lid_open = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lid_detent_engaged = torch.zeros_like(self.ever_lid_open)
        self.ever_pressed = torch.zeros_like(self.ever_lid_open)
        self.ever_pressed_closed = torch.zeros_like(self.ever_lid_open)

    # Reset task state and object poses at the start of each episode.
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        for flag in (self.ever_lid_open, self.ever_pressed, self.ever_pressed_closed):
            flag[env_idx] = False
        self.lid_detent_engaged[env_idx] = False

    # Report whether the cover stayed shut, the button was missed, or time expired.
    def _failures(self, state, success, timeout):
        return dict(
            stage=self.ever_lid_open.int() + self.ever_pressed.int(),
            subgoal_lid_open=self.ever_lid_open.clone(),
            never_contacted=~self.ever_contacted,
            lid_closed=self.ever_contacted & ~self.ever_lid_open & timeout,
            pressed_through_lid=self.ever_pressed_closed,
            stopped_short=self.ever_lid_open & ~state["is_pressed"] & timeout,
        )

    # Add task specific state fields to the observation.
    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                button_pose=self.button.pose.raw_pose,
                lid_pose=self.lid.pose.raw_pose,
                module_qpos=self.valve.get_qpos(),
                module_qvel=self.valve.get_qvel(),
                tcp_to_button_pos=self.button.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    # Score progress toward the task goal at the current step.
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp = self.agent.tcp.pose.p
        reach_lid = 1 - torch.tanh(5 * torch.linalg.norm(self.lid.pose.p - tcp, axis=1))
        opening = torch.clamp(info["lid_angle"] / self.config["button_cover"]["open_angle_limit"], 0, 1)
        reward = reach_lid + 2 * opening
        reach_button = 1 - torch.tanh(5 * torch.linalg.norm(self.button.pose.p - tcp, axis=1))
        depth = torch.clamp(info["press_depth"] / -self.config["button_cover"]["press_depth"], 0, 1)
        open_now = info["is_lid_open"]
        reward[open_now] = (3 + reach_button + 2 * depth)[open_now]
        reward[info["success"]] = 7
        return reward

    # Scale the dense reward to the task reward range.
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 7
