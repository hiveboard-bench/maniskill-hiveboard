# Builds the shared HiveBoard scene, places modules, and tracks task states in the FR3 simulator.
# Created: 2026-10-23

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch

from env.fr3_table_env import FRONT_TABLE_TOP_Z, FR3TaskEnv, rgba
from mani_skill.utils.structs.pose import Pose

ASSET_ROOT = Path(__file__).resolve().parents[1] / "fr3_assets" / "hiveboard"


class HiveboardFR3Env(FR3TaskEnv):
    """Operate the HiveBoard module named in objects.module with physical contact; subclasses set config."""

    failure_keys = ("never_contacted", "stopped_short", "reopened", "timeout")

    # Set up the state and resources used by this instance.
    def __init__(self, *args, **kwargs):
        self.config = deepcopy(type(self).config)
        super().__init__(*args, **kwargs)

    # Build the HiveBoard, robot surroundings, and configured task module.
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.module_name = self.config["objects"]["module"]
        builders = {
            "gate_valve_small": self._build_gate_valve,
            "gate_valve_large": self._build_gate_valve,
        }
        self._build_board()
        builders.get(self.module_name, self._build_module)()
        for articulation in self.scene.articulations.values():
            if articulation is not self.agent.robot:
                for physx_articulation in articulation._objs:
                    physx_articulation.set_sleep_threshold(0.0)
        self.board_center_xy = torch.tensor(
            self.config["objects"]["board_center_xy"], dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self.module_cell_offsets = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.module_yaw_degrees = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.module_cell_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.board_yaw_degrees = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.ever_contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_wrong_direction = torch.zeros_like(self.ever_contacted)
        self.ever_reopened = torch.zeros_like(self.ever_contacted)
        self.ever_unwound = torch.zeros_like(self.ever_contacted)
        self.was_holding = torch.zeros_like(self.ever_contacted)
        self.max_angle = torch.zeros(self.num_envs, device=self.device)
        self.grasp_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    SECTIONS = {"ball_valve": "valve", "circuit_breaker": "breaker",
                "gate_valve_small": "gate_valve", "gate_valve_large": "gate_valve"}

    # Load the joint and geometry settings for this task module.
    @property
    def mechanism_config(self):
        return self.config[self.SECTIONS[self.module_name]]

    # Check whether the configured module is a gate valve.
    @property
    def is_gate_valve(self):
        return self.module_name.startswith("gate_valve")

    # Modules modelled upright in their own frame, which mount straight onto the board surface.
    @property
    def upright_module(self):
        return self.is_gate_valve

    # Build the hexagonal board and its module mounting cells.
    def _build_board(self):
        center = self.config["objects"]["board_center_xy"]
        builder = self.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["board_color"]))
        centers = ((0, 0), (0, 0.086603), (0.075, 0.043301), (0.075, -0.043301),
                   (0, -0.086603), (-0.075, -0.043301), (-0.075, 0.043301))
        hollow = self.config["objects"].get("hollow_cells", False)
        for index, (x, y) in enumerate(centers, 1):
            mesh = ASSET_ROOT / "panel" / f"panel_tn__Hexagono_Suporte{index}_bX0_Corpo1_Mesh.obj"
            pose = sapien.Pose(p=[x, y, 0.0])
            builder.add_visual_from_file(str(mesh), pose=pose, scale=[0.001] * 3, material=material)
            if not hollow:
                builder.add_box_collision(pose=sapien.Pose(p=[x, y, 0.01]), half_size=[0.048, 0.041, 0.01])
                continue
            for low, high, inner_radius in ((0.0, 0.005, 0.03845), (0.005, 0.02, 0.04423)):
                outer, inner = 0.05 * np.cos(np.pi / 6), inner_radius * np.cos(np.pi / 6)
                for side in range(6):
                    normal = np.pi / 6 + side * np.pi / 3
                    distance = (outer + inner) / 2
                    builder.add_box_collision(
                        pose=sapien.Pose(p=[x + distance * np.cos(normal), y + distance * np.sin(normal), (low + high) / 2],
                                         q=[np.cos(normal / 2), 0.0, 0.0, np.sin(normal / 2)]),
                        half_size=[(outer - inner) / 2, 0.025, (high - low) / 2],
                    )
        builder.initial_pose = sapien.Pose(p=[center[0], center[1], FRONT_TABLE_TOP_Z])
        self.board = builder.build_kinematic(name="hiveboard")

    # Build the valve wheel, shaft, and revolute joint.
    def _build_gate_valve(self):
        cfg = self.config["gate_valve"]
        mesh = self.config["objects"]["mesh"]
        wheel = self.config["objects"]["wheel"]
        asset_dir = ASSET_ROOT / self.module_name
        shell = sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["valve_base_color"]))
        accent = sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["wheel_color"]))

        builder = self.scene.create_articulation_builder()
        builder.set_name(self.module_name)
        base = builder.create_link_builder()
        base.set_name("gate_valve_base")
        contact = sapien.physx.PhysxMaterial(cfg["contact_friction"], cfg["contact_friction"], 0.0)
        base_pose = sapien.Pose(p=[0, 0, mesh["base_offset_z"]])
        base.add_visual_from_file(str(asset_dir / mesh["base_file"]), pose=base_pose, material=shell)
        base.add_convex_collision_from_file(str(asset_dir / mesh["base_file"]), pose=base_pose, material=contact)

        hub = builder.create_link_builder(base)
        hub.set_name("gate_valve_wheel")
        hub.set_joint_name("gate_valve_joint")
        axis_q = [np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0]
        hub.set_joint_properties(
            "revolute_unwrapped", [[cfg["lower_limit"], cfg["upper_limit"]]],
            sapien.Pose(q=axis_q), sapien.Pose(q=axis_q), friction=cfg["friction"], damping=cfg["damping"],
        )
        hub.add_visual_from_file(str(asset_dir / mesh["wheel_file"]),
                                 pose=sapien.Pose(p=mesh["wheel_offset_xyz"], q=mesh["wheel_offset_quat"]),
                                 material=accent)
        radius = wheel["rim_radius"]
        rim_z = sum(wheel["rim_z_range"]) / 2
        if wheel["collision"] == "coacd":
            wheel_mesh = asset_dir / mesh["wheel_file"]
            hub.add_multiple_convex_collisions_from_file(
                str(wheel_mesh.with_name(wheel_mesh.stem + "_coacd.obj")),
                pose=sapien.Pose(p=mesh["wheel_offset_xyz"], q=mesh["wheel_offset_quat"]), material=contact,
            )
        else:
            upright = [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]
            for part in ("shaft", "hub"):
                half_length = (wheel[part]["z_range"][1] - wheel[part]["z_range"][0]) / 2
                center = sum(wheel[part]["z_range"]) / 2
                hub.add_cylinder_collision(pose=sapien.Pose(p=[0, 0, center], q=upright), material=contact,
                                           radius=wheel[part]["radius"], half_length=half_length)
            tube = wheel["rim_thickness"] / 2
            half_height = (wheel["rim_z_range"][1] - wheel["rim_z_range"][0]) / 2
            for i in range(wheel["rim_segments"]):
                angle = i * 2 * np.pi / wheel["rim_segments"]
                span = radius * np.tan(np.pi / wheel["rim_segments"])
                hub.add_box_collision(
                    pose=sapien.Pose(p=[radius * np.cos(angle), radius * np.sin(angle), rim_z],
                                     q=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]),
                    half_size=[tube, span, half_height], material=contact,
                )
        hub.set_mass_and_inertia(wheel["mass"], sapien.Pose(p=[0, 0, rim_z]),
                                 [wheel["mass"] * radius ** 2 / 4] * 3)

        center = self.config["objects"]["board_center_xy"]
        builder.set_initial_pose(sapien.Pose(p=[center[0], center[1], FRONT_TABLE_TOP_Z]))
        self.valve = builder.build(fix_root_link=True)
        self.valve_lever = self.valve.links_map["gate_valve_wheel"]
        self.pivot_height = FRONT_TABLE_TOP_Z + self.config["objects"]["board_surface_height"] + mesh["pivot_height"]
        self.wheel_height = self.pivot_height + rim_z
        self.wheel_radius = radius

    # Place the board and module, then reset their per episode state.
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        offsets = self.config["objects"]["board_position_offset_xy"]
        self.board_center_xy[env_idx] = torch.stack(
            [self._uniform(env_idx, *offsets[axis]) for axis in range(2)], dim=1
        ).to(self.device) + torch.tensor(
            self.config["objects"]["board_center_xy"], device=self.device
        )
        yaw_choices = self.config["objects"]["module_yaw_choices_degrees"]
        sampled_yaw = np.asarray([
            yaw_choices[self._batched_episode_rng[int(i)].randint(len(yaw_choices))]
            for i in env_idx.tolist()
        ], dtype=np.float32)
        cell_positions = self.config["objects"]["module_cell_positions_xy"]
        sampled_cells = np.asarray([
            self._batched_episode_rng[int(i)].randint(len(cell_positions))
            for i in env_idx.tolist()
        ])
        board_yaw = self._uniform(env_idx, *self.config["objects"]["board_yaw_range_degrees"]).to(self.device)
        self.board_yaw_degrees[env_idx] = board_yaw
        self.module_yaw_degrees[env_idx] = torch.from_numpy(sampled_yaw).to(self.device) + board_yaw
        self.module_cell_index[env_idx] = torch.from_numpy(sampled_cells).to(self.device)
        cells = torch.tensor([cell_positions[index] for index in sampled_cells], dtype=torch.float32, device=self.device)
        cos, sin = torch.cos(torch.deg2rad(board_yaw)), torch.sin(torch.deg2rad(board_yaw))
        self.module_cell_offsets[env_idx] = torch.stack(
            [cos * cells[:, 0] - sin * cells[:, 1], sin * cells[:, 0] + cos * cells[:, 1]], dim=1
        )
        board_p = torch.cat(
            [self.board_center_xy, torch.full((self.num_envs, 1), FRONT_TABLE_TOP_Z, device=self.device)], dim=1
        )
        board_half_yaw = torch.deg2rad(self.board_yaw_degrees) / 2
        board_q = torch.zeros((self.num_envs, 4), device=self.device)
        board_q[:, 0] = torch.cos(board_half_yaw)
        board_q[:, 3] = torch.sin(board_half_yaw)
        self.board.set_pose(Pose.create_from_pq(p=board_p, q=board_q))
        half_yaw = torch.deg2rad(self.module_yaw_degrees + self.config["objects"].get("module_yaw_offset_degrees", 0.0)) / 2
        zero = torch.zeros_like(half_yaw)
        if self.upright_module:
            root_q = torch.stack([torch.cos(half_yaw), zero, zero, torch.sin(half_yaw)], dim=1)
            module_z = self.pivot_height
        else:
            root_q = torch.stack(
                [
                    torch.cos(half_yaw) * np.sqrt(0.5),
                    torch.sin(half_yaw) * np.sqrt(0.5),
                    -torch.cos(half_yaw) * np.sqrt(0.5),
                    torch.sin(half_yaw) * np.sqrt(0.5),
                ], dim=1
            )
            module_z = FRONT_TABLE_TOP_Z + 0.045
        module_z += self.config["objects"].get("module_height_offset", 0.0)
        valve_p = torch.cat(
            [self.board_center_xy + self.module_cell_offsets,
             torch.full((self.num_envs, 1), module_z, device=self.device)], dim=1
        )
        self.valve.set_pose(Pose.create_from_pq(p=valve_p, q=root_q))
        self._initialize_module(env_idx)
        self.ever_contacted[env_idx] = False
        self.ever_wrong_direction[env_idx] = False
        self.ever_reopened[env_idx] = False
        self.max_angle[env_idx] = self.mechanism_config.get("initial_angle", 0.0)
        self.ever_unwound[env_idx] = False
        self.grasp_count[env_idx] = 0
        self.was_holding[env_idx] = False
        prompts = {
            "ball_valve": "close the gripper, hook the ball valve lever, and pull it through a quarter turn",
            "circuit_breaker": "toggle the 3D-printed circuit breaker",
            "gate_valve_small": "grasp the small gate valve handwheel and turn it one full revolution, regrasping as needed",
            "gate_valve_large": "turn the large gate valve handwheel one full revolution, regrasping as needed",
        }
        prompt = prompts.get(self.module_name, self.config["prompt"]["templates"][0])
        for i in env_idx.tolist():
            self.prompts[i] = prompt

    # Create the module bodies, joints, and collision shapes in the HiveBoard scene.
    def _build_module(self):
        raise ValueError(f"unknown hiveboard module {self.module_name!r}")

    # Reset the module parts and joints to their starting poses.
    def _initialize_module(self, env_idx: torch.Tensor):
        qpos = self.valve.get_qpos()
        qpos[env_idx, 0] = self.mechanism_config["initial_angle"]
        self.valve.set_qpos(qpos)
        qvel = self.valve.get_qvel()
        qvel[env_idx] = 0
        self.valve.set_qvel(qvel)

    # Measure module rotation and contact for success tracking.
    def _task_state(self):
        angle = self.valve.get_qpos()[:, 0]
        velocity = self.valve.get_qvel()[:, 0]
        target = self.mechanism_config["target_angle"]
        success = self.config["success"]
        contact = torch.maximum(
            torch.linalg.norm(self.scene.get_pairwise_contact_forces(self.agent.hand, self.valve_lever), axis=1),
            self.agent.finger_contact_force(self.valve_lever),
        )
        tolerance = np.deg2rad(success["angle_tolerance_degrees"])
        reached = angle >= target - tolerance if self.is_gate_valve else torch.abs(angle - target) <= tolerance
        static = torch.abs(velocity) <= success["angular_velocity"]
        return dict(
            is_grasped=torch.zeros_like(reached), is_static=static,
            is_contacted=contact >= success["contact_force"], valve_angle=angle,
            valve_velocity=velocity, valve_target_error=torch.abs(angle - target),
            module_angle=angle, module_velocity=velocity, module_target_error=torch.abs(angle - target),
            turns=angle / (2 * np.pi), grasp_count=self.grasp_count.float(),
            reached_goal=reached, hold_ok=reached & static & ~(
                contact >= success["contact_force"]
            ), off_table=torch.zeros_like(reached),
        )

    # Update the module state after the robot control step.
    def _after_control_step(self):
        super()._after_control_step()
        state = self._task_state()
        self.ever_contacted |= state["is_contacted"]
        if "valve_angle" in state:
            direction = np.sign(self.mechanism_config["target_angle"] - self.mechanism_config["initial_angle"])
            self.ever_wrong_direction |= direction * (
                state["valve_angle"] - self.mechanism_config["initial_angle"]
            ) < -0.03
        fingers = self.agent.robot.get_qpos()[:, 7:9]
        holding = state["is_contacted"] & (fingers.mean(dim=1) <= 0.018)
        self.grasp_count += (holding & ~self.was_holding).int()
        self.was_holding = holding
        if "valve_angle" not in state:
            pass
        elif self.is_gate_valve:
            self.max_angle = torch.maximum(self.max_angle, state["valve_angle"])
            self.ever_unwound |= (self.max_angle - state["valve_angle"]) > self.config["success"]["unwind_tolerance"]
        else:
            self.ever_reopened |= self.ever_contacted & (fingers.mean(dim=1) > 0.006)

    # Report the task failure conditions from the current state.
    def _failures(self, state, success, timeout):
        return dict(
            never_contacted=~self.ever_contacted,
            stopped_short=self.ever_contacted & ~state["reached_goal"] & timeout,
            wrong_direction=self.ever_wrong_direction & ~success,
            reopened=self.ever_reopened,
            unwound=self.ever_unwound & ~success,
        )

    # Expose board and module state in the observation.
    def _get_obs_extra(self, info: dict):
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            valve_angle=self.valve.get_qpos(),
            valve_velocity=self.valve.get_qvel(),
            module_angle=self.valve.get_qpos(),
            module_velocity=self.valve.get_qvel(),
        )

    # Reward contact and progress toward the target module rotation.
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        pivot = self.valve.pose.p
        reach = 1 - torch.tanh(5 * torch.linalg.norm(self.agent.tcp.pose.p - pivot, axis=1))
        cfg = self.mechanism_config
        total = abs(cfg["target_angle"] - cfg["initial_angle"])
        progress = torch.clamp(1 - info["module_target_error"] / total, 0, 1)
        reward = reach + 3 * progress + info["is_contacted"].float()
        reward[info["success"]] = 5
        return reward

    # Scale the dense reward to the task reward range.
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 5
