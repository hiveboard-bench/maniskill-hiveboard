# Defines the FR3 robot and table simulation environment.
# Created: 2026-09-14

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import sapien
import torch
import yaml

from gymnasium import spaces
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    PDJointPosMimicController,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_quaternion,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.building import actors
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


PROJECT_ROOT = Path(__file__).resolve().parents[1]
with (PROJECT_ROOT / "config" / "config.yaml").open() as config_file:
    CONFIG = yaml.safe_load(config_file)

with (PROJECT_ROOT / "config" / "data.yaml").open() as data_file:
    DATA_CONFIG = yaml.safe_load(data_file)

ROBOT_CONFIG = CONFIG["robot"]
REFLEX_KEYS = ("reflex_joint_limit", "reflex_joint_velocity", "reflex_motion", "reflex_torque")
DOMAIN_RANDOMIZATION_OFF = {
    ("robot", "gripper_delay_seconds"): 0.0,
    ("robot", "ee_deadband_meters"): 0.0,
    ("robot", "ee_deadband_degrees"): 0.0,
    ("robot", "ee_lag_seconds"): 0.0,
    ("robot", "ee_rotation_lag_seconds"): 0.0,
    ("robot", "ee_target_stiffness"): ROBOT_CONFIG["arm_stiffness"],
    ("robot", "ee_target_damping"): ROBOT_CONFIG["arm_damping"],
    ("robot", "ee_max_speed"): 10.0,
    ("robot", "ee_max_angular_speed_degrees"): 1e4,
    ("robot", "ee_error_clip_meters"): 1.0,
    ("robot", "ee_error_clip_radians"): np.pi,
    **{("latency", "camera_seconds", name): 0.0 for name in CONFIG["latency"]["camera_seconds"]},
    **{("render", "color_variation", name): 0.0 for name in CONFIG["render"]["color_variation"]},
    ("render", "texture_variation", "white_wall"): 0.0,
    ("render", "texture_variation", "concrete"): 0.0,
}
DOMAIN_RANDOMIZATION_ON = {}
for _path in DOMAIN_RANDOMIZATION_OFF:
    _node = CONFIG
    for _key in _path[:-1]:
        _node = _node[_key]
    DOMAIN_RANDOMIZATION_ON[_path] = _node[_path[-1]]


# Enable or disable the configured robot and image variation.
def set_domain_randomization(enabled: bool):
    CONFIG["domain_randomization"] = bool(enabled)
    for path, value in (DOMAIN_RANDOMIZATION_ON if enabled else DOMAIN_RANDOMIZATION_OFF).items():
        node = CONFIG
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value


set_domain_randomization(CONFIG.get("domain_randomization", True))


# Fraction of the remaining error a first-order lag closes in one step; a zero lag closes all of it.
def lag_fraction(dt: float, lag_seconds: float) -> float:
    return 1.0 - np.exp(-dt / lag_seconds) if lag_seconds > 0 else 1.0


FR3_MJCF = PROJECT_ROOT / ROBOT_CONFIG["asset_path"]
FR3_KINEMATICS_URDF = PROJECT_ROOT / ROBOT_CONFIG["kinematics_path"]
yaw = np.deg2rad(ROBOT_CONFIG["base_yaw_degrees"]) / 2
ROBOT_BASE_POSE = sapien.Pose(
    p=ROBOT_CONFIG["base_position"], q=[np.cos(yaw), 0.0, 0.0, np.sin(yaw)]
)
FRONT_TABLE_TOP_Z = CONFIG["tables"]["front"]["position"][2]
GRIPPER_CAMERA_POSE = sapien.Pose(
    p=CONFIG["cameras"]["gripper"]["position"],
    q=CONFIG["cameras"]["gripper"]["quaternion"],
)


# Return the real robot start joint positions.
def real_start_qpos(name):
    path = PROJECT_ROOT / "fr3_assets" / "data" / (name if isinstance(name, str) else "real_start_qpos.npy")
    if path not in _REAL_STARTS:
        _REAL_STARTS[path] = np.load(path)
    return _REAL_STARTS[path]


_REAL_STARTS = {}


# Fixed multi-scale noise maps in [-1, 1] that give walls a plaster or concrete texture.
def wall_patterns(count, size, seed):
    rng = np.random.default_rng(seed)
    patterns = []
    for _ in range(count):
        noise = np.zeros((size, size))
        for cells, weight in ((4, 0.5), (16, 0.3), (64, 0.2)):
            grid = torch.from_numpy(rng.standard_normal((1, 1, cells, cells)))
            noise += weight * torch.nn.functional.interpolate(grid, size=(size, size), mode="bicubic")[0, 0].numpy()
        patterns.append(noise / np.abs(noise).max())
    return patterns


# Limit vector length while preserving its direction.
def clip_norm(vector: torch.Tensor, limit: float) -> torch.Tensor:
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    return vector * torch.clamp(limit / norm.clamp(min=1e-9), max=1.0)


# Step along error that closes alpha of it beyond the deadband, capped at limit.
def approach(error: torch.Tensor, deadband: float, alpha: float, limit: float) -> torch.Tensor:
    norm = torch.linalg.norm(error, dim=-1, keepdim=True)
    step = torch.clamp(alpha * (norm - deadband), min=0.0, max=limit)
    return error / norm.clamp(min=1e-9) * step


# Base-frame rotation vector taking wxyz source_q to target_q along the short way.
def rotation_between(target_q: torch.Tensor, source_q: torch.Tensor) -> torch.Tensor:
    delta = quaternion_multiply(target_q, quaternion_invert(source_q))
    return quaternion_to_axis_angle(torch.where(delta[:, :1] < 0, -delta, delta))


class LaggedEEPoseController(PDEEPoseController):
    """Absolute TCP target tracked with the deadband, lag, speed caps and error clip measured on the real FR3 stack."""

    _lagged = None
    _anchor = None

    # Define the supported arm and gripper command ranges.
    def _initialize_action_space(self):
        self.single_action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)

    # Reset state for a new run.
    def reset(self):
        super().reset()
        if self._anchor is None:
            self._anchor = torch.ones(self.scene.num_envs, dtype=torch.bool, device=self.device)
        self._anchor[self.scene._reset_mask] = True

    # Prepare a command and set its delayed control target.
    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos
        current = self.ee_pose_at_base
        if self._lagged is None:
            self._lagged = current.raw_pose.clone()
        self._lagged[self._anchor] = current.raw_pose[self._anchor]
        self._anchor[:] = False

        dt = 1.0 / self._control_freq
        position, quaternion = self._lagged[:, :3], self._lagged[:, 3:]
        target_q = action[:, 3:] / torch.linalg.norm(action[:, 3:], dim=1, keepdim=True).clamp(min=1e-9)
        position = position + approach(
            action[:, :3] - position, ROBOT_CONFIG["ee_deadband_meters"],
            lag_fraction(dt, ROBOT_CONFIG["ee_lag_seconds"]), ROBOT_CONFIG["ee_max_speed"] * dt,
        )
        turn = approach(
            rotation_between(target_q, quaternion), np.deg2rad(ROBOT_CONFIG["ee_deadband_degrees"]),
            lag_fraction(dt, ROBOT_CONFIG["ee_rotation_lag_seconds"]),
            np.deg2rad(ROBOT_CONFIG["ee_max_angular_speed_degrees"]) * dt,
        )
        quaternion = quaternion_multiply(axis_angle_to_quaternion(turn), quaternion)
        self._lagged = torch.cat([position, quaternion / torch.linalg.norm(quaternion, dim=1, keepdim=True)], dim=1)

        clip_p, clip_r = ROBOT_CONFIG["ee_error_clip_meters"], ROBOT_CONFIG["ee_error_clip_radians"]
        offset_p = torch.clamp(self._lagged[:, :3] - current.p, -clip_p, clip_p)
        offset_r = torch.clamp(rotation_between(self._lagged[:, 3:], current.q), -clip_r, clip_r)
        self._target_pose = Pose.create_from_pq(
            current.p + offset_p, quaternion_multiply(axis_angle_to_quaternion(offset_r), current.q)
        )
        self._target_qpos = self.kinematics.compute_ik(
            pose=self._target_pose,
            q0=self.articulation.get_qpos(),
            current_pose=current,
            solver_config=self.config.delta_solver_config,
        )
        if self._target_qpos is None:
            self._target_qpos = self._start_qpos
        self.set_drive_targets(self._target_qpos)


@dataclass
class LaggedEEPoseControllerConfig(PDEEPoseControllerConfig):
    controller_cls = LaggedEEPoseController


class RateLimitedGripperController(PDJointPosMimicController):
    """Mimic gripper that acts on each command gripper_delay_seconds late and moves at most gripper_max_speed."""

    _pending = None

    # Reset state for a new run.
    def reset(self):
        super().reset()
        if self._pending is not None:
            self._pending[:, self.scene._reset_mask] = self._target_qpos[self.scene._reset_mask][:, self.control_joint_indices]

    # Prepare an action for the controller.
    def _preprocess_action(self, action: Array):
        action = super()._preprocess_action(action)
        idx = self.control_joint_indices
        delay = int(round(ROBOT_CONFIG["gripper_delay_seconds"] * self._control_freq))
        if delay:
            if self._pending is None:
                self._pending = self._target_qpos[:, idx].repeat(delay, 1, 1)
            self._pending = torch.cat([self._pending, action[None]])
            action, self._pending = self._pending[0], self._pending[1:]
        step = ROBOT_CONFIG["gripper_max_speed"] / self._control_freq
        target = self._target_qpos[:, idx]
        qpos = self.qpos[:, idx]
        opened = torch.minimum(action, torch.maximum(target, qpos) + step)
        closed = torch.maximum(action, torch.minimum(target, qpos) - step)
        return torch.where(action > target, opened, closed)


@dataclass
class RateLimitedGripperControllerConfig(PDJointPosMimicControllerConfig):
    controller_cls = RateLimitedGripperController


@register_agent()
class FR3(BaseAgent):
    """Franka Research 3 with the standard two-finger Franka Hand."""

    uid = "fr3"
    mjcf_path = str(FR3_MJCF)
    disable_self_collisions = True

    joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
    gripper_joint_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
    keyframes = {
        "rest": Keyframe(
            pose=sapien.Pose(),
            qpos=np.array(ROBOT_CONFIG["rest_qpos"]),
        )
    }
    force_limits = ROBOT_CONFIG["arm_force_limits"]

    # Return the robot controller configurations.
    @property
    def _controller_configs(self):
        joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            lower=None,
            upper=None,
            stiffness=ROBOT_CONFIG["arm_stiffness"],
            damping=ROBOT_CONFIG["arm_damping"],
            force_limit=self.force_limits,
            normalize_action=False,
        )
        joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            lower=-ROBOT_CONFIG["arm_delta_radians"],
            upper=ROBOT_CONFIG["arm_delta_radians"],
            stiffness=ROBOT_CONFIG["arm_stiffness"],
            damping=ROBOT_CONFIG["arm_damping"],
            force_limit=self.force_limits,
            normalize_action=True,
            use_delta=True,
        )
        gripper = RateLimitedGripperControllerConfig(
            self.gripper_joint_names,
            lower=ROBOT_CONFIG["gripper_lower"],
            upper=ROBOT_CONFIG["gripper_upper"],
            stiffness=ROBOT_CONFIG["gripper_stiffness"],
            damping=ROBOT_CONFIG["gripper_damping"],
            force_limit=ROBOT_CONFIG["gripper_force_limit"],
            normalize_action=False,
            mimic={"fr3_finger_joint2": {"joint": "fr3_finger_joint1"}},
        )
        ee_delta_pose = PDEEPoseControllerConfig(
            self.joint_names,
            pos_lower=-ROBOT_CONFIG["ee_delta_meters"],
            pos_upper=ROBOT_CONFIG["ee_delta_meters"],
            rot_lower=-ROBOT_CONFIG["ee_delta_radians"],
            rot_upper=ROBOT_CONFIG["ee_delta_radians"],
            stiffness=ROBOT_CONFIG["arm_stiffness"],
            damping=ROBOT_CONFIG["arm_damping"],
            force_limit=self.force_limits,
            ee_link=ROBOT_CONFIG["tcp_link"],
            urdf_path=str(FR3_KINEMATICS_URDF),
        )
        ee_target_pose = LaggedEEPoseControllerConfig(
            self.joint_names,
            pos_lower=-1.0,
            pos_upper=1.0,
            stiffness=ROBOT_CONFIG["ee_target_stiffness"],
            damping=ROBOT_CONFIG["ee_target_damping"],
            force_limit=self.force_limits,
            ee_link=ROBOT_CONFIG["tcp_link"],
            urdf_path=str(FR3_KINEMATICS_URDF),
            normalize_action=False,
        )
        return deepcopy(
            {
                "pd_joint_pos": {"arm": joint_pos, "gripper": gripper},
                "pd_joint_delta_pos": {
                    "arm": joint_delta_pos,
                    "gripper": gripper,
                },
                "pd_ee_delta_pose": {"arm": ee_delta_pose, "gripper": gripper},
                "pd_ee_target_pose": {"arm": ee_target_pose, "gripper": gripper},
            }
        )

    # Configure the loaded robot articulation.
    def _after_loading_articulation(self):
        bodies = ET.parse(FR3_MJCF).getroot().iter("body")
        inertials = {body.get("name"): body.find("inertial") for body in bodies}
        for link in self.robot.links:
            inertial = inertials.get(link.name)
            if inertial is None:
                continue
            pose = sapien.Pose(
                p=[float(x) for x in inertial.get("pos").split()],
                q=[float(x) for x in inertial.get("quat", "1 0 0 0").split()],
            )
            for body in link._bodies:
                body.mass = float(inertial.get("mass"))
                body.cmass_local_pose = pose
                body.inertia = [float(x) for x in inertial.get("diaginertia").split()]
        for name in ("fr3_leftfinger", "fr3_rightfinger"):
            for body in self.robot.links_map[name]._bodies:
                for shape in body.get_collision_shapes():
                    if isinstance(shape, sapien.physx.PhysxCollisionShapeBox):
                        material = shape.physical_material
                        material.static_friction = material.dynamic_friction = ROBOT_CONFIG["finger_friction"]
                    shape.contact_offset = 0.005

    # Finish environment initialization.
    def _after_init(self):
        self.hand = self.robot.links_map["fr3_hand"]
        self.left_finger = self.robot.links_map["fr3_leftfinger"]
        self.right_finger = self.robot.links_map["fr3_rightfinger"]
        self.tcp = self.robot.links_map[ROBOT_CONFIG["tcp_link"]]

    # Return the largest contact force magnitude between either finger and the object.
    def finger_contact_force(self, object: Actor):
        forces = [
            torch.linalg.norm(self.scene.get_pairwise_contact_forces(link, object), axis=1)
            for link in (self.left_finger, self.right_finger)
        ]
        return torch.maximum(*forces)

    # Return whether both fingers press on the object from opposite sides.
    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        flags = []
        for link in (self.left_finger, self.right_finger):
            force = self.scene.get_pairwise_contact_forces(link, object)
            direction = link.pose.to_transformation_matrix()[..., :3, 1]
            angle = torch.rad2deg(common.compute_angle_between(direction, force))
            flags.append((torch.linalg.norm(force, axis=1) >= min_force) & (angle <= max_angle))
        return flags[0] & flags[1]


@register_env("FR3Table-v1", max_episode_steps=CONFIG["simulation"]["max_episode_steps"])
class FR3TableEnv(BaseEnv):
    """FR3 with a Franka Hand and two worktables."""

    SUPPORTED_ROBOTS = ["fr3"]
    agent: FR3

    def __init__(self, *args, robot_uids="fr3", robot_init_qpos_noise=0.0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        kwargs.setdefault("enable_shadow", CONFIG["render"]["shadows"])
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Return the default sensor configurations.
    @property
    def _default_sensor_configs(self):
        camera = CONFIG["cameras"]["front"]
        gripper = CONFIG["cameras"]["gripper"]
        pose = sapien_utils.look_at(eye=camera["eye"], target=camera["target"])
        return [
            CameraConfig(
                "front_camera",
                pose,
                camera["width"],
                camera["height"],
                np.deg2rad(camera["fov_degrees"]),
                camera["near"],
                camera["far"],
            ),
            CameraConfig(
                "gripper_camera",
                GRIPPER_CAMERA_POSE,
                gripper["width"],
                gripper["height"],
                np.deg2rad(gripper["fov_degrees"]),
                gripper["near"],
                gripper["far"],
                mount=self.agent.hand,
            ),
        ]

    # Load the FR3 model and attach its controllers and sensors.
    def _load_agent(self, options: dict):
        super()._load_agent(options, ROBOT_BASE_POSE)

    # Build tables, lighting, and robot surroundings for each environment.
    def _load_scene(self, options: dict):
        tables = CONFIG["tables"]
        self.table_materials = [
            (sapien.render.RenderMaterial(base_color=tables["top_color"]),
             sapien.render.RenderMaterial(base_color=tables["frame_color"]))
            for _ in range(self.num_envs)
        ]
        self._control_step = 0
        self._static_poses = {}
        self.main_table = self._build_table("table", tables["main"])
        self.front_table = self._build_table("front_table", tables["front"])
        self._build_robot_power_light()
        self._build_back_walls()
        ground = CONFIG["ground"]
        self.ground = build_ground(
            self.scene, floor_width=ground["width"], altitude=ground["altitude"]
        )
        floor_builder = self.scene.create_actor_builder()
        floor_material = sapien.render.RenderMaterial(base_color=ground["color"])
        floor_builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, ground["altitude"] - 0.005]),
            half_size=[ground["width"] / 2, ground["width"] / 2, 0.006],
            material=floor_material,
        )
        floor_builder.initial_pose = sapien.Pose()
        self.floor_visual = floor_builder.build_static(name="gray_floor")
        self.front_camera_pose = self._default_sensor_configs[0].pose.sp
        self.gripper_camera_pose = GRIPPER_CAMERA_POSE
        self.front_camera_marker = self._build_camera_marker(
            "front_camera_marker", self.front_camera_pose
        )
        self.gripper_camera_marker = self._build_camera_marker(
            "gripper_camera_marker", sapien.Pose()
        )
        self._hidden_objects.extend(
            [self.front_camera_marker, self.gripper_camera_marker]
        )

    # Always-on green status LEDs and a small spill light at the FR3 base.
    def _build_robot_power_light(self):
        config = CONFIG["robot_power_light"]
        material = sapien.render.RenderMaterial(base_color=config["led_color"])
        material.emission = config["led_color"]
        builder = self.scene.create_actor_builder()
        for position in config["local_led_positions"]:
            builder.add_box_visual(
                pose=sapien.Pose(p=position),
                half_size=config["led_half_size"],
                material=material,
            )
        builder.initial_pose = ROBOT_BASE_POSE
        self.robot_power_leds = builder.build_static(name="robot_power_leds")
        for glow_position, glow_target in zip(
            config["local_glow_positions"], config["local_glow_targets"]
        ):
            glow_pose = ROBOT_BASE_POSE * sapien.Pose(p=glow_position)
            target_pose = ROBOT_BASE_POSE * sapien.Pose(p=glow_target)
            self.scene.add_spot_light(
                glow_pose.p, np.asarray(target_pose.p) - np.asarray(glow_pose.p),
                inner_fov=0.35, outer_fov=0.8,
                color=config["glow_color"], shadow=False,
            )
        self.glow_lights = [
            [entity.find_component_by_type(sapien.render.RenderSpotLightComponent)
             for entity in scene.entities if entity.name == "spot_light"]
            for scene in self.scene.sub_scenes
        ]

    # Visual workshop corner behind the table, based on the real setup.
    def _build_back_walls(self):
        wall = CONFIG["back_wall"]
        texture = CONFIG["render"]["texture_variation"]
        self.wall_patterns = wall_patterns(texture["variants"], texture["resolution"], 0x7E47)
        self.wall_materials, walls = [], []
        for index in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            materials = []
            for part in wall["parts"]:
                materials.append(sapien.render.RenderMaterial(base_color=part["color"]))
                builder.add_box_visual(
                    pose=sapien.Pose(p=part["position"]),
                    half_size=part["half_size"],
                    material=materials[-1],
                )
            builder.initial_pose = sapien.Pose()
            builder.set_scene_idxs([index])
            walls.append(builder.build_static(name=f"back_wall_{index}"))
            self.wall_materials.append(materials)
        self.back_wall = Actor.merge(walls, "back_wall")

    # Create lights that can be varied reproducibly at each episode reset.
    def _load_lighting(self, options: dict):
        fill = CONFIG["render"]["shadow_variation"]["fill_intensity"]
        self.scene.add_point_light(
            [0, 0, 2.25], [2.7, 2.7, 2.7], shadow=self.enable_shadow,
            shadow_near=0.1, shadow_far=6, shadow_map_size=CONFIG["render"]["shadow_map_size"],
        )
        self.key_lights = [
            scene.entities[-1].find_component_by_type(sapien.render.RenderPointLightComponent)
            for scene in self.scene.sub_scenes
        ]
        self.scene.add_directional_light([0, 0, -1], [fill] * 3)
        self.scene.set_ambient_light([0.3, 0.3, 0.3])

    # Check whether a render variation applies to this episode.
    def _render_randomization_enabled(self, name):
        return CONFIG["domain_randomization"] and self.config.get("render_randomization", {}).get(name, True)

    # A separate RNG keeps lighting repeatable without changing task sampling.
    def _randomize_lighting(self, env_idx):
        if not self._render_randomization_enabled("lighting"):
            return
        variation = CONFIG["render"]["shadow_variation"]
        for index in env_idx.tolist():
            if self.scene.parallel_in_single_scene and index != 0:
                continue
            seed = int(self._episode_seed[index])
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0x51AD0]))
            position = [
                rng.uniform(*variation["light_x_range"]),
                rng.uniform(*variation["light_y_range"]),
                rng.uniform(*variation["light_height_range"]),
            ]
            light = self.key_lights[0 if self.scene.parallel_in_single_scene else index]
            light.entity_pose = sapien.Pose(p=position)
            intensity = rng.uniform(*variation["key_intensity_range"])
            light.color = [intensity] * 3
            ambient = rng.uniform(*variation["ambient_range"])
            scene = self.scene.sub_scenes[0 if self.scene.parallel_in_single_scene else index]
            scene.set_ambient_light([ambient] * 3)
            glow = rng.uniform(*CONFIG["robot_power_light"]["glow_intensity_range"])
            for glow_light in self.glow_lights[0 if self.scene.parallel_in_single_scene else index]:
                glow_light.color = [glow * value for value in CONFIG["robot_power_light"]["glow_color"]]
            self.lighting_parameters = {
                "seed": seed, "light_position": [float(value) for value in position],
                "intensity": float(intensity),
                "ambient": float(ambient),
                "glow": float(glow),
            }

    # Small, seed-stable color and texture shifts around the reference palette, per env.
    def _randomize_surface_colors(self, env_idx):
        ranges = CONFIG["render"]["color_variation"]
        texture = CONFIG["render"]["texture_variation"]
        wall = CONFIG["back_wall"]
        tables = CONFIG["tables"]
        for env in env_idx.tolist():
            seed = int(self._episode_seed[env])
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC010]))

            def shifted(base, amount, shared=None):
                delta = rng.uniform(-amount, amount) if shared is None else shared
                color = np.asarray(base, dtype=float).copy()
                color[:3] = np.clip(color[:3] + delta, 0, 1)
                return color.tolist()

            concrete_delta = rng.uniform(-ranges["concrete"], ranges["concrete"])
            white_delta = rng.uniform(-ranges["white_wall"], ranges["white_wall"])
            materials = self.wall_materials[env]
            for index, (material, part) in enumerate(zip(materials, wall["parts"])):
                delta = concrete_delta if index < wall["concrete_part_count"] else white_delta
                material.base_color = shifted(part["color"], 0, shared=delta)
            texture_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7E47]))
            patterns = {kind: self.wall_patterns[texture_rng.integers(len(self.wall_patterns))]
                        for kind in ("concrete", "white_wall")}
            for index, material in enumerate(materials):
                kind = "concrete" if index < wall["concrete_part_count"] else "white_wall"
                pixels = np.ones((*patterns[kind].shape, 4), dtype=np.float32)
                pixels[..., :3] = np.clip(np.asarray(material.base_color[:3]) * (1 + texture[kind] * patterns[kind])[..., None], 0, 1)
                material.base_color_texture = sapien.render.RenderTexture2D(pixels, "R32G32B32A32Sfloat")
            top, frame = self.table_materials[env]
            top.base_color = shifted(tables["top_color"], ranges["table_top"])
            frame.base_color = shifted(tables["frame_color"], ranges["table_frame"])

    # Create the worktables and their collision surfaces.
    def _build_table(self, name, table):
        parts = []
        for index, (top_material, frame_material) in enumerate(self.table_materials):
            builder = self.scene.create_actor_builder()
            top_pose = sapien.Pose(p=[0.0, 0.0, -0.02])
            builder.add_box_collision(pose=top_pose, half_size=table["top_half_size"])
            builder.add_box_visual(pose=top_pose, half_size=table["top_half_size"], material=top_material)
            for x, y in table["leg_xy"]:
                leg_pose = sapien.Pose(p=[x, y, table["leg_z"]])
                builder.add_box_collision(pose=leg_pose, half_size=table["leg_half_size"])
                builder.add_box_visual(pose=leg_pose, half_size=table["leg_half_size"], material=frame_material)
            builder.initial_pose = sapien.Pose(p=table["position"])
            builder.set_scene_idxs([index])
            parts.append(builder.build_static(name=f"{name}_{index}"))
        return Actor.merge(parts, name)

    # Add a visible marker at a camera pose.
    def _build_camera_marker(self, name, pose):
        marker = CONFIG["camera_markers"]
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=marker["radius"],
            half_length=marker["half_thickness"],
            material=sapien.render.RenderMaterial(base_color=marker["color"]),
        )
        builder.initial_pose = pose
        return builder.build_kinematic(name=name)

    # Reset the robot and randomized scene for a new episode.
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._randomize_lighting(env_idx)
        self._randomize_surface_colors(env_idx)
        self._randomize_images(env_idx)
        self._camera_reset_rows = env_idx
        randomization = self.config.get("randomization", {}) if hasattr(self, "config") else {}
        with torch.device(self.device):
            batch_size = len(env_idx)
            qpos = torch.as_tensor(
                self.agent.keyframes["rest"].qpos, dtype=torch.float32
            ).repeat(batch_size, 1)
            if randomization.get("real_start_qpos"):
                starts = real_start_qpos(randomization["real_start_qpos"])
                for row, index in enumerate(env_idx.tolist()):
                    start_rng = np.random.default_rng(np.random.SeedSequence([int(self._episode_seed[index]), 0x57A7]))
                    start = starts[start_rng.integers(len(starts))]
                    qpos[row, :7] = torch.from_numpy(start[:7])
                    qpos[row, 7:] = float(start[7]) / 2
            rng = self._batched_episode_rng[common.to_numpy(env_idx)]
            delta = rng.normal(0.0, self.robot_init_qpos_noise, (7,))
            delta = np.clip(delta, -2 * self.robot_init_qpos_noise, 2 * self.robot_init_qpos_noise)
            qpos[:, :7] += torch.from_numpy(delta).float().to(self.device)
            lowering_range = (
                self.config.get("randomization", {}).get("robot_init_lowering_radians")
                if hasattr(self, "config") else None
            )
            if lowering_range is not None:
                for row, index in enumerate(env_idx.tolist()):
                    seed = int(self._episode_seed[index])
                    lowering_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x10AE]))
                    lowering = float(lowering_rng.uniform(*lowering_range))
                    qpos[row, 1] += lowering
                    qpos[row, 3] -= lowering
            limits = torch.as_tensor(ROBOT_CONFIG["joint_limits"], dtype=qpos.dtype, device=self.device)
            qpos[:, :7] = torch.minimum(torch.maximum(qpos[:, :7], limits[:, 0] + 0.01), limits[:, 1] - 0.01)
            gripper_probability = (
                self.config.get("randomization", {}).get("gripper_initial_open_probability")
                if hasattr(self, "config") else None
            )
            if gripper_probability is not None:
                for row, index in enumerate(env_idx.tolist()):
                    seed = int(self._episode_seed[index])
                    grip_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x6A1F]))
                    opening = (
                        ROBOT_CONFIG["gripper_upper"] if grip_rng.random() < gripper_probability
                        else ROBOT_CONFIG["gripper_lower"]
                    )
                    qpos[row, 7:] = opening
            self.agent.reset(qpos)
            self.agent.robot.set_pose(ROBOT_BASE_POSE)
            self._set_randomized_cameras(env_idx)
            self._update_camera_markers()

    # Apply sampled camera poses to the active sensors.
    def _set_randomized_cameras(self, env_idx):
        base_front, base_wrist = self._default_sensor_configs[:2]
        front = self._sensors["base_camera" if "base_camera" in self.config["cameras"] else "front_camera"].camera
        wrist = self._sensors["wrist_camera" if "wrist_camera" in self.config["cameras"] else "gripper_camera"].camera
        if not self._render_randomization_enabled("camera_pose"):
            for index in env_idx.tolist():
                row = index if len(front._render_cameras) > 1 else 0
                front._render_cameras[row].set_local_pose(base_front.pose.sp)
                wrist._render_cameras[row].set_local_pose(base_wrist.pose.sp)
            front._cached_local_pose = wrist._cached_local_pose = None
            self.front_camera_pose = base_front.pose.sp
            self.gripper_camera_pose = base_wrist.pose.sp
            self._update_camera_markers()
            return
        ranges = CONFIG["cameras"]["randomization"]

        def turn(rng, degrees):
            angles = np.deg2rad(rng.uniform(-1, 1, 3) * degrees)
            pose = sapien.Pose()
            for axis, angle in zip(np.eye(3), angles):
                pose = pose * sapien.Pose(q=[np.cos(angle / 2), *(axis * np.sin(angle / 2))])
            return pose

        for index in env_idx.tolist():
            rng = np.random.default_rng(np.random.SeedSequence([int(self._episode_seed[index]), 0xCA4E]))
            front_pose = base_front.pose.sp * sapien.Pose(p=rng.uniform(-1, 1, 3) * ranges["front_translation_m"]) \
                * turn(rng, ranges["front_rotation_degrees"])
            offset = [rng.uniform(*bounds) for bounds in ranges["wrist_offset_m"]]
            wrist_pose = sapien.Pose(p=base_wrist.pose.sp.p + offset, q=base_wrist.pose.sp.q) * turn(rng, ranges["wrist_rotation_degrees"])
            row = index if len(front._render_cameras) > 1 else 0
            front._render_cameras[row].set_local_pose(front_pose)
            wrist._render_cameras[row].set_local_pose(wrist_pose)
            if index == 0:
                self.front_camera_pose, self.gripper_camera_pose = front_pose, wrist_pose
        front._cached_local_pose = wrist._cached_local_pose = None
        self._update_camera_markers()

    # Apply image randomization.
    def _randomize_images(self, env_idx):
        if not self._render_randomization_enabled("camera_response"):
            return
        variation = CONFIG["render"]["image_variation"]
        if not variation["enabled"]:
            return
        if not hasattr(self, "image_parameters"):
            self.image_parameters = {name: torch.zeros((self.num_envs, 7)) for name in self._sensors}
        for index in env_idx.tolist():
            seed = int(self._episode_seed[index])
            for camera_index, name in enumerate(self.image_parameters):
                rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1A6E, camera_index]))
                balance = 1.0 + rng.uniform(-1, 1, 3) * variation["white_balance"]
                self.image_parameters[name][index] = torch.tensor([
                    rng.uniform(*variation["gain_range"]), *balance, rng.uniform(*variation["gamma_range"]),
                    rng.uniform(*variation["blur_sigma_range"]), rng.uniform(*variation["distortion_k1_range"]),
                ])
        self.image_noise = None
        self.image_noise_seed = int(self._episode_seed[0]) + 0x1A6E

    # Apply the camera response to a (B, H, W, 3) uint8 batch; noise is per pixel and per frame.
    def _vary_image(self, rgb, parameters):
        variation = CONFIG["render"]["image_variation"]
        parameters = parameters.to(rgb.device)
        batch, height, width, _ = rgb.shape
        image = rgb.permute(0, 3, 1, 2).float() / 255.0
        aspect = height / width
        y, x = torch.meshgrid(
            torch.linspace(-aspect, aspect, height, device=rgb.device),
            torch.linspace(-1.0, 1.0, width, device=rgb.device),
            indexing="ij",
        )
        k1 = parameters[:, 6, None, None]
        scale = (1.0 - k1 * (x * x + y * y)) / (1.0 - k1 * (1.0 + aspect * aspect))
        grid = torch.stack([x * scale, y * scale * width / height], dim=-1)
        image = torch.nn.functional.grid_sample(image, grid, align_corners=True, padding_mode="border")
        radius = int(np.ceil(3 * variation["blur_sigma_range"][1]))
        if radius:
            offsets = torch.arange(-radius, radius + 1, device=rgb.device, dtype=torch.float32)
            kernel = torch.exp(-0.5 * (offsets / parameters[:, 5:6].clamp(min=1e-3)) ** 2)
            kernel = (kernel / kernel.sum(1, keepdim=True)).repeat_interleave(3, dim=0)
            image = image.reshape(1, batch * 3, height, width)
            image = torch.nn.functional.conv2d(
                torch.nn.functional.pad(image, (radius, radius, 0, 0), mode="replicate"),
                kernel[:, None, None, :], groups=batch * 3,
            )
            image = torch.nn.functional.conv2d(
                torch.nn.functional.pad(image, (0, 0, radius, radius), mode="replicate"),
                kernel[:, None, :, None], groups=batch * 3,
            ).reshape(batch, 3, height, width)
        gain = (parameters[:, 0:1] * parameters[:, 1:4])[:, :, None, None]
        image = (image * gain).clamp(0, 1) ** parameters[:, 4, None, None, None]
        if self.image_noise is None or self.image_noise.device != rgb.device:
            self.image_noise = torch.Generator(device=rgb.device).manual_seed(self.image_noise_seed)
        noise = torch.randn(image.shape, generator=self.image_noise, device=rgb.device)
        image = image + noise * variation["noise_std"] / 255.0
        return (image.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)

    # Apply latency and image variation to camera observations.
    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        data = super()._get_obs_sensor_data(apply_texture_transforms)
        self._delay_images(data)
        self._blur_wrist(data)
        if (CONFIG["render"]["image_variation"]["enabled"]
                and self._render_randomization_enabled("camera_response")
                and hasattr(self, "image_parameters")):
            for name, sensor in data.items():
                if "rgb" in sensor:
                    sensor["rgb"] = self._vary_image(sensor["rgb"], self.image_parameters[name])
        return data

    # Weak motion blur on the hand camera along the image motion of a point at motion_blur_depth_meters.
    def _blur_wrist(self, data):
        variation = CONFIG["render"]["image_variation"]
        name = "wrist_camera" if "wrist_camera" in data else "gripper_camera"
        exposure = variation["wrist_motion_blur_seconds"]
        if (not variation["enabled"] or not self._render_randomization_enabled("camera_response")
                or not exposure or name not in data or "rgb" not in data[name]):
            return
        rgb = data[name]["rgb"]
        batch, height, width, _ = rgb.shape
        camera = self.agent.hand.pose * self.gripper_camera_pose
        world_to_camera = quaternion_to_matrix(camera.q).transpose(1, 2)
        velocity = (world_to_camera @ self.agent.hand.get_linear_velocity()[..., None])[..., 0]
        spin = (world_to_camera @ self.agent.hand.get_angular_velocity()[..., None])[..., 0]
        focal = height / 2 / np.tan(self._sensors[name].config.fov / 2)
        depth = variation["motion_blur_depth_meters"]
        streak = torch.stack([velocity[:, 1] / depth + spin[:, 2], velocity[:, 2] / depth - spin[:, 1]], dim=1)
        streak = (streak * focal * exposure).to(rgb.device).float()
        radius = int(min(7, np.ceil(float(streak.norm(dim=1).max()) / 2)))
        if radius == 0:
            return
        offsets = torch.arange(-radius, radius + 1, device=rgb.device, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(offsets, offsets, indexing="xy"), dim=-1).reshape(1, -1, 2)
        start, end = -streak[:, None] / 2, streak[:, None] / 2
        along = ((grid - start) * (end - start)).sum(-1) / ((end - start) ** 2).sum(-1).clamp(min=1e-6)
        distance = (grid - (start + along.clamp(0, 1)[..., None] * (end - start))).norm(dim=-1)
        kernel = (1.0 - distance).clamp(min=0.0).reshape(batch, 1, 2 * radius + 1, 2 * radius + 1)
        kernel = (kernel / kernel.sum(dim=(2, 3), keepdim=True)).repeat_interleave(3, dim=0)
        image = rgb.permute(0, 3, 1, 2).float().reshape(1, batch * 3, height, width)
        image = torch.nn.functional.conv2d(
            torch.nn.functional.pad(image, (radius,) * 4, mode="replicate"), kernel, groups=batch * 3
        )
        data[name]["rgb"] = image.reshape(batch, 3, height, width).permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)

    # Apply image latency.
    def _delay_images(self, data):
        if not hasattr(self, "camera_history"):
            self.camera_history = {}
        rows = getattr(self, "_camera_reset_rows", None)
        self._camera_reset_rows = None
        for name, sensor in data.items():
            delay = int(round(CONFIG["latency"]["camera_seconds"].get(name, 0.0) * self.control_freq))
            if not delay or "rgb" not in sensor:
                continue
            history = self.camera_history.setdefault(name, deque(maxlen=delay + 1))
            if not history:
                history.extend(sensor["rgb"].clone() for _ in range(delay))
            elif rows is not None:
                for frame in history:
                    frame[rows] = sensor["rgb"][rows]
            history.append(sensor["rgb"].clone())
            sensor["rgb"] = history[0]

    # Update task state after a control step.
    def _after_control_step(self):
        self._control_step += 1
        self._update_camera_markers()

    # Actor.is_static, plus, on the GPU, an actor whose pose did not change over the last control step.
    def is_static(self, actor: Actor, lin_thresh=1e-2, ang_thresh=1e-1) -> torch.Tensor:
        static = actor.is_static(lin_thresh=lin_thresh, ang_thresh=ang_thresh)
        if not self.gpu_sim_enabled:
            return static
        history = self._static_poses.setdefault(actor.name, {})
        if history.get("step") != self._control_step:
            pose = actor.pose.raw_pose.clone()
            history.update(step=self._control_step, previous=history.get("current", pose), current=pose)
        previous, current = history["previous"], history["current"]
        linear = torch.linalg.norm(current[:, :3] - previous[:, :3], dim=1) * self.control_freq
        cos_half = torch.abs((current[:, 3:] * previous[:, 3:]).sum(dim=1)).clamp(max=1.0)
        angular = 2 * torch.arccos(cos_half) * self.control_freq
        return static | ((linear <= lin_thresh) & (angular <= ang_thresh))

    # Move visible markers to the current camera poses.
    def _update_camera_markers(self):
        self.front_camera_marker.set_pose(self.front_camera_pose)
        self.gripper_camera_marker.set_pose(
            self.agent.hand.pose * self.gripper_camera_pose
        )

    # Apply requested front and wrist camera positions.
    def set_camera_positions(self, front_position, gripper_position, front_target=None):
        cameras = self.config["cameras"] if hasattr(self, "config") else CONFIG["cameras"]
        base_name = "base_camera" if "base_camera" in cameras else "front"
        wrist_name = "wrist_camera" if "wrist_camera" in cameras else "gripper"
        target = cameras[base_name]["target"] if front_target is None else front_target
        self.front_camera_pose = sapien_utils.look_at(front_position, target).sp
        self.gripper_camera_pose = sapien.Pose(
            p=gripper_position,
            q=cameras[wrist_name]["quaternion"],
        )
        self._sensors[base_name].camera.set_local_pose(self.front_camera_pose)
        self._sensors[wrist_name].camera.set_local_pose(
            self.gripper_camera_pose
        )
        self._update_camera_markers()

    # Set camera marker visibility.
    def show_camera_markers(self):
        self.front_camera_marker.show_visual()
        self.gripper_camera_marker.show_visual()

    # Return success and failure flags for the current task.
    def evaluate(self):
        return {}

    # Add robot and task state to the observation.
    def _get_obs_extra(self, info: dict):
        return {"hand_pose": self.agent.hand.pose.raw_pose}

    # Score task progress for reinforcement learning.
    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    # Scale the task reward to its expected range.
    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs, action, info)


COLORS = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "orange": [1.0, 0.5, 0.0],
    "purple": [0.5, 0.0, 0.8],
    "cyan": [0.0, 1.0, 1.0],
    "white": [1.0, 1.0, 1.0],
    "gray": [0.5, 0.5, 0.5],
    "black": [0.0, 0.0, 0.0],
}
OOD_SHAPES = {
    "bar": ([([0.05, 0.012, 0.012], [0.0, 0.0, 0.0])], 0.05, 0.012),
    "l_shape": ([([0.04, 0.012, 0.012], [0.0, 0.0, 0.0]), ([0.012, 0.03, 0.012], [0.028, 0.042, 0.0])], 0.08, 0.012),
    "plate": ([([0.035, 0.035, 0.006], [0.0, 0.0, 0.0])], 0.05, 0.006),
    "block": ([([0.015, 0.025, 0.035], [0.0, 0.0, 0.0])], 0.03, 0.035),
}


def rgba(color):
    rgb = COLORS[color] if isinstance(color, str) else color
    return [*rgb[:3], 1.0]


# Return a named color, mapping an RGB list to the nearest entry in COLORS.
def color_name(color):
    if isinstance(color, str):
        return color
    return min(COLORS, key=lambda name: np.linalg.norm(np.subtract(COLORS[name], color[:3])))


# Load the config.yaml stored next to a task module.
def load_task_config(task_file):
    with (Path(task_file).parent / "config.yaml").open() as config_file:
        return yaml.safe_load(config_file)


class FR3TaskEnv(FR3TableEnv):
    """FR3 tabletop task with per-task config, two cameras, and typed failure signals."""

    config: dict
    failure_keys: tuple
    task_objects: tuple = ()
    zone_targets: tuple = ()
    info_keys = (
        "success",
        "fail",
        "is_grasped",
        "ever_grasped",
        "is_static",
        "hold_steps",
        "dropped",
        "misaligned",
        "unstable",
        "rotation_failure",
        "unstable_hold",
        "never_contacted",
        "stopped_short",
        "overshoot",
        "tipped",
        "disturbed",
        "never_grasped",
        "timeout",
        "grasped_distractor",
        "wrong_zone",
        "swapped",
    )

    def __init__(self, *args, showcase_camera=False, **kwargs):
        self.showcase_camera = showcase_camera
        noise = self.config["randomization"]["robot_init_qpos_noise"]
        kwargs.setdefault("robot_init_qpos_noise", noise)
        super().__init__(*args, **kwargs)

    # The showcase video's exterior camera, built only when asked for since it renders at a high resolution.
    @property
    def _default_human_render_camera_configs(self):
        if not self.showcase_camera:
            return []
        cfg = CONFIG["showcase_video"]
        return CameraConfig(
            "showcase_camera", self.showcase_pose(0.0), cfg["size"], cfg["size"], np.deg2rad(cfg["fov_degrees"]),
            0.01, 10.0, shader_pack=cfg["shader_pack"],
        )

    # Showcase camera pose after seconds of episode time: a slow sinusoidal sweep around the board.
    @staticmethod
    def showcase_pose(seconds: float) -> Pose:
        cfg = CONFIG["showcase_video"]
        sweep = np.sin(2 * np.pi * seconds / cfg["sweep_seconds"]) * cfg["sweep_degrees"] / 2
        azimuth = np.deg2rad(cfg["azimuth_degrees"] + sweep)
        target = np.asarray(cfg["target"], dtype=float)
        eye = target + [cfg["radius"] * np.cos(azimuth), cfg["radius"] * np.sin(azimuth), cfg["height"]]
        return sapien_utils.look_at(eye, target)

    # Return the default sensor configurations.
    @property
    def _default_sensor_configs(self):
        cameras = self.config["cameras"]
        resolution = cameras.get("resolution", DATA_CONFIG["cameras"])
        size = (resolution["width"], resolution["height"])
        base = cameras["base_camera"]
        wrist = cameras["wrist_camera"]
        return [
            CameraConfig(
                "base_camera",
                sapien_utils.look_at(base["eye"], base["target"]),
                *size,
                np.deg2rad(base["fov_degrees"]),
                base["near"],
                base["far"],
            ),
            CameraConfig(
                "wrist_camera",
                sapien.Pose(wrist["position"], wrist["quaternion"]),
                *size,
                np.deg2rad(wrist["fov_degrees"]),
                wrist["near"],
                wrist["far"],
                mount=self.agent.hand,
            ),
        ]

    # CPU float samples of shape (len(env_idx), *size) drawn from each env's own seeded RNG.
    def _uniform(self, env_idx, low, high, size=()):
        rng = self._batched_episode_rng[common.to_numpy(env_idx)]
        return torch.from_numpy(rng.uniform(low, high, size)).float().cpu()

    # Wxyz quaternions of a random rotation about z within +-half_range, one per env, on the sim device.
    def _yaw_quaternions(self, env_idx, half_range=np.pi):
        half_yaw = self._uniform(env_idx, -half_range, half_range) / 2
        zeros = torch.zeros_like(half_yaw)
        return torch.stack([torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)], dim=1).to(self.device)

    # Sample a reachable tabletop position within configured bounds.
    def _sample_workspace_xy(self, env_idx, valid=None, enforce_reachability="both"):
        rand = self.config["randomization"]
        low = [rand["x_range"][0], rand["y_range"][0]]
        high = [rand["x_range"][1], rand["y_range"][1]]
        env_idx = common.to_numpy(env_idx)
        with torch.device("cpu"):
            base = torch.tensor(ROBOT_BASE_POSE.p[:2], dtype=torch.float32)
            xy = torch.zeros((len(env_idx), 2))
            pending = torch.ones(len(env_idx), dtype=torch.bool)
            for _ in range(1000):
                if not pending.any():
                    return xy
                idx = torch.nonzero(pending)[:, 0]
                candidate = self._uniform(env_idx[idx.numpy()], low, high, (2,))
                if enforce_reachability is True or enforce_reachability == "both":
                    radius = torch.linalg.norm(candidate - base, axis=1)
                    ok = (radius >= rand["min_radius"]) & (radius <= rand["max_radius"])
                elif enforce_reachability == "minimum":
                    radius = torch.linalg.norm(candidate - base, axis=1)
                    ok = radius >= rand["min_radius"]
                else:
                    ok = torch.ones(len(candidate), dtype=torch.bool)
                if valid is not None:
                    ok &= valid(candidate, idx)
                xy[idx[ok]] = candidate[ok]
                pending[idx[ok]] = False
        raise RuntimeError("could not sample a valid workspace position; check the randomization config")

    # Build tables, lighting, and robot surroundings for each environment.
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.ever_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reached_goal = torch.zeros_like(self.ever_grasped)
        self.hold_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.reflex = {key: torch.zeros_like(self.ever_grasped) for key in REFLEX_KEYS}
        self._reflex_fresh = torch.ones_like(self.ever_grasped)
        self._torque_saturated = torch.zeros_like(self.hold_steps)
        self._last_qvel = torch.zeros((self.num_envs, 7), device=self.device)
        self._last_tcp = torch.zeros((self.num_envs, 7), device=self.device)
        self.prompts = [""] * self.num_envs
        self._load_distractors()

    # Return colors for distractor objects.
    def _distractor_colors(self, colors):
        used = {color_name(v) for k, v in self.config["objects"].items() if k.endswith("_color")}
        allowed = [c for c in colors if color_name(c) not in used]
        if not allowed:
            raise ValueError(f"all distractor colors {colors} are used by task objects")
        return allowed

    # Add randomized objects that are not part of the task.
    def _load_distractors(self):
        cfg = self.config["distractors"]
        self.distractors = []
        self.wrong_zones = []
        cubes = cfg["cubes"]
        if cubes["enabled"]:
            colors = self._distractor_colors(cubes["colors"])
            h = cubes["half_size"]
            for i in range(cubes["count"]):
                cube = actors.build_cube(
                    self.scene, half_size=h, color=rgba(colors[i % len(colors)]),
                    name=f"distractor_cube_{i}", initial_pose=sapien.Pose(p=[i, -2.0, 0.1]),
                )
                self.distractors.append((cube, np.sqrt(2) * h, h))
        ood = cfg["ood_objects"]
        if ood["enabled"]:
            colors = self._distractor_colors(ood["colors"])
            for i in range(ood["count"]):
                shape = ood["shapes"][i % len(ood["shapes"])]
                boxes, radius, half_height = OOD_SHAPES[shape]
                material = sapien.render.RenderMaterial(base_color=rgba(colors[i % len(colors)]))
                builder = self.scene.create_actor_builder()
                for half_size, offset in boxes:
                    builder.add_box_collision(sapien.Pose(p=offset), half_size=half_size)
                    builder.add_box_visual(sapien.Pose(p=offset), half_size=half_size, material=material)
                builder.initial_pose = sapien.Pose(p=[i, -2.5, 0.1])
                self.distractors.append((builder.build(name=f"distractor_{shape}_{i}"), radius, half_height))
        ycb = cfg.get("ycb")
        if ycb and ycb["enabled"]:
            root = PROJECT_ROOT / "fr3_assets/data/assets/mani_skill2_ycb"
            with (root / "info_pick_v0.json").open() as file:
                metadata = json.load(file)
            choices = ycb["objects"]
            for i in range(ycb["count"]):
                choice = choices[i % len(choices)]
                model = metadata[choice["asset_id"]]
                scale = np.broadcast_to(np.asarray(choice["scale"], dtype=float), 3)
                extent = (np.asarray(model["bbox"]["max"]) - np.asarray(model["bbox"]["min"])) * scale / 2
                model_dir = root / "models" / choice["asset_id"]
                builder = self.scene.create_actor_builder()
                builder.add_multiple_convex_collisions_from_file(
                    filename=str(model_dir / "collision.ply"), scale=scale.tolist()
                )
                builder.add_visual_from_file(
                    filename=str(model_dir / "textured.obj"),
                    scale=scale.tolist(),
                    material=(
                        sapien.render.RenderMaterial(base_color=rgba(choice["color"]))
                        if choice.get("color") is not None else None
                    ),
                )
                builder.initial_pose = sapien.Pose(p=[i, -2.7, 0.1])
                actor = builder.build(name=f"distractor_{choice['asset_id']}_{i}")
                self.distractors.append((actor, float(np.linalg.norm(extent[:2])), float(extent[2])))
        zones = cfg["wrong_zones"]
        if zones["enabled"]:
            colors = self._distractor_colors(zones["colors"])
            for i in range(zones["count"]):
                builder = self.scene.create_actor_builder()
                builder.add_box_visual(
                    half_size=[zones["half_size"], zones["half_size"], 5e-4],
                    material=sapien.render.RenderMaterial(base_color=rgba(colors[i % len(colors)])),
                )
                builder.initial_pose = sapien.Pose(p=[i, -3.0, 0.0])
                self.wrong_zones.append(builder.build_kinematic(name=f"wrong_zone_{i}"))
        self.wrong_zone_xy = torch.zeros((self.num_envs, len(self.wrong_zones), 2), device=self.device)

    # Reset the robot and randomized scene for a new episode.
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        for flag in self.reflex.values():
            flag[env_idx] = False
        self._reflex_fresh[env_idx] = True
        self._torque_saturated[env_idx] = 0
        self.ever_grasped[env_idx] = False
        self.reached_goal[env_idx] = False
        self.hold_steps[env_idx] = 0
        prompt = self.config["prompt"]
        fields = {k: color_name(v) for k, v in self.config["objects"].items() if k.endswith("_color")}
        for i in env_idx.tolist():
            choice = self._batched_episode_rng[i].randint(len(prompt["templates"])) if prompt["randomize"] else 0
            self.prompts[i] = prompt["templates"][choice].format(**fields)

    # Place distractors in free reachable space; call after the task objects are posed.
    def _place_distractors(self, env_idx: torch.Tensor):
        if not self.distractors and not self.wrong_zones:
            return
        b = len(env_idx)
        gap = self.config["distractors"]["clearance"]
        circles = [(actor.pose.p[env_idx, :2].cpu(), radius) for actor, radius in self.task_objects]
        squares = []
        segments = []
        for cube, center in self.zone_targets:
            center = center.cpu()
            end = center.expand(b, 2) if center.ndim == 1 else center[common.to_numpy(env_idx)]
            start = cube.pose.p[env_idx, :2].cpu()
            direction = (end - start) / torch.linalg.norm(end - start, axis=1, keepdim=True).clamp(min=1e-6)
            squares.append((end, self.zone_half_size))
            segments.append((start - direction * 0.08, end))

        def clear(xy, idx, radius):
            ok = torch.ones(len(idx), dtype=torch.bool)
            for center, r in circles:
                ok &= torch.linalg.norm(xy - center[idx], axis=1) >= r + radius + gap
            for center, half in squares:
                ok &= (xy - center[idx]).abs().amax(dim=1) >= half + radius + gap
            for start, end in segments:
                seg = end[idx] - start[idx]
                t = torch.clamp(((xy - start[idx]) * seg).sum(1) / (seg * seg).sum(1).clamp(min=1e-9), 0, 1)
                ok &= torch.linalg.norm(xy - (start[idx] + t[:, None] * seg), axis=1) >= radius + gap
            return ok

        with torch.device(self.device):
            half = self.config["distractors"]["wrong_zones"]["half_size"]
            for i, zone in enumerate(self.wrong_zones):
                xy = self._sample_workspace_xy(env_idx, valid=lambda xy, idx: clear(xy, idx, np.sqrt(2) * half))
                xyz = torch.full((b, 3), FRONT_TABLE_TOP_Z + 1e-3)
                xyz[:, :2] = xy
                zone.set_pose(Pose.create_from_pq(p=xyz))
                self.wrong_zone_xy[env_idx, i] = xy.to(self.device)
                squares.append((xy, half))
            for actor, radius, half_height in self.distractors:
                xy = self._sample_workspace_xy(env_idx, valid=lambda xy, idx: clear(xy, idx, radius))
                xyz = torch.full((b, 3), FRONT_TABLE_TOP_Z + half_height)
                xyz[:, :2] = xy
                q = self._yaw_quaternions(env_idx)
                actor.set_pose(Pose.create_from_pq(p=xyz, q=q))
                circles.append((xy, radius))

    # Flag simulated motions that exceed real robot reflex limits.
    def _update_reflex(self):
        cfg = ROBOT_CONFIG["reflex"]
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        q, qd = self.agent.robot.get_qpos()[:, :7], self.agent.robot.get_qvel()[:, :7]
        tcp = self.agent.tcp.pose.raw_pose
        limits = torch.as_tensor(ROBOT_CONFIG["joint_limits"], dtype=q.dtype, device=q.device)
        margin = torch.minimum(q - limits[:, 0], limits[:, 1] - q).amin(dim=1)
        self.reflex["reflex_joint_limit"] |= margin < cfg["joint_limit_margin"]
        velocity_limits = torch.as_tensor(cfg["joint_velocity_limits"], dtype=q.dtype, device=q.device)
        self.reflex["reflex_joint_velocity"] |= (qd.abs() > velocity_limits).any(dim=1)
        acceleration = (qd - self._last_qvel).abs().amax(dim=1) * self.control_freq
        speed = torch.linalg.norm(tcp[:, :3] - self._last_tcp[:, :3], dim=1) * self.control_freq
        cos_half = torch.abs((tcp[:, 3:] * self._last_tcp[:, 3:]).sum(dim=1)).clamp(max=1.0)
        spin = 2 * torch.arccos(cos_half) * self.control_freq
        jump = (
            (acceleration > cfg["max_joint_acceleration"]) | (speed > cfg["max_tcp_speed"])
            | (spin > np.deg2rad(cfg["max_tcp_angular_speed_degrees"]))
        )
        self.reflex["reflex_motion"] |= jump & ~self._reflex_fresh
        arm = getattr(self.agent.controller, "controllers", {}).get("arm")
        target = getattr(arm, "_target_qpos", None)
        if target is not None:
            gains = [torch.as_tensor(arm.config.stiffness, dtype=q.dtype, device=q.device),
                     torch.as_tensor(arm.config.damping, dtype=q.dtype, device=q.device)]
            torque = gains[0] * (target.to(q.device) - q) - gains[1] * qd
            force_limits = torch.as_tensor(ROBOT_CONFIG["arm_force_limits"], dtype=q.dtype, device=q.device)
            saturated = (torque.abs() >= force_limits).any(dim=1)
            self._torque_saturated = torch.where(saturated, self._torque_saturated + 1, 0)
            self.reflex["reflex_torque"] |= self._torque_saturated >= cfg["torque_saturation_steps"]
        self._last_qvel = qd.clone()
        self._last_tcp = tcp.clone()
        self._reflex_fresh[:] = False

    # Update task state after a control step.
    def _after_control_step(self):
        super()._after_control_step()
        self._update_reflex()
        state = self._task_state()
        self.ever_grasped |= state["is_grasped"]
        self.reached_goal |= state["reached_goal"]
        self.hold_steps = torch.where(state["hold_ok"], self.hold_steps + 1, 0)

    # Return the current task state.
    def _task_state(self) -> dict:
        raise NotImplementedError

    # Return the task-specific failure flags named in failure_keys.
    def _failures(self, state: dict, success: torch.Tensor, timeout: torch.Tensor) -> dict:
        raise NotImplementedError

    # Return success and failure flags for the current task.
    def evaluate(self):
        state = self._task_state()
        success = self.hold_steps >= self.config["success"]["hold_steps"]
        max_steps = self.config["episode"]["max_episode_steps"]
        timeout = (self.elapsed_steps >= max_steps) & ~success
        info = {key: torch.zeros_like(success) for key in self.info_keys}
        info.update({key: value for key, value in state.items() if key not in ("reached_goal", "hold_ok")})
        info.update(
            success=success,
            fail=state["off_table"],
            ever_grasped=self.ever_grasped.clone(),
            hold_steps=self.hold_steps.clone(),
            never_grasped=~self.ever_grasped,
            timeout=timeout,
        )
        info.update(self._failures(state, success, timeout))
        info.update({key: flag.clone() for key, flag in self.reflex.items()})
        info["reflex"] = torch.stack(list(self.reflex.values())).any(dim=0)
        for actor, _, _ in self.distractors:
            info["grasped_distractor"] = info["grasped_distractor"] | self.agent.is_grasping(actor)
        if self.wrong_zones:
            half = self.config["distractors"]["wrong_zones"]["half_size"]
            for cube, _ in self.zone_targets:
                offset = (cube.pose.p[:, None, :2] - self.wrong_zone_xy).abs().amax(dim=2)
                info["wrong_zone"] |= (offset <= half).any(dim=1) & state["is_static"]
        return info
