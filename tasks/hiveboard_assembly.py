# Builds a fixed HiveBoard support and a loose part from URDF geometry for assembly tasks.
# Created: 2026-10-22

from typing import Any

import numpy as np
import sapien
import torch
import trimesh
from scipy.spatial.transform import Rotation

from env.fr3_table_env import FRONT_TABLE_TOP_Z, rgba
from mani_skill.utils.structs.pose import Pose
from tasks.hiveboard import ASSET_ROOT, HiveboardFR3Env

CYLINDER_TO_SAPIEN = np.eye(4)
CYLINDER_TO_SAPIEN[:3, :3] = Rotation.from_euler("y", -np.pi / 2).as_matrix()


# 4x4 transform of a URDF origin given as {xyz, rpy} (rpy as in URDF: fixed-axis roll, pitch, yaw).
def urdf_transform(origin):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", origin.get("rpy", [0.0, 0.0, 0.0])).as_matrix()
    transform[:3, 3] = origin.get("xyz", [0.0, 0.0, 0.0])
    return transform


# Vertex arrays of the objects (`o` blocks) in an OBJ file.
def obj_pieces(path):
    pieces = [[]]
    for line in path.read_text().splitlines():
        if line.startswith("o ") and pieces[-1]:
            pieces.append([])
        elif line.startswith("v "):
            pieces[-1].append([float(value) for value in line.split()[1:4]])
    return pieces


# Convert the input to a pose.
def to_pose(transform):
    x, y, z, w = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return sapien.Pose(p=transform[:3, 3], q=[w, x, y, z])


class HiveboardAssemblyFR3Env(HiveboardFR3Env):
    """A fixed support in the centre board cell and a loose part beside the board, both from the HiveBoard URDF.

    Config `assembly` holds the URDF geometry: `urdf_to_module` maps the URDF frame to the module frame (z up, origin at
    the bottom centre of the hexagonal base plate), and `support` / `part` list visual and collision entries with their
    URDF origins. The part keeps its assembled orientation turned by `rest_rpy` and rests on its lowest point.
    """

    upright_module = True
    failure_keys = ("never_grasped", "dropped", "timeout")

    # Load the joint and geometry settings for this task module.
    @property
    def mechanism_config(self):
        return {}

    # Load a fixed support and loose part from their URDF geometry.
    def _build_module(self):
        cfg = self.config["assembly"]
        objects = self.config["objects"]
        asset_dir = ASSET_ROOT / cfg["asset_dir"]
        module_frame = urdf_transform(cfg["urdf_to_module"])
        contact = sapien.physx.PhysxMaterial(cfg["contact_friction"], cfg["contact_friction"], 0.0)

        builder = self.scene.create_actor_builder()
        self._add_shapes(builder, cfg["support"], asset_dir, module_frame, objects["support_color"], contact)
        center = objects["board_center_xy"]
        builder.initial_pose = sapien.Pose(p=[center[0], center[1], FRONT_TABLE_TOP_Z])
        self.valve = builder.build_kinematic(name=f"{self.module_name}_support")
        self.pivot_height = FRONT_TABLE_TOP_Z + objects["board_surface_height"]

        part = cfg["part"]
        rest = np.eye(4)
        rest[:3, :3] = Rotation.from_euler("xyz", part.get("rest_rpy", [0.0, 0.0, 0.0])).as_matrix()
        rested = rest @ module_frame @ urdf_transform(part.get("link", {}))
        meshes = self._collision_meshes(part, asset_dir, rested)
        points = np.concatenate([mesh.vertices for mesh in meshes])
        center_xy = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
        body_frame = rested.copy()
        body_frame[:3, 3] -= [center_xy[0], center_xy[1], points[:, 2].min()]
        density = part["mass"] / sum(mesh.volume for mesh in meshes)
        self.part_body_frame, self.part_density, self.part_contact = body_frame, density, contact
        builder = self.scene.create_actor_builder()
        self._add_shapes(builder, part, asset_dir, body_frame, objects["part_color"], contact, density)
        builder.initial_pose = sapien.Pose(p=[0.3, 0.3, FRONT_TABLE_TOP_Z + 0.1])
        self.part = builder.build(name=f"{self.module_name}_part")
        for body in self.part._bodies:
            body.inertia = body.inertia * part["mass"] / body.mass
            body.mass = part["mass"]
        self.part_radius = float(np.linalg.norm(points[:, :2] - center_xy, axis=1).max())
        self.part_seated_pose = to_pose(module_frame @ urdf_transform(part.get("link", {})) @ np.linalg.inv(body_frame))
        self.task_objects = ((self.part, self.part_radius),)

    # Attach visual and collision geometry to a module body.
    @staticmethod
    def _add_shapes(builder, spec, asset_dir, frame, color, contact, density=None):
        material = sapien.render.RenderMaterial(base_color=rgba(color))
        default_scale = spec.get("mesh_scale", 1.0)
        for visual in spec["visuals"]:
            builder.add_visual_from_file(
                str(asset_dir / visual["file"]), pose=to_pose(frame @ urdf_transform(visual)),
                scale=[visual.get("scale", default_scale)] * 3, material=material,
            )
        physics = dict(material=contact) if density is None else dict(material=contact, density=density)
        for collision in spec["collisions"]:
            transform = frame @ urdf_transform(collision)
            if "box" in collision:
                builder.add_box_collision(pose=to_pose(transform), half_size=np.asarray(collision["box"]) / 2, **physics)
            elif "cylinder" in collision:
                radius, length = collision["cylinder"]
                builder.add_cylinder_collision(pose=to_pose(transform @ CYLINDER_TO_SAPIEN), radius=radius,
                                               half_length=length / 2, **physics)
            elif "capsule" in collision:
                radius, length = collision["capsule"]
                builder.add_capsule_collision(pose=to_pose(transform @ CYLINDER_TO_SAPIEN), radius=radius,
                                              half_length=length / 2 - radius, **physics)
            else:
                add = builder.add_multiple_convex_collisions_from_file if collision.get("convex_parts") else \
                    builder.add_convex_collision_from_file
                add(str(asset_dir / collision["file"]), pose=to_pose(transform),
                    scale=[collision.get("scale", default_scale)] * 3, **physics)

    # The part's collision shapes as convex trimeshes in `frame`, for its resting height, footprint and volume.
    @staticmethod
    def _collision_meshes(spec, asset_dir, frame):
        meshes = []
        default_scale = spec.get("mesh_scale", 1.0)
        for collision in spec["collisions"]:
            if "box" in collision:
                mesh = trimesh.creation.box(collision["box"])
            elif "cylinder" in collision:
                mesh = trimesh.creation.cylinder(radius=collision["cylinder"][0], height=collision["cylinder"][1])
            else:
                scale = collision.get("scale", default_scale)
                path = asset_dir / collision["file"]
                pieces = obj_pieces(path) if collision.get("convex_parts") else [trimesh.load(path, force="mesh").vertices]
                for points in pieces:
                    hull = trimesh.convex.convex_hull(np.asarray(points) * scale)
                    meshes.append(hull.apply_transform(frame @ urdf_transform(collision)))
                continue
            mesh.apply_transform(frame @ urdf_transform(collision))
            meshes.append(mesh)
        return meshes

    # Place the loose part beside the board and reset its task state.
    def _initialize_module(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            near, far = self.config["randomization"]["part_board_distance"]
            board = self.board_center_xy[env_idx].cpu()

            # Compute a resting pose for the loose part beside the board.
            def beside_board(xy, idx):
                distance = torch.linalg.norm(xy - board[idx], axis=1) - self.part_radius
                return (distance >= near) & (distance <= far)

            xyz = torch.full((b, 3), FRONT_TABLE_TOP_Z + 5e-4)
            xyz[:, :2] = self._sample_workspace_xy(env_idx, valid=beside_board)
            self.part.set_pose(Pose.create_from_pq(p=xyz, q=self._yaw_quaternions(env_idx)))
            self.part.set_linear_velocity(torch.zeros((b, 3)))
            self.part.set_angular_velocity(torch.zeros((b, 3)))

    # Measure grasp, part placement, and assembled alignment.
    def _task_state(self):
        tol = self.config["success"]
        grasped = self.agent.is_grasping(self.part)
        no_goal = torch.zeros_like(grasped)
        return dict(
            is_grasped=grasped,
            is_static=self.part.is_static(lin_thresh=tol["linear_velocity"], ang_thresh=tol["part_angular_velocity"]),
            is_contacted=self.agent.finger_contact_force(self.part) >= tol["contact_force"],
            part_to_module_dist=torch.linalg.norm(self.part.pose.p[:, :2] - self.valve.pose.p[:, :2], axis=1),
            off_table=self.part.pose.p[:, 2] < FRONT_TABLE_TOP_Z - 0.05,
            reached_goal=no_goal,
            hold_ok=no_goal,
        )

    # Report the task failure conditions from the current state.
    def _failures(self, state, success, timeout):
        return dict(dropped=self.ever_grasped & ~state["is_grasped"] & state["is_static"] & timeout)

    # Expose support and loose part poses in the observation.
    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                part_pose=self.part.pose.raw_pose,
                module_pose=self.valve.pose.raw_pose,
                tcp_to_part_pos=self.part.pose.p - self.agent.tcp.pose.p,
                part_to_module_pos=self.valve.pose.p - self.part.pose.p,
            )
        return obs

    # Reward grasping and positioning the loose part for assembly.
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reach = 1 - torch.tanh(5 * torch.linalg.norm(self.part.pose.p - self.agent.tcp.pose.p, axis=1))
        approach = 1 - torch.tanh(5 * info["part_to_module_dist"])
        return torch.where(info["is_grasped"], 2 + approach, reach)

    # Scale the dense reward to the task reward range.
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 3
