# Simulates lifting the free eye of a shock absorber and fitting it over a nearby peg.
# Created: 2026-10-23

from typing import Any

import numpy as np
import sapien
import torch
import trimesh
from scipy.spatial.transform import Rotation

from env.fr3_table_env import FRONT_TABLE_TOP_Z, load_task_config, rgba
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from tasks.hiveboard import ASSET_ROOT, HiveboardFR3Env

CONFIG = load_task_config(__file__)
CLAMP_COLLISION_BIT = 24
X_TO_Z = [np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0]
X_TO_Y = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
UPRIGHT_CYLINDER = [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]


@register_env("ShockAbsorberFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class ShockAbsorberFR3Env(HiveboardFR3Env):
    """Stages (the protocol's stage_reached): free end grasped (1), its eye aligned over the peg (2), fully inserted
    onto it (3); success once inserted and released."""

    config = CONFIG
    upright_module = True
    failure_keys = ("never_grasped", "not_inserted", "timeout")

    # Load the joint and geometry settings for this task module.
    @property
    def mechanism_config(self):
        return {}

    # Build the clamped eye, spring segments, free eye, and target peg.
    def _build_module(self):
        cfg = self.config["shock_absorber"]
        objects = self.config["objects"]
        support = sapien.render.RenderMaterial(base_color=rgba(objects["support_color"]))
        part = sapien.render.RenderMaterial(base_color=rgba(objects["part_color"]))
        contact = sapien.physx.PhysxMaterial(2.0, 2.0, 0.0)
        m8, peg, shock = ASSET_ROOT / "thread_m8", ASSET_ROOT / "peg_insertion", ASSET_ROOT / "shock_absorber"
        center = objects["board_center_xy"]

        builder = self.scene.create_articulation_builder()
        builder.set_name("shock_absorber")
        base = builder.create_link_builder()
        base.set_name("shock_absorber_clamp")
        base.add_visual_from_file(str(m8 / "M8_Screw_Base_Corpo1_Mesh.obj"), material=support)
        base.add_box_collision(pose=sapien.Pose(p=[0, 0, 0.0075]), half_size=[0.038, 0.038, 0.0075], material=contact)
        for radius, low, high in ((0.0053, 0.015, 0.0254), (0.0036, 0.0254, 0.045)):
            base.add_cylinder_collision(pose=sapien.Pose(p=[0, 0, (low + high) / 2], q=UPRIGHT_CYLINDER), radius=radius,
                                        half_length=(high - low) / 2, material=contact)
        low, high = cfg["washer_z"]
        washer = dict(pose=sapien.Pose(p=[0, 0, (low + high) / 2], q=UPRIGHT_CYLINDER), radius=cfg["washer_radius"],
                      half_length=(high - low) / 2)
        base.add_cylinder_visual(**washer, material=support)
        base.add_cylinder_collision(**washer, material=contact)
        nut = sapien.Pose(p=[0, 0, cfg["nut_z"]])
        nut_material = sapien.render.RenderMaterial(base_color=rgba(cfg["nut_color"]))
        base.add_visual_from_file(str(m8 / "M8_Nut_Corpo1_Mesh.obj"), pose=nut, material=nut_material)
        base.add_multiple_convex_collisions_from_file(str(m8 / "M8_Nut_Corpo1_Mesh_coacd.obj"), pose=nut, material=contact)

        clamped, free = np.asarray(cfg["clamped_eye_mm"]) / 1000, np.asarray(cfg["free_eye_mm"]) / 1000
        pivot = cfg["axis_height"] + clamped[0] - free[0]
        swing = builder.create_link_builder(base)
        swing.set_name("shock_absorber_swing")
        swing.set_joint_name("shock_absorber_swing_joint")
        swing.set_joint_properties("revolute_unwrapped", [[-2 * np.pi, 2 * np.pi]],
                                   sapien.Pose(p=[0, 0, pivot], q=X_TO_Z), sapien.Pose(q=X_TO_Z),
                                   friction=cfg["swing_friction"], damping=cfg["swing_damping"])
        swing.set_mass_and_inertia(1e-3, sapien.Pose(), [1e-7] * 3)
        rotation = Rotation.from_matrix([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        x, y, z, w = rotation.as_quat()
        offset = -rotation.apply(clamped)
        mesh_pose = sapien.Pose(p=offset, q=[w, x, y, z])
        volumes = [trimesh.load(shock / f"{name}.obj", force="mesh").volume for name in cfg["segments"]]
        self.free_eye_offset = rotation.apply(free) + offset
        length = float(self.free_eye_offset[0])
        joints = len(cfg["segments"]) - 1
        low, high = cfg["spring_range"]
        parent = swing
        self.shock_links = []
        for index, name in enumerate(cfg["segments"]):
            link = builder.create_link_builder(parent)
            link.set_name(f"shock_absorber_segment{index}")
            if index == 0:
                link.set_joint_name("shock_absorber_lift_joint")
                link.set_joint_properties("revolute", [[-np.deg2rad(cfg["max_lift_degrees"]), 0.0]],
                                          sapien.Pose(q=X_TO_Y), sapien.Pose(q=X_TO_Y),
                                          friction=cfg["lift_friction"], damping=cfg["lift_damping"])
            else:
                link.set_joint_name(f"shock_absorber_spring_joint{index}")
                link.set_joint_properties("prismatic", [[low / joints, high / joints]], sapien.Pose(), sapien.Pose())
            link.add_visual_from_file(str(shock / f"{name}.obj"), pose=mesh_pose, scale=[0.001] * 3, material=part)
            if index == len(cfg["segments"]) - 1:
                self._add_free_end(link, shock, mesh_pose, contact)
            else:
                link.add_multiple_convex_collisions_from_file(str(shock / f"{name}_coacd.obj"), pose=mesh_pose,
                                                              scale=[0.001] * 3, material=contact)
            mass = cfg["mass"] * volumes[index] / sum(volumes)
            centre = length * (index + 0.5) / len(cfg["segments"])
            link.set_mass_and_inertia(mass, sapien.Pose(p=[centre, 0, -0.002]), [mass * 0.015 ** 2 / 2] * 3)
            parent = link
        self.shock_link_names = [f"shock_absorber_segment{index}" for index in range(len(cfg["segments"]))]
        builder.set_initial_pose(sapien.Pose(p=[center[0], center[1], FRONT_TABLE_TOP_Z]))
        self.valve = builder.build(fix_root_link=True)
        self.shock = self.valve.links_map[self.shock_link_names[-1]]
        self.shock_segments = [self.valve.links_map[name] for name in self.shock_link_names]
        self.valve_lever = self.shock
        for link in [self.valve.links_map["shock_absorber_clamp"], *self.shock_segments]:
            link.set_collision_group_bit(2, CLAMP_COLLISION_BIT, 1)
            for body in link._bodies:
                for shape in body.collision_shapes:
                    shape.set_contact_offset(cfg["contact_offset"])
        for index in range(1, joints + 1):
            spring = self.valve.joints_map[f"shock_absorber_spring_joint{index}"]
            spring.set_drive_properties(cfg["spring_stiffness"] * joints, cfg["spring_damping"] * joints)
            for joint in spring._objs:
                joint.set_drive_target(0.0)
        self.swing_joint = self.valve.joints_map["shock_absorber_swing_joint"]
        self.valve.joints_map["shock_absorber_swing_joint"].set_friction(cfg["swing_friction"])
        self.valve.joints_map["shock_absorber_lift_joint"].set_friction(cfg["lift_friction"])
        self.pivot_height = FRONT_TABLE_TOP_Z + objects["board_surface_height"]
        self.rest_eye_height = self.pivot_height + pivot + float(self.free_eye_offset[2])

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(str(peg / "Base_principal_Corpo1.obj"), material=support)
        builder.add_multiple_convex_collisions_from_file(str(peg / "Base_principal_Corpo1_coacd.obj"), material=contact)
        builder.initial_pose = sapien.Pose(p=[center[0], center[1] + 0.1, FRONT_TABLE_TOP_Z])
        self.peg_base = builder.build_kinematic(name="shock_absorber_peg_base")
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(str(peg / "Al__a_Corpo1.obj"), material=support)
        builder.add_capsule_collision(pose=sapien.Pose(p=[0, 0, (0.015 + cfg["peg_top"]) / 2], q=UPRIGHT_CYLINDER),
                                      radius=cfg["peg_radius"],
                                      half_length=(cfg["peg_top"] - 0.015) / 2 - cfg["peg_radius"], material=contact)
        builder.initial_pose = sapien.Pose(p=[center[0], center[1] + 0.1, FRONT_TABLE_TOP_Z])
        self.peg = builder.build_kinematic(name="shock_absorber_peg")
        self.peg_cell_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.peg_hole_index = torch.zeros_like(self.peg_cell_index)

    # Attach the free end geometry and its graspable collision shapes.
    def _add_free_end(self, link, shock, mesh_pose, contact):
        eye = self.config["shock_absorber"]["free_eye_collision"]
        for part in ("disc", "neck"):
            link.add_convex_collision_from_file(str(shock / f"Corpo3_seg4_{part}_hull.obj"), pose=mesh_pose,
                                                scale=[0.001] * 3, material=contact)
        inner, outer, half_thickness = eye["inner_radius"], eye["outer_radius"], eye["half_thickness"]
        count = eye["boxes"]
        half = np.array([half_thickness, (outer - inner) / 2, outer * np.tan(np.pi / count)]) / 1000
        centre = np.array(self.config["shock_absorber"]["free_eye_mm"]) / 1000
        for k in range(count):
            angle = 2 * np.pi * k / count
            radial = np.array([0.0, np.cos(angle), np.sin(angle)])
            box = sapien.Pose(p=centre + radial * (inner + outer) / 2000, q=[np.cos(angle / 2), np.sin(angle / 2), 0, 0])
            link.add_box_collision(pose=mesh_pose * box, half_size=half, material=contact)

    # Reset the spring and the free eye to their starting positions.
    def _initialize_module(self, env_idx: torch.Tensor):
        cfg = self.config["shock_absorber"]
        cells = self.config["objects"]["peg_cell_positions_xy"]
        yaws = self.config["objects"]["module_yaw_choices_degrees"]
        holes = np.asarray(cfg["peg_holes_xy"])
        low, high = cfg["reachable_spring_change"]
        rest_spacing = float(self.free_eye_offset[0])
        board_yaw = self.board_yaw_degrees[env_idx].cpu().numpy()
        board_xy = self.board_center_xy[env_idx].cpu().numpy()
        clamp_xy = (self.board_center_xy + self.module_cell_offsets)[env_idx].cpu().numpy()
        module_yaw = self.module_yaw_degrees[env_idx].cpu().numpy()

        def turn(xy, degrees):
            c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
            return np.array([c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]])

        plate_p, plate_yaw, peg_p, swing = [], [], [], []
        for row, i in enumerate(env_idx.tolist()):
            rng = self._batched_episode_rng[i]
            cell, hole = rng.randint(len(cells)), rng.randint(len(holes))
            centre = board_xy[row] + turn(cells[cell], board_yaw[row])
            reachable = [yaw for yaw in yaws if low <= np.linalg.norm(
                centre + turn(holes[hole], board_yaw[row] + yaw) - clamp_xy[row]) - rest_spacing <= high]
            yaw = board_yaw[row] + reachable[rng.randint(len(reachable))]
            peg = centre + turn(holes[hole], yaw)
            to_peg = np.arctan2(*(peg - clamp_xy[row])[::-1]) - np.deg2rad(module_yaw[row])
            start = np.deg2rad(cfg["start_offset_choices_degrees"][rng.randint(len(cfg["start_offset_choices_degrees"]))])
            self.peg_cell_index[i], self.peg_hole_index[i] = cell, hole
            plate_p.append(centre)
            plate_yaw.append(yaw)
            peg_p.append(peg)
            swing.append(np.arctan2(np.sin(to_peg + start), np.cos(to_peg + start)))

        def poses(xy, yaw):
            p = torch.cat([torch.tensor(np.array(xy), dtype=torch.float32),
                           torch.full((len(xy), 1), self.pivot_height)], dim=1)
            half = torch.deg2rad(torch.tensor(np.array(yaw), dtype=torch.float32)) / 2
            zeros = torch.zeros_like(half)
            q = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=1)
            return Pose.create_from_pq(p=p.to(self.device), q=q.to(self.device))

        self.peg_base.set_pose(poses(plate_p, plate_yaw))
        self.peg.set_pose(poses(peg_p, plate_yaw))
        qpos = self.valve.get_qpos()
        qpos[env_idx, 0] = torch.tensor(swing, dtype=torch.float32, device=self.device)
        qpos[env_idx, 1:] = 0.0
        self.valve.set_qpos(qpos)
        qvel = self.valve.get_qvel()
        qvel[env_idx] = 0
        self.valve.set_qvel(qvel)

    # World position of the free eye's centre.
    def free_eye(self):
        offset = torch.as_tensor(self.free_eye_offset, dtype=torch.float32, device=self.device)
        return (self.shock.pose * Pose.create_from_pq(p=offset.expand(self.num_envs, 3))).p

    # Track grasp, alignment over the peg, and final insertion.
    def _task_state(self):
        tol = self.config["success"]
        eye = self.free_eye()
        to_peg = torch.linalg.norm(eye[:, :2] - self.peg.pose.p[:, :2], axis=1)
        aligned = to_peg <= tol["insert_radius"]
        inserted = aligned & (eye[:, 2] <= self.rest_eye_height + tol["insert_height"])
        grasped = torch.stack([self.agent.is_grasping(link) for link in self.shock_segments]).any(dim=0)
        touch = torch.stack([self.agent.finger_contact_force(link) for link in self.shock_segments]).amax(dim=0)
        speed = torch.linalg.norm(self.valve.get_qvel(), axis=1)
        released = ~grasped & (touch < tol["contact_force"])
        return dict(
            is_grasped=grasped,
            is_static=speed <= tol["angular_velocity"],
            is_contacted=touch >= tol["contact_force"],
            is_aligned=aligned,
            is_inserted=inserted,
            spring_extension=self.valve.get_qpos()[:, 2:].sum(dim=1),
            eye_to_peg_dist=to_peg,
            eye_height_above_rest=eye[:, 2] - self.rest_eye_height,
            off_table=torch.zeros_like(grasped),
            reached_goal=inserted,
            hold_ok=inserted & released,
        )

    # Add the module and task objects to the shared robot scene.
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.ever_inserted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ever_aligned = torch.zeros_like(self.ever_inserted)

    # Reset task state and object poses at the start of each episode.
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.ever_inserted[env_idx] = False
        self.ever_aligned[env_idx] = False

    # Update insertion progress from the spring and free eye poses.
    def _after_control_step(self):
        super()._after_control_step()
        state = self._task_state()
        self.ever_aligned |= state["is_aligned"] & state["is_grasped"]
        self.ever_inserted |= state["is_inserted"]

    # Report the task failure conditions from the current state.
    def _failures(self, state, success, timeout):
        aligned = self.ever_grasped & self.ever_aligned
        return dict(
            stage_reached=self.ever_grasped.int() + aligned.int() + (aligned & self.ever_inserted).int(),
            subgoal_aligned=self.ever_aligned.clone(),
            subgoal_inserted=self.ever_inserted.clone(),
            not_inserted=self.ever_grasped & ~self.ever_inserted & timeout,
        )

    # Add task specific state fields to the observation.
    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            eye = self.free_eye()
            obs.update(
                shock_pose=self.shock.pose.raw_pose,
                shock_qpos=self.valve.get_qpos(),
                free_eye_pos=eye,
                peg_pos=self.peg.pose.p,
                tcp_to_eye_pos=eye - self.agent.tcp.pose.p,
                eye_to_peg_pos=self.peg.pose.p - eye,
            )
        return obs

    # Reward lifting the free eye, aligning it, and placing it on the peg.
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        eye = self.free_eye()
        reach = 1 - torch.tanh(5 * torch.linalg.norm(eye - self.agent.tcp.pose.p, axis=1))
        approach = 1 - torch.tanh(10 * info["eye_to_peg_dist"])
        reward = torch.where(info["is_grasped"], 2 + approach, reach)
        reward[info["is_inserted"]] = 4
        reward[info["success"]] = 5
        return reward

    # Scale the dense reward to the task reward range.
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 5
