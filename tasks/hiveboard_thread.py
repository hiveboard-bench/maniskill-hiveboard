# Simulates picking up a loose threaded part, engaging it with the screw, and turning it down.
# Created: 2026-10-22

from typing import Any

import numpy as np
import sapien
import torch

from env.fr3_table_env import CONFIG as BASE_CONFIG, FRONT_TABLE_TOP_Z
from mani_skill.utils.structs.pose import Pose
from tasks.hiveboard import ASSET_ROOT
from tasks.hiveboard_assembly import HiveboardAssemblyFR3Env

X_TO_Z = sapien.Pose(q=[np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0])
THREADED_NUT_COLLISION_BIT = 27


# Turn about z and tilt of the z axis from vertical of wxyz quaternions.
def quaternion_yaw_tilt(q: torch.Tensor):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    tilt = torch.arccos((1 - 2 * (x * x + y * y)).clamp(-1.0, 1.0))
    return yaw, tilt


class ThreadFR3Env(HiveboardAssemblyFR3Env):
    """Subgoals: nut grasped, set on the screw (engaged), turned down; success once it is seated on the plate."""

    failure_keys = ("timeout",)

    # Build the fixed screw, loose nut, and engaged threaded joint.
    def _build_module(self):
        super()._build_module()
        cfg = self.config["thread"]
        part = self.config["assembly"]["part"]
        name = self.module_name
        seat = self.part_seated_pose

        builder = self.scene.create_articulation_builder()
        builder.set_name(f"{name}_thread")
        root = builder.create_link_builder()
        root.set_name(f"{name}_thread_axis")
        lift = builder.create_link_builder(root)
        lift.set_name(f"{name}_thread_lift")
        lift.set_joint_name(f"{name}_lift_joint")
        lift.set_joint_properties("prismatic", [[cfg["park_depth"], cfg["travel"]]], X_TO_Z, X_TO_Z)
        lift.set_mass_and_inertia(0.01, sapien.Pose(), [1e-6] * 3)
        nut = builder.create_link_builder(lift)
        nut.set_name(f"{name}_threaded_nut")
        nut.set_joint_name(f"{name}_spin_joint")
        nut.set_joint_properties("revolute_unwrapped", [[-1e3, 1e3]], X_TO_Z, seat.inv() * X_TO_Z)
        self._add_shapes(nut, part, ASSET_ROOT / self.config["assembly"]["asset_dir"], self.part_body_frame,
                         self.config["objects"]["part_color"], self.part_contact, self.part_density)
        center = self.config["objects"]["board_center_xy"]
        builder.set_initial_pose(sapien.Pose(p=[center[0], center[1], FRONT_TABLE_TOP_Z]))
        self.thread = builder.build(fix_root_link=True)
        self.threaded_nut = self.thread.links_map[f"{name}_threaded_nut"]
        for body in self.threaded_nut._bodies:
            body.inertia = body.inertia * part["mass"] / body.mass
            body.mass = part["mass"]
        self.lift_joint = self.thread.joints_map[f"{name}_lift_joint"]
        self.lift_joint.set_drive_properties(stiffness=cfg["lift_stiffness"], damping=cfg["lift_damping"],
                                             force_limit=cfg["lift_force_limit"])
        spin = self.thread.joints_map[f"{name}_spin_joint"]
        spin.set_friction(cfg["spin_friction"])
        spin.set_drive_properties(stiffness=0.0, damping=cfg["spin_damping"])
        for joint in spin._objs:
            joint.set_armature(np.array([cfg["spin_armature"]], dtype=np.float32))
        self.threaded_nut.set_collision_group_bit(2, THREADED_NUT_COLLISION_BIT, 1)
        self.valve.set_collision_group_bit(2, THREADED_NUT_COLLISION_BIT, 1)
        self.seat_pose_inv = Pose.create(seat.inv())
        self.thread_engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.engage_lift = torch.zeros(self.num_envs, device=self.device)
        self.engage_spin = torch.zeros_like(self.engage_lift)

    # Reset the nut, screw joint, and thread engagement state.
    def _initialize_module(self, env_idx: torch.Tensor):
        super()._initialize_module(env_idx)
        park = self.config["thread"]["park_depth"]
        self.thread.set_pose(self.valve.pose)
        qpos = self.thread.get_qpos()
        qpos[env_idx] = torch.tensor([park, 0.0], device=self.device)
        self.thread.set_qpos(qpos)
        self.thread.set_qvel(torch.zeros_like(qpos))
        self.thread_engaged[env_idx] = False
        self._set_lift_targets(qpos[:, 0])

    # Set the joint lift from the current position along the thread.
    def _set_lift_targets(self, lift):
        park = self.config["thread"]["park_depth"]
        self.lift_joint.set_drive_target(torch.where(self.thread_engaged, lift, torch.full_like(lift, park)))

    # Where the thread holds the engaged part: height above the seat along the helix for its present turn.
    def helix_height(self) -> torch.Tensor:
        cfg = self.config["thread"]
        helix = self.engage_lift + cfg["pitch"] * (self.thread.get_qpos()[:, 1] - self.engage_spin) / (2 * np.pi)
        return helix.clamp(0.0, cfg["travel"])

    # A nut pose relative to its seated pose on the screw: (xy offset, height above the seat, tilt, turn).
    def nut_on_axis(self, pose: Pose):
        offset = self.valve.pose.inv() * pose * self.seat_pose_inv
        turn, tilt = quaternion_yaw_tilt(offset.q)
        return torch.linalg.norm(offset.p[:, :2], axis=1), offset.p[:, 2], tilt, turn

    # World pose of whichever nut is in play: the threaded one once engaged, the loose one before.
    @property
    def nut_pose(self) -> Pose:
        engaged = self.thread_engaged[:, None]
        threaded, loose = self.threaded_nut.pose.raw_pose, self.part.pose.raw_pose
        return Pose.create(torch.where(engaged, threaded, loose))

    # Move the loose nut out of the scene (on the floor, far off) while the threaded nut stands in for it.
    def _park_loose_nut(self, rows):
        far = torch.tensor([3.0, 3.0, BASE_CONFIG["ground"]["altitude"] + 0.05], device=self.device)
        raw = self.part.pose.raw_pose.clone()
        raw[rows, :3] = far
        self.part.set_pose(Pose.create(raw))
        velocity = self.part.get_linear_velocity()
        velocity[rows] = 0
        self.part.set_linear_velocity(velocity)
        spin = self.part.get_angular_velocity()
        spin[rows] = 0
        self.part.set_angular_velocity(spin)

    # Engage or release the nut and follow its turn along the helix.
    def _update_thread(self):
        cfg = self.config["thread"]
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        qpos = self.thread.get_qpos()
        lift, spin = qpos[:, 0], qpos[:, 1]
        helix = self.engage_lift + cfg["pitch"] * (spin - self.engage_spin) / (2 * np.pi)
        released = self.thread_engaged & (helix > cfg["travel"] + cfg["engage_height"])
        if released.any():
            raw = self.part.pose.raw_pose.clone()
            raw[released] = self.threaded_nut.pose.raw_pose[released]
            self.part.set_pose(Pose.create(raw))
            self.thread_engaged &= ~released
            qpos[released, 0] = cfg["park_depth"]
        offset, height, tilt, turn = self.nut_on_axis(self.part.pose)
        engage = ~self.thread_engaged & (offset <= cfg["engage_radius"]) & (
            tilt <= np.deg2rad(cfg["engage_tilt_degrees"])) & (height <= cfg["travel"] + 0.0005) & (
            height >= -cfg["engage_height"])
        if engage.any():
            qpos[engage, 0] = height[engage].clamp(0.0, cfg["travel"])
            qpos[engage, 1] = turn[engage]
            self.engage_lift[engage] = qpos[engage, 0]
            self.engage_spin[engage] = turn[engage]
            self.thread_engaged |= engage
            self._park_loose_nut(engage)
        if released.any() or engage.any():
            changed = released | engage
            self.thread.set_qpos(qpos)
            qvel = self.thread.get_qvel()
            qvel[changed] = 0
            self.thread.set_qvel(qvel)
        helix = self.engage_lift + cfg["pitch"] * (qpos[:, 1] - self.engage_spin) / (2 * np.pi)
        seated = self.thread_engaged & (helix < 0)
        self.engage_lift[seated] = 0.0
        self.engage_spin[seated] = qpos[seated, 1]
        self._set_lift_targets(helix.clamp(0.0, cfg["travel"]))
        if self.gpu_sim_enabled:
            if released.any() or engage.any():
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
            else:
                self.scene.px.gpu_apply_articulation_target_position()

    # Synchronize the threaded joint with the latest robot motion.
    def _after_control_step(self):
        self._update_thread()
        super()._after_control_step()

    # Measure nut alignment, engagement, and distance from the seat.
    def _task_state(self):
        state = super()._task_state()
        cfg, tol = self.config["thread"], self.config["success"]
        engaged = self.thread_engaged
        depth = torch.where(engaged, cfg["travel"] - self.helix_height(), torch.zeros_like(state["part_to_module_dist"]))
        seated = engaged & (depth >= cfg["travel"] - cfg["seat_tolerance"])
        nut = self.nut_pose
        state.update(
            is_grasped=state["is_grasped"] | self.agent.is_grasping(self.threaded_nut),
            is_contacted=state["is_contacted"] | (self.agent.finger_contact_force(self.threaded_nut) >= tol["contact_force"]),
            is_static=torch.where(engaged, self.thread.get_qvel().abs().amax(dim=1) <= tol["part_angular_velocity"],
                                  state["is_static"]),
            part_to_module_dist=torch.linalg.norm(nut.p[:, :2] - self.valve.pose.p[:, :2], axis=1),
            off_table=~engaged & state["off_table"],
            thread_engaged=engaged.clone(),
            thread_depth=depth,
            thread_turns=depth / cfg["pitch"],
            thread_progress=depth / cfg["travel"],
            is_seated=seated,
            reached_goal=seated,
            hold_ok=seated,
        )
        return state

    # Report the task failure conditions from the current state.
    def _failures(self, state, success, timeout):
        return {}

    # Add task specific state fields to the observation.
    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            nut = self.nut_pose
            obs.update(
                part_pose=nut.raw_pose,
                module_pose=self.valve.pose.raw_pose,
                tcp_to_part_pos=nut.p - self.agent.tcp.pose.p,
                part_to_module_pos=self.valve.pose.p - nut.p,
                thread_depth=info["thread_depth"][:, None],
            )
        return obs

    # Reward grasping, engaging, and turning the nut toward its seat.
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reach = 1 - torch.tanh(5 * torch.linalg.norm(self.nut_pose.p - self.agent.tcp.pose.p, axis=1))
        approach = 1 - torch.tanh(5 * info["part_to_module_dist"])
        reward = torch.where(info["is_grasped"], 2 + approach, reach)
        reward = torch.where(info["thread_engaged"], 3 + 2 * info["thread_progress"], reward)
        reward[info["success"]] = 6
        return reward

    # Scale the dense reward to the task reward range.
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 6
