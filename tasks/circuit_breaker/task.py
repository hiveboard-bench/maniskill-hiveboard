# Simulates pushing a circuit breaker toggle to its opposite state with a closed gripper.
# Created: 2026-10-22

import numpy as np
import sapien

from env.fr3_table_env import FRONT_TABLE_TOP_Z, load_task_config, rgba
from mani_skill.utils.registration import register_env
from tasks.hiveboard import ASSET_ROOT, HiveboardFR3Env

CONFIG = load_task_config(__file__)


@register_env("CircuitBreakerFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class CircuitBreakerFR3Env(HiveboardFR3Env):
    config = CONFIG

    # Create the module bodies, joints, and collision shapes in the HiveBoard scene.
    def _build_module(self):
        center = self.config["objects"]["board_center_xy"]
        cfg = self.config["breaker"]
        builder = self.scene.create_articulation_builder()
        builder.set_name("circuit_breaker")
        base = builder.create_link_builder()
        base.set_name("circuit_breaker_base")
        base_mesh = ASSET_ROOT / "circuit_breaker" / "breaker_tn__corpo1_dC_Corpo1_Mesh.obj"
        mesh_q = [0.5, -0.5, 0.5, -0.5]
        base_pose = sapien.Pose(p=[-0.0350036, 0, 0], q=mesh_q)
        base.add_visual_from_file(
            str(base_mesh), pose=base_pose,
            material=sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["breaker_base_color"])),
        )
        base.add_convex_collision_from_file(str(base_mesh), pose=base_pose)

        toggle = builder.create_link_builder(base)
        toggle.set_name("circuit_breaker_toggle")
        toggle.set_joint_name("circuit_breaker_joint")
        axis_q = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        toggle.set_joint_properties(
            "revolute", [[cfg["lower_limit"], cfg["upper_limit"]]],
            sapien.Pose(q=axis_q), sapien.Pose(q=axis_q),
            friction=cfg["friction"], damping=cfg["damping"],
        )
        toggle_mesh = ASSET_ROOT / "circuit_breaker" / "breaker_tn__Alavanca1_zH_Corpo1_Mesh.obj"
        toggle_pose = sapien.Pose(p=[0.0081193, 0.0200116, 0.0099949], q=mesh_q)
        toggle.add_visual_from_file(
            str(toggle_mesh), pose=toggle_pose,
            material=sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["breaker_toggle_color"])),
        )
        toggle.add_convex_collision_from_file(str(toggle_mesh), pose=toggle_pose)
        toggle.set_mass_and_inertia(
            0.0264, sapien.Pose(p=[0.02, 0.01, -0.0015]), [3.2758e-6, 5.9158e-6, 4.4e-6]
        )
        root_q = [np.sqrt(0.5), 0, -np.sqrt(0.5), 0]
        builder.set_initial_pose(sapien.Pose(
            p=[center[0], center[1], FRONT_TABLE_TOP_Z + 0.045], q=root_q
        ))
        self.valve = builder.build(fix_root_link=True)
        self.valve_lever = self.valve.links_map["circuit_breaker_toggle"]
        self.valve.joints_map["circuit_breaker_joint"].set_friction(cfg["friction"])
