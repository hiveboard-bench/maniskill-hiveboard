# Simulates hooking a ball valve lever with the FR3 gripper and pulling it through a quarter turn.
# Created: 2026-10-22

import numpy as np
import sapien

from env.fr3_table_env import FRONT_TABLE_TOP_Z, load_task_config, rgba
from mani_skill.utils.registration import register_env
from tasks.hiveboard import ASSET_ROOT, HiveboardFR3Env

CONFIG = load_task_config(__file__)


@register_env("BallValveFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class BallValveFR3Env(HiveboardFR3Env):
    config = CONFIG

    # Create the module bodies, joints, and collision shapes in the HiveBoard scene.
    def _build_module(self):
        center = self.config["objects"]["board_center_xy"]
        cfg = self.config["valve"]
        builder = self.scene.create_articulation_builder()
        builder.set_name("ball_valve")
        base = builder.create_link_builder()
        base.set_name("ball_valve_base")
        base_mesh = ASSET_ROOT / "ball_valve" / "valve_Mesh.obj"
        base_pose = sapien.Pose(p=[-0.0400277, 0, 0], q=[0.5, 0.5, 0.5, 0.5])
        base.add_visual_from_file(str(base_mesh), pose=base_pose,
                                  material=sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["valve_base_color"])))
        base.add_convex_collision_from_file(str(base_mesh), pose=base_pose)

        lever = builder.create_link_builder(base)
        lever.set_name("ball_valve_lever")
        lever.set_joint_name("ball_valve_joint")
        lever.set_joint_properties(
            "revolute", [[cfg["lower_limit"], cfg["upper_limit"]]],
            sapien.Pose(), sapien.Pose(), friction=cfg["friction"], damping=cfg["damping"],
        )
        lever_mesh = ASSET_ROOT / "ball_valve" / "valve_Mesh_1.obj"
        lever_pose = sapien.Pose(p=[-0.0468898, 0.0175836, -0.0114654], q=[0.5, 0.5, 0.5, 0.5])
        lever.add_visual_from_file(str(lever_mesh), pose=lever_pose,
                                   material=sapien.render.RenderMaterial(base_color=rgba(self.config["objects"]["valve_lever_color"])))
        lever.add_convex_collision_from_file(str(lever_mesh), pose=lever_pose)
        lever.set_mass_and_inertia(0.123, sapien.Pose(p=[0.033278, 0.0115, 0.052028]),
                                   [2.18765e-5, 1.99351e-4, 1.88323e-4])
        half_yaw = 0.0
        root_q = [
            np.cos(half_yaw) * np.sqrt(0.5),
            np.sin(half_yaw) * np.sqrt(0.5),
            -np.cos(half_yaw) * np.sqrt(0.5),
            np.sin(half_yaw) * np.sqrt(0.5),
        ]
        builder.set_initial_pose(sapien.Pose(
            p=[center[0], center[1], FRONT_TABLE_TOP_Z + 0.045], q=root_q
        ))
        self.valve = builder.build(fix_root_link=True)
        self.valve_lever = self.valve.links_map["ball_valve_lever"]
        self.valve.joints_map["ball_valve_joint"].set_friction(cfg["friction"])
