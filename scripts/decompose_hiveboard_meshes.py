# Decomposes HiveBoard meshes for simulation.
# Created: 2026-10-22

from pathlib import Path

import sys

import coacd
import trimesh
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parents[1] / "fr3_assets" / "hiveboard"
MESHES = {
    "drawer/tn__Gaveta1_aE_Corpo1_Mesh.obj": 0.03,
    "thread_m8/M8_Nut_Corpo1_Mesh.obj": 0.03,
    "thread_m30/Nut_Corpo1.obj": 0.03,
    "light_bulb/base_pivot_Mesh.obj": 0.03,
    "peg_insertion/Base_principal_Corpo1.obj": 0.03,
    "button_cover/button-cover_tn__Tampa1_dC_Corpo1_Mesh.obj": 0.02,
    "button_cover/button-cover_tn__corpo1_dC_Corpo1_Mesh.obj": 0.03,
    "gate_valve_small/small-valve_Mesh_22.obj": 0.02,
    "gate_valve_large/high-valve_Corpo1__1_.obj": 0.02,
}
BULB_RPY = (-1.5434489, -1.4170299, 1.5429671)
COMBINED = {
    "light_bulb/bulb_coacd.obj": (0.02, [("light_bulb/Corpo1_Mesh.obj", BULB_RPY),
                                         ("light_bulb/tn__Corpo11_j7m1J_Mesh.obj", BULB_RPY),
                                         ("light_bulb/tn__Corpo12_j7m1J_Mesh.obj", BULB_RPY)]),
}

SPLIT = {"shock_absorber/Corpo3.obj": (0.04, [4.5, 22.2, 39.8, 57.5])}


# Save convex mesh pieces for collision geometry.
def write_pieces(mesh, threshold, out):
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold,
                            preprocess_resolution=60, max_convex_hull=48)
    lines, offset = [], 1
    for index, (vertices, faces) in enumerate(parts):
        lines.append(f"o piece_{index}")
        lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
        lines += [f"f {a + offset} {b + offset} {c + offset}" for a, b, c in faces]
        offset += len(vertices)
    out.write_text("\n".join(lines) + "\n")
    print(f"{out.relative_to(ASSETS)}: {len(parts)} convex pieces")


def main():
    coacd.set_log_level("error")
    only = set(sys.argv[1:])
    for name, threshold in MESHES.items():
        path = ASSETS / name
        if only and str(path.with_name(path.stem + "_coacd.obj").relative_to(ASSETS)) not in only:
            continue
        write_pieces(trimesh.load(path, force="mesh"), threshold, path.with_name(path.stem + "_coacd.obj"))
    for name, (threshold, sources) in COMBINED.items():
        if only and name not in only:
            continue
        meshes = []
        for source, rpy in sources:
            mesh = trimesh.load(ASSETS / source, force="mesh")
            transform = trimesh.transformations.identity_matrix()
            transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            meshes.append(mesh.apply_transform(transform))
        write_pieces(trimesh.util.concatenate(meshes), threshold, ASSETS / name)
    for name, (threshold, cuts) in SPLIT.items():
        path = ASSETS / name
        mesh = trimesh.load(path, force="mesh")
        edges = [-float("inf"), *cuts, float("inf")]
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            out = path.with_name(f"{path.stem}_seg{index}.obj")
            if only and str(out.with_name(out.stem + "_coacd.obj").relative_to(ASSETS)) not in only:
                continue
            segment = mesh
            if low > -float("inf"):
                segment = segment.slice_plane([0, 0, low], [0, 0, 1], cap=True)
            if high < float("inf"):
                segment = segment.slice_plane([0, 0, high], [0, 0, -1], cap=True)
            segment.export(out)
            print(f"{out.relative_to(ASSETS)}: z {segment.bounds[:, 2].round(1).tolist()} mm, watertight {segment.is_watertight}")
            write_pieces(segment, threshold, out.with_name(out.stem + "_coacd.obj"))


if __name__ == "__main__":
    main()
