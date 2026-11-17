# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import argparse
import time
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# Import numba (used by furniture_bench.utils.transform) before Isaac Sim starts: Isaac's bundled
# pip_prebundle ships an old `coverage` that shadows the env's one and breaks numba's import.
import numba  # noqa: F401

# import isaacsim
# from omni.isaac.kit import SimulationApp
from omni.isaac.lab.app import AppLauncher

from src.visualization.render_mp4 import unpickle_data
from ipdb import set_trace as bp

# from furniture_bench.utils.pose import get_mat, rot_mat

# add argparse arguments
parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--headless",
#     action="store_true",
#     default=False,
#     help="Force display off at all times.",
# )
parser.add_argument("-i", "--demo-index", type=int, default=0)
parser.add_argument("-sub", "--sub-steps", type=int, default=0)
parser.add_argument("--load-dir", type=str, required=True)
parser.add_argument("--save", action="store_true")
parser.add_argument("--save-dir", type=str, default=None)
parser.add_argument("--num-parts", type=int, required=True)
parser.add_argument("--furniture", type=str, required=True)
parser.add_argument(
    "--robot", choices=["fr3", "panda"], default="fr3",
    help="arm model to render: fr3 (Isaac Sim asset Robots/Franka/FR3/fr3.usd) or Isaac Lab's panda",
)
parser.add_argument("--render-viewer", action="store_true")
parser.add_argument(
    "--robot-mask", action="store_true",
    help="also save robot_mask_front (bool HxW): front-camera pixels that change when the arm is hidden",
)
parser.add_argument("-fw", "--flipped-wrist-camera", action="store_true")
parser.add_argument("-dr", "--domain-rand", action="store_true")

AppLauncher.add_app_launcher_args(parser)

# Create a subparser for randomization arguments
subparsers = parser.add_subparsers(dest="subcommand")

# Create the parser for the "rand" command
rand_parser = subparsers.add_parser("rand", help="Randomization arguments")
rand_parser.add_argument(
    "--part-random", type=str, default="base", help="Randomize part colors"
)
rand_parser.add_argument(
    "--table-random", type=str, default="base", help="Randomize table colors"
)
rand_parser.add_argument(
    "--different-part-colors", action="store_true", help="Use different part colors"
)
rand_parser.add_argument(
    "--random-frame-freq", type=int, default=1, help="Randomize every X frames"
)

# AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# args_cli = parser.parse_args()

# # launch omniverse app
# simulation_app = SimulationApp(
#     {
#         "headless": args_cli.headless,
#         "width": 640,
#         "height": 480,
#     }
# )
# """Rest everything follows."""

import numpy as np
import torch
from torchvision.transforms import functional as F, InterpolationMode
from pxr import Gf, UsdGeom, UsdShade, Vt
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from omni.isaac.core.simulation_context import SimulationContext

from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.replicator.isaac")
enable_extension("omni.kit.window.viewport")
import omni.replicator.core as rep
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab_assets import FRANKA_PANDA_CFG
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from pxr import Usd
from omni.isaac.lab.assets import Articulation

import furniture_bench.utils.transform as T
from furniture_bench.config import config

from src.sim2real.part_config_render import part_config_dict

from typing import Dict, Union, Tuple

# folder to load from
DEMO_DIR = Path(args_cli.load_dir)

# Catch all files in all subfolder that end with .pkl or .pkl.xz
FILES = sorted(list(DEMO_DIR.rglob("*.pkl")) + list(DEMO_DIR.rglob("*.pkl.xz")))

# Set the random seed then shuffle the files
np.random.seed(0)
np.random.shuffle(FILES)

FILE = str(FILES[args_cli.demo_index])

PART_COLOR_BASE = (
    "white" if "PART_COLOR_BASE" not in os.environ else os.environ["PART_COLOR_BASE"]
)

# constants
april_to_sim_mat = np.array(
    [
        [6.1232343e-17, 1.0000000e00, 1.2246469e-16, 1.4999807e-03],
        [1.0000000e00, -6.1232343e-17, -7.4987988e-33, 0.0000000e00],
        [0.0000000e00, 1.2246469e-16, -1.0000000e00, 4.1500002e-01],
        [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
    ]
)
front_435_camera_matrix = np.array(
    [
        [603.70898438, 0.0, 322.38723755],
        [0.0, 603.42315674, 249.26660156],
        [0.0, 0.0, 1.0],
    ]
)
wrist_435_camera_matrix = np.array(
    [
        [607.46057129, 0.0, 316.16723633],
        [0.0, 606.96691895, 243.04000854],
        [0.0, 0.0, 1.0],
    ]
)


def build_camera_params(
    width: int = 640, height: int = 480, pixel_size: float = (3.0 * 1e-3)
) -> Dict[str, Dict[str, float]]:

    cam_params = {}
    for name, mat in zip(
        ["front", "wrist"], [front_435_camera_matrix, wrist_435_camera_matrix]
    ):
        camera_matrix = mat
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix

        focal_length_x = fx * pixel_size
        focal_length_y = fy * pixel_size

        horizontal_aperture = pixel_size * width
        vertical_aperture = pixel_size * height

        params = dict(
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
            focal_length=(focal_length_x + focal_length_y) / 2,
            f_stop=200.0 if name == "front" else 0.0,
            focus_distance=0.6 if name == "front" else 0.0,
        )
        cam_params[name] = params

    return cam_params


def run_until_quit(simulation_app, world):
    import omni.appwindow
    import carb.input

    global RP_RUNNING, RP_PAUSE
    RP_RUNNING = True
    RP_PAUSE = False

    def on_kb_input(e):
        global RP_RUNNING, RP_PAUSE
        if e.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if e.input == carb.input.KeyboardInput.Q:
                print(f'[Run until quit] Caught "Q" keypress, closing app')
                RP_RUNNING = False
            elif e.input == carb.input.KeyboardInput.P:
                print(f'[Run until quit] Caught "P" keypress, pausing')
                RP_PAUSE = True

    app_window = omni.appwindow.get_default_app_window()
    kb = app_window.get_keyboard()
    carb_input = carb.input.acquire_input_interface()
    _ = carb_input.subscribe_to_keyboard_events(kb, on_kb_input)

    while RP_RUNNING:
        # simulation_app.update()
        world.render()
        time.sleep(1 / 120.0)

        if RP_PAUSE:
            print(f"[Run until quit] Paused!")
            from IPython import embed

            embed()
            RP_PAUSE = False

    return


def gen_prim(usd_path: str, prim_name: str, init_pos, init_ori):
    furniture_assets_root_path = Path(
        f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets/furniture/mesh/usd"
    )
    # furniture_assets_root_path = Path(f"assets/furniture/mesh/usd")
    usd_path = str(furniture_assets_root_path / usd_path)
    assert os.path.exists(usd_path), f"Could not find USD file: {usd_path}"

    pose = april_to_sim_mat @ (T.to_homogeneous(init_pos, init_ori))
    pos, ori = T.mat2pose(pose)
    ori = T.convert_quat(ori, to="wxyz")

    prim_utils.create_prim(
        prim_name, usd_path=usd_path, translation=pos, orientation=ori
    )

    view = XFormPrimView(prim_name, reset_xform_properties=False)
    return view


light_prim_paths = [
    "/World/Table/Lights/RectLight",
    "/World/Table/Lights/RectLight_01",
    "/World/Table/Lights/CylinderLight",
    "/World/Table/Lights/CylinderLight_01",
    "/World/Table/Lights/CylinderLight_02",
    "/World/Table/Lights/CylinderLight_03",
]


part_mat_paths = [
    "/World/ObstacleFront/Looks/OmniPBR_ClearCoat",
    "/World/ObstacleRight/Looks/OmniPBR_ClearCoat",
    "/World/ObstacleLeft/Looks/OmniPBR_ClearCoat",
]


table_mat_paths = [
    "/World/Table/Looks/Material_010",
]


def _env_vec(name, default):
    v = os.getenv(name)
    return tuple(float(x) for x in v.split(",")) if v else default


# Scene appearance matched to our real cell (black foam yoga mat on the table, dark blue-teal fabric
# around it). Colours are linear albedo; each can be overridden per run with an environment variable.
SCENE_TABLE_WIDTH = float(os.getenv("SCENE_TABLE_WIDTH", 0.78))  # m along y (asset: 0.8 x 1.35 = 1.08)
SCENE_TABLE_BACK_CUT = float(os.getenv("SCENE_TABLE_BACK_CUT", 0.53))  # m removed from the far (-x) end
SCENE_TABLE_COLOR = _env_vec("SCENE_TABLE_COLOR", (0.036, 0.042, 0.038))
SCENE_TABLE_ROUGHNESS = float(os.getenv("SCENE_TABLE_ROUGHNESS", 0.9))
SCENE_BG_COLOR = _env_vec("SCENE_BG_COLOR", (0.0065, 0.026, 0.043))
SCENE_BG_ROUGHNESS = float(os.getenv("SCENE_BG_ROUGHNESS", 1.0))
# One flat fabric panel behind the robot instead of background.usda's three-part folding screen.
BACK_PANEL_X = float(os.getenv("BACK_PANEL_X", -0.8))  # world x of the panel (old back wall; base x -0.5)
BACK_PANEL_SIZE = _env_vec("BACK_PANEL_SIZE", (3.0, 2.0))  # width along y, height from the floor (m)
ROBOT_WHITE = float(os.getenv("ROBOT_WHITE", 0.55))  # albedo of the FR3's white covers and the hand's white plastic
ROBOT_WHITE_ROUGHNESS = float(os.getenv("ROBOT_WHITE_ROUGHNESS", 0.3))


def apply_scene_appearance():
    """Narrow the table top, make it a matte black mat, and give the walls a dark fabric material.

    The background walls bind Material_002/003/004, which background.usda never defines (so they
    rendered default grey), and the room walls in table_room.usda have no material at all.
    """
    stage = prim_utils.get_current_stage()
    top = stage.GetPrimAtPath("/World/Table/table2")
    # The asset top is 1.3 x 0.8 m and /World/Table is scaled (1.7, 1.35, 1): 2.21 m long, x = -1.105..1.105
    # in the world. Cut SCENE_TABLE_BACK_CUT off the far end and keep the front edge where it is.
    length = 2.21 - SCENE_TABLE_BACK_CUT
    top.GetAttribute("xformOp:scale").Set(Gf.Vec3f(length / 2.21, SCENE_TABLE_WIDTH / 1.08, 1.0))
    top.GetAttribute("xformOp:translate").Set(Gf.Vec3d(SCENE_TABLE_BACK_CUT / 2 / 1.7, 0.0, 0.4))

    shader = prim_utils.get_prim_at_path(table_mat_paths[0]).GetChildren()[0]
    shader.GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*SCENE_TABLE_COLOR))
    shader.GetAttribute("inputs:diffuse_tint").Set(Gf.Vec3f(1.0, 1.0, 1.0))
    shader.GetAttribute("inputs:reflection_roughness_constant").Set(SCENE_TABLE_ROUGHNESS)

    fabric = "/World/Looks/BackgroundFabric"
    mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=SCENE_BG_COLOR, roughness=SCENE_BG_ROUGHNESS, metallic=0.0)
    mat_cfg.func(fabric, mat_cfg)
    walls = [c for root in ("/World/Background", "/World/Table/room")
             for c in Usd.PrimRange(stage.GetPrimAtPath(root)) if c.IsA(UsdGeom.Mesh)]
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/Background")).MakeInvisible()
    panel = UsdGeom.Cube.Define(stage, "/World/BackPanel")  # unit cube spans -1..1, so scale = half extents
    panel.AddTranslateOp().Set(Gf.Vec3d(BACK_PANEL_X, 0.0, BACK_PANEL_SIZE[1] / 2))
    panel.AddScaleOp().Set(Gf.Vec3f(0.005, BACK_PANEL_SIZE[0] / 2, BACK_PANEL_SIZE[1] / 2))
    walls = [w for w in walls if not str(w.GetPath()).startswith("/World/Background")] + [panel.GetPrim()]
    for w in walls:
        UsdShade.MaterialBindingAPI.Apply(w).Bind(
            UsdShade.Material(stage.GetPrimAtPath(fabric)), UsdShade.Tokens.strongerThanDescendants
        )
    print(f"[scene] table {length:.3f} x {SCENE_TABLE_WIDTH:.3f} m (back cut {SCENE_TABLE_BACK_CUT} m) colour {SCENE_TABLE_COLOR}, "
          f"fabric {SCENE_BG_COLOR} on {len(walls)} wall meshes, back panel {BACK_PANEL_SIZE} m at x={BACK_PANEL_X}")


def split_link6_pad(stage, mesh_path, white, pad_min_y=0.05):
    """link6's color_64_64_64 subset holds both the pilot pad (dark on the real arm, at y > 5 cm in the
    link frame) and the joint band around the axis (white on the real arm). Keep the pad dark in its own
    subset and turn the rest white."""
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    dark = [c for c in mesh.GetPrim().GetChildren() if c.GetTypeName() == "GeomSubset"
            and "color_64_64_64" in (UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial()[0].GetPath().name or "")]
    if not dark:
        return 0
    dark = dark[0]; dark_mat = UsdShade.MaterialBindingAPI(dark).ComputeBoundMaterial()[0]
    M = np.array(UsdGeom.Xformable(mesh).GetLocalTransformation()).T  # visuals -> link frame
    pts = np.array(mesh.GetPointsAttr().Get(), float); pts = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]
    cnt = np.array(mesh.GetFaceVertexCountsAttr().Get()); idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    off = np.r_[0, np.cumsum(cnt)]
    faces = np.array(dark.GetAttribute("indices").Get())
    cy = np.array([pts[idx[off[f]:off[f + 1]], 1].mean() for f in faces])
    pad, band = faces[cy > pad_min_y], faces[cy <= pad_min_y]
    dark.GetAttribute("indices").Set(Vt.IntArray(band.tolist()))
    UsdShade.MaterialBindingAPI(dark).Bind(white)
    pad_subset = UsdGeom.Subset.CreateGeomSubset(
        mesh, "pilot_pad_dark", UsdGeom.Tokens.face, Vt.IntArray(pad.tolist()), familyName="materialBind"
    )
    UsdShade.MaterialBindingAPI.Apply(pad_subset.GetPrim()).Bind(dark_mat)
    return 1


def apply_fr3_appearance(root="/World/Robot_2"):
    """Make Isaac's FR3 asset look like our arm: one matte white (ROBOT_WHITE) on the covers, including
    the ones the CAD colours dark grey (color_64_64_64) on links 1-5 and link6's joint band, brushed
    aluminium where the CAD says 192,192,192 (flange ring), dark kept where the real arm is dark (link0
    logo ring and base parts, link6 pilot pad, link7 button and side oval), and the Franka Hand look (Isaac Lab's Panda
    hand and fingers, same frames as fr3_hand / fr3_*finger) instead of the asset's own hand."""
    stage = prim_utils.get_current_stage()
    white_path = "/World/Looks/RobotWhite"
    white_cfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(ROBOT_WHITE,) * 3, roughness=ROBOT_WHITE_ROUGHNESS, metallic=0.0
    )
    white_cfg.func(white_path, white_cfg)
    white = UsdShade.Material(stage.GetPrimAtPath(white_path))
    alu_path = "/World/Looks/RobotAluminum"  # the flange ring and base fittings (CAD colour 192,192,192)
    alu_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.47), roughness=0.4, metallic=0.3)
    alu_cfg.func(alu_path, alu_cfg)
    alu = UsdShade.Material(stage.GetPrimAtPath(alu_path))

    def rebind(prims, names, mat=None):
        n = 0
        for c in prims:
            m = UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial()[0]
            if m and any(k in m.GetPath().name for k in names):
                UsdShade.MaterialBindingAPI.Apply(c).Bind(mat or white)
                n += 1
        return n

    recoloured = 0
    whites = ("color_202_209_238", "color_255_255_255")
    for i in range(0, 8):
        mesh = stage.GetPrimAtPath(f"{root}/fr3_link{i}/visuals")
        parts = [mesh] + [c for c in mesh.GetChildren() if c.GetTypeName() == "GeomSubset"]
        # Dark grey (color_64_64_64) on the real arm: link0 base parts, the pilot pad on top of link6 and
        # the button / side oval on link7. Everywhere else (links 1-5, link6's joint band) it is white.
        dark_to_white = ("color_64_64_64",) if 1 <= i <= 5 else ()
        recoloured += rebind(parts, whites + dark_to_white)
        recoloured += rebind(parts, ("color_192_192_192",), alu)
    recoloured += split_link6_pad(stage, f"{root}/fr3_link6/visuals", white)
    props = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Props"
    for link, panda in (("fr3_hand", "panda_hand"), ("fr3_leftfinger", "panda_leftfinger"), ("fr3_rightfinger", "panda_rightfinger")):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{root}/{link}/visuals")).MakeInvisible()
        vis = stage.DefinePrim(f"{root}/{link}/panda_visuals", "Xform")
        vis.GetReferences().AddReference(f"{props}/{panda}.usd", "/Root")
        vis.SetInstanceable(False)  # so its white plastic can be rebound below
        recoloured += rebind([c for c in Usd.PrimRange(vis) if c.GetTypeName() in ("Mesh", "GeomSubset")], ("PlasticWhite",))
    print(f"[robot] fr3 appearance: {recoloured} parts rebound (RobotWhite {ROBOT_WHITE} / RobotAluminum), Panda hand/finger visuals")

camera_prim_paths = ["/World/CameraSensor"]


def rnd_high_low(high: np.ndarray, low: np.ndarray):
    rnd = np.random.random(high.shape[0])
    return low + (high - low) * rnd


def rnd_about_nominal(
    nom: np.ndarray,
    dist_type: str = "gaussian",
    uniform_scale: float = None,
    variance: float = None,
):
    assert dist_type in ["gaussian", "uniform"], f"Unrecognized dist type: {dist_type}"
    if dist_type == "gaussian":
        assert variance is not None, f"Must set variance if using Gaussian"
    if dist_type == "uniform":
        assert uniform_scale is not None, f"Must set uniform_scale if using uniform"

    if dist_type == "gaussian":
        rnd = np.random.normal(scale=np.sqrt(variance), size=nom.shape)
    elif dist_type == "uniform":
        rnd = (np.random.random(size=nom.shape) - 0.5) * uniform_scale

    return nom + rnd


def rnd_isaac_pose_about_nominal(
    nom_pose: Tuple[torch.Tensor, torch.Tensor],
    dist_type: str = "gaussian",
    uniform_scale_xyz: float = None,
    uniform_scale_rpyaw: float = None,
    variance_xyz: float = None,
    variance_rpyaw: float = None,
):
    assert dist_type in ["gaussian", "uniform"], f"Unrecognized dist type: {dist_type}"
    if dist_type == "gaussian":
        assert variance_xyz is not None, f"Must set variance if using Gaussian"
        assert variance_rpyaw is not None, f"Must set variance if using Gaussian"
    if dist_type == "uniform":
        assert uniform_scale_xyz is not None, f"Must set uniform_scale if using uniform"
        assert (
            uniform_scale_rpyaw is not None
        ), f"Must set uniform_scale if using uniform"

    if dist_type == "gaussian":
        rnd_xyz = torch.normal(mean=0.0, std=np.sqrt(variance_xyz), size=(3,))
        rnd_rpyaw = torch.normal(mean=0.0, std=np.sqrt(variance_rpyaw), size=(3,))
    elif dist_type == "uniform":
        rnd_xyz = (torch.rand(size=(3,)) - 0.5) * uniform_scale_xyz
        rnd_rpyaw = (torch.rand(size=(3,)) - 0.5) * uniform_scale_rpyaw

    # add the position part
    nom_pos = nom_pose[0]
    new_pos = nom_pos + rnd_xyz

    # convert orientation part properly
    nom_quat = T.convert_quat(nom_pose[1], to="xyzw")
    rnd_quat = T.euler2quat(rnd_rpyaw)
    new_quat = T.quat_multiply(nom_quat, rnd_quat)
    new_quat = T.convert_quat(new_quat, to="wxyz")

    if isinstance(new_quat, np.ndarray):
        new_quat = torch.from_numpy(new_quat)

    return new_pos, new_quat


def np2Vec3f(arr):
    assert arr.shape[0] == 3, f"Must be 3D to make Vec3f"
    vec = Gf.Vec3f()
    for i, val in enumerate(arr):
        vec[i] = val
    return vec


def resize(img: Union[np.ndarray, torch.Tensor]):
    """Resizes `img` into ..."""
    # th, tw = 240, 320
    # th, tw = 480, 640
    th, tw = 720, 960
    was_numpy = False

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
        was_numpy = True

    if isinstance(img, torch.Tensor):
        # Move channels in front (B, H, W, C) -> (B, C, H, W)
        if len(img.shape) == 4:
            img = img.permute(0, 3, 1, 2)
        else:
            img = img.permute(2, 0, 1)

    img = F.resize(
        img, (th, tw), interpolation=InterpolationMode.BILINEAR, antialias=True
    )

    if isinstance(img, torch.Tensor):
        # Move channels back (B, C, H, W) -> (B, H, W, C)
        if len(img.shape) == 4:
            img = img.permute(0, 2, 3, 1)
        else:
            img = img.permute(1, 2, 0)

    if was_numpy:
        img = img.numpy()

    return img


def resize_dict(img_dict: dict):
    img_dict["rgb"] = resize(img_dict["rgb"][:, :, :3])
    return img_dict


@dataclass
class RandomizationConfig:
    part_random: str = "base"
    table_random: str = "base"
    different_part_colors: bool = False
    random_frame_freq: int = 1


class RandomizationHelper:
    def __init__(self, rand_config: RandomizationConfig):
        self.rand_config = rand_config

        assert self.rand_config.part_random in ["base", "full"]
        self.part_random = self.rand_config.part_random

        assert self.rand_config.table_random in ["base", "full"]
        self.table_random = self.rand_config.table_random

        self.different_part_colors = self.rand_config.different_part_colors

        self._setup()

    def _setup(self):

        # get the nominal color for the table

        table_mat_prim = prim_utils.get_prim_at_path(table_mat_paths[0])
        # print(f"Part material prim: {part_mat_prim} at path: {pmpp}")
        assert (
            table_mat_prim.IsValid()
        ), f"Prim {table_mat_prim} at path {table_mat_paths[0]} is not valid"
        table_shader_prim = table_mat_prim.GetChildren()[0]
        self.nominal_color = np.asarray(
            table_shader_prim.GetAttribute("inputs:diffuse_color_constant").Get()
        )

        # set different low levels for part colors
        self.part_low_high = np.array([0.984, 0.889, 0.843])
        self.part_low_middle = np.array([0.5, 0.5, 0.5])

        self.part_color_low = (
            self.part_low_middle if self.part_random == "full" else self.part_low_high
        )

        self.part_high_middle = np.array([0.5, 0.5, 0.5])
        self.part_high_low = np.array([0.02, 0.02, 0.02])

        self.part_color_high = (
            self.part_high_middle if self.part_random == "full" else self.part_high_low
        )

        self.global_cams = []
        self.global_cam_poses = []
        self.local_cams = []
        self.local_cam_poses = []

    def set_global_cams(self, global_cams):
        self.global_cams = global_cams

        self.global_cam_poses = []
        for cam in self.global_cams:
            pos, ori = cam._view.get_world_poses()
            self.global_cam_poses.append((pos[0], ori[0]))

    def set_local_cams(self, local_cams):
        self.local_cams = local_cams
        self.local_cam_poses = []
        for cam in self.local_cams:
            pos, ori = cam._view.get_local_poses()
            self.local_cam_poses.append((pos[0], ori[0]))

    def random_table_colors(self):
        # set table color rand function
        if self.table_random == "full":
            return self.random_table_colors_full()
        else:
            return self.random_table_colors_nominal()

    def toggle_lights(self):
        if np.random.random() > 0.75:
            return
        for lpp in light_prim_paths:
            light_prim = prim_utils.get_prim_at_path(lpp)
            # print(f"Light prim: {light_prim} at path: {lpp}")
            assert light_prim.IsValid(), f"Prim {light_prim} at path {lpp} is not valid"
            visible_attr = light_prim.GetAttribute("visibility")

            if np.random.random() > 0.5:
                visible_attr.Set("invisible")
            else:
                visible_attr.Set("inherited")

    def random_light_colors(self):
        high = np.array([1.0, 1.0, 1.0])
        low = np.array([0.789, 0.715, 0.622])
        for lpp in light_prim_paths:
            if "03" in lpp:
                # TODO: for some reason this is broken on the last light...
                continue
            # print(f"Light prim: {light_prim} at path: {lpp}")
            light_prim = prim_utils.get_prim_at_path(lpp)
            assert light_prim.IsValid(), f"Prim {light_prim} at path {lpp} is not valid"
            color_attr = light_prim.GetAttribute("color")
            color_to_set_gf = np2Vec3f(rnd_high_low(high, low))
            color_attr.Set(color_to_set_gf)

    def random_light_intensity(self):
        high = np.array([8000])
        low = np.array([3000])
        for lpp in light_prim_paths:
            # print(f"Light prim: {light_prim} at path: {lpp}")
            light_prim = prim_utils.get_prim_at_path(lpp)
            assert light_prim.IsValid(), f"Prim {light_prim} at path {lpp} is not valid"
            intensity_attr = light_prim.GetAttribute("intensity")
            intensity_attr.Set(rnd_high_low(high, low)[0])

    def random_part_colors(self):

        if PART_COLOR_BASE == "white":
            high = np.array([1.0, 1.0, 1.0])
            # low = np.array([0.984, 0.889, 0.843])
            # low = np.array([0.5, 0.5, 0.5])
            low = self.part_color_low
        elif PART_COLOR_BASE == "black":
            high = self.part_color_high
            low = np.array([0.0, 0.0, 0.0])
        color_to_set_gf = np2Vec3f(rnd_high_low(high, low))
        for pmpp in part_mat_paths:
            part_mat_prim = prim_utils.get_prim_at_path(pmpp)
            # print(f"Part material prim: {part_mat_prim} at path: {pmpp}")
            assert (
                part_mat_prim.IsValid()
            ), f"Prim {part_mat_prim} at path {pmpp} is not valid"
            shader_prim = part_mat_prim.GetChildren()[0]
            color_attr = shader_prim.GetAttribute("inputs:diffuse_color_constant")
            if self.different_part_colors:
                color_to_set_gf = np2Vec3f(rnd_high_low(high, low))
            color_attr.Set(color_to_set_gf)

    def random_table_colors_nominal(self):

        table_mat_prim = prim_utils.get_prim_at_path(table_mat_paths[0])
        # print(f"Part material prim: {part_mat_prim} at path: {pmpp}")
        assert (
            table_mat_prim.IsValid()
        ), f"Prim {table_mat_prim} at path {table_mat_paths[0]} is not valid"
        table_shader_prim = table_mat_prim.GetChildren()[0]

        # nominal_color = np.asarray(
        #     table_shader_prim.GetAttribute("inputs:diffuse_color_constant").Get()
        # )
        nominal_color = self.nominal_color
        dist_type = "gaussian"
        variance = 0.0001

        color_attr = table_shader_prim.GetAttribute("inputs:diffuse_color_constant")
        color_to_set_gf = np2Vec3f(
            rnd_about_nominal(nom=nominal_color, dist_type=dist_type, variance=variance)
        )
        color_attr.Set(color_to_set_gf)

    def random_table_colors_full(self):

        table_mat_prim = prim_utils.get_prim_at_path(table_mat_paths[0])
        # print(f"Part material prim: {part_mat_prim} at path: {pmpp}")
        assert (
            table_mat_prim.IsValid()
        ), f"Prim {table_mat_prim} at path {table_mat_paths[0]} is not valid"
        table_shader_prim = table_mat_prim.GetChildren()[0]

        high = np.array([1.0, 1.0, 1.0])
        low = np.array([0.5, 0.5, 0.5])

        color_attr = table_shader_prim.GetAttribute("inputs:diffuse_color_constant")
        color_to_set_gf = np2Vec3f(rnd_high_low(high, low))
        color_attr.Set(color_to_set_gf)

    def random_camera_pose(self):

        dist_type = "gaussian"
        std_xyz = 0.0025
        std_rpyaw = np.deg2rad(0.5)
        for i, cam in enumerate(self.global_cams):
            nom_world_pose = self.global_cam_poses[i]
            new_world_pose = rnd_isaac_pose_about_nominal(
                nom_pose=nom_world_pose,
                dist_type=dist_type,
                variance_xyz=std_xyz**2,
                variance_rpyaw=std_rpyaw**2,
            )
            new_pos, new_quat = new_world_pose
            # cam._sensor_xform.set_world_pose(new_pos, new_quat)
            cam._view.set_world_poses(
                new_pos.reshape(1, 3),
                new_quat.reshape(1, 4),
            )

        std_xyz = 0.00075
        std_rpyaw = np.deg2rad(0.25)
        for i, cam in enumerate(self.local_cams):
            nom_local_pose = self.local_cam_poses[i]
            new_local_pose = rnd_isaac_pose_about_nominal(
                nom_pose=nom_local_pose,
                dist_type=dist_type,
                variance_xyz=std_xyz**2,
                variance_rpyaw=std_rpyaw**2,
            )
            new_pos, new_quat = new_local_pose
            # cam._sensor_xform.set_local_pose(new_pos, new_quat)
            cam._view.set_local_poses(
                new_pos.reshape(1, 3),
                new_quat.reshape(1, 4),
            )


def main():

    s = time.time()

    data = unpickle_data(FILE)
    """Spawns a single arm manipulator and applies random joint commands."""

    # Load kit helper
    # sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    sim = SimulationContext(physics_dt=0.0001, rendering_dt=0.0001, backend="torch")

    # get camera params
    cam_params = build_camera_params()

    # Setup camera sensor (front)
    camera_cfg = CameraCfg(
        height=480,
        width=640,
        # width=1440,
        # height=1080,
        # height=720,
        # width=1280,
        prim_path="/World/CameraSensor",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam_params["front"]["focal_length"],
            horizontal_aperture=cam_params["front"]["horizontal_aperture"],
            clipping_range=(0.1, 1.0e5),
            f_stop=cam_params["front"]["f_stop"],
            focus_distance=cam_params["front"]["focus_distance"],
        ),
    )
    camera = Camera(cfg=camera_cfg)

    # Table
    usd_base_path = Path(
        f"{os.getenv('RARL_SOURCE_DIR')}/sim2real/assets/furniture/mesh/usd"
    )
    table_z_offset = 0.415

    prim_utils.create_prim(
        "/World/Table", usd_path=str(usd_base_path / "table_room.usda")
    )
    table_view = XFormPrimView("/World/Table", reset_xform_properties=False)
    table_view.set_local_scales(torch.Tensor([[1.7, 1.35, 1.0]]))

    views = []

    # Background
    prim_utils.create_prim(
        "/World/Background",
        translation=(-0.8, 0, 0.75),
        usd_path=str((usd_base_path / "background.usda")),
    )

    # Fixture geometry, overridable per run with environment variables so it can match the real
    # printed fixture. Defaults: side-bar inner faces 35 cm apart (measured on our real fixture;
    # the FurnitureBench sim uses 33 cm), front bar 35 x 2 x 3 cm, side bars 17 x 2 x 3 cm.
    obst_side_y = float(os.getenv("OBST_SIDE_Y", 0.185))  # side-bar centre |y| from front-bar centre (sim: 0.175)
    obst_side_dx = float(os.getenv("OBST_SIDE_DX", 0.075))  # side-bar centre behind front-bar centre (sim: 0.075)
    obst_front_len = float(os.getenv("OBST_FRONT_LEN", 0.35))  # front-bar length in m (mesh is 0.35)
    print(
        f"[fixture] side |y|={obst_side_y:.3f} m (inner gap {2 * obst_side_y - 0.02:.3f} m), "
        f"side dx={obst_side_dx:.3f} m, front length={obst_front_len:.3f} m"
    )

    # Obstacle front.
    obs_quat_wxyz = np.array([0.7071067690849304, 0, 0, 0.7071067690849304])
    prim_utils.create_prim(
        "/World/ObstacleFront",
        translation=(0.3815, 0.0, table_z_offset),
        orientation=obs_quat_wxyz,
        usd_path=str((usd_base_path / "obstacle_front.usda")),
    )

    # Obstacle side.
    prim_utils.create_prim(
        "/World/ObstacleRight",
        translation=(0.306, -0.175, table_z_offset),
        orientation=obs_quat_wxyz,
        usd_path=str((usd_base_path / "obstacle_side.usda")),
    )
    prim_utils.create_prim(
        "/World/ObstacleLeft",
        translation=(0.306, 0.175, table_z_offset),
        orientation=obs_quat_wxyz,
        usd_path=str((usd_base_path / "obstacle_side.usda")),
    )
    obstacle_center_view = XFormPrimView(
        "/World/ObstacleFront", reset_xform_properties=False
    )
    # The front-bar mesh is long along its local x (extent +-0.175 m), so stretch only that axis.
    obstacle_center_view.set_local_scales(torch.Tensor([[obst_front_len / 0.35, 1.0, 1.0]]))
    obstacle_right_view = XFormPrimView(
        "/World/ObstacleRight", reset_xform_properties=False
    )
    obstacle_left_view = XFormPrimView(
        "/World/ObstacleLeft", reset_xform_properties=False
    )

    # Parts
    part_config = part_config_dict[args_cli.furniture]

    furn_name = part_config["furniture"]
    for i, part_name in enumerate(part_config["names"]):
        prim_path = part_config["prim_paths"][i]
        usd_path = furn_name + "/" + part_config["usd_names"][i]
        pos = config["furniture"][furn_name][part_name]["reset_pos"][0]
        ori = config["furniture"][furn_name][part_name]["reset_ori"][0][:3, :3]
        view = gen_prim(usd_path, prim_path, pos, ori)
        views.append(view)

    # Robot
    franka_from_origin_mat = np.array(
        [
            [1.0, 0.0, 0.0, -0.3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, table_z_offset],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pos, ori = T.mat2pose(franka_from_origin_mat)
    robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot_2")
    if args_cli.robot == "fr3":
        # Same kinematics as the Panda; Isaac Sim's FR3 asset names joints/links fr3_* instead of panda_*,
        # and fr3_hand sits on fr3_link8 rotated -45 deg about z with no offset, exactly like panda_hand.
        robot_cfg.spawn = robot_cfg.spawn.replace(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/FR3/fr3.usd")
        robot_cfg.init_state = robot_cfg.init_state.replace(
            joint_pos={k.replace("panda_", "fr3_"): v for k, v in robot_cfg.init_state.joint_pos.items()}
        )
        robot_cfg.actuators = {
            k: v.replace(joint_names_expr=[e.replace("panda_", "fr3_") for e in v.joint_names_expr])
            for k, v in robot_cfg.actuators.items()
        }
    robot_prefix = "fr3" if args_cli.robot == "fr3" else "panda"
    robot_cfg.init_state.pos = pos
    robot = Articulation(cfg=robot_cfg)
    apply_scene_appearance()
    if args_cli.robot == "fr3":
        apply_fr3_appearance()

    # Setup camera sensor (wrist)
    wrist_camera_cfg = CameraCfg(
        height=480,
        width=640,
        # width=1440,
        # height=1080,
        prim_path=f"/World/Robot_2/{robot_prefix}_hand/camera",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam_params["wrist"]["focal_length"],
            horizontal_aperture=cam_params["wrist"]["horizontal_aperture"],
            clipping_range=(1.0e-5, 1.0e5),
            f_stop=cam_params["wrist"]["f_stop"],
            focus_distance=cam_params["wrist"]["focus_distance"],
        ),
    )
    # wrist_camera_cfg.offset.pos = torch.from_numpy(np.array([-0.05, 0.0, 0.04]))
    # wrist_camera_cfg.offset.rot = torch.from_numpy(
    #     T.convert_quat(
    #         R.from_euler("xyz", np.deg2rad([0, 200, -90])).as_quat(),
    #         to="wxyz",
    #     )
    # )
    # wrist_camera_cfg.offset.convention = "world"

    wrist_camera = Camera(cfg=wrist_camera_cfg)

    # Play the simulator
    sim.reset()

    # The pickles store 7 arm joints followed by finger 1 and finger 2; write them by name, not by index.
    expected_joints = [f"{robot_prefix}_joint{i}" for i in range(1, 8)] + [
        f"{robot_prefix}_finger_joint1",
        f"{robot_prefix}_finger_joint2",
    ]
    joint_ids = [robot.joint_names.index(n) for n in expected_joints]
    print(f"[robot] {args_cli.robot}: joints {robot.joint_names}, wrist camera on /World/Robot_2/{robot_prefix}_hand")

    cam_pos = torch.from_numpy(np.array([0.787, 0.0, 0.809]).reshape(1, 3))
    cam_quat = torch.from_numpy(
        T.convert_quat(
            R.from_euler("XYZ", np.deg2rad([-0.145, 64.17, 90.161])).as_quat(), to="wxyz"
        ).reshape(1, 4)
    )
    camera._view.set_world_poses(cam_pos, cam_quat)
    # camera._sensor_xform.set_world_pose(cam_pos, cam_quat)

    # Wrist RGB optical centre in the hand frame (FurnitureBench mount, D435 on its two rear M3 holes, mount
    # screwed to the hand back on the centre line). y: the camera body centre sits 28.5 mm to the hand's -y
    # (mount STL: M3-hole midpoint vs counterbored hole) and the RGB lens is 32.5 mm to the camera's left
    # = hand +y (D400 datasheet: 17.5 mm 1/4-20 centreline -> left imager, +15 mm -> RGB), so +4 mm;
    # +5 mm after comparing with the real wrist image.
    local_wrist_pos = torch.from_numpy(np.array([-0.05, 0.005, 0.04])).reshape(1, 3)
    local_wrist_rot = torch.from_numpy(
        T.convert_quat(
            # 180 + 17: view along the fingers, tilted 17 deg toward them (mount nominal is 20 deg)
            R.from_euler("XYZ", np.deg2rad([0, 197, -90])).as_quat(),
            to="wxyz",
        )
    ).reshape(1, 4)
    wrist_camera._view.set_local_poses(local_wrist_pos, local_wrist_rot)

    # Create replicator writer
    demo_date = FILE.split("/")[-1].replace(".pkl", "")
    output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "output",
        "camera",
        # f"{demo_date}_substeps_{args_cli.sub_steps}",
        f"{demo_date}_substeps_{args_cli.sub_steps}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    )
    if args_cli.domain_rand:
        output_dir += "_domain_rand"

    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    wrist_output_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "output",
        "wrist_camera",
        # f"{demo_date}_substeps_{args_cli.sub_steps}",
        f"{demo_date}_substeps_{args_cli.sub_steps}_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
    )
    if args_cli.domain_rand:
        wrist_output_dir += "_domain_rand"

    wrist_rep_writer = rep.BasicWriter(output_dir=wrist_output_dir, frame_padding=3)

    if args_cli.render_viewer:
        rp3 = rep.create.render_product(
            "/OmniverseKit_Persp", (int(640 * 3), int(480 * 3))
        )
        rp_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rp_rgb.attach(rp3)
        viewer_rep_writer = rep.BasicWriter(
            output_dir=output_dir + "_viewer", frame_padding=3
        )

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    ep_step_count = 0

    sim_steps = 1
    print(f"Sim steps: {sim_steps}")

    prev_goal_pos = torch.tensor(
        data["observations"][0]["robot_state"]["joint_positions"]
    )

    parts_prev_goal_pos = []
    parts_prev_goal_ori = []
    # part_idx_offset = 1
    part_idx_offset = 0

    fj1 = data["observations"][0]["robot_state"]["gripper_finger_1_pos"]
    fj2 = data["observations"][0]["robot_state"]["gripper_finger_2_pos"]

    # Initialize robot state
    robot.write_joint_state_to_sim(
        torch.concat(
            [
                torch.tensor(data["observations"][0]["robot_state"]["joint_positions"]),
                torch.from_numpy(
                    fj1.reshape(
                        1,
                    )
                ).float(),
                torch.from_numpy(
                    fj2.reshape(
                        1,
                    )
                ).float(),
            ]
        ),
        torch.concat(
            [
                torch.tensor(
                    data["observations"][0]["robot_state"]["joint_velocities"]
                ),
                torch.zeros((1,)),
                torch.zeros((1,)),
            ]
        ),
        joint_ids=joint_ids,
    )

    # initialize parts
    for i in range(args_cli.num_parts):
        pos = data["observations"][part_idx_offset]["parts_poses"][7 * i : 7 * i + 3]
        ori = data["observations"][part_idx_offset]["parts_poses"][
            7 * i + 3 : 7 * i + 7
        ]
        parts_prev_goal_pos.append(pos)
        parts_prev_goal_ori.append(ori)

    # initialize obstacles
    obstacle_center_pose = data["observations"][1]["parts_poses"][-7:]
    obstacle_right_pose = obstacle_center_pose.copy()
    obstacle_left_pose = obstacle_center_pose.copy()

    for obs_name, obs_view, obs_pose in zip(
        ["center", "right", "left"],
        [obstacle_center_view, obstacle_right_view, obstacle_left_view],
        [obstacle_center_pose, obstacle_right_pose, obstacle_left_pose],
    ):

        rot = T.quat2mat(obs_pose[3:])
        obs_pose_world = april_to_sim_mat @ (T.to_homogeneous(obs_pose[:3], rot))
        obs_pose_pos, obs_pose_ori = T.mat2pose(obs_pose_world)
        obs_pose_ori = T.convert_quat(obs_pose_ori, to="wxyz")

        if obs_name == "right":
            obs_pose_pos[0] -= obst_side_dx
            obs_pose_pos[1] -= obst_side_y
        elif obs_name == "left":
            obs_pose_pos[0] -= obst_side_dx
            obs_pose_pos[1] += obst_side_y

        obs_pose_pos = torch.from_numpy(obs_pose_pos).unsqueeze(0)
        obs_pose_ori = torch.from_numpy(obs_pose_ori).unsqueeze(0)

        obs_view.set_world_poses(positions=obs_pose_pos, orientations=obs_pose_ori)

    # run_until_quit(simulation_app=simulation_app, world=sim)

    for _ in range(100):
        sim.step()
        time.sleep(0.01)

    # setup re-saving
    if args_cli.save:
        assert args_cli.save_dir is not None, f"Must set --save-dir if --save is True!"
        demo_save_dir = Path(args_cli.save_dir)
        demo_save_dir.mkdir(exist_ok=True, parents=True)
        pkl_path = demo_save_dir / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"

    episode_data = {}
    episode_data["observations"] = []
    episode_data["actions"] = data["actions"]
    episode_data["task"] = data.get("task", data.get("furniture"))
    # assume all real world demos that we actually save are success
    episode_data["success"] = True
    # episode_data["args"] = data["args"]
    episode_data["args_rerender"] = args_cli.__dict__
    if "metadata" in data:
        episode_data["metadata"] = data["metadata"]
    else:
        episode_data["metadata"] = None

    fps_list = []
    last_time = time.time()

    # Use the subcommand arguments
    dr_config = RandomizationConfig()
    if args_cli.subcommand == "rand":
        rand_args = args_cli
        for rand_key in dr_config.__dict__.keys():
            print(f"Rand key: {rand_key}, args value: {rand_args.__dict__[rand_key]}")
            dr_config.__dict__[rand_key] = rand_args.__dict__[rand_key]

    dr_helper = RandomizationHelper(rand_config=dr_config)
    dr_helper.set_global_cams([camera])
    dr_helper.set_local_cams([wrist_camera])

    # Try to automatically retrieve all the material prims
    for i, prim_path in enumerate(part_config["prim_paths"]):
        looks_prim_path = prim_path + "/Looks"
        assert prim_utils.is_prim_path_valid(
            looks_prim_path
        ), f"Could not find Looks prim: {looks_prim_path}"
        looks_prim = prim_utils.get_prim_at_path(looks_prim_path)
        for looks_child_prim in looks_prim.GetChildren():
            looks_child_path = looks_child_prim.GetPath().pathString
            if (
                "Material" in looks_child_path
                and looks_child_path not in part_mat_paths
            ):
                print(
                    f"Adding Material path: {looks_child_path} to materials for randomization"
                )
                part_mat_paths.append(looks_child_path)

    print(f"Found material prims for all parts: {part_mat_paths}")

    for obs_idx, obs in enumerate(data["observations"]):

        if args_cli.domain_rand:
            #     if obs_idx % dr_config.random_frame_freq == 0:

            # if False:
            # if True:
            dr_helper.toggle_lights()
            dr_helper.random_light_colors()
            dr_helper.random_light_intensity()
            dr_helper.random_part_colors()
            dr_helper.random_table_colors()
            dr_helper.random_camera_pose()

        goal_pos = torch.tensor(obs["robot_state"]["joint_positions"])
        dx = (goal_pos - prev_goal_pos) / sim_steps

        part_idx = (
            obs_idx + part_idx_offset
            if obs_idx + part_idx_offset < len(data["observations"])
            else len(data["observations"]) - 1
        )

        # # if obs_idx == 150:
        # if obs_idx == int(len(data["observations"]) * 0.5):
        #     print(f"Stopping")
        #     run_until_quit(simulation_app=simulation_app, world=sim)
        #     from pdb import set_trace

        #     set_trace()
        #     # from IPython import embed

        #     # embed()
        #     assert False

        for i in range(int(sim_steps)):
            interp_goal = prev_goal_pos + (i + 1) * dx

            # If simulation is paused, then skip.
            if not sim.is_playing():
                sim.step(render=not args_cli.headless)
                continue

            # Update camera data
            camera.update(dt=0.0)
            wrist_camera.update(dt=0.0)

            if not args_cli.save:
                # "trigger_outputs"] = {"on_time": camera.frame[camera_index]}
                # front_to_write = convert_dict_to_backend(
                #     camera.data.output, backend="numpy"
                # )
                front_cam_data = convert_dict_to_backend(
                    camera.data.output, backend="numpy"
                )
                front_cam_data["rgb"] = front_cam_data["rgb"][:, :, :3]
                front_cam_info = camera.data.info[0]
                front_rep_output = {"annotators": {}}
                for key, cam_data, info in zip(
                    front_cam_data.keys(),
                    front_cam_data.values(),
                    front_cam_info.values(),
                ):
                    if info is not None:
                        front_rep_output["annotators"][key] = {
                            "render_product": {"data": cam_data, **info}
                        }
                    else:
                        front_rep_output["annotators"][key] = {
                            "render_product": {"data": cam_data}
                        }
                front_rep_output["trigger_outputs"] = {"on_time": 0}
                # rep_writer.write(front_rep_output)

                wrist_cam_data = convert_dict_to_backend(
                    wrist_camera.data.output, backend="numpy"
                )
                wrist_cam_data["rgb"] = wrist_cam_data["rgb"][:, :, :3]
                wrist_cam_info = wrist_camera.data.info[0]
                wrist_rep_output = {"annotators": {}}
                for key, cam_data, info in zip(
                    wrist_cam_data.keys(),
                    wrist_cam_data.values(),
                    wrist_cam_info.values(),
                ):
                    if info is not None:
                        wrist_rep_output["annotators"][key] = {
                            "render_product": {"data": cam_data, **info}
                        }
                    else:
                        wrist_rep_output["annotators"][key] = {
                            "render_product": {"data": cam_data}
                        }
                wrist_rep_output["trigger_outputs"] = {"on_time": 0}
                # wrist_rep_writer.write(wrist_rep_output)

                if args_cli.render_viewer:
                    viewer_rep_writer.write(
                        {"rgb": rp_rgb.get_data(), "trigger_outputs": {"on_time": 0}}
                    )
                # rep_writer.write(resize_dict(camera.data.output))
                # wrist_rep_writer.write(resize_dict(wrist_camera.data.output))

            elapsed = time.time() - last_time
            fps = 1.0 / elapsed
            fps_list.append(fps)
            if i % 10 == 0:
                fps_avg = np.mean(fps_list[-10:])
                print(f"Frames per second: {fps_avg}")
            last_time = time.time()

            # robot.set_dof_state(
            robot.write_joint_state_to_sim(
                torch.concat(
                    [
                        interp_goal,
                        torch.from_numpy(
                            fj1.reshape(
                                1,
                            )
                        ).float(),
                        torch.from_numpy(
                            fj2.reshape(
                                1,
                            )
                        ).float(),
                    ]
                ),
                torch.concat(
                    [
                        torch.tensor(obs["robot_state"]["joint_velocities"]),
                        torch.zeros((1,)),
                        torch.zeros((1,)),
                    ]
                ),
                joint_ids=joint_ids,
            )

            for j in range(args_cli.num_parts):
                # pos = obs['parts_poses'][7 * i:7 * i + 3]
                goal_pos = data["observations"][part_idx]["parts_poses"][
                    7 * j : 7 * j + 3
                ]
                part_dx = (goal_pos - parts_prev_goal_pos[j]) / sim_steps
                pos = torch.tensor(parts_prev_goal_pos[j] + (i + 1) * part_dx)

                goal_ori = data["observations"][part_idx]["parts_poses"][
                    7 * j + 3 : 7 * j + 7
                ]
                interp_fraction = i / sim_steps
                ori = T.quat_slerp(
                    parts_prev_goal_ori[j], goal_ori, fraction=interp_fraction
                )

                rot = T.quat2mat(ori)
                pose = april_to_sim_mat @ (T.to_homogeneous(pos, rot))
                pos, ori = T.mat2pose(pose)
                ori = T.convert_quat(ori, to="wxyz")
                pos = torch.from_numpy(pos).unsqueeze(0)
                ori = torch.from_numpy(ori).unsqueeze(0)
                views[j].set_world_poses(positions=pos, orientations=ori)

            # perform step
            sim.step()
            for _ in range(args_cli.sub_steps):
                sim.render()

            # update sim-time
            sim_time += sim_dt
            ep_step_count += 1

        prev_goal_pos = torch.tensor(obs["robot_state"]["joint_positions"])
        fj1 = obs["robot_state"]["gripper_finger_1_pos"]
        fj2 = obs["robot_state"]["gripper_finger_2_pos"]
        parts_prev_goal_pos = []
        parts_prev_goal_ori = []
        for i in range(args_cli.num_parts):
            pos = data["observations"][part_idx]["parts_poses"][7 * i : 7 * i + 3]
            ori = data["observations"][part_idx]["parts_poses"][7 * i + 3 : 7 * i + 7]
            parts_prev_goal_pos.append(pos)
            parts_prev_goal_ori.append(ori)

        if args_cli.save:
            # log the new observation
            wrist_image = (
                resize(wrist_camera.data.output["rgb"][0, :, :, :3]).cpu().numpy()
            )
            front_image = resize(camera.data.output["rgb"][0, :, :, :3]).cpu().numpy()

            # print(f"front_image shape: {front_image.shape}")

            # log the new observation
            wrist_image = wrist_camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
            front_image2 = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()

            # print(f"front_image shape: {front_image.shape}")

            new_obs = dict(
                color_image1=wrist_image,
                color_image2=front_image,
                color_image3=front_image2,
                robot_state=obs["robot_state"],
            )
            if args_cli.robot_mask:
                # On a CPU sim, .cpu().numpy() aliases the camera buffer; copy before re-rendering.
                front_image2 = front_image2.copy()
                wrist_image = wrist_image.copy()
                new_obs["color_image1"], new_obs["color_image3"] = wrist_image, front_image2
                # Render the same frame once more with the arm hidden; pixels that change are the robot
                # (and its shadow). Then show the arm again so the next frames are unaffected.
                robot_root = prim_utils.get_prim_at_path("/World/Robot_2")
                robot_links = [UsdGeom.Imageable(c) for c in robot_root.GetChildren() if c.IsA(UsdGeom.Xformable)]
                for lk in robot_links:
                    lk.MakeInvisible()
                sim.step()  # camera buffers only refresh after a physics step (0.1 ms here, negligible)
                for _ in range(max(args_cli.sub_steps, 1) + 2):
                    sim.render()
                camera.update(dt=0.0)
                no_robot = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().copy()
                for lk in robot_links:
                    lk.MakeVisible()
                sim.step()  # camera buffers only refresh after a physics step (0.1 ms here, negligible)
                for _ in range(max(args_cli.sub_steps, 1) + 2):
                    sim.render()
                diff = np.abs(front_image2.astype(np.int16) - no_robot.astype(np.int16)).max(-1)
                new_obs["robot_mask_front"] = diff > 12
            episode_data["observations"].append(new_obs)

    e = time.time()
    print(f"Time taken: {e-s}")

    if args_cli.save:
        print(f"Saving new pickle to: {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(episode_data, f)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
