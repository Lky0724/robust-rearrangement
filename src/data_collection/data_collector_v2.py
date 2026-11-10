"""Scripted demo collector with a modified one_leg expert. Run with:

    python -m src.data_collection.data_collector_v2 -f one_leg -r med -n 100 --headless --save-failure --save-mp4

Differences from data_collector.py, all done here by swapping the part FSMs at runtime (furniture-bench is untouched):
  - Motion: moves blend translation, rotation and height instead of going through separate xy, turn and descend
    phases and fixed waypoints. Larger per-step limits make the arm about twice as fast at turning.
  - Screw: each grasp turns the leg 180 degrees (+90 -> -90) instead of 90, so the gripper opens and closes less.
  - Gripper timing: only opens/closes once the arm has reached its target and stopped, not when a phase times out.
  - Tabletop: squared up to the fixture, then pressed into the corner, closing the loop on the tabletop's own pose.
  - After success: opens the gripper and sweeps to a centered pose, gripper square to the workspace, in one motion.
"""
import argparse
import lzma
import math
import types
import time
import pickle
from datetime import datetime
from pathlib import Path
import gzip

import gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.data.collect_enum import CollectEnum
from furniture_bench.sim_config import sim_config
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action
from furniture_bench.config import config
from furniture_bench.furniture.parts.square_table_leg import SquareTableLeg
from furniture_bench.furniture.parts.square_table_top import SquareTableTop
from furniture_bench.utils.pose import get_mat, rot_mat
import furniture_bench.controllers.control_utils as C

from src.common.files import trajectory_save_dir
from src.data_processing.utils import resize, resize_crop
from src.gym import turn_off_april_tags
from src.visualization.render_mp4 import pickle_data, create_mp4, data_to_video


# Per-step action limits. Commanded motion sets the speed in this controller: the arm covers about a fifth of a
# commanded turn and a third of a commanded move per step. process_pickles clips position deltas to 2.5 cm per axis.
POS_BOUND_M = 0.025
ORI_BOUND_DEG = 30

SETTLE_STEPS = 2  # Consecutive steps at the target before the gripper may change

# Approaching a part: height drops as the gripper lines up, and it only moves sideways once it is clear of the table
HOVER_HEIGHT = 0.08
HOVER_PER_POS_ERROR = 1.5
HOVER_PER_ANGLE_ERROR = 0.1  # Meters per radian
APPROACH_POS_TOLERANCE = 0.004
APPROACH_ANGLE_TOLERANCE = math.radians(5)
CLEAR_Z = 0.04
CLEAR_BAND = 0.03
NEAR_GOAL_DIST = 0.03  # Within this, sideways motion is always allowed
APPROACH_TIMEOUT = 70
GRASP_POS_NOISE = 0.003  # Varies the grasps, like the stock script
GRASP_YAW_NOISE = math.radians(3)

# Gripper
OPEN_CLEAR_LEG = 0.045  # Wide enough to move around the leg, so there's no need to wait for fully open
OPEN_CLEAR_TAB = 0.05
GRIP_MIN_STEPS = 6
GRIP_TIMEOUT = 15

# Carrying the leg to the hole
CARRY_HEIGHT = 0.12
CARRY_LIFT_START = 0.02  # Above the height it was picked at
CARRY_LIFT_BAND = 0.04
CARRY_POS_TOLERANCE = 0.015
CARRY_ANGLE_TOLERANCE = math.radians(8)
CARRY_TIMEOUT = 70
ALIGN_HOVER = 0.03
ALIGN_HOVER_PER_ERROR = 3.0
ALIGN_TIMEOUT = 60
LEG_POS_TOLERANCE = 0.003
INSERT_TIMEOUT = 40

# Screw. Joint 7 is about 1.0 at yaw 0 and 2.4 at -90 (limit 2.9), so a 180 degree turn has to start from +90.
SCREW_START_YAW = math.pi / 2
SCREW_END_YAW = -math.pi / 2
YAW_STEP = math.radians(ORI_BOUND_DEG)  # How far ahead of the current yaw to aim, so a turn can't go the short way around
SCREW_YAW_STEP = math.radians(15)  # Turning the leg any faster doesn't let the thread catch, and it pops back out
YAW_GAIN = 2.0
YAW_TOLERANCE = math.radians(2)
ROTATE_TIMEOUT = 90
# After the insertion the gripper is still tilted around a leg that is only just seated. Swinging upright fast knocks
# it out of the hole, so straighten up slowly and center on the leg as the tilt goes away, then turn at full speed.
STRAIGHTEN_ANGLE = math.radians(5)
STRAIGHTEN_MAX_AIM = math.radians(15)
STRAIGHTEN_CENTER_RANGE = math.radians(36)

# Tabletop
WALL_HALF_THICKNESS = 0.01
PUSH_CLEARANCE = 0.015  # Square up this far from both walls, so the corners don't scrape on the way in
PUSH_OVERSHOOT = 0.003  # Aim slightly past the corner to press into it
PUSH_POS_TOLERANCE = 0.002
PUSH_YAW_TOLERANCE = math.radians(1.5)
PUSH_STALL_DIST = 0.0005
PUSH_STALL_STEPS = 8
PUSH_TIMEOUT = 60

# Return to a centered pose after success
HOME_POS = (0.5, 0.0, 0.2)  # Over the middle of the workspace, on the fixture's center line
HOME_YAW = 0.0  # Gripper square to the workspace
HOME_LEG_CLEARANCE = 0.065  # Above the height the leg was held at
HOME_CLEAR_BAND = 0.03
HOME_POS_TOLERANCE = 0.01
HOME_ANGLE_TOLERANCE = math.radians(4)
HOME_TIMEOUT = 80

MAX_ROTATION_AIM = math.radians(40)


def _rot_z(yaw, device):
    c, s = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=torch.float32
    )


def _yaw_of(mat):
    return math.atan2(mat[1, 0].item(), mat[0, 0].item())


def _gripper_down_ori(yaw, device):
    """Gripper pointing straight down, turned by yaw. Same as rot_mat([pi, 0, yaw])."""
    flip = torch.tensor([1.0, -1.0, -1.0], device=device).diag()
    return _rot_z(yaw, device) @ flip


def _yaw_toward(ee_yaw, goal_yaw, step=YAW_STEP):
    """Next yaw on the way to goal_yaw. Stays on the side of the circle through 0, so it never wraps."""
    return ee_yaw + float(np.clip(YAW_GAIN * (goal_yaw - ee_yaw), -step, step))


def _angle_between(a, b):
    cos = ((a.T @ b).trace().item() - 1) / 2
    return math.acos(float(np.clip(cos, -1.0, 1.0)))


def _rotate_toward(ee_ori, goal_ori, fraction=1.0, gain=YAW_GAIN, max_aim=MAX_ROTATION_AIM):
    """Orientation to aim for to turn fraction of the way to goal_ori, without slowing to a crawl near the end."""
    rotvec = R.from_matrix((ee_ori.T @ goal_ori).cpu().numpy().astype(np.float64)).as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-6:
        return goal_ori
    aim = min(gain * fraction * angle, max_aim)
    rel = R.from_rotvec(rotvec / angle * aim).as_matrix()
    return ee_ori @ torch.tensor(rel, device=ee_ori.device, dtype=torch.float32)


def _smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def _approach(ee_pos, ee_ori, goal_pos, goal_ori):
    """One blended step toward a grasp. Returns the target pose and whether the gripper has arrived."""
    xy_error = (goal_pos[:2] - ee_pos[:2]).norm().item()
    angle = _angle_between(ee_ori, goal_ori)
    hover = min(
        HOVER_HEIGHT,
        HOVER_PER_POS_ERROR * max(0.0, xy_error - APPROACH_POS_TOLERANCE)
        + HOVER_PER_ANGLE_ERROR * max(0.0, angle - APPROACH_ANGLE_TOLERANCE),
    )
    sideways = max(
        _smoothstep((ee_pos[2].item() - CLEAR_Z) / CLEAR_BAND),
        1 - min(xy_error / NEAR_GOAL_DIST, 1.0),
    )
    target_pos = ee_pos.clone()
    target_pos[:2] += sideways * (goal_pos[:2] - ee_pos[:2])
    target_pos[2] = goal_pos[2] + hover
    arrived = (
        xy_error < APPROACH_POS_TOLERANCE
        and abs((goal_pos[2] - ee_pos[2]).item()) < APPROACH_POS_TOLERANCE
        and angle < APPROACH_ANGLE_TOLERANCE
    )
    # Turning waits for the same clearance as moving sideways, or the fingers knock the part it just let go of
    target_ori = _rotate_toward(ee_ori, goal_ori, fraction=sideways)
    return C.to_homogeneous(target_pos, target_ori), arrived


def _steps_in_state(part):
    """Steps since the FSM entered its current state. part.curr_cnt isn't reset between episodes, so keep our own."""
    if getattr(part, "_v2_state", None) != part._state:
        part._v2_state = part._state
        part._v2_steps = 0
        part._v2_settled = 0
        part._v2_stall_xy = None
        part._v2_stall_steps = 0
    else:
        part._v2_steps += 1
    return part._v2_steps


def _settled(part, at_target):
    part._v2_settled = part._v2_settled + 1 if at_target else 0
    return part._v2_settled >= SETTLE_STEPS


def _leg_fsm_step(
    self,
    ee_pos,
    ee_quat,
    gripper_width,
    rb_states,
    part_idxs,
    sim_to_april_mat,
    april_to_robot,
    assemble_to,
):
    """Replaces Leg.fsm_step, with the same states up to the insertion, minus the fixed waypoints."""
    device = ee_pos.device
    steps = _steps_in_state(self)
    next_state = self._state

    ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
    ee_ori = ee_pose[:3, :3]
    ee_yaw = _yaw_of(ee_pose)
    table_pose = sim_to_april_mat @ C.to_homogeneous(
        rb_states[part_idxs[assemble_to]][0][:3],
        C.quat2mat(rb_states[part_idxs[assemble_to]][0][3:7]),
    )
    leg_pose = sim_to_april_mat @ C.to_homogeneous(
        rb_states[part_idxs[self.name]][0][:3],
        C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
    )
    leg_pose_robot = april_to_robot @ leg_pose
    leg_pos_robot = leg_pose_robot[:3, 3]
    hole = (
        april_to_robot
        @ table_pose
        @ torch.tensor(
            get_mat(self.default_assembled_pose[:3, 3], [0.0, 0.0, 0.0]), device=device
        ).float()
    )
    tilt = torch.tensor(rot_mat([0, -np.pi / 5, 0], hom=True), device=device).float()

    def leg_pose_for_hole():
        """The leg's pose, among its 4 symmetric ones, closest to facing front."""
        best = leg_pose_robot.clone()
        tmp = leg_pose_robot
        rot = torch.tensor(rot_mat([0, -np.pi / 2, 0], hom=True), device=device).float()
        for _ in range(3):
            tmp = tmp @ rot
            if best[0, 0] < tmp[0, 0]:
                best = tmp
        return best

    def goal_leg_pose(height):
        return torch.tensor(
            [
                [1.0, 0.0, 0.0, hole[0, 3]],
                [0.0, 0.0, -1.0, hole[1, 3]],
                [0.0, 1.0, 0.0, table_pose[2, 3] + height],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )

    if self._state == "reach_leg_floor_xy":
        if steps == 0:
            self._v2_noise = torch.normal(
                torch.zeros(2, device=device), GRASP_POS_NOISE
            )
        down_pose = self._find_down_z(leg_pose).clone().to(device)
        goal_pos = (april_to_robot @ down_pose[:4, 3])[:3]
        goal_pos[1] += 0.01
        goal_pos[0] += self.grasp_margin_x
        goal_pos[:2] += self._v2_noise
        goal_pos[2] = leg_pos_robot[2]
        rot = torch.tensor(rot_mat([np.pi / 2, -np.pi / 2, 0], hom=True), device=device).float()
        goal_ori = (tilt @ april_to_robot @ rot)[:3, :3]
        target, arrived = _approach(ee_pos, ee_ori, goal_pos, goal_ori)
        if _settled(self, arrived) or steps >= APPROACH_TIMEOUT:
            self.prev_pose = C.to_homogeneous(goal_pos, goal_ori)
            next_state = "pick_leg"
    elif self._state == "pick_leg":
        target = self.prev_pose
        self.gripper_action = 1
        closed = gripper_width <= 2 * self.half_width + 0.002
        if (steps >= GRIP_MIN_STEPS and _settled(self, closed)) or steps >= GRIP_TIMEOUT:
            self.pick_z = ee_pos[2].item()
            next_state = "lift_up"
    elif self._state == "lift_up":
        # Rise, turn the leg upright and head for the hole, all at once. Sideways and turning fade in with height.
        progress = _smoothstep(
            (ee_pos[2].item() - self.pick_z - CARRY_LIFT_START) / CARRY_LIFT_BAND
        )
        goal_ori = (tilt @ torch.tensor(rot_mat([np.pi, 0, 0], hom=True), device=device).float())[:3, :3]
        xy_error = hole[:2, 3] - leg_pos_robot[:2]
        target_pos = ee_pos.clone()
        target_pos[:2] += progress * xy_error
        target_pos[2] = CARRY_HEIGHT
        target = C.to_homogeneous(
            target_pos, _rotate_toward(ee_ori, goal_ori, fraction=progress)
        )
        if (
            (
                xy_error.norm().item() < CARRY_POS_TOLERANCE
                and _angle_between(ee_ori, goal_ori) < CARRY_ANGLE_TOLERANCE
            )
            or steps >= CARRY_TIMEOUT
        ):
            next_state = "reach_table_top_xy"
    elif self._state == "reach_table_top_xy":
        # Fine alignment on the leg's own pose, coming down as the error shrinks
        current = leg_pose_for_hole()
        xy_error = (hole[:2, 3] - current[:2, 3]).norm().item()
        height = 0.09 + min(
            ALIGN_HOVER, ALIGN_HOVER_PER_ERROR * max(0.0, xy_error - LEG_POS_TOLERANCE)
        )
        goal = goal_leg_pose(height)
        target = goal @ torch.linalg.inv(current) @ ee_pose
        if (
            xy_error < LEG_POS_TOLERANCE
            and abs((goal_leg_pose(0.09)[2, 3] - current[2, 3]).item()) < 0.005
        ) or steps >= ALIGN_TIMEOUT:
            next_state = "insert_wait"
    elif self._state == "insert_wait":
        # Wait until the leg is actually over the hole before letting go
        current = leg_pose_for_hole()
        goal = goal_leg_pose(0.084)
        target = goal @ torch.linalg.inv(current) @ ee_pose
        error = (goal[:3, 3] - current[:3, 3]).abs().max().item()
        if _settled(self, error < LEG_POS_TOLERANCE) or steps >= INSERT_TIMEOUT:
            self.prev_pose = target
            next_state = "insert_release"
    elif self._state == "insert_release":
        # All the way open: the gripper is still tilted and straightening up swings the fingers past the leg
        target = self.prev_pose
        self.gripper_action = -1
        max_width = config["robot"]["max_gripper_width"]["square_table"] - 0.001
        if gripper_width >= max_width or steps >= GRIP_TIMEOUT:
            next_state = "insert"
    elif self._state == "release":
        target = self.prev_pose
        self.gripper_action = -1
        if gripper_width >= OPEN_CLEAR_LEG or steps >= GRIP_TIMEOUT:
            next_state = "pre_grasp"
    elif self._state == "insert":
        # Dummy transition state for skill complete.
        target = self.prev_pose
        next_state = "pre_grasp"
    elif self._state == "pre_grasp":
        target_pos = leg_pos_robot.clone()
        target_pos[2] += self.grasp_margin_z
        tilt_angle = math.acos(float(np.clip(-ee_ori[2, 2].item(), -1.0, 1.0)))
        if tilt_angle > STRAIGHTEN_ANGLE:
            center = 1 - min(tilt_angle / STRAIGHTEN_CENTER_RANGE, 1.0)
            target_pos = ee_pos + center * (target_pos - ee_pos)
            target = C.to_homogeneous(
                target_pos,
                _rotate_toward(
                    ee_ori,
                    _gripper_down_ori(ee_yaw, device),
                    max_aim=STRAIGHTEN_MAX_AIM,
                ),
            )
        else:
            target = C.to_homogeneous(
                target_pos,
                _gripper_down_ori(_yaw_toward(ee_yaw, SCREW_START_YAW), device),
            )
        at_target = (
            abs(ee_yaw - SCREW_START_YAW) < YAW_TOLERANCE
            and (ee_pos - target_pos).abs().max().item() < LEG_POS_TOLERANCE
        )
        if _settled(self, at_target) or steps >= ROTATE_TIMEOUT:
            self.prev_pose = target
            next_state = "screw_grasp"
    elif self._state == "screw_grasp":
        target = self.prev_pose
        self.gripper_action = 1
        closed = gripper_width <= 2 * self.half_width + 0.002
        if (steps >= GRIP_MIN_STEPS and _settled(self, closed)) or steps >= GRIP_TIMEOUT:
            next_state = "screw"
    elif self._state == "screw":
        # Hold x and y where the turn began: if they drift, the leg drags the tabletop out of the corner.
        # z has no error, so the arm follows the leg down the thread.
        if steps == 0:
            self.screw_xy = ee_pos[:2].clone()
        target_pos = ee_pos.clone()
        target_pos[:2] = self.screw_xy
        target = C.to_homogeneous(
            target_pos,
            _gripper_down_ori(_yaw_toward(ee_yaw, SCREW_END_YAW, SCREW_YAW_STEP), device),
        )
        at_target = abs(ee_yaw - SCREW_END_YAW) < YAW_TOLERANCE
        if _settled(self, at_target) or steps >= ROTATE_TIMEOUT:
            self.prev_pose = target
            next_state = "release"

    skill_complete = self.may_transit_state(next_state)
    return (
        target[:3, 3],
        C.mat2quat(target[:3, :3]),
        torch.tensor([self.gripper_action], device=device),
        skill_complete,
    )


def _table_pre_assemble(
    self,
    original_pre_assemble,
    ee_pos,
    ee_quat,
    gripper_width,
    rb_states,
    part_idxs,
    sim_to_april_mat,
    april_to_robot,
):
    """Replaces TableTop.pre_assemble, apart from its final "done" state.

    The stock push moves the gripper straight at the corner and lets go within 2 cm of it. The tabletop is only held
    by a thin tab, so it pivots when a corner touches a wall, and then stops short at an angle. Instead, control the
    tabletop's own pose: square it up clear of the walls, press it into the corner, then let go.
    """
    if self._state == "done":
        return original_pre_assemble(
            ee_pos,
            ee_quat,
            gripper_width,
            rb_states,
            part_idxs,
            sim_to_april_mat,
            april_to_robot,
        )

    device = ee_pos.device
    steps = _steps_in_state(self)
    next_state = self._state

    ee_pose = C.to_homogeneous(ee_pos, C.quat2mat(ee_quat))
    ee_ori = ee_pose[:3, :3]
    sim_to_robot = april_to_robot @ sim_to_april_mat

    def body_pose(name):
        return sim_to_robot @ C.to_homogeneous(
            rb_states[part_idxs[name]][0][:3], C.quat2mat(rb_states[part_idxs[name]][0][3:7])
        )

    if self._state == "reach_body_grasp_xy":
        if steps == 0:
            self._v2_noise = torch.normal(torch.zeros(2, device=device), GRASP_POS_NOISE)
            self._v2_yaw_noise = float(np.random.normal(0, GRASP_YAW_NOISE))
        table_april = sim_to_april_mat @ C.to_homogeneous(
            rb_states[part_idxs[self.name]][0][:3],
            C.quat2mat(rb_states[part_idxs[self.name]][0][3:7]),
        )
        edge = self._find_closest_y(table_april)
        rot = edge @ torch.tensor(rot_mat([np.pi / 2, 0, 0], hom=True), device=device).float()
        goal_pos = (april_to_robot @ edge)[:3, 3].clone()
        goal_pos[:2] += self._v2_noise
        goal_pos[2] = (april_to_robot @ table_april)[2, 3]
        goal_ori = _rot_z(self._v2_yaw_noise, device) @ (april_to_robot @ rot)[:3, :3]
        target, arrived = _approach(ee_pos, ee_ori, goal_pos, goal_ori)
        if _settled(self, arrived) or steps >= APPROACH_TIMEOUT:
            self.prev_pose = C.to_homogeneous(goal_pos, goal_ori)
            next_state = "pick_body"
    elif self._state == "pick_body":
        target = self.prev_pose
        self.gripper_action = 1
        # The tab is thin and the fingers close at a fixed rate, so wait them out rather than watch the width
        if steps >= GRIP_TIMEOUT - 3:
            next_state = "push"
    elif self._state == "release":
        target = self.prev_pose
        self.gripper_action = -1
        if gripper_width >= OPEN_CLEAR_TAB or steps >= GRIP_TIMEOUT:
            next_state = "done"
    else:
        table_pose = body_pose(self.name)
        # The tabletop is square, so only its yaw modulo 90 degrees matters
        yaw_error = (_yaw_of(table_pose) + math.pi / 4) % (math.pi / 2) - math.pi / 4
        corner = torch.stack(
            [
                body_pose("obstacle_front")[0, 3] - WALL_HALF_THICKNESS - self.half_width,
                body_pose("obstacle_left")[1, 3] - WALL_HALF_THICKNESS - self.half_width,
            ]
        )
        table_xy = table_pose[:2, 3]

        def target_for(goal_xy):
            """Gripper pose that would put the tabletop at goal_xy, squared up, if it moves rigidly with the gripper."""
            rot = _rot_z(-yaw_error, device)
            rel = torch.eye(4, device=device)
            rel[:3, :3] = rot
            rel[:2, 3] = goal_xy - rot[:2, :2] @ table_xy
            target = rel @ ee_pose
            target[2, 3] = self.prev_pose[2, 3]  # Keep the grasp height
            return target

        if self._state == "push":
            goal_xy = corner - PUSH_CLEARANCE
            target = target_for(goal_xy)
            # No need to stop here: it's only a point to pass through on the way into the corner
            if (
                (goal_xy - table_xy).abs().max().item() < 2 * PUSH_POS_TOLERANCE
                and abs(yaw_error) < 2 * PUSH_YAW_TOLERANCE
            ) or steps >= PUSH_TIMEOUT:
                next_state = "push_press"
        elif self._state == "push_press":
            target = target_for(corner + PUSH_OVERSHOOT)
            in_corner = (
                (corner - table_xy).abs().max().item() < PUSH_POS_TOLERANCE
                and abs(yaw_error) < PUSH_YAW_TOLERANCE
            )
            # Stop pressing once the walls stop it, or the tabletop just slides in the fingers
            if (
                self._v2_stall_xy is None
                or (table_xy - self._v2_stall_xy).abs().max().item() > PUSH_STALL_DIST
            ):
                self._v2_stall_xy = table_xy.clone()
                self._v2_stall_steps = 0
            else:
                self._v2_stall_steps += 1
            if (
                in_corner
                or self._v2_stall_steps >= PUSH_STALL_STEPS
                or steps >= PUSH_TIMEOUT
            ):
                self.prev_pose = ee_pose.clone()
                next_state = "push_hold"
        elif self._state == "push_hold":
            # Let the arm stop before the gripper opens
            target = self.prev_pose
            if _settled(self, True):
                next_state = "release"

    skill_complete = self.may_transit_state(next_state)
    return (
        target[:3, 3],
        C.mat2quat(target[:3, :3]),
        torch.tensor([self.gripper_action], device=device),
        skill_complete,
    )


def patch_expert(furniture):
    """Swap in the modified FSMs on one_leg's parts. Other furniture keeps the stock script."""
    for part in furniture.parts:
        if isinstance(part, SquareTableLeg):
            part.fsm_step = types.MethodType(_leg_fsm_step, part)
            # The stock script jitters the action during these states, which also shakes the leg as it is let go
            part.state_no_noise = lambda: False
        elif isinstance(part, SquareTableTop):

            def pre_assemble(self, *args, _original=part.pre_assemble):
                return _table_pre_assemble(self, _original, *args)

            part.pre_assemble = types.MethodType(pre_assemble, part)
            # Leaving the tabletop no longer has its own go_up state
            part.skill_complete_next_states = ["push", "done"]


class DataCollector:
    """Demonstration collection class.

    Only used for collecting scripted demos in the simulator

    For teleoperation demos, use `DataCollectorSM` instead.

    For now, only stores pickle files, either compressed or uncompressed.
    """

    def __init__(
        self,
        is_sim: bool,
        data_path: str,
        device_interface: DeviceInterface,
        furniture: str,
        headless: bool,
        draw_marker: bool,
        manual_label: bool,
        scripted: bool,
        randomness: Randomness.LOW,
        compute_device_id: int,
        graphics_device_id: int,
        save_failure: bool = False,
        num_demos: int = 100,
        resize_sim_img: bool = False,
        ctrl_mode: str = "osc",
        compress_pickles: bool = False,
        save_mp4: bool = False,
        verbose: bool = False,
        max_env_steps: int = None,
        return_home: bool = True,
    ):
        """
        Args:
            is_sim (bool): Whether to use simulator or real world environment.
            data_path (str): Path to save data.
            device_interface (DeviceInterface): Keyboard and/or Oculus interface.
            furniture (str): Name of the furniture.
            headless (bool): Whether to use headless mode.
            draw_marker (bool): Whether to draw AprilTag marker.
            manual_label (bool): Whether to manually label the reward.
            scripted (bool): Whether to use scripted function for getting action.
            randomness (str): Initialization randomness level.
            compute_device_id (int): GPU device ID used for simulation.
            graphics_device_id (int): GPU device ID used for rendering.
            pkl_only (bool): Whether to save only `pkl` files (i.e., exclude *.mp4 and *.png).
            save_failure (bool): Whether to save failure trajectories.
            num_demos (int): The maximum number of demonstrations to collect in this run. Internal loop will be terminated when this number is reached.
            resize_sim_img (bool): Read resized image
            ctrl_mode (str): 'osc' (joint torque, with operation space control) or 'diffik' (joint impedance, with differential inverse kinematics control)
            compress_pickles (bool): Whether to compress the pickle files with gzip.
            save_mp4 (bool): Also save an mp4 of both cameras (wrist | front) next to each pickle.
            max_env_steps (int): Steps allowed to finish the assembly. Defaults to sim_config["scripted_timeout"].
            return_home (bool): After a scripted success, keep recording while the arm returns to a centered pose.
        """
        if is_sim:
            self.env = gym.make(
                "FurnitureSimFull-v0",
                furniture=furniture,
                max_env_steps=max_env_steps
                or (sim_config["scripted_timeout"][furniture] if scripted else 3000),
                headless=headless,
                num_envs=1,  # Only support 1 for now.
                manual_done=False if scripted else True,
                resize_img=resize_sim_img,
                np_step_out=False,  # Always output Tensor in this setting. Will change to numpy in this code.
                channel_first=False,
                randomness=randomness,
                compute_device_id=compute_device_id,
                graphics_device_id=graphics_device_id,
                ctrl_mode=ctrl_mode,
            )
        else:
            if randomness == "med":
                randomness = Randomness.MEDIUM_COLLECT
            elif randomness == "high":
                randomness = Randomness.HIGH_COLLECT

            self.env = gym.make(
                "FurnitureBench-v0",
                furniture=furniture,
                resize_img=False,
                manual_done=True,
                with_display=not headless,
                draw_marker=draw_marker,
                randomness=randomness,
            )

        self.is_sim = is_sim
        self.data_path = Path(data_path)
        self.device_interface = device_interface
        self.headless = headless
        self.manual_label = manual_label
        self.furniture = furniture
        self.num_demos = num_demos
        self.scripted = scripted

        self.traj_counter = 0
        self.num_success = 0
        self.num_fail = 0

        self.save_failure = save_failure
        self.resize_sim_img = resize_sim_img
        self.compress_pickles = compress_pickles
        self.save_mp4 = save_mp4
        self.return_home = return_home
        self.home_phase = None

        if scripted and is_sim:
            patch_expert(self.env.furniture)
        self.verbose = verbose

        self._reset_collector_buffer()

    def _append_obstacle_pose(self, obs):
        """Append the U-fixture (obstacle_front) pose to parts_poses, in place.

        The RL env's parts_poses is the parts followed by the fixture, all in the
        AprilTag frame (42-D for one_leg), while FurnitureSimEnv returns only the parts.
        """
        if not self.is_sim or "parts_poses" not in obs:
            return

        rb_idx = self.env.part_idxs["obstacle_front"][0]
        pose = self.env.rb_states[rb_idx, :7]
        pos, quat = C.mat2pose(
            self.env.sim_coord_to_april_coord(
                C.pose2mat(pose[:3], pose[3:7], device=self.env.device)
            )
        )
        obstacle_pose = torch.cat([pos, quat]).to(obs["parts_poses"].device)
        obs["parts_poses"] = torch.cat(
            [obs["parts_poses"], obstacle_pose.view(1, -1)], dim=-1
        )

    def _home_action(self):
        """Next action of the return to the centered pose. Sets self.home_phase to None when it is finished."""
        ee_pos, ee_quat = self.env.get_ee_pose()
        ee_pos, ee_quat = ee_pos.squeeze(), ee_quat.squeeze()
        device = ee_pos.device
        ee_ori = C.quat2mat(ee_quat).to(device)
        self.home_steps += 1

        goal_pos, goal_ori = ee_pos.clone(), ee_ori
        gripper = -1.0
        if self.home_phase == "settle":
            # Success usually lands mid-turn, so stop the arm before opening the gripper
            gripper = 1.0
            if self.home_steps >= SETTLE_STEPS:
                self.home_phase, self.home_steps = "release", 0
        elif self.home_phase == "release":
            if self.env.gripper_width() >= OPEN_CLEAR_LEG or self.home_steps >= GRIP_TIMEOUT:
                self.home_phase, self.home_steps = "move", 0
                self.home_clear_z = ee_pos[2].item() + HOME_LEG_CLEARANCE
        else:
            # One sweep: rise and turn at once, and fade in the sideways motion as the fingers clear the leg
            home_pos = torch.tensor(HOME_POS, device=device, dtype=torch.float32)
            home_ori = _gripper_down_ori(HOME_YAW, device)
            sideways = _smoothstep((ee_pos[2].item() - self.home_clear_z) / HOME_CLEAR_BAND)
            goal_pos[:2] += sideways * (home_pos[:2] - ee_pos[:2])
            goal_pos[2] = home_pos[2]
            goal_ori = _rotate_toward(ee_ori, home_ori)
            at_home = (
                (home_pos - ee_pos).norm().item() < HOME_POS_TOLERANCE
                and _angle_between(ee_ori, home_ori) < HOME_ANGLE_TOLERANCE
            )
            self.home_settled = self.home_settled + 1 if at_home else 0
            if self.home_settled >= SETTLE_STEPS or self.home_steps >= HOME_TIMEOUT:
                self.home_phase = None

        action = torch.cat(
            [
                (goal_pos - ee_pos) * 2,  # Same gain as get_assembly_action
                C.quat_mul(C.quat_conjugate(ee_quat), C.mat2quat(goal_ori).to(device)),
                torch.tensor([gripper], device=device),
            ]
        )
        return action.unsqueeze(0)

    def _verbose_print(self, msg):
        if self.verbose:
            print(msg)

    def collect(self):
        print("[data collection] Start collecting the data!")

        obs = self.reset()
        done = False

        while self.num_success < self.num_demos:
            # The assembly just succeeded: keep recording while the arm returns to the centered pose
            if (
                self.scripted
                and self.return_home
                and done
                and self.home_phase is None
                and not self.home_done
                and self.env.furnitures[0].all_assembled()
            ):
                self.home_phase, self.home_steps, self.home_settled = "settle", 0, 0
                done = False

            # Get an action.
            if self.scripted:
                if self.home_phase is not None:
                    action, skill_complete = self._home_action(), 0
                    if self.home_phase is None:
                        self.home_done = True
                        done = True
                else:
                    action, skill_complete = self.env.get_assembly_action()
                pos_bounds_m = POS_BOUND_M
                ori_bounds_deg = ORI_BOUND_DEG if self.env.ctrl_mode == "diffik" else 20
                action = scale_scripted_action(
                    action.detach().cpu().clone(),
                    pos_bounds_m=pos_bounds_m,
                    ori_bounds_deg=ori_bounds_deg,
                    device=self.env.device,
                )
                collect_enum = CollectEnum.DONE_FALSE
            else:
                action, collect_enum = self.device_interface.get_action()
                skill_complete = int(collect_enum == CollectEnum.SKILL)
                if skill_complete == 1:
                    self.skill_set.append(skill_complete)

            if collect_enum == CollectEnum.TERMINATE:
                print("Terminate the program.")
                break

            # An episode is done.
            if done or collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                if self.is_sim:
                    # Convert it to numpy.
                    for k, v in next_obs.items():
                        if isinstance(v, dict):
                            for k1, v1 in v.items():
                                v[k1] = v1.squeeze().cpu().numpy()
                        elif k == "color_image1":
                            next_obs[k] = resize(next_obs[k]).squeeze().cpu().numpy()
                        elif k == "color_image2":
                            next_obs[k] = (
                                resize_crop(next_obs[k]).squeeze().cpu().numpy()
                            )
                        else:
                            next_obs[k] = v.squeeze().cpu().numpy()

                self.org_obs.append(next_obs)

                n_ob = {}
                n_ob["color_image1"] = next_obs["color_image1"]
                n_ob["color_image2"] = next_obs["color_image2"]
                n_ob["robot_state"] = next_obs["robot_state"]
                n_ob["parts_poses"] = next_obs["parts_poses"]
                self.obs.append(n_ob)

                if done and not self.env.furnitures[0].all_assembled():
                    if self.save_failure:
                        print("Saving failure trajectory.")
                        collect_enum = CollectEnum.FAIL
                        obs = self.save_and_reset(collect_enum, {})
                    else:
                        print("Failed to assemble the furniture, reset without saving.")
                        obs = self.reset()
                        collect_enum = CollectEnum.SUCCESS
                    self.num_fail += 1
                else:
                    if done:
                        collect_enum = CollectEnum.SUCCESS

                    obs = self.save_and_reset(collect_enum, {})
                    self.num_success += 1
                self.traj_counter += 1
                print(f"Success: {self.num_success}, Fail: {self.num_fail}")
                done = False
                continue

            # Execute action.
            next_obs, rew, done, info = self.env.step(action)
            self._append_obstacle_pose(next_obs)

            # The env reports done from the moment of success on, which doesn't end the return
            if self.home_phase is not None:
                done = torch.zeros_like(done)

            if rew == 1:
                self.last_reward_idx = len(self.acts)

            # Label reward.
            if collect_enum == CollectEnum.REWARD:
                rew = self.env.furniture.manual_assemble_label(
                    self.device_interface.rew_key
                )
                if rew == 0:
                    # Correction the label.
                    self.rews[self.last_reward_idx] = 0
                    rew = 1

            # Error handling.
            if not info["obs_success"]:
                print("Getting observation failed, save trajectory.")
                # Pop the last reward and action so that obs has length plus 1 then those of actions and rewards.
                self.rews.pop()
                self.acts.pop()
                obs = self.save_and_reset(CollectEnum.FAIL, info)
                continue

            # Logging a step.
            self.step_counter += 1
            self._verbose_print(
                f"{[self.step_counter]} assembled: {self.env.furniture.assembled_set} num assembled: {len(self.env.furniture.assembled_set)} Skill: {len(self.skill_set)}"
            )

            # Store a transition.
            if info["action_success"]:
                if self.is_sim:
                    for k, v in obs.items():
                        if isinstance(v, dict):
                            for k1, v1 in v.items():
                                v[k1] = v1.squeeze().cpu().numpy()
                        elif k == "color_image1":
                            obs[k] = resize(obs[k]).squeeze().cpu().numpy()
                        elif k == "color_image2":
                            obs[k] = resize_crop(obs[k]).squeeze().cpu().numpy()
                        else:
                            obs[k] = v.squeeze().cpu().numpy()
                    if isinstance(rew, torch.Tensor):
                        rew = float(rew.squeeze().cpu())

                self.org_obs.append(obs.copy())
                ob = {}
                # if (not self.is_sim) or (not self.resize_sim_img):
                #     # Resize for every real world images, or for sim didn't resize in simulation side.
                #     ob["color_image1"] = resize(obs["color_image1"])
                #     ob["color_image2"] = resize_crop(obs["color_image2"])
                # else:
                #     ob["color_image1"] = obs["color_image1"]
                #     ob["color_image2"] = obs["color_image2"]
                ob["color_image1"] = obs["color_image1"]
                ob["color_image2"] = obs["color_image2"]
                ob["robot_state"] = obs["robot_state"]
                ob["parts_poses"] = obs["parts_poses"]
                self.obs.append(ob)

                if self.is_sim:
                    if isinstance(action, torch.Tensor):
                        action = action.squeeze().cpu().numpy()
                    else:
                        action = action.squeeze()
                self.acts.append(action)
                self.rews.append(rew)
                self.skills.append(skill_complete)
            obs = next_obs

        print(
            f"Collected {self.traj_counter} / {self.num_demos} successful trajectories!"
        )

    def save_and_reset(self, collect_enum: CollectEnum, info):
        """Saves the collected data and reset the environment."""
        self.save(collect_enum, info)
        print(f"Saved {self.traj_counter} trajectories in this run.")
        return self.reset()

    def reset(self):
        obs = self.env.reset()
        self._append_obstacle_pose(obs)
        self._reset_collector_buffer()
        self.home_phase = None
        self.home_done = False

        print("Start collecting the data!")
        if not self.scripted:
            print("Press enter to start")
            while True:
                if input() == "":
                    break
            time.sleep(0.2)

        return obs

    def _reset_collector_buffer(self):
        self.obs = []
        self.org_obs = []
        self.acts = []
        self.rews = []
        self.skills = []
        self.step_counter = 0
        self.last_reward_idx = -1
        self.skill_set = []

    def save(self, collect_enum: CollectEnum, info):
        print(f"Length of trajectory: {len(self.obs)}")

        # Save transitions with resized images.
        data = {}
        data["observations"] = self.obs
        data["actions"] = self.acts
        data["rewards"] = self.rews
        data["skills"] = self.skills
        data["success"] = True if collect_enum == CollectEnum.SUCCESS else False
        data["furniture"] = self.furniture

        if "error" in info:
            data["error_description"] = info["error"].value
            data["error"] = True
        else:
            data["error"] = False
            data["error_description"] = ""

        # Save data.
        demo_path = self.data_path / ("success" if data["success"] else "failure")
        demo_path.mkdir(parents=True, exist_ok=True)

        path = demo_path / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"

        if self.compress_pickles:
            # Add the suffix .gz if we are compressing the pickle files
            path = path.with_suffix(".pkl.xz")

        pickle_data(data, path)

        if self.save_mp4:
            mp4_path = path.with_name(path.name.split(".")[0] + ".mp4")
            create_mp4(data_to_video(data), mp4_path)
            print(f"Video saved at {mp4_path}")

        print(f"Data saved at {path}")

    def __del__(self):
        del self.env

        if self.device_interface is not None:
            self.device_interface.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomness", "-r", type=str, default="low")
    parser.add_argument("--num-demos", "-n", type=int, default=100)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--furniture", "-f", type=str, required=True)
    parser.add_argument("--save-failure", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--ctrl-mode", "-c", type=str, default="diffik", choices=["diffik", "osc"]
    )
    parser.add_argument(
        "--keep-april-tags",
        action="store_true",
        help="Keep AprilTags on the part meshes (ResiP renders and trains without them)",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Also save an mp4 (wrist | front camera) next to each pickle",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=1000,
        help="Steps allowed to finish the assembly",
    )
    parser.add_argument(
        "--no-return-home",
        action="store_true",
        help="End the episode at success instead of returning to the centered pose",
    )
    args = parser.parse_args()

    # Load the tagless part meshes, like the RL env and the rest of the pipeline
    if not args.keep_april_tags:
        turn_off_april_tags()

    data_path = trajectory_save_dir(
        controller=args.ctrl_mode,
        domain="sim",
        task=args.furniture,
        demo_source="scripted",
        randomness=args.randomness,
        suffix=args.suffix,
    )
    print(f"Saving data to directory: {data_path}")

    collector = DataCollector(
        is_sim=True,
        data_path=data_path,
        furniture=args.furniture,
        device_interface=None,
        headless=args.headless,
        manual_label=False,
        scripted=True,
        draw_marker=False,
        randomness=args.randomness,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        resize_sim_img=False,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        ctrl_mode=args.ctrl_mode,
        compress_pickles=True,
        save_mp4=args.save_mp4,
        max_env_steps=args.max_env_steps,
        return_home=not args.no_return_home,
    )
    collector.collect()
