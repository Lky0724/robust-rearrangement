"""
FR3 version of teleop_sm.py (the original is left unchanged).

Changes from teleop_sm.py, see docs/FR3_ORIGINAL_CODE_TELEOP_TEST.md §8:
- No furniture_bench import (it was unused).
- Polymetis server address is an option (--server_ip, default localhost), not the authors' IP.
- DiffIKWrapper is built with the arguments the rdt lars-dev constructor requires.
- --no_cameras: no RealSense cameras; black 640x480 placeholder images are saved instead.
- --no_gripper: no gripper server; a stand-in gripper reports the commanded width.
- The ` key (CollectEnum.SKILL) marks the insertion start: the next recorded step gets 1
  in the pickle's `skills`, every other step 0. segment_demos.py cuts phases at the first mark.
- --show_cameras: live wrist | front window built from the frames the loop already captured
  (no extra camera reads, recorded data unchanged).
- Loop timing (work per step, display, late steps) is printed at the end of every run.
- --twist_deadzone: while twisting the SpaceMouse cap (rotation about z, for screwing), the
  other five axes get this larger deadzone, so the leg stays vertical.
- --max_twist_speed: speed of the twist (rotation about z) on its own; tilts keep --max_rot_speed.
- --interp_hz (off by default): spreads each 10 Hz joint target over the step instead of jumping
  to it, which removes most of the stop-and-go. Decisions and recording stay at 10 Hz.
- Camera exposure, gain and white balance come from src/real/camera_settings.py (fixed values
  instead of auto exposure, which clips the white parts); the applied values are saved in each
  episode's metadata["camera_settings"].
"""

from pathlib import Path
import pickle
import time
from multiprocessing.managers import SharedMemoryManager
import torch
import numpy as np
from datetime import datetime
import meshcat
import scipy.spatial.transform as st
import pyrealsense2 as rs

from polymetis import GripperInterface

from rdt.spacemouse.spacemouse_shared_memory import Spacemouse

from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.interfaces.diffik import DiffIKWrapper
from rdt.common import mc_util
from rdt.common.keyboard_interface import KeyboardInterface
from rdt.common.demo_util import CollectEnum
from rdt.image.factory import enable_single_realsense

# from furniture_bench.utils.scripted_demo_mod import scale_scripted_action  # unused

from src.real.serials import FRONT_CAMERA_SERIAL, WRIST_CAMERA_SERIAL
from src.real.camera_settings import COLOR_SETTINGS, apply_color_settings

from ipdb import set_trace as bp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port_vis", type=int, default=6000)
parser.add_argument("--frequency", type=int, default=10)  # 30
parser.add_argument("--command_latency", type=float, default=0.01)
parser.add_argument("--deadzone", type=float, default=0.15)
parser.add_argument("--max_pos_speed", type=float, default=0.3)
parser.add_argument("--max_rot_speed", type=float, default=0.5)
parser.add_argument("--use_lcm", action="store_true")
parser.add_argument("--save_dir", required=True)
parser.add_argument("--furniture", type=str, required=True)
parser.add_argument("--server_ip", type=str, default="localhost")  # Polymetis server
parser.add_argument("--no_cameras", action="store_true")
parser.add_argument("--no_gripper", action="store_true")
parser.add_argument("--show_cameras", action="store_true")  # live wrist | front window
# Deadzone for the other five axes while twisting (0 = off)
parser.add_argument("--twist_deadzone", type=float, default=0.3)
# Twist (rotation about z) speed, same units as --max_rot_speed; default: --max_rot_speed
parser.add_argument("--max_twist_speed", type=float, default=0.8)
# Joint target interpolation rate within each 10 Hz step, e.g. 50 (0 = off: one jump per step)
parser.add_argument("--interp_hz", type=float, default=0.0)

args = parser.parse_args()

poly_util = PolymetisHelper()


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


def convert_reference_frame_mat(
    pose_source_mat, pose_frame_target_mat, pose_frame_source_mat
):

    # transform that maps from target to source (S = XT)
    target2source_mat = np.matmul(
        pose_frame_source_mat, np.linalg.inv(pose_frame_target_mat)
    )

    # obtain source pose in target frame
    pose_source_in_target_mat = np.matmul(target2source_mat, pose_source_mat)
    return pose_source_in_target_mat


def convert_tip2wrist(tip_pose_mat):

    tip2wrist_tf_mat = np.eye(4)
    tip2wrist_tf_mat[:-1, -1] = np.array([0.0, 0.0, -0.1034])
    tip2wrist_tf_mat[:-1, :-1] = st.Rotation.from_quat(
        [0.0, 0.0, 0.3826834323650898, 0.9238795325112867]
    ).as_matrix()

    wrist_pose_mat = convert_reference_frame_mat(
        pose_source_mat=tip2wrist_tf_mat,
        pose_frame_target_mat=np.eye(4),
        pose_frame_source_mat=tip_pose_mat,
    )

    return wrist_pose_mat


def convert_wrist2tip(wrist_pose_mat):

    wrist2tip_tf_mat = np.eye(4)
    wrist2tip_tf_mat[:-1, -1] = np.array([0.0, 0.0, 0.1034])
    wrist2tip_tf_mat[:-1, :-1] = st.Rotation.from_quat(
        [0.0, 0.0, -0.3826834323650898, 0.9238795325112867]
    ).as_matrix()

    tip_pose_mat = convert_reference_frame_mat(
        pose_source_mat=wrist2tip_tf_mat,
        pose_frame_target_mat=np.eye(4),
        pose_frame_source_mat=wrist_pose_mat,
    )

    return tip_pose_mat


class NoGripper:
    """Stand-in for polymetis.GripperInterface when no gripper is mounted (--no_gripper).

    Commands return immediately; get_state().width reports the commanded width, with grasp()
    treated as closing fully (nothing to grasp), so recorded episodes keep their format.
    """

    class _State:
        def __init__(self, width):
            self.width = width

    def __init__(self, width=0.065):
        self._width = width

    def goto(self, width, speed, force, blocking=True):
        self._width = width

    def grasp(self, speed, force, blocking=True):
        self._width = 0.0

    def get_state(self):
        return self._State(self._width)


def apply_twist_deadzone(sm_state: np.ndarray, deadzone: float) -> np.ndarray:
    """While twisting (rotation about z, index 5), zero the other axes below `deadzone`, so
    accidental pushes and tilts don't move or tilt the leg while screwing. Twisting counts when
    the twist is at least `deadzone` or the strongest input, so a slight accidental twist during
    a normal move changes nothing. A deliberate push above `deadzone` still goes through.
    Values are in [-1, 1] after the SpaceMouse's own deadzone."""
    s = np.array(sm_state, dtype=float)
    twist = abs(s[5])
    if deadzone > 0 and twist > 0 and (twist >= deadzone or twist >= np.abs(s[:5]).max()):
        s[:5][np.abs(s[:5]) < deadzone] = 0.0
    return s


class CameraView:
    """Live wrist | front window for --show_cameras.

    Shows the frames get_observation() already captured, so there are no extra camera reads,
    and nothing it does changes the recorded data. Each camera is shown at 320x240 as the
    policy data has it: the wrist frame resized whole (process_pickles' resize), the front frame
    cut to its central 4:3 part first (resize_crop). The yellow box on the front view is the
    224x224 center crop the policy sees at deployment. Any display error turns the window off
    instead of ending the episode.
    """

    WIN = "teleop cameras: wrist | front"

    def __init__(self):
        self.enabled = False
        try:
            import os
            import cv2  # only needed with --show_cameras

            # opencv-python points Qt at a font folder it doesn't ship, which floods the
            # terminal with warnings; use the system's DejaVu fonts when they are there
            if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
                os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"
            self.cv2 = cv2
            cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)  # can be resized with the mouse
            cv2.resizeWindow(self.WIN, 960, 393)
            self.enabled = True
        except Exception as e:
            print(f"--show_cameras: can't open a window, continuing without it ({e})")

    def _fit(self, img, crop_43):
        h, w = img.shape[:2]
        if crop_43 and w * 3 > h * 4:  # keep the central 4:3 part, as resize_crop does
            w43 = h * 4 // 3
            x0 = (w - w43) // 2
            img = img[:, x0 : x0 + w43]
        return self.cv2.resize(img, (320, 240), interpolation=self.cv2.INTER_AREA)

    def show(self, obs, n_recorded, recorded, gripper_open, mark_text, loop_ms):
        if not self.enabled:
            return
        cv2 = self.cv2
        try:
            wrist = self._fit(obs["color_image1"], crop_43=False)
            front = self._fit(obs["color_image2"], crop_43=True)
            frame = cv2.cvtColor(np.hstack([wrist, front]), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (320 + 48, 8), (320 + 271, 231), (0, 255, 255), 1)
            frame = np.vstack([frame, np.zeros((22, 640, 3), dtype=np.uint8)])

            width_cm = 100 * float(np.ravel(obs["robot_state"]["gripper_width"])[0])
            items = [
                (f"REC {n_recorded}" if recorded else f"idle {n_recorded}",
                 (0, 0, 255) if recorded else (160, 160, 160)),
                (f"grip {'open' if gripper_open else 'closed'} {width_cm:.1f}cm", (255, 255, 255)),
                (mark_text, (0, 255, 0) if mark_text.startswith("mark") else (160, 160, 160)),
                (f"loop {loop_ms:.0f}ms", (0, 165, 255) if loop_ms > 50 else (255, 255, 255)),
            ]
            x = 6
            for text, color in items:
                cv2.putText(frame, text, (x, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
                            cv2.LINE_AA)
                x += 160
            cv2.imshow(self.WIN, frame)
            # Lets the window refresh (~6 ms, most of the cost). Keys still reach the global
            # keyboard listener while the window has focus.
            cv2.pollKey()
        except Exception as e:
            print(f"--show_cameras: display error, turning the window off ({e})")
            self.enabled = False

    def close(self):
        if self.enabled:
            self.cv2.destroyAllWindows()


def diffik_joint_target(robot, ee_pose_mat, dt):
    """Joint target for a flange pose target: the same computation as rdt's
    DiffIKWrapper.update_desired_ee_pose (scalar 1), without sending it to the robot."""
    q = robot.get_joint_positions()
    jacobian = robot.robot_model.compute_jacobian(q)
    ee_velocity_desired = robot.compute_ee_vel_desired(ee_pose_mat, dt_pos=dt, dt_rot=dt)
    joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired).solution
    return q + joint_vel_desired * dt


class JointTargetInterpolator:
    """--interp_hz: instead of jumping to each step's joint target, send a ramp of targets.

    The ramp starts from the last target sent (so the target never jumps) and reaches the new
    one after SPAN x dt, i.e. before the next step's camera read; the sub-targets go out while
    the loop would otherwise sleep. If the next step's target arrives first, the new ramp
    starts wherever this one got to.
    """

    SPAN = 0.7  # fraction of the step the ramp takes

    def __init__(self, robot, hz, dt):
        self.robot = robot
        self.period = 1.0 / hz
        self.n = max(1, round(self.SPAN * dt / self.period))  # sub-targets per step
        self.q_last = robot.get_joint_positions().clone()  # the impedance controller holds here
        self.q_from = self.q_to = self.q_last
        self.k = self.n
        self.next_t = None
        self.send_ms = []
        self.completed = []  # per step: did the ramp reach its target before the next one?
        self.started = False

    def set_target(self, q_new):
        if self.started:
            self.completed.append(self.k >= self.n)
        self.started = True
        self.q_from, self.q_to = self.q_last, q_new
        self.k = 0
        self.next_t = time.monotonic()
        self._send_due(self.next_t)  # first sub-target right away

    def _send_due(self, t_now):
        while self.next_t is not None and self.next_t <= t_now and self.k < self.n:
            self.k += 1
            q = self.q_from + (self.q_to - self.q_from) * (self.k / self.n)
            a = time.monotonic()
            self.robot.update_desired_joint_positions(q)
            self.send_ms.append(1000 * (time.monotonic() - a))
            self.q_last = q
            self.next_t = self.next_t + self.period if self.k < self.n else None

    def wait(self, t_end):
        """Wait until t_end, sending the sub-targets that fall due meanwhile."""
        while self.next_t is not None and self.next_t < t_end:
            precise_wait(self.next_t)
            self._send_due(time.monotonic())
        precise_wait(t_end)

    def summary(self):
        if not self.send_ms:
            return
        ms = np.array(self.send_ms)
        done = np.mean(self.completed) * 100 if self.completed else 100.0
        print(
            f"  interpolation: {self.n} sub-targets per step, {len(ms)} sent, "
            f"send mean {ms.mean():.1f} ms max {ms.max():.1f} ms, ramps completed {done:.0f}%"
        )


def print_loop_timing(work_ms, display_ms, late_ms, dt):
    """Summary of the 10 Hz loop: time spent per step and how late steps started."""
    if not work_ms:
        return
    work, late = np.array(work_ms), np.array(late_ms)
    print(
        f"Loop timing over {len(work)} steps: work mean {work.mean():.1f} ms, "
        f"p95 {np.percentile(work, 95):.1f} ms, max {work.max():.1f} ms (budget {1000 * dt:.0f} ms)"
    )
    if display_ms:
        disp = np.array(display_ms)
        print(f"  camera window: mean {disp.mean():.1f} ms, max {disp.max():.1f} ms")
    print(
        f"  steps that started more than 5 ms late: {(late > 5).sum()} "
        f"(max {late.max():.1f} ms late)"
    )


class ActionContainer:
    def __init__(
        self,
        current_pose_mat: np.ndarray,
        next_pose_mat: np.ndarray,
        grasp_flag: int,
        action_taken: bool,
        collect_enum: CollectEnum,
        is_gripper_open: bool,
        toggle_gripper: bool,
    ):
        self.current_pose_mat = current_pose_mat
        self.next_pose_mat = next_pose_mat
        self.grasp_flag = grasp_flag
        self.action_taken = action_taken
        self.collect_enum = collect_enum
        self.is_gripper_open = is_gripper_open
        self.toggle_gripper = toggle_gripper


# Setup observation and action helpers
class ObsActHelper:
    def __init__(
        self, sm, keyboard, robot, gripper, front_image_pipeline, wrist_image_pipeline
    ):

        self.sm = sm
        self.keyboard = keyboard

        self.robot = robot
        self.gripper = gripper

        self.front_image_pipeline = front_image_pipeline
        self.wrist_image_pipeline = wrist_image_pipeline

        self._setup()

    def set_constants(
        self,
        max_pos_speed: float,
        max_rot_speed: float,
        sm_dpos_scalar: float,
        sm_drot_scalar: float,
        frequency: float,
    ):
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.sm_dpos_scalar = sm_dpos_scalar
        self.sm_drot_scalar = sm_drot_scalar
        self.frequency = frequency

    def _setup(self):
        self.grasp_flag = -1
        self.gripper_open = True
        self.last_grip_step = 0
        self.steps_since_grasp = 0
        self.record_latency_when_grasping = 15

    @staticmethod
    def to_isaac_dpose_from_abs(current_pose_mat, goal_pose_mat, grasp_flag, rm=True):
        """
        Convert from absolute current and desired pose to delta pose

        Args:
            rm (bool): 'rm' stands for 'right multiplication' - If True, assume commands send as right multiply (local rotations)
        """
        if rm:
            delta_rot_mat = (
                np.linalg.inv(current_pose_mat[:-1, :-1]) @ goal_pose_mat[:-1, :-1]
            )
        else:
            delta_rot_mat = goal_pose_mat[:-1:-1] @ np.linalg.inv(
                current_pose_mat[:-1, :-1]
            )

        target_translation = goal_pose_mat[:-1, -1] - current_pose_mat[:-1, -1]
        target_quat_xyzw = st.Rotation.from_matrix(delta_rot_mat).as_quat()

        target_dpose = np.concatenate(
            (target_translation, target_quat_xyzw, np.array([grasp_flag])), axis=-1
        )

        return target_dpose

    @staticmethod
    def to_pose_mat(pose_):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = pose_[:3]
        pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_[3:]).as_matrix()
        return pose_mat

    def set_target_pose(self, target_pose: np.ndarray):
        # [x, y, z, dx, dy, dz] (rotvec!)
        self.target_pose = target_pose

    def get_action(self):
        # get teleop command
        sm_state = self.sm.get_motion_state_transformed()
        sm_state = apply_twist_deadzone(sm_state, args.twist_deadzone)

        # scale pos command
        dpos = (
            sm_state[:3] * (self.max_pos_speed / self.frequency) * self.sm_dpos_scalar
        )

        # convert and scale rot command
        # Tilts (about x, y) at --max_rot_speed, the twist (about z) at --max_twist_speed
        twist_speed = args.max_twist_speed if args.max_twist_speed is not None else self.max_rot_speed
        rot_speeds = np.array([self.max_rot_speed, self.max_rot_speed, twist_speed])
        drot_xyz = sm_state[3:] * (rot_speeds / self.frequency)
        drot_rotvec = st.Rotation.from_euler("xyz", drot_xyz).as_rotvec()
        drot_rotvec *= self.sm_drot_scalar
        drot = st.Rotation.from_rotvec(drot_rotvec)

        # get keyboard actions/flags
        keyboard_action, collect_enum = self.keyboard.get_action()

        # check if action is taken
        if np.allclose(dpos, 0.0) and np.allclose(drot_xyz, 0.0):
            action_taken = False
        else:
            action_taken = True

        # manage grasping
        self.steps_since_grasp += 1
        if self.steps_since_grasp < self.record_latency_when_grasping:
            action_taken = True

        self.last_grip_step += 1
        is_gripper_open = self.gripper_open
        if (
            self.sm.is_button_pressed(0)
            or self.sm.is_button_pressed(1)
            and self.last_grip_step > 10
        ):
            toggle_gripper = True

            self.gripper_open = not self.gripper_open
            self.last_grip_step = 0
            self.grasp_flag = -1 * self.grasp_flag
            self.steps_since_grasp = 0
        else:
            toggle_gripper = False

        # Make a delta action of xyz + quat_xyzw that we can scale before sending to robot
        delta_action = np.concatenate(
            [dpos, drot.as_quat(), np.array([self.grasp_flag])]
        )

        # overwrite action from keyboard action (for screwing)
        kb_taken = False
        if not (np.allclose(keyboard_action[3:6], 0.0)):
            delta_action[3:7] = keyboard_action[3:7]
            kb_taken = True
            action_taken = True

        pos_bounds_m = 0.025
        ori_bounds_deg = 20

        # delta_action = (
        #     scale_scripted_action(
        #         torch.from_numpy(delta_action).unsqueeze(0),
        #         pos_bounds_m=pos_bounds_m,
        #         ori_bounds_deg=ori_bounds_deg,
        #     )
        #     .squeeze()
        #     .numpy()
        # )

        # write out action
        new_target_pose = self.target_pose.copy()
        dpos, drot = delta_action[:3], st.Rotation.from_quat(delta_action[3:7])
        new_target_pose[:3] += dpos
        # new_target_pose[3:] = (
        #     drot * st.Rotation.from_rotvec(self.target_pose[3:])
        # ).as_rotvec()
        if kb_taken:
            # right multiply (more intuitive for screwing)
            new_target_pose[3:] = (
                st.Rotation.from_rotvec(self.target_pose[3:]) * drot
            ).as_rotvec()
        else:
            # left multiply (more intuitive for spacemouse)
            new_target_pose[3:] = (
                drot * st.Rotation.from_rotvec(self.target_pose[3:])
            ).as_rotvec()
        new_target_pose_mat = self.to_pose_mat(new_target_pose)
        current_pose_mat = self.to_pose_mat(self.target_pose)

        action_struct = ActionContainer(
            current_pose_mat=current_pose_mat,
            next_pose_mat=new_target_pose_mat,
            grasp_flag=self.grasp_flag,
            action_taken=action_taken,
            collect_enum=collect_enum,
            is_gripper_open=is_gripper_open,
            toggle_gripper=toggle_gripper,
        )
        return action_struct

    def get_rgbd_rs(self, pipe) -> dict:
        if pipe is None:  # --no_cameras: black images in the same format as the D435s
            return dict(
                rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                depth=np.zeros((480, 640), dtype=np.uint16),
            )
        align_to = rs.stream.color
        align = rs.align(align_to)

        try:
            # Get frameset of color and depth
            frames = pipe.wait_for_frames(100)  # 100
            # frames = pipe.wait_for_frames(100)
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            print(
                f"Couldn't get frame for device: {pipe.get_active_profile().get_device()}"
            )
            # continue
            raise

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # the .copy() here is super important!
        img_dict = dict(rgb=color_image.copy(), depth=depth_image.copy())

        return img_dict

    def get_observation(self):
        # get the rgb images
        front_rgbd = self.get_rgbd_rs(self.front_image_pipeline)
        wrist_rgbd = self.get_rgbd_rs(self.wrist_image_pipeline)

        # get the robot state
        # wrist pose
        # current_ee_pose = torch.cat(self.robot.get_ee_pose(), dim=-1)

        # convert to tip
        current_ee_wrist_pose_mat = poly_util.polypose2mat(self.robot.get_ee_pose())
        current_ee_tip_pose_mat = convert_wrist2tip(current_ee_wrist_pose_mat)
        current_ee_tip_pose = poly_util.mat2polypose(current_ee_tip_pose_mat)
        current_ee_pose = torch.cat(current_ee_tip_pose, dim=-1)

        current_joint_positions = self.robot.get_joint_positions()
        jacobian = self.robot.robot_model.compute_jacobian(current_joint_positions)
        ee_spatial_velocity = jacobian @ self.robot.get_joint_velocities()

        robot_state_dict = dict(
            ee_pos=current_ee_pose[:3].numpy(),
            ee_quat=current_ee_pose[3:7].numpy(),
            ee_pos_vel=ee_spatial_velocity[:3].numpy(),
            ee_ori_vel=ee_spatial_velocity[3:6].numpy(),
            gripper_width=self.gripper.get_state().width,
        )

        # pack these
        obs = dict(
            color_image1=wrist_rgbd["rgb"],
            color_image2=front_rgbd["rgb"],
            robot_state=robot_state_dict,
        )

        return obs

    def posemat2action(self, target_pose_mat: np.ndarray, grasp_flag: int):
        action = np.zeros(8)
        action[:3] = target_pose_mat[:-1, -1]
        action[3:7] = st.Rotation.from_matrix(target_pose_mat[:-1, :-1]).as_quat()
        action[-1] = grasp_flag

        return action


def main():
    # some main args
    frequency = args.frequency
    dt = 1 / frequency
    command_latency = args.command_latency

    # setup robot
    franka_ip = args.server_ip  # Polymetis server address, not the robot's IP
    # robot = RobotInterface(ip_address=franka_ip)
    # lars-dev DiffIKWrapper stores robot_home/Kq/Kqd only for its own reset(), unused here
    robot = DiffIKWrapper(robot_home=None, Kq=None, Kqd=None, ip_address=franka_ip)
    if args.no_gripper:
        print("--no_gripper: no gripper server, using a stand-in gripper")
        gripper = NoGripper()
    else:
        gripper = GripperInterface(ip_address=franka_ip)

    # manual home
    gripper.goto(0.065, 0.05, 0.1, blocking=False)
    # robot_home = torch.Tensor([-0.1363, -0.0406, -0.0460, -2.1322, 0.0191, 2.0759, 0.5])
    robot_home = torch.Tensor(
        [-0.0931, 0.0382, 0.1488, -2.3811, -0.0090, 2.4947, 0.1204]
    )
    home_noise = (2 * torch.rand(7) - 1) * np.deg2rad(5)
    robot_home = robot_home + home_noise

    robot.move_to_joint_positions(robot_home)

    # sm_dpos_scalar = np.array([1.5] * 3)
    # sm_drot_scalar = np.array([2.25] * 3)
    sm_dpos_scalar = np.array([1.8] * 3)
    sm_drot_scalar = np.array([4.0] * 3)

    twist_speed = args.max_twist_speed if args.max_twist_speed is not None else args.max_rot_speed
    # The command is re-anchored to the measured pose every step, and the arm covers only about
    # 30% of each step's rotation within the step (joint 7 has the softest gains, Kq 40 / Kqd 8),
    # in this data and the authors'. The real rate is therefore ~0.3x the commanded one.
    for label, v in [("tilt", args.max_rot_speed), ("twist", twist_speed)]:
        cmd = np.rad2deg(v * sm_drot_scalar[0])
        print(f"SpaceMouse {label} at full deflection: commanded {cmd:.0f} deg/s, arm turns about {0.3 * cmd:.0f} deg/s")
    # Joint 7 torque at the start of a full-deflection twist step: Kq[6] x commanded step
    j7_torque = 40.0 * twist_speed * sm_drot_scalar[0] / args.frequency
    if j7_torque > 10.0:
        print(f"WARNING: a full twist asks joint 7 for about {j7_torque:.0f} Nm (FR3 limit 12 Nm)")

    Kq_new = torch.Tensor([150.0, 120.0, 160.0, 100.0, 110.0, 100.0, 40.0])
    Kqd_new = torch.Tensor([20.0, 20.0, 20.0, 20.0, 12.0, 12.0, 8.0])

    robot.start_joint_impedance(Kq=Kq_new, Kqd=Kqd_new, adaptive=True)

    # setup visuals
    zmq_url = f"tcp://127.0.0.1:{args.port_vis}"
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis["scene"].delete()

    # Setup camera streams
    camera_settings = None
    if args.no_cameras:
        print("--no_cameras: not opening any camera, saving black placeholder images")
        front_image_pipeline = wrist_image_pipeline = None
    else:
        rs_cfg = get_default_multi_realsense_cfg()
        resolution_width = rs_cfg.WIDTH  # pixels
        resolution_height = rs_cfg.HEIGHT  # pixels
        frame_rate = rs_cfg.FRAME_RATE  # fps

        ctx = rs.context()  # Create librealsense context for managing devices

        print(
            f"Enabling devices with serial numbers: {FRONT_CAMERA_SERIAL} (front) and {WRIST_CAMERA_SERIAL} (wrist)"
        )
        front_image_pipeline = enable_single_realsense(
            FRONT_CAMERA_SERIAL, ctx, resolution_width, resolution_height, frame_rate
        )
        time.sleep(1.0)
        wrist_image_pipeline = enable_single_realsense(
            WRIST_CAMERA_SERIAL, ctx, resolution_width, resolution_height, frame_rate
        )
        # Fixed exposure/white balance from camera_settings.py (None there = auto)
        camera_settings = dict(
            front=apply_color_settings(front_image_pipeline, COLOR_SETTINGS["front"], "front"),
            wrist=apply_color_settings(wrist_image_pipeline, COLOR_SETTINGS["wrist"], "wrist"),
        )
        time.sleep(1.0)

    # Setup data saving
    demo_save_dir = Path(args.save_dir)
    demo_save_dir.mkdir(exist_ok=True, parents=True)
    pkl_path = demo_save_dir / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"

    episode_data = {}
    episode_data["observations"] = []
    episode_data["actions"] = []
    # 1 on the step recorded after each ` press (insertion start), 0 elsewhere
    episode_data["skills"] = []
    episode_data["furniture"] = args.furniture
    # assume all real world demos that we actually save are success
    episode_data["success"] = True
    episode_data["args"] = args.__dict__

    # initial metadata dict
    metadata = dict(
        sm_dpos_scalar=sm_dpos_scalar,
        sm_drot_scalar=sm_drot_scalar,
        Kq=Kq_new.cpu().numpy(),
        Kqd=Kqd_new.cpu().numpy(),
        camera_settings=camera_settings,  # values read back from the cameras; None without cameras
    )
    episode_data["metadata"] = metadata

    # Setup other interfaces
    keyboard = KeyboardInterface()
    skill_enum = getattr(CollectEnum, "SKILL", None)
    if skill_enum is None:
        print("WARNING: this CollectEnum has no SKILL, the insertion start can't be marked")
    else:
        print("Press ` with the leg held about 5 cm above the hole to mark the insertion start")

    def polypose2target(poly_pose):
        translation, quat_xyzw = poly_pose[0], poly_pose[1]
        rotvec = st.Rotation.from_quat(quat_xyzw.numpy()).as_rotvec()
        target_pose = np.array([*translation.numpy(), *rotvec])
        return target_pose

    def to_pose_mat(pose_):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = pose_[:3]
        pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_[3:]).as_matrix()
        return pose_mat

    def wrist_target_to_tip(wrist_target_pose_rv):
        wrist_target_pose_mat = to_pose_mat(wrist_target_pose_rv)
        tip_target_pose_mat = convert_wrist2tip(wrist_target_pose_mat)
        tip_target_pos = tip_target_pose_mat[:-1, -1]
        tip_target_rv = st.Rotation.from_matrix(
            tip_target_pose_mat[:-1, :-1]
        ).as_rotvec()
        tip_target_pose_rv = np.array([*tip_target_pos, *tip_target_rv])
        return tip_target_pose_rv

    translation, quat_xyzw = robot.get_ee_pose()
    rotvec = st.Rotation.from_quat(quat_xyzw.numpy()).as_rotvec()
    target_pose = np.array([*translation.numpy(), *rotvec])
    tip_target_pose = wrist_target_to_tip(target_pose)

    def execute_gripper_action(toggle_gripper: bool, gripper_open: bool):
        if not toggle_gripper:
            return
        if gripper_open:
            gripper.grasp(0.07, 70, blocking=False)
        else:
            # gripper.goto(0.08, 0.05, 0.1, blocking=False)
            gripper.goto(0.065, 0.05, 0.1, blocking=False)

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone) as sm:
            t_start = time.monotonic()
            iter_idx = 0
            stop = False

            obs_act_helper = ObsActHelper(
                sm=sm,
                keyboard=keyboard,
                robot=robot,
                gripper=gripper,
                front_image_pipeline=front_image_pipeline,
                wrist_image_pipeline=wrist_image_pipeline,
            )

            # obs_act_helper.set_target_pose(target_pose)
            obs_act_helper.set_target_pose(tip_target_pose)
            obs_act_helper.set_constants(
                max_pos_speed=args.max_pos_speed,
                max_rot_speed=args.max_rot_speed,
                sm_dpos_scalar=sm_dpos_scalar,
                sm_drot_scalar=sm_drot_scalar,
                frequency=args.frequency,
            )
            global_start_time = time.time()
            print(f"Start collecting!")
            # A ` press usually comes while the arm is still, and still steps aren't
            # recorded, so the mark waits for the next recorded step
            mark_pending = False
            prev_skill = False
            view = CameraView() if args.show_cameras else None
            interp = None
            if args.interp_hz > 0:
                interp = JointTargetInterpolator(robot, args.interp_hz, dt)
                print(f"--interp_hz {args.interp_hz:g}: {interp.n} joint sub-targets per {1000 * dt:.0f} ms step")
            wait = interp.wait if interp is not None else precise_wait
            work_ms, display_ms, late_ms = [], [], []
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                # t_command_target = t_cycle_end + dt
                wait(t_sample)
                t_work = time.monotonic()
                late_ms.append(1000 * (t_work - t_sample))

                # get robot state/image observation
                observation = obs_act_helper.get_observation()

                # print(f"EE xyz: {observation['robot_state']['ee_pos'].round(3)}")
                # At pick tabletop-time, EE xyz: [0.433 0.009 0.127]
                # At pick table leg-time, EE xyz: [0.437 0.212 0.051]

                # get and unpack action
                action_struct = obs_act_helper.get_action()
                action_current_pose_mat = action_struct.current_pose_mat
                action_next_pose_mat = action_struct.next_pose_mat
                grasp_flag = action_struct.grasp_flag
                action_taken = action_struct.action_taken
                collect_enum = action_struct.collect_enum
                is_gripper_open = action_struct.is_gripper_open
                toggle_gripper = action_struct.toggle_gripper

                if collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                    break

                # Count a held or repeated SKILL as one press
                is_skill = skill_enum is not None and collect_enum == skill_enum
                if is_skill and not prev_skill:
                    mark_pending = True
                    n_marks = sum(episode_data["skills"]) + 1
                    print(
                        f"Mark {n_marks}: insertion start at step {len(episode_data['actions'])}"
                        + ("" if n_marks == 1 else " (segment_demos uses the first mark)")
                    )
                prev_skill = is_skill

                # send command to the robot
                # robot.update_desired_ee_pose(action_next_pose_mat, dt=dt)
                if interp is None:
                    robot.update_desired_ee_pose(
                        convert_tip2wrist(action_next_pose_mat), dt=dt
                    )
                else:  # same joint target, reached through a ramp
                    interp.set_target(
                        diffik_joint_target(robot, convert_tip2wrist(action_next_pose_mat), dt)
                    )
                execute_gripper_action(toggle_gripper, is_gripper_open)

                # log the data
                if action_taken:
                    # convert to delta actions (where action quat is a right mult.)
                    action = obs_act_helper.to_isaac_dpose_from_abs(
                        current_pose_mat=action_current_pose_mat,
                        goal_pose_mat=action_next_pose_mat,
                        grasp_flag=grasp_flag,
                        rm=True,
                    )
                    episode_data["actions"].append(action)
                    episode_data["observations"].append(observation)
                    episode_data["skills"].append(int(mark_pending))
                    mark_pending = False

                target_pose = polypose2target(robot.get_ee_pose())
                tip_target_pose = wrist_target_to_tip(target_pose)
                # obs_act_helper.set_target_pose(target_pose)
                obs_act_helper.set_target_pose(tip_target_pose)

                # Draw the current and target pose (in meshcat)
                mc_util.meshcat_frame_show(
                    mc_vis,
                    f"scene/target_pose_wrist",
                    convert_tip2wrist(action_next_pose_mat),
                )
                mc_util.meshcat_frame_show(
                    mc_vis, f"scene/target_pose_tip", action_next_pose_mat
                )
                mc_util.meshcat_frame_show(
                    mc_vis,
                    f"scene/current_pose",
                    poly_util.polypose2mat(robot.get_ee_pose()),
                )
                work_ms.append(1000 * (time.monotonic() - t_work))

                if view is not None:
                    t_disp = time.monotonic()
                    marks = [i for i, s in enumerate(episode_data["skills"]) if s]
                    mark_text = (
                        f"mark @ step {marks[0]}" if marks
                        else "mark pending" if mark_pending else "no mark"
                    )
                    view.show(
                        observation,
                        n_recorded=len(episode_data["actions"]),
                        recorded=action_taken,
                        gripper_open=obs_act_helper.gripper_open,
                        mark_text=mark_text,
                        loop_ms=work_ms[-1] + (display_ms[-1] if display_ms else 0.0),
                    )
                    display_ms.append(1000 * (time.monotonic() - t_disp))

                wait(t_cycle_end)
                iter_idx += 1
            if view is not None:
                view.close()

    global_total_time = time.time() - global_start_time
    print(f"Time elapsed: {global_total_time}")
    print_loop_timing(work_ms, display_ms, late_ms, dt)
    if interp is not None:
        interp.summary()
    # Steps without input aren't recorded, so the episode is shorter than the loop
    n_rec, n_loop = len(episode_data["actions"]), len(work_ms)
    print(
        f"Recorded steps: {n_rec} of {n_loop} loop steps ({100 * n_rec / max(n_loop, 1):.0f}%), "
        f"{n_rec * dt:.1f} s of motion at {frequency} Hz"
    )
    if collect_enum == CollectEnum.SUCCESS:
        marks = [i for i, s in enumerate(episode_data["skills"]) if s]
        if marks:
            print(f"Insertion marks at steps {marks} of {len(episode_data['actions'])}")
        else:
            print(
                "WARNING: no insertion mark. segment_demos skips this demo unless "
                "it is listed in a --marks CSV"
            )
        # save the data
        with open(pkl_path, "wb") as f:
            pickle.dump(episode_data, f)
        print(f"Saved {pkl_path}")


if __name__ == "__main__":
    main()
