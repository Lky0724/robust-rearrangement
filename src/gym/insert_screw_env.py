"""Task one_leg_insert_screw: the insert + screw part of one_leg, started from random mid-task states.

The furniture, the success check and the observations are those of one_leg (the env is a
FurnitureRLSimEnv created with furniture="one_leg"). Only the reset differs. Instead of the
parts lying on the table and the arm at its home pose, every reset puts the scene where the
insert + screw segment of the one_leg demos begins (segment_demos.py --phase insert --through screw):

- fixture (obstacle): shifted at random as in one_leg
- legs 1-3: their one_leg reset poses, pushed at random as in one_leg
- table top: snug in the fixture corner, in one of the two orientations seen in the demos
- leg 4: in the hand, with a random grasp and hand orientation, its tip at a random spot above
  the hole in the fixture corner
- arm: the joint angles for that hand pose (inverse kinematics, seeded at the demos' arm pose)
- gripper: closed on the leg

The ranges of the random parameters (DEMO_RANGES) are fitted to the first frames of the 41
training demos (scripts/check_insert_screw_env.py prints them) and scaled about their centre by
START_RANGE_SCALE[randomness]: low narrows them, med widens them a little, high more. Samples
that put the leg into the table top, tilt it or stretch the arm further than the (scaled) demos,
or have no IK solution within the joint limits are drawn again. After the scene has settled for
a few physics steps, envs where the leg slipped in the hand or the table top was pushed away
are reset again.

Frames: "robot" is the robot base frame (axes as the sim world, z up, origin at the base),
"sim" is the sim frame of the root/rigid body state tensors (robot frame + franka_pose.p).
"""

import math
from collections import Counter
from typing import Dict, Tuple

from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv  # imports isaacgym before torch
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.sim_config import sim_config
from isaacgym import gymtorch
import furniture_bench.controllers.control_utils as C
import torch

# ---------------------------------------------------------------------------------------------
# Scene geometry (m)

HELD_PART, BASE_PART = 4, 0  # leg 4 and the table top, indices into furniture.parts
LEG_TIP = (0.0, -0.05625, 0.0)  # thread end in the leg frame; the leg's long axis is +y
# Lowest points of the leg when it hangs tip down: the tip and the 4 bottom corners of the
# 30 mm square body, which starts 25 mm above the tip
LEG_LOW_POINTS = [LEG_TIP] + [(sx * 0.015, -0.03125, sz * 0.015) for sx in (-1, 1) for sz in (-1, 1)]
# Leg positions when assembled, table top frame (furniture.assembled_rel_poses, rotation identity)
TOP_HOLES = [(sx * 0.05625, 0.046875, sz * 0.05625) for sx in (-1, 1) for sz in (-1, 1)]
TOP_HALF_THICKNESS = 0.0156
TOP_Z_ABOVE_FIXTURE = 0.0007  # table top centre above the fixture front piece's centre when resting
MIN_LEG_CLEARANCE = 0.003  # lowest leg point above the table top surface
# Finger joint positions at reset: GRIPPER_WIDTH / 2 -+ (grasp_z + FINGER_OFFSET), so the fingers
# start just outside the 30 mm leg and close on it while settling. Tuned in the sim: fingers that
# start in contact or inside the leg squeeze it out sideways in the first physics step (with
# the offset the demos' finger positions suggest, +0.65 mm, 60% of the starts lost the leg).
GRIPPER_WIDTH = 0.0303
FINGER_OFFSET = -0.0002

# ---------------------------------------------------------------------------------------------
# Start state parameters: (min, max) over the first frames of the 41 training demos
# (data/raw/diffik/sim/one_leg_insert_screw/teleop/med/own51/success, the insert + screw cut of
# our own med teleop demos). Robot frame; m and deg.

DEMO_RANGES = {
    # Table top centre - fixture front piece centre. The top is snug against the front piece at
    # the max of board_dx and against the left side piece at the max of board_dy, so these two
    # are only widened downwards (away from the fixture).
    "board_dx": (-0.0986, -0.0911),
    "board_dy": (0.0820, 0.0838),
    "board_yaw_noise": (-0.63, 0.63),  # about the up axis, around one of BOARD_YAWS
    # Leg tip - the tip's position when assembled in the fixture-corner hole
    "tip_dx": (-0.0966, 0.0222),
    "tip_dy": (-0.0297, 0.0368),
    "tip_dz": (0.0311, 0.1155),
    # Hand orientation: R_EE = R_DOWN @ Rx(ee_rx) @ Ry(ee_ry) @ Rz(ee_rz), R_DOWN = pointing
    # straight down with the hand's x along the robot's x
    "ee_rx": (-9.38, 11.86),
    "ee_ry": (-28.70, 20.73),
    "ee_rz": (-4.81, 28.26),
    # Grasp. The fingers close along the leg's z axis (the same face in all demos); the leg is
    # turned by grasp_pitch about the closing axis, away from the approach axis.
    "grasp_pitch": (13.95, 37.56),
    # The hand's reference point (k_ee_link) in the leg frame: x across the leg in the finger
    # plane, y along the leg (from the centre of the body), z along the closing axis
    "grasp_x": (-0.0114, 0.0031),
    "grasp_y": (-0.0033, 0.0263),
    "grasp_z": (-0.0102, 0.0032),
    # Not sampled, but samples outside these (scaled) ranges are drawn again, since the
    # parameters above are drawn independently: the leg's tilt from vertical (deg), and the
    # distance from the shoulder (joint 2) to the wrist (link 7's origin), which keeps the arm
    # away from full stretch (~0.80 m)
    "leg_tilt": (1.11, 57.76),
    "reach": (0.550, 0.745),
}
ONE_SIDED_DOWN = ("board_dx", "board_dy")
SHOULDER = (0.0, 0.0, 0.333)  # joint 2 in the robot frame
WRIST_FROM_EE = 0.107 + 0.103399  # link 7's origin, back along the approach axis from k_ee_link
BOARD_YAWS = (-90.0, 0.0)  # table top orientations about the up axis (deg)
BOARD_YAW_WEIGHTS = (37.0, 4.0)  # how many training demos have each
START_RANGE_SCALE = {Randomness.LOW: 0.5, Randomness.MEDIUM: 1.2, Randomness.HIGH: 1.5}

# Median arm pose of the training demos' first frames, the IK seed and null-space target
ARM_SEED = (0.23, 0.55, -0.01, -1.78, 0.10, 2.18, 0.97)
R_DOWN = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))


def scaled_ranges(scale: float) -> Dict[str, Tuple[float, float]]:
    """DEMO_RANGES scaled about their centre (one-sided ones from their max downwards)."""
    out = {}
    for k, (lo, hi) in DEMO_RANGES.items():
        if k in ONE_SIDED_DOWN:
            out[k] = (hi - (hi - lo) * scale, hi)
        else:
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            out[k] = (mid - half * scale, mid + half * scale)
    return out


# ---------------------------------------------------------------------------------------------
# Rotations (batched, (..., 3, 3))


def rot_axis(angle: torch.Tensor, axis: int) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    if axis == 0:
        m = [o, z, z, z, c, -s, z, s, c]
    elif axis == 1:
        m = [c, z, s, z, o, z, -s, z, c]
    else:
        m = [c, -s, z, s, c, z, z, z, o]
    return torch.stack(m, -1).view(*angle.shape, 3, 3)


def euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """Angles (a, b, c) with R = Rx(a) @ Ry(b) @ Rz(c), in radians."""
    b = torch.asin(R[..., 0, 2].clamp(-1, 1))
    a = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
    c = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
    return torch.stack([a, b, c], -1)


def grasp_rotation(pitch: torch.Tensor) -> torch.Tensor:
    """Leg orientation in the hand frame: leg z along the closing axis (hand y), leg y (its
    long axis, tip end at -y) turned by `pitch` about hand y from pointing back along -z."""
    c, s, z, o = torch.cos(pitch), torch.sin(pitch), torch.zeros_like(pitch), torch.ones_like(pitch)
    return torch.stack([c, s, z, z, z, o, s, -c, z], -1).view(*pitch.shape, 3, 3)


# ---------------------------------------------------------------------------------------------
# Franka kinematics, from the sim's URDF (franka_description_ros/.../franka_panda.urdf):
# panda_link0 -> panda_joint1..7 (all about their local z) -> panda_hand -> k_ee_link


class FrankaKinematics:
    # (xyz, rpy) of each joint's origin in its parent link
    JOINTS = [
        ((0, 0, 0.333), (0, 0, 0)),
        ((0, 0, 0), (-math.pi / 2, 0, 0)),
        ((0, -0.316, 0), (math.pi / 2, 0, 0)),
        ((0.0825, 0, 0), (math.pi / 2, 0, 0)),
        ((-0.0825, 0.384, 0), (-math.pi / 2, 0, 0)),
        ((0, 0, 0), (math.pi / 2, 0, 0)),
        ((0.088, 0, 0), (math.pi / 2, 0, 0)),
    ]
    EE = ((0, 0, 0.107 + 0.103399), (0, 0, -math.pi / 4))  # panda_hand_joint + k_ee
    LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
    UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)

    def __init__(self, device, dtype=torch.float64):
        def tf(xyz, rpy):
            t = torch.tensor(rpy, dtype=dtype)
            R = rot_axis(t[0], 0)
            R = rot_axis(t[2], 2) @ rot_axis(t[1], 1) @ R  # URDF rpy: R = Rz(y) Ry(p) Rx(r)
            M = torch.eye(4, dtype=dtype)
            M[:3, :3], M[:3, 3] = R, torch.tensor(xyz, dtype=dtype)
            return M

        self.device, self.dtype = device, dtype
        self.joint_tf = torch.stack([tf(*j) for j in self.JOINTS]).to(device)
        self.ee_tf = tf(*self.EE).to(device)
        self.lower = torch.tensor(self.LOWER, dtype=dtype, device=device)
        self.upper = torch.tensor(self.UPPER, dtype=dtype, device=device)

    def fk(self, q: torch.Tensor):
        """q (N, 7) -> EE pose (N, 4, 4) in the robot frame, joint axes (N, 7, 3) and origins (N, 7, 3)."""
        q = q.to(self.dtype)
        T = torch.eye(4, dtype=self.dtype, device=q.device).expand(q.shape[0], 4, 4)
        axes, origins = [], []
        for i in range(7):
            T = T @ self.joint_tf[i]
            axes.append(T[:, :3, 2])
            origins.append(T[:, :3, 3])
            Rz = torch.eye(4, dtype=self.dtype, device=q.device).repeat(q.shape[0], 1, 1)
            Rz[:, :3, :3] = rot_axis(q[:, i], 2)
            T = T @ Rz
        return T @ self.ee_tf, torch.stack(axes, 1), torch.stack(origins, 1)

    def ik(self, pos, rot, q_seed, iters=60, polish=40, damping=0.02, null_gain=0.3, max_step=0.3):
        """Damped least squares IK for EE poses (pos (N, 3), rot (N, 3, 3), robot frame), with a
        null-space pull to q_seed for the first `iters` steps (the damped pseudo-inverse leaks
        it into the EE pose, so the last `polish` steps go without it). Returns q (N, 7),
        position error (m) and rotation error (rad)."""
        pos, rot = pos.to(self.dtype), rot.to(self.dtype)
        q0 = q_seed.to(self.dtype).expand(pos.shape[0], 7)
        q = q0.clone()
        eye6 = torch.eye(6, dtype=self.dtype, device=pos.device)
        eye7 = torch.eye(7, dtype=self.dtype, device=pos.device)
        for it in range(iters + polish):
            T, z, o = self.fk(q)
            dp = pos - T[:, :3, 3]
            dw = C.matrix_to_axis_angle(rot @ T[:, :3, :3].transpose(1, 2))
            J = torch.cat([torch.cross(z, T[:, None, :3, 3] - o, dim=-1), z], -1).transpose(1, 2)  # (N, 6, 7)
            J_pinv = J.transpose(1, 2) @ torch.linalg.inv(J @ J.transpose(1, 2) + damping**2 * eye6)
            dq = (J_pinv @ torch.cat([dp, dw], -1)[..., None])[..., 0]
            if it < iters:
                dq = dq + ((eye7 - J_pinv @ J) @ (null_gain * (q0 - q))[..., None])[..., 0]
            q = q + dq.clamp(-max_step, max_step)
        T, _, _ = self.fk(q)
        pos_err = (pos - T[:, :3, 3]).norm(dim=-1)
        rot_err = C.matrix_to_axis_angle(rot @ T[:, :3, :3].transpose(1, 2)).norm(dim=-1)
        return q, pos_err, rot_err

    def within_limits(self, q: torch.Tensor, margin=0.05) -> torch.Tensor:
        return ((q >= self.lower + margin) & (q <= self.upper - margin)).all(-1)


# ---------------------------------------------------------------------------------------------
# Sampling and decomposing start states (robot frame)


def _uniform(n, lo_hi, device, deg=False):
    lo, hi = lo_hi
    x = lo + (hi - lo) * torch.rand(n, device=device, dtype=torch.float64)
    return torch.deg2rad(x) if deg else x


def _hole_frame(top_pos, top_rot):
    """Position of the table top hole nearest the fixture corner (+x, +y), robot frame."""
    holes = torch.tensor(TOP_HOLES, dtype=top_pos.dtype, device=top_pos.device)  # (4, 3)
    holes = top_pos[:, None] + (top_rot[:, None] @ holes[None, :, :, None])[..., 0]  # (N, 4, 3)
    k = (holes[..., 0] - top_pos[:, None, 0] + holes[..., 1] - top_pos[:, None, 1]).argmax(1)
    return holes[torch.arange(len(k), device=k.device), k]


REJECT_REASONS = {1: "leg in table top", 2: "leg tilt or reach", 3: "IK failed", 4: "joint limits"}


def _leg_tilt(leg_rot):
    return torch.rad2deg(torch.acos(leg_rot[:, 2, 1].clamp(-1, 1)))  # leg's long axis vs up


def _reach(ee_pos, ee_rot):
    shoulder = torch.tensor(SHOULDER, dtype=ee_pos.dtype, device=ee_pos.device)
    return (ee_pos - ee_rot[:, :, 2] * WRIST_FROM_EE - shoulder).norm(dim=-1)


def sample_starts(obstacle_pos: torch.Tensor, ranges: Dict, kin: FrankaKinematics) -> Dict[str, torch.Tensor]:
    """One candidate start state per fixture position (N, 3), robot frame. `valid` marks the
    candidates that can be used; `reason` says why the others can't (0 ok, else REJECT_REASONS)."""
    dev, n = obstacle_pos.device, obstacle_pos.shape[0]
    f64 = torch.float64
    u = lambda k, deg=False: _uniform(n, ranges[k], dev, deg)  # noqa: E731
    lift = torch.tensor(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=f64, device=dev)  # top's y (legs side) up

    # Table top in the fixture corner
    w = torch.tensor(BOARD_YAW_WEIGHTS, dtype=f64, device=dev)
    yaw = torch.tensor(BOARD_YAWS, dtype=f64, device=dev)[torch.multinomial(w / w.sum(), n, replacement=True)]
    top_rot = rot_axis(torch.deg2rad(yaw) + u("board_yaw_noise", deg=True), 2) @ lift
    top_pos = obstacle_pos.to(f64) + torch.stack(
        [u("board_dx"), u("board_dy"), torch.full((n,), TOP_Z_ABOVE_FIXTURE, dtype=f64, device=dev)], -1
    )

    # Hand orientation and grasp -> leg orientation
    R_down = torch.tensor(R_DOWN, dtype=f64, device=dev)
    ee_rot = R_down @ rot_axis(u("ee_rx", True), 0) @ rot_axis(u("ee_ry", True), 1) @ rot_axis(u("ee_rz", True), 2)
    G_rot = grasp_rotation(u("grasp_pitch", True))
    leg_rot = ee_rot @ G_rot
    e = torch.stack([u("grasp_x"), u("grasp_y"), u("grasp_z")], -1)  # hand point in the leg frame
    g = -(G_rot @ e[..., None])[..., 0]  # leg origin in the hand frame

    # Leg tip relative to where it is when assembled -> leg and hand positions
    tip_local = torch.tensor(LEG_TIP, dtype=f64, device=dev)
    tip_asm = _hole_frame(top_pos, top_rot) + top_rot @ tip_local
    tip = tip_asm + torch.stack([u("tip_dx"), u("tip_dy"), u("tip_dz")], -1)
    leg_pos = tip - leg_rot @ tip_local
    ee_pos = leg_pos - (ee_rot @ g[..., None])[..., 0]

    # Checks: leg above the table top, leg tilt and reach as in the demos, IK, joint limits
    low = torch.tensor(LEG_LOW_POINTS, dtype=f64, device=dev)  # (5, 3)
    low_z = (leg_pos[:, None] + (leg_rot[:, None] @ low[None, :, :, None])[..., 0])[..., 2].min(1).values
    clear = low_z >= top_pos[:, 2] + TOP_HALF_THICKNESS + MIN_LEG_CLEARANCE
    within = lambda x, k: (x >= ranges[k][0]) & (x <= ranges[k][1])  # noqa: E731
    demo_like = within(_leg_tilt(leg_rot), "leg_tilt") & within(_reach(ee_pos, ee_rot), "reach")
    q, pos_err, rot_err = kin.ik(ee_pos, ee_rot, torch.tensor(ARM_SEED, dtype=f64, device=dev))
    ik_ok = (pos_err < 5e-4) & (rot_err < math.radians(0.5))
    in_lim = kin.within_limits(q)
    reason = torch.zeros(n, dtype=torch.long, device=dev)
    for r, ok in ((4, in_lim), (3, ik_ok), (2, demo_like), (1, clear)):  # the first failed check wins
        reason[~ok] = r

    closing = e[:, 2] + FINGER_OFFSET
    fingers = torch.stack([GRIPPER_WIDTH / 2 - closing, GRIPPER_WIDTH / 2 + closing], -1)
    return dict(
        top_pos=top_pos, top_rot=top_rot, leg_pos=leg_pos, leg_rot=leg_rot, ee_pos=ee_pos,
        ee_rot=ee_rot, q=q, fingers=fingers, valid=reason == 0, reason=reason,
    )


def decompose_starts(ee_pos, ee_rot, top_pos, top_rot, leg_pos, leg_rot, obstacle_pos) -> Dict[str, torch.Tensor]:
    """The parameters of DEMO_RANGES (m, deg) for given states (robot frame), the inverse of
    sample_starts."""
    f64 = torch.float64
    ee_pos, ee_rot, top_pos, top_rot, leg_pos, leg_rot, obstacle_pos = (
        x.to(f64) for x in (ee_pos, ee_rot, top_pos, top_rot, leg_pos, leg_rot, obstacle_pos)
    )
    dev = ee_pos.device
    out = {}
    d = top_pos - obstacle_pos
    out["board_dx"], out["board_dy"] = d[:, 0], d[:, 1]
    yaw = torch.rad2deg(torch.atan2(top_rot[:, 1, 0], top_rot[:, 0, 0]))
    yaws = torch.tensor(BOARD_YAWS, dtype=f64, device=dev)
    diff = (yaw[:, None] - yaws[None] + 180) % 360 - 180
    out["board_yaw_noise"] = diff.gather(1, diff.abs().argmin(1, keepdim=True))[:, 0]

    tip_local = torch.tensor(LEG_TIP, dtype=f64, device=dev)
    tip = leg_pos + leg_rot @ tip_local
    dt = tip - (_hole_frame(top_pos, top_rot) + top_rot @ tip_local)
    out["tip_dx"], out["tip_dy"], out["tip_dz"] = dt[:, 0], dt[:, 1], dt[:, 2]

    R_down = torch.tensor(R_DOWN, dtype=f64, device=dev)
    eul = torch.rad2deg(euler_xyz(R_down.T @ ee_rot))
    out["ee_rx"], out["ee_ry"], out["ee_rz"] = eul[:, 0], eul[:, 1], eul[:, 2]

    G_rot = ee_rot.transpose(1, 2) @ leg_rot
    g = (ee_rot.transpose(1, 2) @ (leg_pos - ee_pos)[..., None])[..., 0]
    a = G_rot[..., 1]  # leg axis in the hand frame
    out["grasp_pitch"] = torch.rad2deg(torch.atan2(a[:, 0], -a[:, 2]))
    e = -(G_rot.transpose(1, 2) @ g[..., None])[..., 0]
    out["grasp_x"], out["grasp_y"], out["grasp_z"] = e[:, 0], e[:, 1], e[:, 2]

    out["leg_tilt"] = _leg_tilt(leg_rot)
    out["reach"] = _reach(ee_pos, ee_rot)
    return out


# ---------------------------------------------------------------------------------------------


class FurnitureRLSimEnvInsertScrew(FurnitureRLSimEnv):
    """one_leg, reset to a random start of the insert + screw segment (see the module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.furniture_name == "one_leg", "one_leg_insert_screw needs furniture='one_leg'"
        self.task_name = "one_leg_insert_screw"

        if self.randomness not in START_RANGE_SCALE:
            raise ValueError(f"No start ranges for randomness {self.randomness}")
        self.start_ranges = scaled_ranges(START_RANGE_SCALE[self.randomness])
        self.kin = FrankaKinematics(self.device)
        p = self.franka_pose.p
        self.robot_origin = torch.tensor([p.x, p.y, p.z], device=self.device)
        self.ee_rb_idxs = torch.tensor(self.ee_idxs, device=self.device, dtype=torch.long)
        self.part_rb_idxs = self.furniture_rb_indices.to(self.device).long()  # (num_envs, num_parts)

        # The one_leg reset push, for the parts still lying on the table only
        self.push_mask = torch.ones_like(self.force_multiplier)
        self.push_mask[[BASE_PART, HELD_PART]] = 0

        self.settle_steps = 2 * self.sim_steps  # physics steps before the first observation
        self.max_reset_rounds = 4  # resets of envs whose start didn't hold
        self.max_slip = 0.005  # leg move in the hand while settling (m)
        self.max_slip_rot = math.radians(5)
        self.max_part_move = 0.005  # table top move while settling (m)
        self.reset_stats = Counter()  # rejected samples and resets, by reason

        print(
            f"one_leg_insert_screw start ranges ({self.randomness.name}, scale "
            f"{START_RANGE_SCALE[self.randomness]}): "
            + ", ".join(f"{k} [{lo:.4g}, {hi:.4g}]" for k, (lo, hi) in self.start_ranges.items())
        )

    @torch.no_grad()
    def reset(self, env_idxs: torch.Tensor = None):
        if env_idxs is None:
            env_idxs = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        assert env_idxs.numel() > 0

        self.already_assembled[env_idxs] = 0
        todo = env_idxs.long()
        for round_ in range(self.max_reset_rounds):
            want = self._place_start(todo)
            self._settle()
            err = self.start_errors(todo, want)
            if round_ == 0:
                self.first_start_errors = err  # all envs, before any were reset again (for checks)
            bad = (err["slip"] > self.max_slip) | (err["slip_rot"] > self.max_slip_rot) | (
                err["top_move"] > self.max_part_move
            )
            self.reset_stats["envs reset"] += len(todo)
            if not bad.any():
                break
            self.reset_stats["start didn't hold"] += int(bad.sum())
            if round_ == self.max_reset_rounds - 1:
                print(f"[one_leg_insert_screw] {int(bad.sum())} envs keep a start that didn't hold")
                break
            todo = todo[bad]

        self.env_steps[env_idxs] = 0
        self.refresh()
        obs = self.get_observation()
        if self.record:
            self.recorder.restart_recording()
            self.recorder.record_frame(obs)
        return obs

    def _sample_valid(self, obstacle_pos: torch.Tensor, max_rounds=20) -> Dict[str, torch.Tensor]:
        """A valid start for each fixture position, drawing again where a candidate is invalid."""
        n = obstacle_pos.shape[0]
        out = new = sample_starts(obstacle_pos, self.start_ranges, self.kin)
        for _ in range(max_rounds):
            self.reset_stats["sample: drawn"] += len(new["reason"])
            for r, name in REJECT_REASONS.items():
                self.reset_stats[f"sample: {name}"] += int((new["reason"] == r).sum())
            bad = ~out["valid"]
            if not bad.any():
                return out
            new = sample_starts(obstacle_pos[bad], self.start_ranges, self.kin)
            for k in out:
                out[k][bad] = new[k]
        raise RuntimeError(f"No valid one_leg_insert_screw start after {max_rounds} draws for {int(bad.sum())}/{n} envs")

    def _place_start(self, idxs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Write a new start state for envs idxs (long). Returns the intended poses, sim frame."""
        # Fixture and every part to their one_leg reset poses, the fixture shifted at random
        self._reset_part_poses(idxs.int())
        front = self.root_pos[idxs, self.obstacles_idx_list[0].long()] - self.robot_origin
        s = self._sample_valid(front.to(torch.float64))

        to_sim = lambda p: (p + self.robot_origin.double()).float()  # noqa: E731
        quat = lambda R: C.matrix_to_quaternion_xyzw(R).float()  # noqa: E731
        parts = self.parts_idx_list.long()
        self.root_pos[idxs, parts[BASE_PART]] = to_sim(s["top_pos"])
        self.root_quat[idxs, parts[BASE_PART]] = quat(s["top_rot"])
        self.root_pos[idxs, parts[HELD_PART]] = to_sim(s["leg_pos"])
        self.root_quat[idxs, parts[HELD_PART]] = quat(s["leg_rot"])
        actor_idxs = self.actor_idx_by_env[idxs].reshape(-1)
        self.root_tensor[actor_idxs.long(), 7:13] = 0.0
        assert self.isaac_gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_tensor), gymtorch.unwrap_tensor(actor_idxs), len(actor_idxs)
        ), "Failed to set the part states"

        # Arm at the IK solution and held there by its joint drives, gripper closed on the leg
        self.dof_pos[idxs, :7] = s["q"].float()
        self.dof_pos[idxs, 7:9] = s["fingers"].float()
        self.dof_vel[idxs] = 0.0
        franka_idxs = self.franka_actor_idxs_all_t[idxs]
        assert self.isaac_gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_states), gymtorch.unwrap_tensor(franka_idxs), len(franka_idxs)
        ), "Failed to set the franka state"
        assert self.isaac_gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_pos.contiguous()), gymtorch.unwrap_tensor(franka_idxs), len(franka_idxs)
        ), "Failed to set the franka targets"
        torque = torch.zeros_like(self.dof_pos)
        torque[idxs, 7:9] = -float(sim_config["robot"]["gripper_torque"])  # closing, as in simulate_step
        assert self.isaac_gym.set_dof_actuation_force_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(torque), gymtorch.unwrap_tensor(franka_idxs), len(franka_idxs)
        ), "Failed to set the gripper torque"
        self.last_grasp[idxs] = 1.0  # the gripper stays closed until the policy opens it

        # The one_leg reset push, without the table top and the held leg
        fm, tm = self.force_multiplier, self.torque_multiplier
        self.force_multiplier, self.torque_multiplier = fm * self.push_mask, tm * self.push_mask
        self._apply_forces_to_parts(idxs.int(), self.max_force_magnitude, self.max_torque_magnitude)
        self.force_multiplier, self.torque_multiplier = fm, tm

        return dict(
            leg_in_ee=(s["ee_rot"].transpose(1, 2), s["ee_pos"], s["leg_rot"], s["leg_pos"]),
            top_pos=to_sim(s["top_pos"]),
        )

    def _settle(self):
        for _ in range(self.settle_steps):
            self.isaac_gym.simulate(self.sim)
        self.isaac_gym.fetch_results(self.sim, True)
        self.isaac_gym.refresh_dof_state_tensor(self.sim)
        self.isaac_gym.refresh_dof_force_tensor(self.sim)
        self.isaac_gym.refresh_rigid_body_state_tensor(self.sim)
        self.isaac_gym.refresh_jacobian_tensors(self.sim)

    def _leg_in_hand(self, idxs):
        """Leg position (N, 3) and rotation (N, 3, 3) in the hand frame, now."""
        ee = self.rb_states[self.ee_rb_idxs[idxs]]
        leg = self.rb_states[self.part_rb_idxs[idxs, HELD_PART]]
        Ree = C.quaternion_to_matrix(ee[:, 3:7]).transpose(1, 2)
        return (Ree @ (leg[:, :3] - ee[:, :3])[..., None])[..., 0], Ree @ C.quaternion_to_matrix(leg[:, 3:7])

    def start_errors(self, idxs, want) -> Dict[str, torch.Tensor]:
        """How far the settled state is from the intended one, per env."""
        ee_rot_T, ee_pos, leg_rot, leg_pos = want["leg_in_ee"]
        want_p = (ee_rot_T @ (leg_pos - ee_pos)[..., None])[..., 0].float()
        want_R = (ee_rot_T @ leg_rot).float()
        p, R = self._leg_in_hand(idxs)
        slip_rot = C.matrix_to_axis_angle(R @ want_R.transpose(1, 2)).norm(dim=-1)
        top = self.rb_states[self.part_rb_idxs[idxs, BASE_PART], :3]
        return dict(
            slip=(p - want_p).norm(dim=-1),
            slip_rot=slip_rot,
            top_move=(top - want["top_pos"]).norm(dim=-1),
        )
