# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from typing import Dict, Tuple
from src.dataset.normalizer import LinearNormalizer
from src.common.geometry import proprioceptive_quat_xyzw_to_rot_6d
import torch


from ipdb import set_trace as bp

# Set the gym logger to not print to console
from gymnasium import Env
import gymnasium as gym


class FurnitureEnvRLWrapper:

    def __init__(
        self,
        env: Env,
        max_env_steps=300,
        chunk_size=1,
        reset_on_success=False,
        normalize_reward=False,
        reward_clip=5.0,
        device="cuda",
    ):
        # super(FurnitureEnvWrapper, self).__init__(env)
        self.env = env
        self.chunk_size: int = chunk_size
        self.reset_on_success = reset_on_success
        self.device = device
        self.normalizer = LinearNormalizer()

        # Define a new action space
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(chunk_size, self.env.action_space.shape[-1])
        )

        robot_state_dim = self.env.observation_space["robot_state"].shape[-1]

        if robot_state_dim == 14:
            robot_state_dim = 16

        parts_poses_dim = self.env.observation_space["parts_poses"].shape[-1]

        self.observation_space = gym.spaces.Box(
            -float("inf"),
            float("inf"),
            shape=(robot_state_dim + parts_poses_dim,),
        )

        # Define the maximum number of steps in the environment
        self.max_env_steps = max_env_steps
        self.num_envs = self.env.num_envs

        self.reward_normalizer = (
            RunningMeanStdClip(shape=(1,), clip_value=reward_clip)
            if normalize_reward
            else None
        )

        self.no_rotation_or_gripper = torch.tensor(
            [[0, 0, 0, 1, -1]], device=device, dtype=torch.float32
        ).repeat(self.num_envs, 1)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def process_obs(self, obs: Dict[str, torch.Tensor]):
        # Robot state is [pos, ori_quat, pos_vel, ori_vel, gripper]
        robot_state = obs["robot_state"]

        # Parts poses is [pos, ori_quat] for each part
        parts_poses = obs["parts_poses"]

        # Make the robot state have 6D proprioception
        if robot_state.shape[-1] == 14:
            robot_state = proprioceptive_quat_xyzw_to_rot_6d(robot_state)

        robot_state = self.normalizer(robot_state, "robot_state", forward=True)
        parts_poses = self.normalizer(parts_poses, "parts_poses", forward=True)

        nobs = torch.cat([robot_state, parts_poses], dim=-1)

        # Clamp the observation to be bounded to [-5, 5]
        nobs = torch.clamp(nobs, -5, 5)

        return nobs

    def process_action(self, action: torch.Tensor):
        """
        Done any desired processing to the action before
        it is passed to the environment.
        """
        return action

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.process_obs(obs)

    def jerkinesss_penalty(self, action: torch.Tensor):
        # Get the current end-effector velocity
        ee_velocity = self.env.rb_states[self.env.ee_idxs, 7:10]

        # Calculate the dot product between the action and the end-effector velocity
        dot_product = torch.sum(action[..., :3] * ee_velocity, dim=1, keepdim=True)

        # Calculate the velocity-based penalty
        velocity_penalty = torch.where(dot_product < 0, -0.01, 0.0)

        # Add the velocity-based penalty to the rewards
        return velocity_penalty

    def _inner_step(self, action_chunk: torch.Tensor):
        total_reward = torch.zeros(action_chunk.shape[0], device=action_chunk.device)
        dones = torch.zeros(
            action_chunk.shape[0], dtype=torch.bool, device=action_chunk.device
        )
        for i in range(self.chunk_size):
            # The dimensions of the action_chunk are (num_envs, chunk_size, action_dim)
            obs, reward, done, info = self.env.step(action_chunk[:, i, :])
            total_reward += reward.squeeze()
            dones = dones | done.squeeze()

        return obs, total_reward, dones, info

    def step(
        self, naction_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        assert naction_chunk.shape[-2:] == self.action_space.shape

        # First denormalize the action
        action_chunk = self.normalizer(naction_chunk, "action", forward=False)

        # Move the robot
        obs, reward, done, info = self._inner_step(action_chunk)

        # Episodes that received reward are terminated
        terminated = reward > 0

        # Check if any envs have reached the max number of steps
        truncated = self.env.env_steps >= self.max_env_steps

        done = terminated | truncated

        if self.reward_normalizer is not None:
            reward = self.reward_normalizer(reward)

        obs = self.process_obs(obs)

        return obs, reward, done, info

    def increment_randomness(self):
        self.env.increment_randomness()

    @property
    def force_magnitude(self):
        return self.env.max_force_magnitude

    @property
    def torque_magnitude(self):
        return self.env.max_torque_magnitude


class RunningMeanStdClip:
    def __init__(self, epsilon=1e-4, shape=(), clip_value=10.0, device="cuda"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.clip_value = clip_value

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count

    def __call__(self, x):
        self.update(x)
        x_normalized = x / torch.sqrt(self.var + 1e-8)
        return torch.clamp(x_normalized, -self.clip_value, self.clip_value)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of xyzw quaternions."""
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of xyzw quaternions."""
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    v = aw * bv + bw * av + torch.cross(av, bv, dim=-1)
    w = aw * bw - (av * bv).sum(dim=-1, keepdim=True)
    return torch.cat([v, w], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v by xyzw quaternions q."""
    qv, qw = q[..., :3], q[..., 3:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


class RLPolicyEnvWrapper:

    def __init__(
        self,
        env: Env,
        max_env_steps=300,
        normalize_reward=False,
        reset_on_success=True,
        reset_on_failure=False,
        reward_clip=5.0,
        sample_perturbations=False,
        torque_penalty_coef=0.0,
        torque_threshold=0.5,
        terminate_torque_ratio=None,
        min_insert_twist_turns=None,
        min_held_twist_turns=None,
        max_insert_base_move_mm=None,
        device="cuda",
    ):
        self.env = env
        self.reset_on_success = reset_on_success
        self.reset_on_failure = reset_on_failure
        self.device = device
        self.reward_normalizer = (
            RunningMeanStdClip(shape=(1,), clip_value=reward_clip)
            if normalize_reward
            else None
        )
        self.sample_perturbations = sample_perturbations

        # Safety diagnostics and the optional excessive-force penalty.
        # The arm has gravity disabled, so measured joint torques only grow when the
        # robot pushes against something (e.g. forcing a part into place).
        self.torque_penalty_coef = torque_penalty_coef
        self.torque_threshold = torque_threshold
        self.track_safety = all(
            hasattr(self.env, attr)
            for attr in ("forces", "pairs_to_assemble", "get_ee_pose", "act_low")
        )
        if self.track_safety:
            # Franka joint torque limits (Nm) for joints 1-7
            self.torque_limits = torch.tensor(
                [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], device=self.device
            )
            # Part that gets inserted/screwed: second part of the first assembly pair,
            # and the part it goes into: the first one
            self.tracked_part_idx = self.env.pairs_to_assemble[0][1]
            self.base_part_idx = self.env.pairs_to_assemble[0][0]
            # Long axis of the tracked part in its mesh frame (square_table legs: local y)
            self.part_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self.prev_part_quat = None
        self.prev_rel_quat = None

        # Fail an episode once its torque ratio exceeds this value before success.
        # The failure is latched: the env stays done and gets no success reward.
        self.terminate_torque_ratio = terminate_torque_ratio
        self.force_failed = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.bool
        )
        self.succeeded = torch.zeros_like(self.force_failed)

        # Stricter success check: the tracked part must twist at least this many turns about
        # its long axis while within `insert_radius` of an assembled position. Otherwise the
        # assembled pose was reached by forcing the part in, and the episode fails (latched).
        self.min_insert_twist_turns = min_insert_twist_turns
        # Two more checks, latched the same way:
        # - min_held_twist_turns: as above, but only steps where the gripper holds the part count,
        #   so turning it by pushing with the fingertips doesn't
        # - max_insert_base_move_mm: the base part's peak horizontal move after the tracked part
        #   first comes within `insert_radius` must stay below this
        self.min_held_twist_turns = min_held_twist_turns
        self.max_insert_base_move_mm = max_insert_base_move_mm
        self.insert_radius = 0.03  # the threads span the last ~2 cm
        # Gripper width (m) that counts as holding the part: the square_table leg is ~3 cm wide,
        # the gripper closed on nothing ~0 and open 8 cm
        self.held_width = (0.02, 0.036)
        # The insertion diagnostics are logged whether or not the checks are on
        self.track_insertion = self.track_safety and all(
            hasattr(self.env, attr) for attr in ("assembled_rel_poses", "gripper_width")
        )
        checks = (min_insert_twist_turns, min_held_twist_turns, max_insert_base_move_mm)
        if any(c is not None for c in checks) and not self.track_insertion:
            raise ValueError("The insertion success checks need an env with part and EE states")
        if self.track_insertion:
            # Assembled positions of the tracked part in the base part's frame
            self.assembled_pos = self.env.assembled_rel_poses[0, :, :3, 3].to(self.device)
        self.insert_twist = torch.zeros(self.env.num_envs, device=self.device)
        self.twist_failed = torch.zeros_like(self.force_failed)
        self.held_twist = torch.zeros_like(self.insert_twist)
        self.held_twist_failed = torch.zeros_like(self.force_failed)
        # Base part's horizontal position when the tracked part first came within insert_radius
        # (NaN before), and its peak horizontal distance from there (m)
        self.base_anchor = torch.full(
            (self.env.num_envs, 2), float("nan"), device=self.device
        )
        self.peak_base_move = torch.zeros_like(self.insert_twist)
        self.base_move_failed = torch.zeros_like(self.force_failed)

        self.env_success = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.bool
        )

        # Define a new action space
        self.action_space = gym.spaces.Box(
            -1, 1, shape=(self.env.action_space.shape[-1],)
        )

        robot_state_dim = self.env.observation_space["robot_state"].shape[-1]

        if robot_state_dim == 14:
            robot_state_dim = 16

        parts_poses_dim = self.env.observation_space["parts_poses"].shape[-1]

        self.observation_space = gym.spaces.Box(
            -float("inf"),
            float("inf"),
            shape=(robot_state_dim + parts_poses_dim,),
        )

        # Define the maximum number of steps in the environment
        self.max_env_steps = max_env_steps
        self.num_envs = self.env.num_envs

    def reset(self, **kwargs):
        obs = self.env.reset()
        self.env_success = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # Re-read the tracked part's orientation on the first step after a reset
        self.prev_part_quat = None
        self.prev_rel_quat = None
        self.force_failed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.succeeded = torch.zeros_like(self.force_failed)
        self.insert_twist = torch.zeros(self.num_envs, device=self.device)
        self.twist_failed = torch.zeros_like(self.force_failed)
        self.held_twist = torch.zeros_like(self.insert_twist)
        self.held_twist_failed = torch.zeros_like(self.force_failed)
        self.base_anchor = torch.full((self.num_envs, 2), float("nan"), device=self.device)
        self.peak_base_move = torch.zeros_like(self.insert_twist)
        self.base_move_failed = torch.zeros_like(self.force_failed)
        return obs

    def _safety_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-env max joint torque as a fraction of its limit, and the tracked
        part's rotation (rad) since the previous step."""
        torque_ratio = (
            self.env.forces[:, :7].abs() / self.torque_limits
        ).max(dim=1).values

        part_quat = self.env.rb_states[self.env.furniture_rb_indices][
            :, self.tracked_part_idx, 3:7
        ].clone()
        if self.prev_part_quat is None:
            part_rotation = torch.zeros(self.num_envs, device=self.device)
        else:
            # Angle between orientations, 4 * atan2(|q1 - q2|, |q1 + q2|) with both
            # quats in the same hemisphere. Unlike 2 * acos(|q1.q2|), this stays
            # accurate near 0, so per-step float error doesn't add up over an episode.
            sign = torch.sign((part_quat * self.prev_part_quat).sum(dim=-1, keepdim=True))
            prev = self.prev_part_quat * torch.where(sign == 0, 1.0, sign)
            part_rotation = 4.0 * torch.atan2(
                (part_quat - prev).norm(dim=-1), (part_quat + prev).norm(dim=-1)
            )
        self.prev_part_quat = part_quat

        return torque_ratio, part_rotation

    def _part_twist(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Signed rotation (rad) of the tracked part about its long axis since the
        previous step, relative to the base part, and the tracked part's position
        in the base part's frame."""
        states = self.env.rb_states[self.env.furniture_rb_indices]
        part = states[:, self.tracked_part_idx]
        base = states[:, self.base_part_idx]
        base_quat_inv = _quat_conj(base[:, 3:7])
        rel_quat = _quat_mul(base_quat_inv, part[:, 3:7])
        rel_pos = _quat_rotate(base_quat_inv, part[:, :3] - base[:, :3])

        if self.prev_rel_quat is None:
            twist = torch.zeros(self.num_envs, device=self.device)
        else:
            # Rotation since the previous step, in the tracked part's own frame
            dq = _quat_mul(_quat_conj(self.prev_rel_quat), rel_quat)
            dq = dq * torch.where(dq[:, 3:4] < 0, -1.0, 1.0)
            # Twist component about the long axis (swing-twist decomposition).
            # Signed, so back-and-forth jitter cancels out when summed.
            twist = 2.0 * torch.atan2((dq[:, :3] * self.part_axis).sum(dim=-1), dq[:, 3])
        self.prev_rel_quat = rel_quat

        return twist, rel_pos

    def step(self, action: torch.Tensor):
        assert action.shape[1:] == self.action_space.shape

        if self.track_safety:
            # End-effector position the step's target is set from
            ee_pos_before = self.env.get_ee_pose()[0].clone()

        # Move the robot
        obs, reward, termination, info = self.env.step(
            action, sample_perturbations=self.sample_perturbations
        )
        reward = reward.squeeze()
        termination = termination.squeeze()

        if self.track_safety:
            torque_ratio, part_rotation = self._safety_stats()
            info["torque_ratio"] = torque_ratio
            info["part_rotation"] = part_rotation
            info["part_twist"], info["part_rel_pos"] = self._part_twist()
            # Target position (as clipped by the env) minus the end-effector position.
            # The arm's PD drive pushes toward it, so a large offset means a large force.
            target_pos = torch.clamp(
                action[:, :3], self.env.act_low[..., :3], self.env.act_high[..., :3]
            )
            info["ee_goal_offset"] = target_pos - ee_pos_before

            if self.terminate_torque_ratio is not None:
                # Only count steps before success; a failed env never succeeds later
                self.force_failed |= (
                    torque_ratio > self.terminate_torque_ratio
                ) & ~self.succeeded
                self.succeeded |= termination & ~self.force_failed
                reward = torch.where(self.force_failed, 0.0, reward)
                termination = termination | self.force_failed
                info["force_failed"] = self.force_failed.clone()

        if self.track_insertion:
            inserting = (
                (info["part_rel_pos"].unsqueeze(1) - self.assembled_pos)
                .norm(dim=-1)
                .min(dim=1)
                .values
                < self.insert_radius
            )
            self.insert_twist += torch.where(inserting, info["part_twist"], 0.0)
            width = self.env.gripper_width().view(-1)
            held = inserting & (width > self.held_width[0]) & (width < self.held_width[1])
            self.held_twist += torch.where(held, info["part_twist"], 0.0)
            base_xy = self.env.rb_states[self.env.furniture_rb_indices][
                :, self.base_part_idx, :2
            ]
            first_insert = inserting & self.base_anchor.isnan().any(dim=-1)
            self.base_anchor = torch.where(
                first_insert.unsqueeze(1), base_xy, self.base_anchor
            )
            self.peak_base_move = torch.maximum(
                self.peak_base_move,
                torch.nan_to_num((base_xy - self.base_anchor).norm(dim=-1), nan=0.0),
            )
            info["held_twist"] = self.held_twist.clone()
            info["peak_base_move"] = self.peak_base_move.clone()

            # Checked on the step the env reports the part as newly assembled (reward > 0).
            # The env stays done afterwards and pays that reward only once, so a failed
            # env gets no success reward. Twists by magnitude only, so the checks don't depend
            # on the threads' handedness or the axis' sign convention.
            assembled = reward > 0
            if self.min_insert_twist_turns is not None:
                self.twist_failed |= assembled & (
                    self.insert_twist.abs() < self.min_insert_twist_turns * 2 * torch.pi
                )
                info["twist_failed"] = self.twist_failed.clone()
            if self.min_held_twist_turns is not None:
                self.held_twist_failed |= assembled & (
                    self.held_twist.abs() < self.min_held_twist_turns * 2 * torch.pi
                )
                info["held_twist_failed"] = self.held_twist_failed.clone()
            if self.max_insert_base_move_mm is not None:
                self.base_move_failed |= assembled & (
                    self.peak_base_move > self.max_insert_base_move_mm / 1000
                )
                info["base_move_failed"] = self.base_move_failed.clone()
            reward = torch.where(
                self.twist_failed | self.held_twist_failed | self.base_move_failed,
                0.0,
                reward,
            )

        # Keep the task reward (0/1 per assembled pair, minus force failures) for success statistics
        info["task_reward"] = reward.clone()

        if self.track_safety and self.torque_penalty_coef > 0:
            reward = reward - self.torque_penalty_coef * torch.clamp(
                torque_ratio - self.torque_threshold, min=0.0
            )

        if self.reward_normalizer is not None:
            reward = self.reward_normalizer(reward)

        truncation = self.env.env_steps >= self.max_env_steps

        # Clip the obs
        obs["robot_state"] = torch.clamp(obs["robot_state"], -3, 3)
        obs["parts_poses"] = torch.clamp(obs["parts_poses"], -3, 3)

        return obs, reward, termination, truncation, info

    def increment_randomness(self):
        self.env.increment_randomness()

    @property
    def force_magnitude(self):
        return self.env.max_force_magnitude

    @property
    def torque_magnitude(self):
        return self.env.max_torque_magnitude
