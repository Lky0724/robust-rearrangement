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
        self.track_safety = hasattr(self.env, "forces") and hasattr(
            self.env, "pairs_to_assemble"
        )
        if self.track_safety:
            # Franka joint torque limits (Nm) for joints 1-7
            self.torque_limits = torch.tensor(
                [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], device=self.device
            )
            # Part that gets inserted/screwed: second part of the first assembly pair
            self.tracked_part_idx = self.env.pairs_to_assemble[0][1]
        self.prev_part_quat = None

        # Fail an episode once its torque ratio exceeds this value before success.
        # The failure is latched: the env stays done and gets no success reward.
        self.terminate_torque_ratio = terminate_torque_ratio
        self.force_failed = torch.zeros(
            self.env.num_envs, device=self.device, dtype=torch.bool
        )
        self.succeeded = torch.zeros_like(self.force_failed)

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
        self.force_failed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.succeeded = torch.zeros_like(self.force_failed)
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

    def step(self, action: torch.Tensor):
        assert action.shape[1:] == self.action_space.shape

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

            if self.terminate_torque_ratio is not None:
                # Only count steps before success; a failed env never succeeds later
                self.force_failed |= (
                    torque_ratio > self.terminate_torque_ratio
                ) & ~self.succeeded
                self.succeeded |= termination & ~self.force_failed
                reward = torch.where(self.force_failed, 0.0, reward)
                termination = termination | self.force_failed
                info["force_failed"] = self.force_failed.clone()

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
