"""Start evaluation episodes from recorded states instead of the env's own reset.

A policy trained on one segment of a task (e.g. insert + screw, cut out by
src/data_processing/segment_demos.py) has never seen the task's initial states, so it
has to be evaluated from states where its segment begins. This module takes the first
observation of each raw pickle in a folder and puts the FurnitureRLSimEnv into those
states after its normal reset:

    states = load_start_states("data/raw/diffik/sim/one_leg/teleop/med/own51_insert_screw/success")
    obs = reset_to_start_states(env, states, offset=0)

Env i of a round gets state (offset + i) % n_states, so with offset = round * num_envs
every state is used about equally often.

The env's own reset_env_to() can't be used for this: it resets one env at a time and
needs a physics step after each call, so the envs reset first would be simulated for
hundreds of steps before the episode starts. Here all envs are written in one call per
tensor, followed by a single physics step, as FurnitureRLSimEnv.reset() does.

The raw pickles are needed because the processed zarrs don't store joint positions.
load_start_states also reads the .npz files written by save_start_states, e.g. the
synthetic start states made by scripts/gen_start_states.py.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from isaacgym import gymtorch
from scipy.spatial.transform import Rotation as R

from furniture_bench.sim_config import sim_config
from src.visualization.render_mp4 import unpickle_data


@dataclass
class StartStates:
    dof_pos: torch.Tensor  # (N, 9): 7 arm joints + 2 gripper fingers
    parts_poses: np.ndarray  # (N, 42): parts and obstacle, AprilTag frame, as observed
    gripper_closed: torch.Tensor  # (N,) bool, from the gripper command of the first action
    paths: List[str]

    def __len__(self) -> int:
        return len(self.paths)


def load_start_states(path: Union[str, Path], device="cuda") -> StartStates:
    """Load start states from a .npz written by save_start_states, or from a folder of raw
    pickles (.pkl, .pkl.xz, .pkl.gz), taking the first observation of each."""
    path = Path(path)
    if path.suffix == ".npz":
        return _load_npz(path, device)

    folder = path
    paths = sorted(
        p for p in folder.iterdir() if p.name.endswith((".pkl", ".pkl.xz", ".pkl.gz"))
    )
    if not paths:
        raise FileNotFoundError(f"No .pkl, .pkl.xz or .pkl.gz files in {folder}")

    dof_pos, parts_poses, gripper_closed = [], [], []
    for path in paths:
        data = unpickle_data(path)
        obs0: Dict = data["observations"][0]
        rs = obs0["robot_state"]
        dof_pos.append(
            np.concatenate(
                [
                    np.asarray(rs["joint_positions"], dtype=np.float32).reshape(7),
                    np.asarray(
                        [rs["gripper_finger_1_pos"], rs["gripper_finger_2_pos"]],
                        dtype=np.float32,
                    ).reshape(2),
                ]
            )
        )
        parts_poses.append(np.asarray(obs0["parts_poses"], dtype=np.float32).reshape(-1))
        # The env closes the gripper for a positive command (FurnitureRLSimEnv.simulate_step)
        gripper_closed.append(float(np.asarray(data["actions"][0])[-1]) > 0)

    return StartStates(
        dof_pos=torch.from_numpy(np.stack(dof_pos)).to(device),
        parts_poses=np.stack(parts_poses),
        gripper_closed=torch.tensor(gripper_closed, device=device),
        paths=[str(p) for p in paths],
    )


def save_start_states(
    states: StartStates, path: Union[str, Path], meta: Optional[Dict] = None, **arrays
) -> None:
    """Save start states to a .npz that load_start_states reads. Extra arrays (e.g. how each
    state was made) and a JSON-serializable meta dict are stored alongside for reference."""
    path = Path(path)
    assert path.suffix == ".npz", f"Expected a .npz path, got {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dof_pos=states.dof_pos.cpu().numpy().astype(np.float32),
        parts_poses=np.asarray(states.parts_poses, dtype=np.float32),
        gripper_closed=states.gripper_closed.cpu().numpy().astype(bool),
        paths=np.asarray(states.paths),
        meta=np.asarray(json.dumps(meta or {})),
        **arrays,
    )


def _load_npz(path: Path, device) -> StartStates:
    data = np.load(path)
    return StartStates(
        dof_pos=torch.from_numpy(data["dof_pos"].astype(np.float32)).to(device),
        parts_poses=data["parts_poses"].astype(np.float32),
        gripper_closed=torch.from_numpy(data["gripper_closed"].astype(bool)).to(device),
        paths=[str(p) for p in data["paths"]],
    )


@torch.no_grad()
def reset_to_start_states(env, states: StartStates, offset: int = 0) -> Dict:
    """Reset all envs, then move each into a recorded state. Returns the new observation."""
    env.reset()

    n_envs = env.num_envs
    idx = (torch.arange(n_envs) + offset) % len(states)
    dof_pos = states.dof_pos[idx.to(states.dof_pos.device)].to(env.device)
    closed = states.gripper_closed[idx.to(states.gripper_closed.device)].to(env.device)
    env_idxs = torch.arange(n_envs, device=env.device)

    # Robot: joint positions with zero velocity, and the drive targets at the same pose
    env.dof_pos[:, :] = dof_pos
    env.dof_vel[:, :] = 0.0
    franka_idxs = env.franka_actor_idxs_all_t[env_idxs].reshape(-1, 1)
    assert env.isaac_gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_states),
        gymtorch.unwrap_tensor(franka_idxs),
        len(franka_idxs),
    ), "Failed to set franka state"

    # Gripper drive as FurnitureRLSimEnv.simulate_step sets it for the same command,
    # so a grasped part stays squeezed during the settling step below
    pos_target = dof_pos.clone()
    pos_target[:, 7:9] = torch.where(
        closed.unsqueeze(1),
        torch.zeros_like(pos_target[:, 7:9]),
        torch.full_like(pos_target[:, 7:9], env.max_gripper_width / 2),
    )
    sign = closed.float() * 2 - 1  # +1 closed, -1 open, the command convention of the env
    torque = torch.zeros_like(dof_pos)
    torque[:, 7:9] = -float(sim_config["robot"]["gripper_torque"]) * sign.unsqueeze(1)
    env.isaac_gym.set_dof_position_target_tensor(
        env.sim, gymtorch.unwrap_tensor(pos_target.contiguous())
    )
    env.isaac_gym.set_dof_actuation_force_tensor(
        env.sim, gymtorch.unwrap_tensor(torque.contiguous())
    )
    # Keep the gripper in this state until the policy's command changes sign
    env.last_grasp = sign

    # Parts and obstacles. The recorded poses are in the AprilTag frame, the root state is in
    # the sim frame. The env's own _reset_parts can't be reused: it indexes the obstacles by
    # their sim-wide rigid body indices instead of their actor index within the env, which
    # goes out of range (a CUDA device-side assert) as soon as there is more than one env.
    n_parts = len(env.furniture.parts)
    poses = states.parts_poses[idx.numpy()].reshape(n_envs, -1, 7).astype(np.float64)
    assert poses.shape[1] == n_parts + 1, (
        f"Expected {n_parts} parts + the obstacle in parts_poses, got {poses.shape[1]} poses"
    )
    april_to_sim = np.asarray(env.april_to_sim_mat, dtype=np.float64)
    pos_sim = poses[..., :3] @ april_to_sim[:3, :3].T + april_to_sim[:3, 3]
    rot_sim = april_to_sim[:3, :3] @ R.from_quat(poses[..., 3:].reshape(-1, 4)).as_matrix()
    quat_sim = R.from_matrix(rot_sim).as_quat().reshape(n_envs, -1, 4)  # xyzw, like gymapi

    to_t = lambda a: torch.tensor(a, device=env.device, dtype=torch.float32)  # noqa: E731
    parts_idx = env.parts_idx_list.long()  # actor index within an env, furniture.parts order
    env.root_pos[:, parts_idx] = to_t(pos_sim[:, :n_parts])
    env.root_quat[:, parts_idx] = to_t(quat_sim[:, :n_parts])

    # The obstacle pose in parts_poses is the front piece; the right and left pieces keep
    # their offsets to it, as in the env's reset
    front = to_t(pos_sim[:, n_parts])
    initial = env.obstacle_initial_pos  # (3, 3): front, right, left
    env.root_pos[:, env.obstacles_idx_list.long()] = initial.unsqueeze(0) + (
        front - initial[0]
    ).unsqueeze(1)

    actor_idxs = env.actor_idx_by_env[env_idxs].view(-1)
    env.root_tensor[actor_idxs.long(), 7:13] = 0.0  # zero velocities
    assert env.isaac_gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_tensor),
        gymtorch.unwrap_tensor(actor_idxs),
        len(actor_idxs),
    ), "Failed to set part states"

    # One physics step applies the new state and refreshes the observation tensors
    env.refresh()
    env.env_steps[env_idxs] = 0
    env.already_assembled[env_idxs] = 0

    return env.get_observation()
