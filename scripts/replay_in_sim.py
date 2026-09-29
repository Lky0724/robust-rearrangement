"""
Replay a recorded episode in the IsaacGym simulator: watch it in the viewer and/or record the
simulated cameras to mp4, and check whether the recorded actions still assemble the parts.

    # raw pickle (one episode)
    python scripts/replay_in_sim.py <episode.pkl> [--viewer] [--record out.mp4]
    # processed zarr (pick an episode)
    python scripts/replay_in_sim.py <success.zarr> --episode 3 [--viewer] [--record out.mp4]

How the start state is restored:
  .pkl   exact: robot joints + part poses from the first observation (env.reset_env_to). The
         recorded 8-D delta actions (pos + quat xyzw + gripper) are turned into absolute targets
         (recorded EE pose + delta), exactly as process_pickles.py does, and replayed.
  .zarr  approximate: zarr files don't store joint angles, so the arm is reset to its default joints,
         then driven to the recorded start EE pose with the controller (--settle-steps), and only
         then are the parts placed at their recorded start poses. Positions and gripper come from
         action/pos. Orientation targets come from the *next recorded state* by default
         (--zarr-rotation state), because the released zarrs' action rotations are scaled down
         by a processing bug (DATA_GUIDE §10); --zarr-rotation action replays them as stored.

Replay is open-loop, so small physics differences can accumulate; divergence from the recording is
itself a data-quality signal. Per-step EE tracking error against the recording is printed at the end.
"""

import argparse
from pathlib import Path

import furniture_bench  # noqa: F401  (imports isaacgym, which must come before torch)
import imageio
import numpy as np
import torch
import zarr
from PIL import Image

from src.common.geometry import (
    np_apply_quat_xyzw,
    np_quat_xyzw_to_rot_6d,
    np_rot_6d_to_quat_xyzw,
)
from src.gym import get_rl_env
from src.visualization.render_mp4 import unpickle_data


def load_episode(path: Path, episode: int, zarr_rotation: str = "state", lead: int = 1):
    """Return a dict with start state and absolute (T, 10) actions: pos + rot6d + gripper."""
    if path.suffix == ".zarr" or path.is_dir():
        z = zarr.open(str(path), mode="r")
        ends = z["episode_ends"][:].astype(int)
        s, e = (0 if episode == 0 else ends[episode - 1]), ends[episode]
        task = z["task"][episode] if "task" in z else z.attrs.get("task")
        actions = z["action/pos"][s:e].copy()  # (T, 10) absolute pos + rot6d + gripper
        if zarr_rotation == "state":
            # Orientation target at step t = recorded orientation at t+lead (end: hold final)
            rot = z["robot_state"][s:e, 3:9]
            actions[:, 3:9] = np.concatenate([rot[lead:], np.repeat(rot[-1:], lead, axis=0)], axis=0)
        return dict(
            kind="zarr",
            task=str(task),
            randomness=z.attrs.get("randomness", "low"),
            actions=actions,
            ee_pos=z["robot_state"][s:e, :3],
            start_ee_rot6d=z["robot_state"][s, 3:9],
            start_parts=z["parts_poses"][s],
            start_gripper=z["action/pos"][s, -1],
        )

    data = unpickle_data(path)
    obs0 = dict(data["observations"][0])
    rs = dict(obs0["robot_state"])
    # reset_env_to expects scalar finger positions; pickles store them as 1-element arrays.
    # Teleop pickles from the older env may lack them, so fall back to half the gripper width.
    half_width = float(np.ravel(rs["gripper_width"])[0]) / 2
    for k in ["gripper_finger_1_pos", "gripper_finger_2_pos"]:
        rs[k] = float(np.ravel(rs[k])[0]) if k in rs else half_width
    obs0["robot_state"] = rs

    # Absolute targets = recorded EE pose + delta (same as process_pickles.py, without its clipping).
    # Replaying deltas directly would apply them to the *simulated* pose and compound small errors.
    delta = np.asarray(data["actions"], dtype=np.float32)  # (T, 8) delta pos + quat xyzw + gripper
    obs = data["observations"][: len(delta)]
    ee_pos = np.array([np.ravel(o["robot_state"]["ee_pos"]) for o in obs], dtype=np.float32)
    ee_quat = np.array([np.ravel(o["robot_state"]["ee_quat"]) for o in obs], dtype=np.float32)
    abs_quat = np_apply_quat_xyzw(ee_quat, delta[:, 3:7])
    actions = np.concatenate(
        [ee_pos + delta[:, :3], np_quat_xyzw_to_rot_6d(abs_quat.astype(np.float32)), np.sign(delta[:, -1:])],
        axis=1,
    ).astype(np.float32)
    return dict(
        kind="pkl",
        task=data.get("task", data.get("furniture")),
        randomness=None,
        actions=actions,  # (T, 10) absolute pos + rot6d + gripper
        ee_pos=np.array([np.ravel(o["robot_state"]["ee_pos"]) for o in data["observations"]]),
        obs0=obs0,
    )


def to_frame(obs, height=240):
    """Side-by-side wrist | front RGB frame from an env observation (env 0)."""
    frames = []
    for key in ["color_image1", "color_image2"]:
        img = obs[key][0].cpu().numpy().astype(np.uint8)
        h, w = img.shape[:2]
        img = np.asarray(Image.fromarray(img).resize((int(w * height / h), height)))
        frames.append(img)
    return np.concatenate(frames, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Raw .pkl episode or processed .zarr")
    parser.add_argument("--episode", type=int, default=0, help="Episode index (zarr only)")
    parser.add_argument("--task", default=None, help="Override the task name stored in the data")
    parser.add_argument("--viewer", action="store_true", help="Open the IsaacGym viewer window")
    parser.add_argument("--record", type=Path, default=None, help="Save simulated wrist | front cameras to this mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=40, help="zarr only: steps to drive the arm to the start pose")
    parser.add_argument("--zarr-rotation", choices=["state", "action"], default="state",
                        help="zarr only: orientation targets from the next recorded state (default) or action/pos")
    parser.add_argument("--rotation-lead", type=int, default=1,
                        help="zarr + state rotation: aim at the recorded orientation this many steps ahead")
    parser.add_argument("--hold-steps", type=int, default=20, help="Repeat the last action this many extra steps")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    ep = load_episode(args.path, args.episode, args.zarr_rotation, args.rotation_lead)
    task = args.task or ep["task"]
    # Variant labels (e.g. one_leg_full) reuse the base task's simulator environment
    for base in ["one_leg", "round_table", "lamp", "mug_rack", "factory_peg_hole"]:
        if task.startswith(base):
            task = base
            break

    render = args.record is not None
    env = get_rl_env(
        gpu_id=args.gpu,
        task=task,
        num_envs=1,
        randomness=ep["randomness"] or "low",
        observation_space="image" if render else "state",
        resize_img=False,
        act_rot_repr="rot_6d",
        action_type="pos",
        headless=not args.viewer,
        max_env_steps=100_000,
    )
    device = env.device
    frames = []

    obs = env.reset()

    if ep["kind"] == "pkl":
        # Exact start state from the first recorded observation
        env.reset_to([ep["obs0"]])
        env.refresh()
    else:
        # 1. Drive the arm to the recorded start EE pose (gripper as at the start), parts at random
        quat = np_rot_6d_to_quat_xyzw(ep["start_ee_rot6d"][None])[0]
        rot6d = ep["start_ee_rot6d"]
        target = np.concatenate([ep["ee_pos"][0], rot6d, [ep["start_gripper"]]]).astype(np.float32)
        target_t = torch.tensor(target, device=device)[None]
        for _ in range(args.settle_steps):
            obs, *_ = env.step(target_t)
        # 2. Place the parts at their recorded start poses (AprilTag frame, as stored)
        env._reset_parts(0, ep["start_parts"])
        env.refresh()
        reached = env.get_ee_pose()[0][0].cpu().numpy()
        err = np.linalg.norm(reached - ep["ee_pos"][0]) * 100
        print(f"Start pose reached within {err:.2f} cm (quat {np.round(quat, 3)})")

    obs = env.get_observation()
    if render:
        frames.append(to_frame(obs))

    actions = torch.tensor(ep["actions"], device=device)
    n_rec = len(actions)
    if args.hold_steps > 0:  # let the controller converge on the final target
        actions = torch.cat([actions, actions[-1:].repeat(args.hold_steps, 1)])
    total_reward = 0.0
    sim_ee = []
    for t in range(len(actions)):
        obs, reward, done, _ = env.step(actions[t : t + 1])
        total_reward += float(reward.sum())
        sim_ee.append(env.get_ee_pose()[0][0].cpu().numpy())
        if render:
            frames.append(to_frame(obs))

    sim_ee = np.array(sim_ee)
    # Recorded EE positions are observations *before* each action; compare the simulated state
    # after step t with the recorded state at t+1 where available
    rec = ep["ee_pos"][1 : n_rec + 1]
    n = min(len(rec), n_rec)
    track = np.linalg.norm(sim_ee[:n] - rec[:n], axis=1) * 100

    print(f"\nReplayed {n_rec} steps (+{len(actions) - n_rec} hold) of {args.path.name}"
          f"{f' episode {args.episode}' if ep['kind'] == 'zarr' else ''} ({task})")
    print(f"  EE tracking vs. recording: mean {track.mean():.2f} cm, max {track.max():.2f} cm, "
          f"final {track[-1]:.2f} cm")
    print(f"  Reward collected: {total_reward:.0f}  "
          f"({'assembled ✓' if total_reward > 0 else 'NOT assembled ✗'})")

    if render:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.record, frames, fps=args.fps)
        print(f"  Saved {args.record} ({len(frames)} frames, simulated wrist | front)")


if __name__ == "__main__":
    main()
