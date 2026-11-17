"""Make synthetic start states from recorded ones by shifting the scene under the robot.

A policy trained on a segment of a task (e.g. insert + screw, cut by
src/data_processing/segment_demos.py) can only be started from where its segment begins,
and the demos give just one such state each. This script makes more of them:

  1. Pick one of the recorded start states at random.
  2. Keep the robot (joints, gripper) and the part in its hand as they are, so the grasp
     stays one that really happened.
  3. Shift everything else (the base part, the other parts and the fixture) by the same random
     horizontal offset, at most --max-shift, with the fixture staying inside the area the
     recorded fixtures cover (plus --fixture-margin). This changes where the hand is relative
     to the target hole, which is what the segment has to cope with. Heights and the fixture's
     orientation are left unchanged, as they don't vary in the recorded states.
  4. Check each candidate in the simulator: after the reset, the robot holds its pose for
     --hold-steps steps with the gripper closed. A candidate is dropped if the reset moved the
     held part by more than --max-held-jump (it overlapped something), if the part slips in
     the hand by more than --max-slip, or if the parts that should stay put moved by more than
     --max-static-move.

The recorded states are checked the same way first, as a reference for the thresholds.
The output .npz is read by src/eval/start_states.load_start_states, so it can be passed
wherever a folder of start pickles is accepted:

    python scripts/gen_start_states.py \\
        --src data/raw/diffik/sim/one_leg/teleop/med/own51_insert_screw/success \\
        --out data/start_states/one_leg_med_own41_insert_screw_shift2cm.npz \\
        --n 4096 --max-shift 0.02

    python -m src.train.bc ... rollout.reset_from=<out.npz>
    python -m src.eval.evaluate_model ... --reset-from <out.npz>
    python -m src.train.residual_ppo ... env.reset_from=<out.npz>
"""

import argparse
from collections import Counter
from pathlib import Path

import furniture_bench  # noqa: F401  (imports isaacgym, which must come before torch)
import numpy as np
import torch

from furniture_bench.config import config
from src.common.geometry import quat_xyzw_to_rot_6d
from src.eval.start_states import (
    StartStates,
    load_start_states,
    reset_to_start_states,
    save_start_states,
)
from src.gym import get_rl_env


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Folder of raw pickles (or a .npz) with the recorded start states")
    p.add_argument("--out", required=True, help="Output .npz")
    p.add_argument("--n", type=int, default=4096, help="Number of start states to keep")
    p.add_argument("--max-shift", type=float, default=0.02, help="Largest horizontal shift of the scene [m]")
    p.add_argument("--fixture-margin", type=float, default=0.0,
                   help="How far the fixture may go beyond the area of the recorded fixtures [m]")
    p.add_argument("--task", default="one_leg")
    p.add_argument("--randomness", default="med")
    p.add_argument("--n-envs", type=int, default=256, help="Candidates checked in the simulator at once")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hold-steps", type=int, default=10)
    p.add_argument("--max-held-jump", type=float, default=0.01,
                   help="Largest move of the held part during the reset [m] (recorded states: up to ~8 mm)")
    p.add_argument("--max-slip", type=float, default=0.005, help="Largest slip of the held part in the hand [m]")
    p.add_argument("--max-static-move", type=float, default=0.003,
                   help="Largest move of any other part or the fixture [m]")
    p.add_argument("--max-batches", type=int, default=100, help="Give up after this many simulator batches")
    return p.parse_args()


def to_robot(poses_april: np.ndarray, A: np.ndarray) -> np.ndarray:
    """(N, P, 7) AprilTag-frame poses -> (N, P, 3) robot-frame positions."""
    return poses_april[..., :3] @ A[:3, :3].T + A[:3, 3]


def held_part(src: StartStates, A: np.ndarray) -> np.ndarray:
    """Index of the part in the hand for each state: the highest part (the others lie on the table)."""
    pos = to_robot(src.parts_poses.reshape(len(src), -1, 7), A)
    return pos[:, :-1, 2].argmax(1)  # the last pose is the fixture


def sample_candidates(src: StartStates, held: np.ndarray, A: np.ndarray, n: int, args, rng):
    """n candidates: a random recorded state with everything but the robot and its part shifted."""
    poses = src.parts_poses.reshape(len(src), -1, 7)
    fixture_xy = to_robot(poses, A)[:, -1, :2]
    lo = fixture_xy.min(0) - args.fixture_margin
    hi = fixture_xy.max(0) + args.fixture_margin

    source, shift = [], []
    while len(source) < n:
        i = rng.integers(len(src))
        r = args.max_shift * np.sqrt(rng.random())  # uniform over the disc
        a = rng.random() * 2 * np.pi
        d = np.array([r * np.cos(a), r * np.sin(a)])
        if np.all(fixture_xy[i] + d >= lo) and np.all(fixture_xy[i] + d <= hi):
            source.append(i)
            shift.append(d)
    source, shift = np.array(source), np.array(shift)

    new = poses[source].astype(np.float64)
    d_april = np.c_[shift, np.zeros(n)] @ A[:3, :3]  # robot-frame shift in the AprilTag frame
    static = np.ones(new.shape[:2], dtype=bool)
    static[np.arange(n), held[source]] = False
    new[..., :3] += np.where(static[..., None], d_april[:, None, :], 0.0)

    cand = StartStates(
        dof_pos=src.dof_pos[torch.as_tensor(source, device=src.dof_pos.device)].clone(),
        parts_poses=new.reshape(n, -1).astype(np.float32),
        gripper_closed=src.gripper_closed[torch.as_tensor(source, device=src.gripper_closed.device)].clone(),
        paths=[f"{src.paths[i]} shifted by ({d[0]:+.4f}, {d[1]:+.4f}) m" for i, d in zip(source, shift)],
    )
    return cand, source, shift


@torch.no_grad()
def check(env, states: StartStates, held: np.ndarray, args):
    """Reset to the states, hold for a while; return per-state metrics in metres."""
    n = env.num_envs
    assert len(states) == n
    A = env.april_to_robot_mat.float()
    rows = torch.arange(n, device=env.device)
    held_t = torch.as_tensor(held, device=env.device)

    def positions(obs):
        p = obs["parts_poses"].view(n, -1, 7)[..., :3]
        return p @ A[:3, :3].T + A[:3, 3]

    obs = reset_to_start_states(env, states, offset=0)
    want = torch.as_tensor(states.parts_poses, device=env.device).view(n, -1, 7)[..., :3]
    want = want @ A[:3, :3].T + A[:3, 3]
    after_reset = positions(obs)
    held_jump = (after_reset[rows, held_t] - want[rows, held_t]).norm(dim=-1)

    def in_hand(obs, pos):
        ee = obs["robot_state"]["ee_pos"]
        return pos[rows, held_t] - ee  # robot frame; the hand doesn't turn while holding

    rel0 = in_hand(obs, after_reset)
    ee0, q0 = obs["robot_state"]["ee_pos"].clone(), obs["robot_state"]["ee_quat"].clone()
    action = torch.cat([ee0, quat_xyzw_to_rot_6d(q0), torch.ones(n, 1, device=env.device)], dim=1)
    for _ in range(args.hold_steps):
        obs, _, _, _ = env.step(action)
    after_hold = positions(obs)
    slip = (in_hand(obs, after_hold) - rel0).norm(dim=-1)

    static = torch.ones(after_hold.shape[:2], dtype=torch.bool, device=env.device)
    static[rows, held_t] = False
    static_move = torch.where(static, (after_hold - want).norm(dim=-1), torch.zeros_like(slip[:, None]))
    static_move = static_move.max(dim=1).values

    return {
        "held_jump": held_jump.cpu().numpy(),
        "slip": slip.cpu().numpy(),
        "static_move": static_move.cpu().numpy(),
    }


def reasons(m, args):
    """Why each state fails, as a list of names (empty list = accepted)."""
    out = []
    for j in range(len(m["slip"])):
        r = []
        if m["held_jump"][j] > args.max_held_jump:
            r.append("held part moved at reset")
        if m["slip"][j] > args.max_slip:
            r.append("held part slipped")
        if m["static_move"][j] > args.max_static_move:
            r.append("other parts moved")
        out.append(r)
    return out


def summary(name, m):
    pct = lambda x: " / ".join(f"{v * 1000:.1f}" for v in np.percentile(x, [50, 90, 100]))  # noqa: E731
    print(f"{name}: [mm, median / p90 / max]  held part at reset {pct(m['held_jump'])}, "
          f"slip in hand {pct(m['slip'])}, other parts {pct(m['static_move'])}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    A = np.asarray(config["robot"]["tag_base_from_robot_base"], dtype=np.float64)

    src = load_start_states(args.src, device=f"cuda:{args.gpu}")
    held = held_part(src, A)
    print(f"Loaded {len(src)} recorded start states from {args.src}; held part index: {Counter(held.tolist())}")

    env = get_rl_env(
        gpu_id=args.gpu, task=args.task, num_envs=args.n_envs, randomness=args.randomness,
        observation_space="state", max_env_steps=5000, resize_img=False,
        act_rot_repr="rot_6d", action_type="pos", headless=True,
    )

    # Reference: the recorded states themselves (repeated to fill the envs)
    idx = np.arange(args.n_envs) % len(src)
    ref = StartStates(
        dof_pos=src.dof_pos[torch.as_tensor(idx, device=src.dof_pos.device)],
        parts_poses=src.parts_poses[idx],
        gripper_closed=src.gripper_closed[torch.as_tensor(idx, device=src.gripper_closed.device)],
        paths=[src.paths[i] for i in idx],
    )
    m = check(env, ref, held[idx], args)
    summary("Recorded states ", {k: v[: len(src)] for k, v in m.items()})
    ref_fail = sum(bool(r) for r in reasons({k: v[: len(src)] for k, v in m.items()}, args))
    print(f"  {ref_fail}/{len(src)} recorded states would fail the checks")

    kept = {"dof_pos": [], "parts_poses": [], "gripper_closed": [], "paths": [], "source": [], "shift": []}
    why = Counter()
    n_cand = 0
    for batch in range(args.max_batches):
        if len(kept["paths"]) >= args.n:
            break
        cand, source, shift = sample_candidates(src, held, A, args.n_envs, args, rng)
        m = check(env, cand, held[source], args)
        r = reasons(m, args)
        ok = np.array([not x for x in r])
        why.update(x for xs in r for x in xs)
        n_cand += len(ok)
        if batch == 0:
            summary("Shifted states  ", m)

        ok_t = torch.as_tensor(ok, device=cand.dof_pos.device)
        kept["dof_pos"].append(cand.dof_pos[ok_t])
        kept["parts_poses"].append(cand.parts_poses[ok])
        kept["gripper_closed"].append(cand.gripper_closed[ok_t])
        kept["paths"] += [p for p, o in zip(cand.paths, ok) if o]
        kept["source"].append(source[ok])
        kept["shift"].append(shift[ok])
        print(f"  batch {batch}: kept {ok.sum()}/{len(ok)}, total {len(kept['paths'])}/{args.n}")
    else:
        if len(kept["paths"]) < args.n:
            raise SystemExit(f"Only {len(kept['paths'])} states passed after {args.max_batches} batches; "
                             "loosen the thresholds or lower --max-shift")

    n = args.n
    out = StartStates(
        dof_pos=torch.cat(kept["dof_pos"])[:n],
        parts_poses=np.concatenate(kept["parts_poses"])[:n],
        gripper_closed=torch.cat(kept["gripper_closed"])[:n],
        paths=kept["paths"][:n],
    )
    source = np.concatenate(kept["source"])[:n]
    shift = np.concatenate(kept["shift"])[:n]
    meta = {
        "args": vars(args),
        "n_candidates": n_cand,
        "rejected": dict(why),
        "source_paths": src.paths,
    }
    save_start_states(out, args.out, meta=meta, source=source, shift=shift)

    print(f"\nSaved {n} start states to {args.out}")
    print(f"  candidates checked: {n_cand}, rejections: {dict(why) or 'none'}")
    print(f"  uses {len(np.unique(source))}/{len(src)} recorded states; "
          f"shift length median {np.median(np.linalg.norm(shift, axis=1)) * 100:.1f} cm, "
          f"max {np.linalg.norm(shift, axis=1).max() * 100:.1f} cm")


if __name__ == "__main__":
    main()
