"""Check the random starts of task one_leg_insert_screw (src/gym/insert_screw_env.py).

1. Demos: the start parameters of the first frame of each demo in --demos (and --eval-demos),
   next to DEMO_RANGES. To refit the ranges to new demos, copy the min / max printed here
   into DEMO_RANGES.
2. Env: resets --n-envs envs --resets times and reports
   - why samples were drawn again and how many envs had to be reset again,
   - how far the leg slipped in the hand and the table top moved while settling,
   - the start parameters of the envs (from their observations) next to the sampled ranges,
     and how many eval demo starts fall inside those ranges.
3. Hold: after the last reset, commands the current hand pose with the gripper closed for
   --hold-steps steps and reports how much the leg slips in the hand meanwhile.

    python scripts/check_insert_screw_env.py --randomness med
"""

import argparse
import glob
import gzip
import lzma
import pickle
import time

import furniture_bench  # noqa: F401  (imports isaacgym, which must come before torch)
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from furniture_bench.config import config
from src.common.geometry import quat_xyzw_to_rot_6d
from src.gym import get_rl_env
from src.gym.insert_screw_env import DEMO_RANGES, decompose_starts

DEMOS = "data/raw/diffik/sim/one_leg_insert_screw/teleop/med/own51/success"
EVAL_DEMOS = "data/raw/diffik/sim/one_leg_insert_screw/teleop/med/own51_eval/success"
T_RT = torch.tensor(np.asarray(config["robot"]["tag_base_from_robot_base"]), dtype=torch.float64)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demos", default=DEMOS, help="Raw pickles whose first frames the ranges are fitted to")
    p.add_argument("--eval-demos", default=EVAL_DEMOS, help="Held-out demos to check the coverage of the ranges ('' to skip)")
    p.add_argument("--randomness", default="med", choices=["low", "med", "high"])
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--resets", type=int, default=4)
    p.add_argument("--hold-steps", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-env", action="store_true", help="Only print the demo parameters")
    return p.parse_args()


def _load(path):
    op = lzma.open if path.endswith(".xz") else gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        return pickle.load(f)


def _april_to_robot(poses):
    """(N, 7) AprilTag frame poses -> rotations (N, 3, 3) and positions (N, 3), robot frame."""
    poses = torch.as_tensor(poses, dtype=torch.float64).cpu()
    rot = torch.as_tensor(R.from_quat(poses[:, 3:].numpy()).as_matrix())
    return T_RT[:3, :3] @ rot, (T_RT[:3, :3] @ poses[:, :3, None])[..., 0] + T_RT[:3, 3]


def params_from_obs(ee_pos, ee_quat, parts_poses):
    """Start parameters from observations: robot_state ee_pos / ee_quat and parts_poses (N, 42)."""
    pp = torch.as_tensor(parts_poses, dtype=torch.float64).cpu().view(len(ee_pos), -1, 7)
    top_rot, top_pos = _april_to_robot(pp[:, 0])
    leg_rot, leg_pos = _april_to_robot(pp[:, 4])
    _, obstacle = _april_to_robot(pp[:, 5])
    ee_rot = torch.as_tensor(R.from_quat(torch.as_tensor(ee_quat).cpu().double().numpy()).as_matrix())
    ee_pos = torch.as_tensor(ee_pos, dtype=torch.float64).cpu()
    return {k: v.numpy() for k, v in decompose_starts(ee_pos, ee_rot, top_pos, top_rot, leg_pos, leg_rot, obstacle).items()}


def demo_params(folder):
    paths = sorted(p for p in glob.glob(f"{folder}/*") if p.endswith((".pkl", ".pkl.xz", ".pkl.gz")))
    if not paths:
        raise SystemExit(f"No raw pickles in {folder}")
    obs = [_load(p)["observations"][0] for p in paths]
    return params_from_obs(
        np.stack([o["robot_state"]["ee_pos"] for o in obs]),
        np.stack([o["robot_state"]["ee_quat"] for o in obs]),
        np.stack([np.asarray(o["parts_poses"]).reshape(-1) for o in obs]),
    )


def print_demo_params(name, params):
    print(f"\n{name}: {len(next(iter(params.values())))} demos, first frame (min / max)  vs DEMO_RANGES")
    for k, v in params.items():
        lo, hi = DEMO_RANGES[k]
        flag = "" if v.min() >= lo - 1e-4 and v.max() <= hi + 1e-4 else "   <- outside"
        print(f"  {k:16s} {v.min():9.4f} {v.max():9.4f}    [{lo:9.4f} {hi:9.4f}]{flag}")


def pct(x, qs=(50, 90, 100), scale=1000.0):
    return " / ".join(f"{v * scale:.2f}" for v in np.percentile(x, qs))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = demo_params(args.demos)
    print_demo_params(args.demos, train)
    held_out = demo_params(args.eval_demos) if args.eval_demos else None
    if held_out is not None:
        print_demo_params(args.eval_demos, held_out)
    if args.no_env:
        return

    env = get_rl_env(
        gpu_id=args.gpu, task="one_leg_insert_screw", num_envs=args.n_envs, randomness=args.randomness,
        observation_space="state", max_env_steps=5000, resize_img=False,
        act_rot_repr="rot_6d", action_type="pos", headless=True,
    )
    ranges = env.start_ranges

    starts, slip, slip_rot, top_move, times = [], [], [], [], []
    for i in range(args.resets):
        t = time.time()
        obs = env.reset()
        torch.cuda.synchronize()
        times.append(time.time() - t)
        rs = obs["robot_state"]
        starts.append(params_from_obs(rs["ee_pos"], rs["ee_quat"], obs["parts_poses"]))
        # Settling errors of the first placement of each env, before failed ones were reset again
        err = env.first_start_errors
        slip.append(err["slip"].cpu().numpy())
        slip_rot.append(np.degrees(err["slip_rot"].cpu().numpy()))
        top_move.append(err["top_move"].cpu().numpy())
    starts = {k: np.concatenate([s[k] for s in starts]) for k in starts[0]}
    n = len(starts["tip_dx"])

    print(f"\nEnv one_leg_insert_screw ({args.randomness}): {args.resets} resets of {args.n_envs} envs, "
          f"{np.mean(times[1:] or times):.2f} s per reset (first {times[0]:.2f} s)")
    stats = env.reset_stats
    drawn = stats["sample: drawn"]
    print(f"  samples drawn: {drawn}, drawn again because: "
          + ", ".join(f"{k[8:]} {v} ({v / max(drawn, 1):.1%})" for k, v in stats.items() if k.startswith("sample: ") and k != "sample: drawn"))
    not_held = stats["start didn't hold"]
    print(f"  envs reset: {stats['envs reset']}, start didn't hold (reset again): {not_held}")
    print(f"  while settling, first placement [median / p90 / max]: leg slip in hand {pct(np.concatenate(slip))} mm, "
          f"{pct(np.concatenate(slip_rot), scale=1)} deg; table top moved {pct(np.concatenate(top_move))} mm")

    print(f"\n  start parameters of the {n} env starts [min / p50 / max]  vs sampled range"
          + ("  | eval demos inside" if held_out is not None else ""))
    for k, v in starts.items():
        lo, hi = ranges[k]
        line = f"  {k:16s} {v.min():9.4f} {np.median(v):9.4f} {v.max():9.4f}    [{lo:9.4f} {hi:9.4f}]"
        if held_out is not None:
            h = held_out[k]
            line += f"  | {np.mean((h >= lo) & (h <= hi)):.0%}"
        print(line)
    if held_out is not None:
        inside = np.ones(len(held_out["tip_dx"]), dtype=bool)
        for k, (lo, hi) in ranges.items():
            inside &= (held_out[k] >= lo) & (held_out[k] <= hi)
        print(f"  eval demos inside all ranges: {inside.sum()}/{len(inside)}")

    # Hold the hand still with the gripper closed
    rs = obs["robot_state"]
    action = torch.cat([rs["ee_pos"], quat_xyzw_to_rot_6d(rs["ee_quat"]), torch.ones(args.n_envs, 1, device=env.device)], 1)
    idxs = torch.arange(args.n_envs, device=env.device)
    p0, R0 = env._leg_in_hand(idxs)
    max_slip = torch.zeros(args.n_envs, device=env.device)
    max_rot = torch.zeros(args.n_envs, device=env.device)
    assembled = torch.zeros(args.n_envs, dtype=torch.bool, device=env.device)
    for _ in range(args.hold_steps):
        _, reward, _, _ = env.step(action)
        assembled |= reward.view(-1) > 0
        p, Rm = env._leg_in_hand(idxs)
        max_slip = torch.maximum(max_slip, (p - p0).norm(dim=-1))
        cos = ((Rm @ R0.transpose(1, 2)).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
        max_rot = torch.maximum(max_rot, torch.rad2deg(torch.acos(cos.clamp(-1, 1))))
    print(f"\nHolding the hand still for {args.hold_steps} steps [median / p90 / max]: leg slip in hand "
          f"{pct(max_slip.cpu().numpy())} mm, {pct(max_rot.cpu().numpy(), scale=1)} deg; "
          f"envs counted as assembled: {int(assembled.sum())}")


if __name__ == "__main__":
    main()
