"""
Check whether recorded robot motion stays inside another robot's joint limits, e.g. whether Panda
demos or policy rollouts would be feasible on a Franka Research 3 (FR3).

    python scripts/check_joint_limits.py <episode.pkl | folder of pickles> [--robot fr3] [--margin-deg 2]

Reads `robot_state/joint_positions` from raw pickles (teleop demos, `evaluate_model --save-rollouts`
output, coverage/distill rollouts). Processed zarrs don't store joint angles, so they can't be checked.

Joint limit sets (rad):
  panda            franka_panda.urdf used by the simulator
  fr3              FR3 datasheet position limits (also in docs/FR3_ORIGINAL_CODE_TELEOP_TEST.md)
  fr3_description  vendored franka_description/robots/fr3/joint_limits.yaml (more conservative)
"""

import argparse
import gzip
import lzma
import pickle
from pathlib import Path

import numpy as np

LIMITS = {
    "panda": (
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    ),
    "fr3": (
        [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
        [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159],
    ),
    "fr3_description": (
        [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895],
        [2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895],
    ),
}


def unpickle(path: Path):
    opener = {".gz": gzip.open, ".xz": lzma.open}.get(path.suffix, open)
    with opener(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="A raw .pkl episode or a folder of pickles (searched recursively)")
    parser.add_argument("--robot", choices=list(LIMITS), default="fr3", help="Joint limits to check against")
    parser.add_argument("--margin-deg", type=float, default=0.0,
                        help="Also flag joints within this many degrees of a limit")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    files = [args.data] if args.data.is_file() else sorted(
        p for p in args.data.rglob("*") if p.name.endswith((".pkl", ".pkl.gz", ".pkl.xz"))
    )
    files = files[: args.max_files] if args.max_files else files
    if not files:
        raise SystemExit(f"No pickles found at {args.data}")

    lo, hi = (np.array(x) for x in LIMITS[args.robot])
    margin = np.radians(args.margin_deg)
    q_min, q_max = np.full(7, np.inf), np.full(7, -np.inf)
    bad_eps, bad_steps, total_steps = [], np.zeros(7, int), 0

    for f in files:
        obs = unpickle(f)["observations"]
        q = np.array([np.ravel(o["robot_state"]["joint_positions"])[:7] for o in obs])
        q_min, q_max = np.minimum(q_min, q.min(0)), np.maximum(q_max, q.max(0))
        out = (q < lo + margin) | (q > hi - margin)
        bad_steps += out.sum(0)
        total_steps += len(q)
        if out.any():
            bad_eps.append((f.name, sorted(set(np.where(out)[1] + 1))))

    print(f"Checked {len(files)} episode(s), {total_steps} steps, against '{args.robot}' limits"
          + (f" with a {args.margin_deg:g}° margin" if args.margin_deg else ""))
    print(f"\n{'joint':>5} {'limit lo':>9} {'limit hi':>9} {'data min':>9} {'data max':>9} {'steps out':>10}")
    for j in range(7):
        flag = "  ✗" if bad_steps[j] else ""
        print(f"{j + 1:>5} {lo[j]:>9.3f} {hi[j]:>9.3f} {q_min[j]:>9.3f} {q_max[j]:>9.3f} {bad_steps[j]:>10}{flag}")
    print(f"\nEpisodes with any joint outside the limits: {len(bad_eps)} / {len(files)}")
    for name, joints in bad_eps[:20]:
        print(f"  {name}: joint(s) {joints}")
    if len(bad_eps) > 20:
        print(f"  … and {len(bad_eps) - 20} more")


if __name__ == "__main__":
    main()
