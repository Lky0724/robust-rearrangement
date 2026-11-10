"""Cut one phase (by default the leg insertion) out of full one_leg demos.

The `skills` annotations are empty in the one_leg data, so the phases are detected
from the leg pose relative to the tabletop and the gripper action:

    setup      start of the demo -> leg grasped (includes pushing the tabletop into the fixture)
    transport  leg grasped -> leg within --start-radius of the hole it ends up in
    insert     end of transport -> first gripper release with the leg in a hole
    screw      that release -> end of the demo (regrasps, screwing and retreat)

Zarrs:
    python -m src.data_processing.segment_demos processed \
        data/processed/diffik/sim/one_leg/teleop/low/success.zarr
    writes data/processed/diffik/sim/one_leg/teleop/low/success/insert.zarr, which
    training reads with `data.suffix=insert`

Pickles:
    python -m src.data_processing.segment_demos raw \
        data/raw/diffik/sim/one_leg/teleop/low/success
    writes data/raw/diffik/sim/one_leg/teleop/low/insert/success/*.pkl, which
    process_pickles reads with `--suffix insert --output-suffix insert`

The two routes give the same data except for the rotation actions: process_pickles
scales the rotations of an episode by one factor that depends on the episode's length
(see clip_quat_xyzw_magnitude), so the actions of a segmented zarr keep the factor of
the full demo, and those of a segmented pickle get a larger one. Don't mix the routes.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as R

from src.visualization.render_mp4 import pickle_data, unpickle_data

PHASES = ("setup", "transport", "insert", "screw")

# Layout of the 42D parts_poses in one_leg: 6 x (pos + quat xyzw)
POSE_DIM = 7
TOP_IDX = 0
LEG_IDX = 4
N_PARTS_POSES = 42

# Seated leg positions in the tabletop frame (SquareTable.assembled_rel_poses).
# The tabletop's y axis is its normal, so y is the hole axis and x, z are lateral.
HOLES = np.array(
    [[x, 0.046875, z] for x in (-0.05625, 0.05625) for z in (-0.05625, 0.05625)]
)

OUTCOMES = ("success", "failure", "partial_success")


def leg_to_holes(parts_poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the leg's lateral and axial offset from each hole, both (T, 4), in meters."""
    top = parts_poses[:, TOP_IDX * POSE_DIM : (TOP_IDX + 1) * POSE_DIM]
    leg = parts_poses[:, LEG_IDX * POSE_DIM : (LEG_IDX + 1) * POSE_DIM]

    top_rot = R.from_quat(top[:, 3:]).as_matrix()

    # A few recorded tabletop quaternions are off by 180 deg about the local x axis,
    # which turns the tabletop upside down for a single step. World z points down.
    upside_down = top_rot[:, 2, 1] > 0
    top_rot[upside_down, :, 1:] *= -1

    leg_in_top = np.einsum("nji,nj->ni", top_rot, leg[:, :3] - top[:, :3])
    offset = leg_in_top[:, None] - HOLES
    return np.hypot(offset[..., 0], offset[..., 2]), offset[..., 1]


def find_phases(
    parts_poses: np.ndarray,
    gripper_action: np.ndarray,
    start_radius: float,
    near_lateral: float,
    near_axial: float,
) -> Optional[Dict[str, Tuple[int, int]]]:
    """Return {phase: (start, end)} with `end` exclusive, or None if no insertion is found."""
    n_steps = len(gripper_action)

    # Policies can flicker the gripper sign for a step or two around a toggle
    gripper = median_filter(np.sign(gripper_action), size=5, mode="nearest")
    toggles = np.flatnonzero(np.diff(gripper)) + 1
    opens, closes = toggles[gripper[toggles] < 0], toggles[gripper[toggles] > 0]

    lateral, axial = leg_to_holes(parts_poses)
    in_hole = (lateral < near_lateral) & (np.abs(axial) < near_axial)

    releases = opens[in_hole[opens].any(axis=1)]
    if len(releases) == 0:
        return None
    release = int(releases[0])
    hole = int(in_hole[release].argmax())

    grasps = closes[closes < release]
    if len(grasps) == 0:
        return None
    grasp = int(grasps[-1])

    outside = np.flatnonzero(lateral[grasp:release, hole] > start_radius)
    start = grasp + int(outside[-1]) + 1 if len(outside) else grasp

    return {
        "setup": (0, grasp),
        "transport": (grasp, start),
        "insert": (start, release),
        "screw": (release, n_steps),
    }


def find_segment(parts_poses, gripper_action, args) -> Optional[Tuple[int, int]]:
    """Return the (start, end) of the chosen phase with its margins, or None."""
    phases = find_phases(
        parts_poses,
        gripper_action,
        args.start_radius,
        args.near_lateral,
        args.near_axial,
    )
    if phases is None:
        return None
    start, end = phases[args.phase]
    return max(start - args.pre_steps, 0), min(end + args.post_steps, len(parts_poses))


def segment_params(args) -> dict:
    return {
        "phase": args.phase,
        "start_radius": args.start_radius,
        "near_lateral": args.near_lateral,
        "near_axial": args.near_axial,
        "pre_steps": args.pre_steps,
        "post_steps": args.post_steps,
    }


def report(segments: List[Tuple[str, int, int, int]], skipped: List[Tuple[str, str]]):
    for name, start, end, n_steps in segments:
        print(f"  {name}: [{start}, {end}) = {end - start} of {n_steps} steps")
    for name, reason in skipped:
        print(f"  SKIPPED {name}: {reason}")

    print(f"Segmented {len(segments)} episodes, skipped {len(skipped)}")
    if not segments:
        raise SystemExit("No episode could be segmented, nothing was written")

    lengths = [end - start for _, start, end, _ in segments]
    print(
        f"Segment length: min {min(lengths)}, median {int(np.median(lengths))}, "
        f"max {max(lengths)}, total {sum(lengths)} steps"
    )


def segment_pickles(args):
    in_dir = Path(args.input)
    paths = sorted(in_dir.glob("*.pkl*"))
    if not paths:
        raise SystemExit(f"No pickles found in {in_dir}")

    # <...>/<randomness>/<outcome> -> <...>/<randomness>/<suffix>/<outcome>
    # which is where get_raw_paths looks for a suffix
    out_dir = (
        Path(args.output)
        if args.output
        else in_dir.parent / args.output_suffix / in_dir.name
    )
    if out_dir.resolve() == in_dir.resolve():
        raise SystemExit("The output directory must differ from the input directory")
    print(f"Output directory: {out_dir}")

    segments, skipped = [], []
    for path in paths:
        out_path = out_dir / path.name
        if out_path.exists() and not (args.overwrite or args.dry_run):
            raise SystemExit(f"{out_path} already exists, use --overwrite to replace it")

        data = unpickle_data(path)
        task = data.get("task", data.get("furniture"))
        n_steps = len(data["actions"])
        obs = data["observations"]

        if task != "one_leg":
            skipped.append((path.name, f"task is {task}, not one_leg"))
            continue
        if np.size(obs[0].get("parts_poses", [])) != N_PARTS_POSES:
            skipped.append((path.name, "no one_leg parts_poses"))
            continue

        parts_poses = np.array([o["parts_poses"] for o in obs[:n_steps]])
        actions = np.array(data["actions"])
        segment = find_segment(parts_poses, actions[:, -1], args)
        if segment is None or segment[0] == segment[1]:
            skipped.append(
                (path.name, f"no {args.phase} phase found in {n_steps} steps")
            )
            continue

        start, end = segment
        segments.append((path.name, start, end, n_steps))
        if args.dry_run:
            continue

        # Simulator demos store one observation more than actions, process_pickles expects it
        data["observations"] = obs[start : end + len(obs) - n_steps]
        for key in ("actions", "rewards", "skills", "augment_states"):
            if key in data and len(data[key]) == n_steps:
                data[key] = data[key][start:end]
        data["segment"] = {
            "source": str(path),
            "start": start,
            "end": end,
            "source_steps": n_steps,
            **segment_params(args),
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        pickle_data(data, out_path)

    report(segments, skipped)


def copy_steps(src: zarr.Array, dst: zarr.Array, ranges: List[Tuple[int, int]]):
    """Copy the ranges from src to dst, writing whole chunks to avoid recompressing them."""
    chunk_len = dst.chunks[0]
    written = 0
    pending = []
    n_pending = 0

    for start, end in ranges:
        pending.append(src[start:end])
        n_pending += end - start
        if n_pending >= chunk_len:
            data = np.concatenate(pending)
            n_write = n_pending // chunk_len * chunk_len
            dst[written : written + n_write] = data[:n_write]
            written += n_write
            pending = [data[n_write:]]
            n_pending -= n_write

    if n_pending:
        dst[written : written + n_pending] = np.concatenate(pending)


def segment_zarr(args):
    in_path = Path(args.input)
    src = zarr.open(str(in_path), "r")

    # <outcome>.zarr -> <outcome>/<suffix>.zarr, which is where get_processed_paths looks
    # for a suffix. A zarr that already has a suffix gets a sibling instead.
    if args.output:
        out_path = Path(args.output)
    elif in_path.stem in OUTCOMES:
        out_path = in_path.with_suffix("") / f"{args.output_suffix}.zarr"
    else:
        out_path = in_path.with_name(f"{in_path.stem}_{args.output_suffix}.zarr")

    if out_path.resolve() == in_path.resolve():
        raise SystemExit("The output zarr must differ from the input zarr")
    if out_path.exists() and not (args.overwrite or args.dry_run):
        raise SystemExit(f"{out_path} already exists, use --overwrite to replace it")
    print(f"Output zarr: {out_path}")

    episode_ends = src["episode_ends"][:]
    n_total = int(episode_ends[-1])
    if src["parts_poses"].shape[1:] != (N_PARTS_POSES,):
        raise SystemExit(f"{in_path} has no one_leg parts_poses")

    parts_poses = src["parts_poses"][:]
    gripper_action = src["action/delta"][:, -1]
    names = src["pickle_file"][:]
    tasks = src["task"][:]

    keep, ranges, segments, skipped = [], [], [], []
    ep_start = 0
    for i, ep_end in enumerate(episode_ends):
        ep_end = int(ep_end)
        name = f"{i} ({names[i]})"
        segment = None
        if tasks[i] == "one_leg":
            segment = find_segment(
                parts_poses[ep_start:ep_end], gripper_action[ep_start:ep_end], args
            )
        n_steps = ep_end - ep_start

        if tasks[i] != "one_leg":
            skipped.append((name, f"task is {tasks[i]}, not one_leg"))
        elif segment is None or segment[0] == segment[1]:
            skipped.append((name, f"no {args.phase} phase found in {n_steps} steps"))
        else:
            start, end = segment
            keep.append(i)
            ranges.append((ep_start + start, ep_start + end))
            segments.append((name, start, end, n_steps))
        ep_start = ep_end

    report(segments, skipped)
    if args.dry_run:
        return

    lengths = [end - start for start, end in ranges]
    n_out = sum(lengths)

    arrays = []
    src.visitvalues(
        lambda value: arrays.append(value) if isinstance(value, zarr.Array) else None
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open(str(out_path), "w")
    for array in arrays:
        name = array.path
        if name == "episode_ends":
            continue

        if array.shape[0] == n_total:
            new_shape = (n_out,) + array.shape[1:]
        elif array.shape[0] == len(episode_ends):
            new_shape = (len(keep),) + array.shape[1:]
        else:
            raise SystemExit(f"Unexpected length of {name}: {array.shape}")

        # zarr stores the codec of an object array as its first filter
        object_codec, filters = None, array.filters
        if array.dtype == object:
            object_codec, *filters = array.filters

        dst = out.create_dataset(
            name,
            shape=new_shape,
            chunks=array.chunks,
            dtype=array.dtype,
            compressor=array.compressor,
            filters=filters,
            object_codec=object_codec,
        )
        print(f"Writing {name} {new_shape}")
        if array.shape[0] == n_total:
            copy_steps(array, dst, ranges)
        else:
            dst[:] = array.get_orthogonal_selection(np.array(keep))

    out.create_dataset(
        "episode_ends",
        data=np.cumsum(lengths),
        dtype=src["episode_ends"].dtype,
        chunks=src["episode_ends"].chunks,
    )

    out.attrs.update(src.attrs.asdict())
    out.attrs["n_episodes"] = len(keep)
    out.attrs["n_timesteps"] = n_out
    out.attrs["mean_episode_length"] = round(n_out / len(keep))
    out.attrs["segment"] = {
        "source": str(in_path),
        "source_episode": keep,
        "start": [start for _, start, _, _ in segments],
        "end": [end for _, _, end, _ in segments],
        "skipped": [name for name, _ in skipped],
        **segment_params(args),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "source",
        choices=["raw", "processed"],
        help="raw: a folder of pickles, processed: a zarr",
    )
    parser.add_argument("input", type=str, help="Folder of pickles, or a .zarr")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder or .zarr. By default it follows the data path grammar",
    )
    parser.add_argument("--output-suffix", type=str, default="insert")
    parser.add_argument("--phase", type=str, choices=PHASES, default="insert")
    parser.add_argument(
        "--start-radius",
        type=float,
        default=0.05,
        help="The insertion starts when the leg stays within this lateral distance "
        "of the hole [m]. A larger value keeps more of the transport",
    )
    parser.add_argument(
        "--near-lateral",
        type=float,
        default=0.02,
        help="Lateral distance from the hole that counts as inserted at release [m]",
    )
    parser.add_argument(
        "--near-axial",
        type=float,
        default=0.02,
        help="Height above the seated pose that counts as inserted at release [m]",
    )
    parser.add_argument(
        "--pre-steps", type=int, default=0, help="Extra steps to keep before the phase"
    )
    parser.add_argument(
        "--post-steps", type=int, default=0, help="Extra steps to keep after the phase"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the segments, write nothing"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.source == "raw":
        segment_pickles(args)
    else:
        segment_zarr(args)
