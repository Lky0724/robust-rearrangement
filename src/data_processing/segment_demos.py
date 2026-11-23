"""Cut one phase, or a run of phases, out of full one_leg demos.

    setup      start of the demo -> leg grasped (includes pushing the tabletop into the fixture)
    transport  leg grasped -> insertion start
    insert     insertion start -> first gripper release after it
    screw      that release -> end of the demo (regrasps, screwing and retreat)

`--phase insert --through screw` keeps insert and screw. The insertion start comes from
one of two places (--detect):

    poses  the leg pose relative to the tabletop: the last time the leg is farther than
           --start-radius from the hole it is released in. Needs the 42D parts_poses, so
           it works on sim data only
    marks  an insertion mark: a row of the --marks CSV, or else the first 1 in the
           episode's `skills` (the ` key in teleop_sm_fr3.py). Works on real demos
    auto   (default) a --marks row, else poses, else the recorded mark

Marks CSV, one row per episode, `name,step` where name is the pickle file name without its
extension and step the first step of the insertion (as shown by visualize_data.py video):

    2026-11-20T10:31:05,212

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
PICKLE_SUFFIXES = (".pkl", ".gz", ".xz")

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

    gripper = filtered_gripper(gripper_action)
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


def filtered_gripper(gripper_action: np.ndarray) -> np.ndarray:
    # Policies can flicker the gripper sign for a step or two around a toggle
    return median_filter(np.sign(gripper_action), size=5, mode="nearest")


def find_phases_from_mark(
    mark: int, gripper_action: np.ndarray
) -> Tuple[Optional[Dict[str, Tuple[int, int]]], Optional[str]]:
    """Phases around an insertion start `mark`. Returns (phases, None) or (None, reason)."""
    n_steps = len(gripper_action)
    if not 0 <= mark < n_steps:
        return None, f"mark {mark} is outside the {n_steps} steps"

    gripper = filtered_gripper(gripper_action)
    if gripper[mark] <= 0:
        return None, f"gripper is open at the mark (step {mark})"

    toggles = np.flatnonzero(np.diff(gripper)) + 1
    opens, closes = toggles[gripper[toggles] < 0], toggles[gripper[toggles] > 0]

    grasps = closes[closes <= mark]
    grasp = int(grasps[-1]) if len(grasps) else 0
    releases = opens[opens > mark]
    release = int(releases[0]) if len(releases) else n_steps

    return {
        "setup": (0, grasp),
        "transport": (grasp, mark),
        "insert": (mark, release),
        "screw": (release, n_steps),
    }, None


def episode_segment(
    args,
    task: str,
    parts_poses: Optional[np.ndarray],
    gripper_action: np.ndarray,
    skills: Optional[np.ndarray],
    csv_mark: Optional[int],
) -> Tuple[Optional[Tuple[int, int]], str]:
    """Return ((start, end), method) for the chosen phases with margins, or (None, reason)."""
    n_steps = len(gripper_action)
    has_poses = parts_poses is not None and parts_poses.shape[1:] == (N_PARTS_POSES,)

    use_poses = args.detect == "poses" or (
        args.detect == "auto" and csv_mark is None and has_poses
    )
    if use_poses:
        if task != "one_leg":
            return None, f"task is {task}, pose detection only knows one_leg"
        if not has_poses:
            return None, "no one_leg parts_poses"
        phases = find_phases(
            parts_poses,
            gripper_action,
            args.start_radius,
            args.near_lateral,
            args.near_axial,
        )
        method = "poses"
        if phases is None:
            return None, "no gripper release with the leg in a hole"
    else:
        if csv_mark is not None:
            mark, method = csv_mark, "csv mark"
        elif skills is not None and len(skills) == n_steps and np.any(skills > 0):
            mark, method = int(np.flatnonzero(skills > 0)[0]), "recorded mark"
        else:
            return None, "no insertion mark (no 1 in skills and no --marks row)"
        phases, reason = find_phases_from_mark(mark, gripper_action)
        if phases is None:
            return None, reason
        method = f"{method} at {mark}"

    start, end = phases[args.phase][0], phases[args.through][1]
    if start >= end:
        return None, f"{args.phase}..{args.through} is empty"
    return (max(start - args.pre_steps, 0), min(end + args.post_steps, n_steps)), method


def episode_key(name: str) -> str:
    """Pickle file name without directories or extensions, the key of a --marks row."""
    name = Path(str(name)).name
    while Path(name).suffix in PICKLE_SUFFIXES:
        name = Path(name).stem
    return name


def read_marks(path: Optional[str]) -> Dict[str, int]:
    if path is None:
        return {}
    marks = {}
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.split("#")[0].strip()
        if not line:
            continue
        fields = [f.strip() for f in line.split(",")]
        try:
            marks[episode_key(fields[0])] = int(fields[1])
        except (IndexError, ValueError):
            if i == 0:  # header
                continue
            raise SystemExit(f"{path}:{i + 1}: expected `name,step`, got {line!r}")
    print(f"Read {len(marks)} marks from {path}")
    return marks


def segment_params(args) -> dict:
    return {
        "phase": args.phase,
        "through": args.through,
        "detect": args.detect,
        "marks": args.marks,
        "start_radius": args.start_radius,
        "near_lateral": args.near_lateral,
        "near_axial": args.near_axial,
        "pre_steps": args.pre_steps,
        "post_steps": args.post_steps,
    }


def report(segments: List[Tuple[str, int, int, int, str]], skipped: List[Tuple[str, str]]):
    for name, start, end, n_steps, method in segments:
        print(f"  {name}: [{start}, {end}) = {end - start} of {n_steps} steps ({method})")
    for name, reason in skipped:
        print(f"  SKIPPED {name}: {reason}")

    print(f"Segmented {len(segments)} episodes, skipped {len(skipped)}")
    if not segments:
        raise SystemExit("No episode could be segmented, nothing was written")

    lengths = [end - start for _, start, end, _, _ in segments]
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

    marks = read_marks(args.marks)
    segments, skipped = [], []
    for path in paths:
        out_path = out_dir / path.name
        if out_path.exists() and not (args.overwrite or args.dry_run):
            raise SystemExit(f"{out_path} already exists, use --overwrite to replace it")

        data = unpickle_data(path)
        task = data.get("task", data.get("furniture"))
        n_steps = len(data["actions"])
        obs = data["observations"]

        parts_poses = None
        if np.size(obs[0].get("parts_poses", [])) == N_PARTS_POSES:
            parts_poses = np.array([o["parts_poses"] for o in obs[:n_steps]])
        skills = np.asarray(data["skills"]) if "skills" in data else None
        actions = np.array(data["actions"])
        segment, method = episode_segment(
            args, task, parts_poses, actions[:, -1], skills, marks.get(episode_key(path.name))
        )
        if segment is None:
            skipped.append((path.name, method))
            continue

        start, end = segment
        segments.append((path.name, start, end, n_steps, method))
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
            "method": method,
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

    # Real demos have no part poses; process_pickles stores an empty array for them
    parts_poses = None
    if src["parts_poses"].shape == (n_total, N_PARTS_POSES):
        parts_poses = src["parts_poses"][:]
    skills = src["skill"][:] if "skill" in src else None
    gripper_action = src["action/delta"][:, -1]
    names = src["pickle_file"][:]
    tasks = src["task"][:]
    marks = read_marks(args.marks)

    keep, ranges, segments, skipped = [], [], [], []
    ep_start = 0
    for i, ep_end in enumerate(episode_ends):
        ep_end = int(ep_end)
        name = f"{i} ({names[i]})"
        n_steps = ep_end - ep_start
        segment, method = episode_segment(
            args,
            tasks[i],
            None if parts_poses is None else parts_poses[ep_start:ep_end],
            gripper_action[ep_start:ep_end],
            None if skills is None else skills[ep_start:ep_end],
            marks.get(episode_key(names[i])),
        )

        if segment is None:
            skipped.append((name, method))
        else:
            start, end = segment
            keep.append(i)
            ranges.append((ep_start + start, ep_start + end))
            segments.append((name, start, end, n_steps, method))
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
        elif array.size == 0:  # e.g. the parts_poses of real demos
            new_shape = array.shape
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
        if array.size == 0:
            continue
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
        "start": [start for _, start, _, _, _ in segments],
        "end": [end for _, _, end, _, _ in segments],
        "method": [method for _, _, _, _, method in segments],
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
        "--through",
        type=str,
        choices=PHASES,
        default=None,
        help="Last phase to keep (default: --phase). --phase insert --through screw "
        "keeps insert and screw",
    )
    parser.add_argument(
        "--detect",
        type=str,
        choices=["auto", "poses", "marks"],
        default="auto",
        help="Where the insertion start comes from: part poses (sim) or insertion "
        "marks (real). auto: a --marks row, else poses, else the recorded mark",
    )
    parser.add_argument(
        "--marks",
        type=str,
        default=None,
        help="CSV of `name,step` insertion starts, overriding recorded marks",
    )
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
    if args.through is None:
        args.through = args.phase
    if PHASES.index(args.through) < PHASES.index(args.phase):
        parser.error(f"--through {args.through} comes before --phase {args.phase}")
    if args.marks and args.detect == "poses":
        parser.error("--marks has no effect with --detect poses")

    if args.source == "raw":
        segment_pickles(args)
    else:
        segment_zarr(args)
