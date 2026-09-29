"""
Inspect recorded demo data as it was recorded: summary, camera videos, and trajectory plots.
Works on processed zarrs and on raw pickles (one .pkl, or a folder of them).

    python scripts/visualize_data.py info  <data>
    python scripts/visualize_data.py video <data> --episodes 0 1 2 [--fps 30] [--out-dir DIR]
    python scripts/visualize_data.py plot  <data> [--episodes 0 1 2] [--out-dir DIR]

<data> is a .zarr directory, a .pkl/.pkl.gz/.pkl.xz file, or a folder of pickles (e.g. raw/.../success/).
For a folder, episodes are the files in sorted order; --max-files limits how many are loaded.

To watch an episode re-simulated in IsaacGym instead, use scripts/replay_in_sim.py.
"""

import argparse
import gzip
import lzma
import pickle
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from PIL import Image, ImageDraw

# Part names for parts_poses (6 × 7 = 42 values) in one_leg / square_table
ONE_LEG_PARTS = ["table_top", "leg1", "leg2", "leg3", "leg4", "fixture"]
PICKLE_SUFFIXES = (".pkl", ".gz", ".xz")


# ---------------------------------------------------------------------------------------------
# Loading: both formats become one flat, zarr-like dataset (episodes concatenated along time)
# ---------------------------------------------------------------------------------------------


def unpickle(path: Path):
    opener = {".gz": gzip.open, ".xz": lzma.open}.get(path.suffix, open)
    with opener(path, "rb") as f:
        return pickle.load(f)


def load_pickle_episode(path: Path):
    """One raw episode → arrays with the same meaning as the zarr fields used here."""
    data = unpickle(path)
    actions = np.asarray(data["actions"], dtype=np.float32)  # (T, 8) delta pos + quat + gripper
    obs = data["observations"][: len(actions)]  # sim stores one extra final observation
    rs = [o["robot_state"] for o in obs]
    ee_pos = np.array([np.ravel(r["ee_pos"]) for r in rs], dtype=np.float32)
    return dict(
        ee_pos=ee_pos,
        gripper_width=np.array([np.ravel(r["gripper_width"])[0] for r in rs], dtype=np.float32),
        # Absolute position target = current EE position + delta (as process_pickles.py does)
        act_pos=ee_pos + actions[:, :3],
        act_grip=np.sign(actions[:, -1]),
        parts=np.array([o["parts_poses"] for o in obs], dtype=np.float32) if "parts_poses" in obs[0] else None,
        img1=np.array([o["color_image1"] for o in obs]),
        img2=np.array([o["color_image2"] for o in obs]),
        success=bool(data.get("success", False)),
    )


class Dataset:
    """Uniform view over a zarr or over raw pickles. Images are read lazily per episode."""

    def __init__(self, path: Path, max_files=None):
        self.path = path
        if path.suffix == ".zarr" or (path.is_dir() and (path / ".zgroup").exists()):
            self.kind = "zarr"
            z = zarr.open(str(path), mode="r")
            self.z = z
            self.attrs = dict(z.attrs)
            self.ends = z["episode_ends"][:].astype(int)
            self.starts = np.concatenate([[0], self.ends[:-1]])
            self.ee_pos = z["robot_state"][:, :3]
            self.gripper_width = z["robot_state"][:, -1]
            act = z["action/pos"][:]
            self.act_pos, self.act_grip = act[:, :3], act[:, -1]
            self.parts = z["parts_poses"][:]
            self.success = z["success"][:].astype(bool)
            self.name = path.parent.name if path.name == "success.zarr" else path.stem
        else:
            self.kind = "pkl"
            files = [path] if path.is_file() else sorted(
                p for p in path.iterdir() if p.name.endswith(PICKLE_SUFFIXES)
            )
            if max_files:
                files = files[:max_files]
            if not files:
                raise SystemExit(f"No pickles found at {path}")
            self.files = files
            eps = [load_pickle_episode(f) for f in files]
            self.eps = eps
            self.attrs = {"n_files": len(files), "source": str(path)}
            lengths = np.array([len(e["ee_pos"]) for e in eps])
            self.ends = np.cumsum(lengths)
            self.starts = np.concatenate([[0], self.ends[:-1]])
            cat = lambda k: np.concatenate([e[k] for e in eps])
            self.ee_pos, self.gripper_width = cat("ee_pos"), cat("gripper_width")
            self.act_pos, self.act_grip = cat("act_pos"), cat("act_grip")
            self.parts = cat("parts") if eps[0]["parts"] is not None else None
            self.success = np.array([e["success"] for e in eps])
            self.name = path.stem if path.is_file() else f"{path.parent.name}_{path.name}"

    def __len__(self):
        return len(self.ends)

    def images(self, ep):
        if self.kind == "zarr":
            s, e = self.starts[ep], self.ends[ep]
            return self.z["color_image1"][s:e], self.z["color_image2"][s:e]  # only these chunks
        return self.eps[ep]["img1"], self.eps[ep]["img2"]


# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------


def cmd_info(ds: Dataset, args):
    print(f"== {ds.path} ({ds.kind})")
    for k, v in ds.attrs.items():
        print(f"   {k}: {v}")

    if ds.kind == "zarr":
        print("\n== arrays")

        def walk(g, prefix=""):
            for name, item in g.items():
                if isinstance(item, zarr.hierarchy.Group):
                    walk(item, prefix + name + "/")
                else:
                    print(f"   {prefix + name:16s} {str(item.shape):24s} {item.dtype}")

        walk(ds.z)
    else:
        e0 = ds.eps[0]
        print(f"\n== first pickle: {ds.files[0].name}")
        print(f"   images: wrist {e0['img1'].shape[1:]}, front {e0['img2'].shape[1:]}")

    lengths = ds.ends - ds.starts
    print(f"\n== episodes: {len(ds)}, steps: {ds.ends[-1]}")
    print(f"   length min/mean/max: {lengths.min()} / {lengths.mean():.0f} / {lengths.max()}")
    print(f"   success: {int(ds.success.sum())}/{len(ds)}")

    print("\n== sanity checks")
    problems = 0
    for key in ["ee_pos", "act_pos", "parts"]:
        x = getattr(ds, key)
        if x is not None and x.size and not np.isfinite(x).all():
            print(f"   ✗ {key} has {(~np.isfinite(x)).sum()} non-finite values")
            problems += 1
    grip = np.unique(ds.act_grip)
    if not np.isin(grip, [-1, 1]).all():
        print(f"   ✗ gripper action has values other than ±1: {grip[:10]}")
        problems += 1
    gap = np.linalg.norm(ds.act_pos - ds.ee_pos, axis=1)
    print(f"   action-to-state position gap: mean {gap.mean() * 100:.2f} cm, max {gap.max() * 100:.2f} cm"
          + (" (zarr: each axis clipped to ±2.5 cm, so ≤ 4.3 cm)" if ds.kind == "zarr" else ""))
    if ds.kind == "zarr" and np.isnan(ds.z["reward"][:]).any():
        print("   (note) reward has NaNs; it is unused for teleop data (DATA_GUIDE §10)")
    print("   ✓ no problems found" if problems == 0 else f"   {problems} problem(s)")


def label(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(img)


def same_height(a, b):
    """Resize image b to a's height (raw real-robot cameras can differ in size)."""
    if a.shape[0] == b.shape[0]:
        return b
    w = int(b.shape[1] * a.shape[0] / b.shape[0])
    return np.asarray(Image.fromarray(b).resize((w, a.shape[0])))


def cmd_video(ds: Dataset, args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in args.episodes:
        wrist, front = ds.images(ep)
        if wrist.shape[1] < 16:
            print(f"Episode {ep}: images are placeholders (state-only rollout); nothing to show. "
                  "Use `plot`, or scripts/replay_in_sim.py to re-render it in the simulator.")
            continue
        s = ds.starts[ep]
        n = len(wrist)
        frames = [
            label(
                np.concatenate([wrist[t], same_height(wrist[t], front[t])], axis=1),
                f"ep {ep}  step {t}/{n}  grip {'close' if ds.act_grip[s + t] > 0 else 'open '}  "
                f"width {ds.gripper_width[s + t] * 100:.1f} cm",
            )
            for t in range(n)
        ]
        path = out_dir / f"{ds.name}_ep{ep:03d}.mp4"
        imageio.mimsave(path, frames, fps=args.fps)
        print(f"Saved {path} ({n} frames, wrist | front)")


def cmd_plot(ds: Dataset, args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-episode time series: EE position (state vs commanded), gripper
    for ep in args.episodes:
        s, e = ds.starts[ep], ds.ends[ep]
        t = np.arange(e - s)
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        for i, ax_name in enumerate("xyz"):
            axes[i].plot(t, ds.ee_pos[s:e, i], label="EE state")
            axes[i].plot(t, ds.act_pos[s:e, i], "--", label="action (target)")
            axes[i].set_ylabel(f"{ax_name} [m]")
        axes[0].legend(loc="upper right")
        axes[3].plot(t, ds.gripper_width[s:e] * 100, label="gripper width [cm]")
        axes[3].step(t, ds.act_grip[s:e], where="post", label="gripper action (±1)")
        axes[3].legend(loc="upper right")
        axes[3].set_xlabel("step (10 Hz)")
        fig.suptitle(f"{ds.name} · episode {ep}")
        fig.tight_layout()
        path = out_dir / f"{ds.name}_ep{ep:03d}_timeseries.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Saved {path}")

    # 2. Dataset overview: EE paths (top-down and side), initial part positions, episode lengths
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for s, e in zip(ds.starts, ds.ends):
        axes[0].plot(ds.ee_pos[s:e, 1], ds.ee_pos[s:e, 0], lw=0.6, alpha=0.6)
        axes[1].plot(ds.ee_pos[s:e, 0], ds.ee_pos[s:e, 2], lw=0.6, alpha=0.6)
    axes[0].set(title="EE path, top-down", xlabel="y [m]", ylabel="x [m]")
    axes[0].invert_xaxis()
    axes[1].set(title="EE path, side", xlabel="x [m]", ylabel="z [m]")

    if ds.parts is not None:
        n_parts = ds.parts.shape[1] // 7
        names = ONE_LEG_PARTS if n_parts == len(ONE_LEG_PARTS) else [f"part{i}" for i in range(n_parts)]
        init = ds.parts[ds.starts].reshape(len(ds), n_parts, 7)
        for p in range(n_parts):
            axes[2].scatter(init[:, p, 0], init[:, p, 1], s=12, label=names[p])
        axes[2].legend(fontsize=8)
        axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].set(title="Initial part positions (AprilTag frame)", xlabel="x [m]", ylabel="y [m]")

    axes[3].hist(ds.ends - ds.starts, bins=20)
    axes[3].set(title="Episode length", xlabel="steps", ylabel="episodes")
    fig.suptitle(f"{ds.name} · {len(ds)} episodes")
    fig.tight_layout()
    path = out_dir / f"{ds.name}_overview.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["info", "video", "plot"])
    parser.add_argument("data", type=Path, help=".zarr, .pkl file, or folder of pickles")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--max-files", type=int, default=None, help="Folder of pickles: load at most this many")
    parser.add_argument("--fps", type=int, default=30, help="Video fps (data is recorded at 10 Hz)")
    parser.add_argument("--out-dir", default="viz")
    args = parser.parse_args()

    ds = Dataset(args.data, args.max_files)
    {"info": cmd_info, "video": cmd_video, "plot": cmd_plot}[args.command](ds, args)


if __name__ == "__main__":
    main()
