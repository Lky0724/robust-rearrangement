"""Shrink re-rendered pickles saved with `isaac_lab_rerender.py --save-size full` to 240x320.

Full-size pickles hold the wrist image at 480x640, the front image upsampled to 720x960 (color_image2) and
at 480x640 (color_image3), ~3.9 MB per frame. process_pickles.py shrinks everything to 240x320 anyway, so
this rewrites each file the way `--save-size 240x320` would have saved it (~0.46 MB per frame):
color_image1 = wrist, color_image2 = front (from the 480x640 color_image3), no color_image3. The resize
is the one process_pickles.py applies (resize for the wrist, resize_crop for the front).

Usage (resip env, repo root):
    python scripts/downsample_rendered.py <dir with rendered .pkl files>
Files already at 240x320 are skipped. Each file is replaced atomically (written to <name>.partial first).
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from src.data_processing.utils import resize, resize_crop
from src.visualization.render_mp4 import unpickle_data


def downsample(path: Path) -> str:
    data = unpickle_data(path)
    obs = data["observations"]
    if "color_image3" not in obs[0]:
        return f"skip (already {np.asarray(obs[0]['color_image2']).shape[:2]})"

    wrist = np.stack([np.asarray(o["color_image1"]) for o in obs])
    front = np.stack([np.asarray(o["color_image3"]) for o in obs])
    assert wrist.shape[1:] == (480, 640, 3) and front.shape[1:] == (480, 640, 3), (wrist.shape, front.shape)
    wrist, front = resize(wrist), resize_crop(front)

    for t, o in enumerate(obs):
        o["color_image1"], o["color_image2"] = wrist[t], front[t]
        del o["color_image3"]
        if "robot_mask_front" in o:
            # A 240x320 pixel is robot if any of its 2x2 pixels in the 480x640 render is
            o["robot_mask_front"] = np.asarray(o["robot_mask_front"]).reshape(240, 2, 320, 2).any(axis=(1, 3))
    if isinstance(data.get("args_rerender"), dict):
        data["args_rerender"]["save_size"] = "240x320 (downsampled from full)"

    tmp = path.with_name(path.stem + ".partial")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)
    return f"{len(obs)} frames -> {path.stat().st_size / 1e9:.2f} GB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    args = parser.parse_args()

    files = sorted(args.dir.glob("*.pkl"))
    print(f"{len(files)} pickles in {args.dir}")
    for f in files:
        size = f.stat().st_size / 1e9
        print(f"{f.name} ({size:.2f} GB): {downsample(f)}", flush=True)


if __name__ == "__main__":
    main()
