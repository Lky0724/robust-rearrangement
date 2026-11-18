"""Sanity-check every re-rendered pickle against its source rollout (usage: check_rendered.py <rendered dir> <source dir>)."""
import sys, os, glob, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor

R, SRC = sys.argv[1], sys.argv[2]

def check(f):
    d = pickle.load(open(f, "rb")); s = pickle.load(open(os.path.join(SRC, os.path.basename(f)), "rb"))
    o, so = d["observations"], s["observations"]
    issues = []
    if len(o) != len(so): issues.append(f"frames {len(o)} vs source {len(so)}")
    w = np.stack([x["color_image1"] for x in o]); fr = np.stack([x["color_image2"] for x in o])
    if w.shape[1:] != (240, 320, 3) or fr.shape[1:] != (240, 320, 3): issues.append(f"shape {w.shape} {fr.shape}")
    stats = {}
    for name, im in (("wrist", w), ("front", fr)):
        m = im.reshape(len(im), -1).mean(1)
        dark, bright = int((m < 5).sum()), int((m > 250).sum())
        same = int((np.abs(np.diff(im.astype(np.int16), axis=0)).reshape(len(im) - 1, -1).max(1) == 0).sum())
        if dark or bright: issues.append(f"{name}: {dark} black / {bright} white frames")
        stats[name] = (m.mean(), same)
    ee = max(np.abs(np.asarray(a["robot_state"]["ee_pos"]) - np.asarray(b["robot_state"]["ee_pos"])).max() for a, b in zip(o, so))
    if ee > 0: issues.append(f"ee_pos differs from source by {ee}")
    if not np.array_equal(np.asarray(d["actions"]), np.asarray(s["actions"])): issues.append("actions differ")
    return os.path.basename(f), len(o), stats, issues

files = sorted(glob.glob(R + "/*.pkl"))
with ProcessPoolExecutor(6) as ex:
    res = list(ex.map(check, files, chunksize=4))
bad = [r for r in res if r[3]]
print(f"{len(res)} episodes, {sum(r[1] for r in res)} frames, {len(bad)} with issues")
for name, n, _, iss in bad[:20]: print("  ", name, n, "; ".join(iss))
for cam in ("wrist", "front"):
    b = np.array([r[2][cam][0] for r in res]); same = np.array([r[2][cam][1] for r in res])
    print(f"{cam}: mean brightness per episode {b.mean():.1f} (min {b.min():.1f}, max {b.max():.1f}); "
          f"identical consecutive frames: total {same.sum()}, max in one episode {same.max()}")
lo = np.argsort([r[2]["front"][0] for r in res])
print("darkest front episodes:", [(res[i][0], round(res[i][2]['front'][0], 1)) for i in lo[:3]])
print("brightest front episodes:", [(res[i][0], round(res[i][2]['front'][0], 1)) for i in lo[-3:]])
