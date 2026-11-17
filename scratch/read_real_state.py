"""Camera-alignment capture, run on the robot PC (polymetis-local env, Polymetis robot server running,
hand on the e-stop). Moves the FR3 to the reference joints, lets it settle, then records in one go:

  scratch/align/real_state.json   measured joints, gripper width, both cameras' colour intrinsics
  scratch/align/real_front.png    front camera, 640x480
  scratch/align/real_wrist.png    wrist camera, 640x480

    python scratch/read_real_state.py              # move, wait, record
    python scratch/read_real_state.py --no-move    # record at the current pose
    python scratch/read_real_state.py --gripper-width 0.08   # no gripper server: give the finger gap (m)

Copy the three files to the Isaac PC's ~/ResiP/scratch/align/ and continue in camera_alignment_guide.ipynb.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from polymetis import RobotInterface

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.real.serials import FRONT_CAMERA_SERIAL, WRIST_CAMERA_SERIAL  # noqa: E402

# Reference pose: the arm is large in the front view and the hand is near the image centre.
# It is the first frame of a one_leg sim rollout with the TCP raised 8 cm (same hand orientation,
# FR3 IK): TCP at base (0.450, 0.055, 0.101) m, about 10 cm above the parts plane, tilted ~23 deg.
JOINTS = [0.1449, 0.3155, 0.0023, -2.2580, -0.0963, 2.1797, 0.0263]
OUT = REPO / "scratch" / "align"


def grab(serial, warmup=30):
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    try:
        for _ in range(warmup):  # let auto-exposure settle
            frames = pipe.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data()).copy()
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return img, dict(fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy)
    finally:
        pipe.stop()


p = argparse.ArgumentParser()
p.add_argument("--no-move", action="store_true")
p.add_argument("--settle", type=float, default=5.0, help="seconds to wait before recording")
p.add_argument("--gripper-width", type=float, default=None, help="finger gap in m if no gripper server")
args = p.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

robot = RobotInterface(ip_address="localhost")
if not args.no_move:
    robot.move_to_joint_positions(torch.tensor(JOINTS, dtype=torch.float32), time_to_go=6.0)
print(f"waiting {args.settle:.0f} s for the arm to settle ...")
time.sleep(args.settle)

q = robot.get_joint_positions().numpy()
err = np.degrees(q - np.array(JOINTS))
print("diff (deg):", np.round(err, 2).tolist(), f"  max |diff| = {np.abs(err).max():.2f} deg")

width = args.gripper_width
if width is None:
    try:
        from polymetis import GripperInterface

        width = float(GripperInterface(ip_address="localhost").get_state().width)
    except Exception as e:
        sys.exit(f"could not read the gripper ({type(e).__name__}); rerun with --gripper-width <metres>")
print(f"gripper width: {width * 100:.2f} cm")

front, k_front = grab(FRONT_CAMERA_SERIAL)
wrist, k_wrist = grab(WRIST_CAMERA_SERIAL)
q_after = robot.get_joint_positions().numpy()
moved = np.degrees(np.abs(q_after - q)).max()
if moved > 0.1:
    print(f"WARNING: the arm moved {moved:.2f} deg while the photos were taken; rerun with --no-move")
cv2.imwrite(str(OUT / "real_front.png"), front[:, :, ::-1])
cv2.imwrite(str(OUT / "real_wrist.png"), wrist[:, :, ::-1])
state = dict(
    joint_positions=q.tolist(), commanded=JOINTS, gripper_width=width,
    intrinsics=dict(front=k_front, wrist=k_wrist),
    serials=dict(front=FRONT_CAMERA_SERIAL, wrist=WRIST_CAMERA_SERIAL),
    time=time.strftime("%Y-%m-%d %H:%M:%S"),
)
(OUT / "real_state.json").write_text(json.dumps(state, indent=1))
print("saved", OUT / "real_state.json", OUT / "real_front.png", OUT / "real_wrist.png")
