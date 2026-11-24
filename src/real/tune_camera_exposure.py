"""
Pick fixed exposure, gain and white balance for the two real cameras.

Opens both cameras in the same mode as teleop (640x480 rgb8 + depth, 30 fps), starting from the
values in src/real/camera_settings.py, and shows them side by side with clipped pixels in red.
Set up the scene as for recording (lights, parts, arm in view) before tuning.

    # polymetis-local env, from the repo root, cameras plugged in, no teleop running
    python -m src.real.tune_camera_exposure            # live tuning
    python -m src.real.tune_camera_exposure --sweep    # exposure sweep -> table + image sheet

Live keys (they act on the selected camera, shown in yellow):
    1 / 2   select front / wrist
    a       auto exposure on/off (switching off keeps the camera's current exposure and gain)
    e / E   exposure down / up (x0.85 / x1.18)
    g / G   gain down / up (8 steps)
    b       auto white balance on/off;  w / W   white balance down / up (100 K)
    z       clipped-pixel overlay on/off
    s       save the two raw frames as PNGs
    p       print the COLOR_SETTINGS block for camera_settings.py
    q, Esc  quit (also prints the block)
Drag a rectangle with the mouse to measure a region, e.g. the tabletop face; right click clears it.

What to aim for: in the measured region on the white parts, p99 about 200-235 and no clipped
pixels, with the same lighting you will record with. The renderer's tabletop face is about 210.
For white balance, lock it (b) and adjust (w / W) until a region on a white part shows about
equal mean R, G and B: lower values make the image bluer, higher ones warmer.
Without a region, the front camera is measured in the policy's crop box and the wrist camera on
the whole frame.
"""

import argparse
import time
from pathlib import Path

import os

import cv2
import numpy as np
import pyrealsense2 as rs

# opencv-python points Qt at a font folder it doesn't ship; use the system fonts (no warnings)
if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

from src.real.camera_settings import COLOR_SETTINGS, apply_color_settings
from src.real.serials import FRONT_CAMERA_SERIAL, WRIST_CAMERA_SERIAL

W, H, FPS = 640, 480, 30
CLIP = 250  # a pixel counts as clipped when its brightest channel reaches this
POLICY_BOX = (96, 16, 544, 464)  # front camera: 224x224 center crop of the 320x240 image, in 640x480
NAMES = ["front", "wrist"]
SERIALS = {"front": FRONT_CAMERA_SERIAL, "wrist": WRIST_CAMERA_SERIAL}


def start(serial):
    """Same streams as rdt.image.factory.enable_single_realsense (used by teleop)."""
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
    pipe = rs.pipeline()
    pipe.start(cfg)
    return pipe


def grab(pipe):
    frames = pipe.wait_for_frames(1000)
    c = frames.get_color_frame()
    meta = {}
    for key, m in [("exp", rs.frame_metadata_value.actual_exposure), ("fps", rs.frame_metadata_value.actual_fps)]:
        if c.supports_frame_metadata(m):
            meta[key] = c.get_frame_metadata(m)
    return np.asanyarray(c.get_data()).copy(), meta


def region_stats(img, box):
    x0, y0, x1, y1 = box
    patch = img[y0:y1, x0:x1]
    v = patch.max(axis=2)
    return dict(mean=float(v.mean()), p99=float(np.percentile(v, 99)), clipped=float((v >= CLIP).mean() * 100),
                rgb=patch.reshape(-1, 3).mean(axis=0))


def default_box(name):
    return POLICY_BOX if name == "front" else (0, 0, W, H)


def settings_block(sensors):
    lines = ["COLOR_SETTINGS = {"]
    for name in NAMES:
        if name not in sensors:
            lines.append(f'    "{name}": dict(exposure=None, gain=None, white_balance=None),')
            continue
        s = sensors[name]
        auto_e = s.get_option(rs.option.enable_auto_exposure) > 0
        auto_w = s.get_option(rs.option.enable_auto_white_balance) > 0
        exp = "None" if auto_e else f"{s.get_option(rs.option.exposure):.0f}"
        gain = "None" if auto_e else f"{s.get_option(rs.option.gain):.0f}"
        wb = "None" if auto_w else f"{s.get_option(rs.option.white_balance):.0f}"
        lines.append(f'    "{name}": dict(exposure={exp}, gain={gain}, white_balance={wb}),')
    lines.append("}")
    return "\n".join(lines)


def nudge(sensor, option, value):
    r = sensor.get_option_range(option)
    value = min(max(value, r.min), r.max)
    if r.step > 0:
        value = r.min + round((value - r.min) / r.step) * r.step
    sensor.set_option(option, float(value))


def live(pipes, sensors, out_dir):
    boxes = {n: None for n in pipes}
    sel = [list(pipes)[0]]
    overlay = [True]
    drag = {}
    names = list(pipes)

    def on_mouse(event, x, y, flags, _):
        idx = min(x // W, len(names) - 1)
        name, lx = names[idx], x - idx * W
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update(name=name, x0=lx, y0=y)
        elif event == cv2.EVENT_LBUTTONUP and drag.get("name") == name:
            x0, x1 = sorted([drag["x0"], lx]); y0, y1 = sorted([drag["y0"], y])
            if x1 - x0 > 4 and y1 - y0 > 4:
                boxes[name] = (x0, y0, x1, y1)
            drag.clear()
        elif event == cv2.EVENT_RBUTTONDOWN:
            boxes[name] = None

    win = "exposure tuner (keys: see the script header)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    last = {}
    while True:
        tiles = []
        for name in names:
            img, meta = grab(pipes[name])
            last[name] = img
            s = sensors[name]
            box = boxes[name] or default_box(name)
            st = region_stats(img, box)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if overlay[0]:
                bgr[img.max(axis=2) >= CLIP] = (0, 0, 255)
            if name == "front":
                cv2.rectangle(bgr, POLICY_BOX[:2], POLICY_BOX[2:], (0, 255, 255), 1)
            if boxes[name]:
                cv2.rectangle(bgr, boxes[name][:2], boxes[name][2:], (255, 0, 255), 2)
            auto_e = s.get_option(rs.option.enable_auto_exposure) > 0
            auto_w = s.get_option(rs.option.enable_auto_white_balance) > 0
            lines = [
                f"{name}{' [selected]' if name == sel[0] else ''}",
                ("exposure AUTO" if auto_e else
                 f"exposure {s.get_option(rs.option.exposure):.0f}  gain {s.get_option(rs.option.gain):.0f}")
                + (f"  (frame exp {meta['exp']})" if "exp" in meta else ""),
                "white balance AUTO" if auto_w else f"white balance {s.get_option(rs.option.white_balance):.0f} K",
                f"{'region' if boxes[name] else 'policy box' if name == 'front' else 'frame'}: "
                f"mean {st['mean']:.0f}  p99 {st['p99']:.0f}  clipped {st['clipped']:.1f}%"
                + (f"  fps {meta['fps']}" if "fps" in meta else ""),
                # On a white or gray region, equal R, G, B means a neutral white balance
                "mean R G B: " + " ".join(f"{c:.0f}" for c in st["rgb"]),
            ]
            cv2.rectangle(bgr, (0, 0), (W, 18 * len(lines) + 6), (0, 0, 0), -1)
            for i, t in enumerate(lines):
                color = (0, 255, 255) if name == sel[0] and i == 0 else (255, 255, 255)
                cv2.putText(bgr, t, (6, 18 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            tiles.append(bgr)
        cv2.imshow(win, np.hstack(tiles))

        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        s = sensors[sel[0]]
        if k == ord("1") and "front" in pipes:
            sel[0] = "front"
        elif k == ord("2") and "wrist" in pipes:
            sel[0] = "wrist"
        elif k == ord("a"):
            if s.get_option(rs.option.enable_auto_exposure) > 0:
                exp, gain = s.get_option(rs.option.exposure), s.get_option(rs.option.gain)
                s.set_option(rs.option.enable_auto_exposure, 0)
                nudge(s, rs.option.exposure, exp)
                nudge(s, rs.option.gain, gain)
            else:
                s.set_option(rs.option.enable_auto_exposure, 1)
        elif k in (ord("e"), ord("E"), ord("g"), ord("G")):
            if s.get_option(rs.option.enable_auto_exposure) > 0:  # adjusting means manual
                s.set_option(rs.option.enable_auto_exposure, 0)
            if k in (ord("e"), ord("E")):
                nudge(s, rs.option.exposure, s.get_option(rs.option.exposure) * (0.85 if k == ord("e") else 1.18))
                if s.get_option(rs.option.exposure) > 10000 / FPS:
                    print(f"note: exposure above {10000 / FPS:.0f} can't keep {FPS} fps")
            else:
                nudge(s, rs.option.gain, s.get_option(rs.option.gain) + (-8 if k == ord("g") else 8))
        elif k == ord("b"):
            on = s.get_option(rs.option.enable_auto_white_balance) > 0
            wb = s.get_option(rs.option.white_balance)
            s.set_option(rs.option.enable_auto_white_balance, 0 if on else 1)
            if on:
                nudge(s, rs.option.white_balance, wb)
        elif k in (ord("w"), ord("W")):
            if s.get_option(rs.option.enable_auto_white_balance) > 0:
                s.set_option(rs.option.enable_auto_white_balance, 0)
            nudge(s, rs.option.white_balance, s.get_option(rs.option.white_balance) + (-100 if k == ord("w") else 100))
        elif k == ord("z"):
            overlay[0] = not overlay[0]
        elif k == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for name, img in last.items():
                p = out_dir / f"{stamp}_{name}.png"
                cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"saved {p}")
        elif k == ord("p"):
            print(settings_block(sensors))
    cv2.destroyAllWindows()
    print("\nPaste into src/real/camera_settings.py:\n" + settings_block(sensors))


def sweep(pipes, sensors, out_dir, exposures, gain):
    rows, sheets = [], []
    for name in pipes:
        s = sensors[name]
        s.set_option(rs.option.enable_auto_exposure, 0)
        g = gain if gain is not None else s.get_option(rs.option.gain)
        nudge(s, rs.option.gain, g)
        tiles = []
        for e in exposures:
            nudge(s, rs.option.exposure, e)
            for _ in range(15):  # let the new exposure reach the frames
                img, _ = grab(pipes[name])
            st = region_stats(img, default_box(name))
            rows.append((name, s.get_option(rs.option.exposure), s.get_option(rs.option.gain), st))
            bgr = cv2.cvtColor(cv2.resize(img, (320, 240)), cv2.COLOR_RGB2BGR)
            bgr[cv2.resize(img, (320, 240)).max(axis=2) >= CLIP] = (0, 0, 255)
            cv2.putText(bgr, f"{name} exp {e:.0f} gain {g:.0f}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(bgr, f"p99 {st['p99']:.0f} clip {st['clipped']:.1f}%", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(bgr)
        sheets.append(np.hstack(tiles))
    print(f"\n{'camera':6s} {'exposure':>8s} {'gain':>5s} {'mean':>6s} {'p99':>5s} {'clipped %':>9s}   (front: policy box, wrist: whole frame)")
    for name, e, g, st in rows:
        print(f"{name:6s} {e:8.0f} {g:5.0f} {st['mean']:6.0f} {st['p99']:5.0f} {st['clipped']:9.2f}")
    for name in pipes:
        ok = [e for n, e, _, st in rows if n == name and st["clipped"] < 0.5 and st["p99"] <= 240]
        print(f"{name}: brightest exposure without clipping (clipped < 0.5%, p99 <= 240): {max(ok):.0f}" if ok
              else f"{name}: every exposure clips; lower the gain or the light")
    p = out_dir / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(p), np.vstack(sheets))
    print(f"saved {p} (clipped pixels in red). Check the parts by eye, then confirm with the live mode.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", choices=["front", "wrist", "both"], default="both")
    parser.add_argument("--sweep", action="store_true", help="step through --exposures instead of live tuning")
    parser.add_argument("--exposures", type=float, nargs="+", default=[40, 60, 80, 110, 150, 200, 260, 330])
    parser.add_argument("--gain", type=float, default=None, help="sweep: fixed gain (default: the camera's current gain)")
    parser.add_argument("--out-dir", type=Path, default=Path("scratch/exposure"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    names = NAMES if args.camera == "both" else [args.camera]
    connected = {d.get_info(rs.camera_info.serial_number) for d in rs.context().devices}
    missing = [f"{n} ({SERIALS[n]})" for n in names if SERIALS[n] not in connected]
    if missing:
        raise SystemExit(f"Camera not connected: {', '.join(missing)}. Connected: {sorted(connected) or 'none'}")

    pipes, sensors = {}, {}
    try:
        for name in names:
            pipes[name] = start(SERIALS[name])
            apply_color_settings(pipes[name], COLOR_SETTINGS[name], name)
            sensors[name] = pipes[name].get_active_profile().get_device().first_color_sensor()
            time.sleep(1.0)
        if args.sweep:
            sweep(pipes, sensors, args.out_dir, args.exposures, args.gain)
        else:
            live(pipes, sensors, args.out_dir)
    finally:
        for p in pipes.values():
            p.stop()


if __name__ == "__main__":
    main()
