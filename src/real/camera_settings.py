"""
Fixed color settings for the two real RealSense cameras.

With the default auto exposure, the mostly dark scene makes the cameras expose for the black
cloth: the white PLA parts clip to 255 (the tabletop face loses its holes and edges), and the
brightness changes as the white arm moves in and out of view. The renderer never clips, so this
is a sim-to-real gap exactly where insertion happens.

Pick the values with the tuner, write them below, and every real script that opens the cameras
applies the same ones:

    python -m src.real.tune_camera_exposure          # polymetis-local env, cameras plugged in

None keeps the camera's automatic control for that setting. exposure is in the sensor's
units (100 us for the D435 color sensor, so 330 = 33 ms, the longest that keeps 30 fps).
Options stay on the camera until it is power-cycled, so apply_color_settings() always sets
every option explicitly, auto included.
"""

import pyrealsense2 as rs

COLOR_SETTINGS = {
    "front": dict(exposure=30, gain=64, white_balance=None),
    "wrist": dict(exposure=30, gain=64, white_balance=None),
}


def _set(sensor, option, value):
    r = sensor.get_option_range(option)
    if not r.min <= value <= r.max:
        raise ValueError(f"{option.name} = {value} is outside [{r.min}, {r.max}]")
    sensor.set_option(option, float(value))


def apply_color_settings(pipe, settings: dict, name: str = "") -> dict:
    """Apply exposure/gain/white balance to a started pipeline's color sensor.

    exposure set: manual exposure, with gain as given (the sensor's default gain if None).
    exposure None: auto exposure. white_balance None: auto white balance.
    Returns the values read back from the camera, for logging in the episode metadata.
    """
    sensor = pipe.get_active_profile().get_device().first_color_sensor()

    if settings.get("exposure") is not None:
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        _set(sensor, rs.option.exposure, settings["exposure"])
        gain = settings.get("gain")
        _set(sensor, rs.option.gain, gain if gain is not None else sensor.get_option_range(rs.option.gain).default)
    else:
        sensor.set_option(rs.option.enable_auto_exposure, 1)

    if settings.get("white_balance") is not None:
        sensor.set_option(rs.option.enable_auto_white_balance, 0)
        _set(sensor, rs.option.white_balance, settings["white_balance"])
    else:
        sensor.set_option(rs.option.enable_auto_white_balance, 1)

    applied = dict(
        auto_exposure=bool(sensor.get_option(rs.option.enable_auto_exposure)),
        exposure=sensor.get_option(rs.option.exposure),
        gain=sensor.get_option(rs.option.gain),
        auto_white_balance=bool(sensor.get_option(rs.option.enable_auto_white_balance)),
        white_balance=sensor.get_option(rs.option.white_balance),
    )
    if applied["auto_exposure"]:
        desc = "auto exposure"
    else:
        desc = f"exposure {applied['exposure']:.0f}, gain {applied['gain']:.0f}"
    desc += ", auto white balance" if applied["auto_white_balance"] else f", white balance {applied['white_balance']:.0f} K"
    print(f"{name} camera: {desc}")
    return applied
