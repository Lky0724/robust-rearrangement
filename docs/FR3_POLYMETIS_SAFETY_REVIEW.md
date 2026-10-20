# FR3 + Polymetis Teleop: Safety Review and Change List

Review of what can make the robot behave unexpectedly when running [src/real/teleop_sm_fr3.py](../src/real/teleop_sm_fr3.py) on the FR3 through Polymetis, as set up in [FR3_ORIGINAL_CODE_TELEOP_TEST.md](FR3_ORIGINAL_CODE_TELEOP_TEST.md). It covers six risks, and a planned change for each.

**Status:** review only. None of the changes below are implemented yet.

**What already keeps teleop bounded:**
- **No command build-up.** Each 10 Hz cycle starts the new target from the arm's **actual** pose.
- **Torque limits.** Torques are clamped (`joint_torques` in `franka_hardware.yaml`) and rate-limited (libfranka `limit_rate`, 100 Hz low-pass).
- **Robot-side safety.** The FR3's own safety reflexes and the stop button work independently of the PC.

The risks below are about motion that is **faster, jumpier or less expected** than one might assume, not about runaway motion.

---

## 1. Risks

### Risk 1: the robot restarts by itself after a stop
When anything stops the robot (a Polymetis safety limit, a timing error, or **the stop button**), Polymetis' robot client tries automatic error recovery: up to 3 times, 1 s apart. The constants are `RECOVERY_MAX_TRIES 3` and `RECOVERY_WAIT_SECS 1` in `fairo/polymetis/polymetis/src/clients/franka_panda_client/franka_panda_client.cpp`.

If the stop button is released within those ~3 s, control **resumes immediately**. The teleop script is still running and sending SpaceMouse commands.

### Risk 2: keyboard keys move the robot from any window, and held keys make large jumps
- **Any window.** The keyboard listener (`pynput`, in `rdt.common.keyboard_interface`) is global. Typing `i k j l u o` in **any** window rotates the gripper. `t`/`n` end the episode.
- **Held keys.** Each key event rotates the target by `INIT_ROT_DELTA` = 0.13 rad (7.4°). Holding a key triggers the OS key repeat (~25–30 events/s), so about **0.4 rad (22°) per 0.1 s step** reaches the robot. That's the largest command step in the system.

### Risk 3: real speeds are higher than the option values
`get_action()` multiplies the speed options by fixed factors: position by `sm_dpos_scalar = 1.8`, rotation by `sm_drot_scalar = 4.0`. SpaceMouse values are divided by `max_value = 500`. The rdt code notes raw maxima of about 300 (wired) and 500 (wireless), so a full deflection gives about ±0.6 (wired) to ±1.0 (wireless).

| Setting | Max position speed | Max rotation speed |
|---|---|---|
| Defaults (`--max_pos_speed 0.3 --max_rot_speed 0.7`) | ~0.32–0.54 m/s | **~1.7–2.8 rad/s** |
| First-session values in the test doc (0.15 / 0.4) | ~0.16–0.27 m/s | **~1.0–1.6 rad/s** |

A fast twist mostly turns joint 7. Polymetis' velocity limit for joint 7 is 2.51 rad/s (`joint_vel` in `franka_hardware.yaml`), and exceeding it **throws and stops the robot** (then see risk 1).

The test doc's wording "halve the defaults (0.3 m/s and 0.7 rad/s)" is misleading.

### Risk 4: no protection against singularities and joint limits in the diff-IK
`DiffIKWrapper.update_desired_ee_pose` (rdt `lars-dev`, `polymetis_robot_utils/interfaces/diffik.py`) computes, once per 0.1 s step:
```
Δq = lstsq(J, error / dt) · dt,   q* = q + Δq
```
There is **no damping and no limit on the joint step**.
- **Singular poses.** Near them (arm stretched far out, or the wrist straight with joint 6 near π), a small Cartesian command becomes a **large joint jump**.
- **Joint limits.** The diff-IK ignores them. Polymetis' soft-limit springs (stiffness 50 Nm/rad, margin 0.2 rad) are weaker than the teleop gains (`Kq` up to 160). The arm can be pulled into the hard limit, which throws a stop, then auto-recovery, then another stop. That's the "reflex error loop" in the test doc's §9.2 f.

### Risk 5: collision detection is less sensitive than Franka's defaults
`franka_hardware.yaml` sets all collision thresholds (`collision_behavior`) to **40 Nm / 40 N**:
- That's about **twice** Franka's own example values (~20; `communication_test` uses `[20, 20, 18, 18, 16, 14, 12]` Nm).
- For the wrist joints (5–7) it's **above their maximum torque** of 12 Nm, so a collision there never triggers the collision reflex.
- In a collision, the arm pushes harder before stopping.

### Risk 6: smaller points
- **6a. SpaceMouse button 0 isn't debounced.** `if is_button_pressed(0) or is_button_pressed(1) and last_grip_step > 10` reads as `A or (B and C)`. So button 0 toggles the gripper **every cycle while held**; only button 1 is debounced. Harmless with `--no_gripper`; with a real gripper it makes the fingers chatter.
- **6b. The startup move to home isn't checked.** `move_to_joint_positions(home ± 5° noise)` starts as soon as the script runs. It's a smooth minimum-jerk motion in joint space (duration from the URDF speed limits, ~3 s), so the path isn't checked against obstacles.
- **6c. Polymetis' robot model is the Panda's.** See §3; information only.
- **6d. Timing stops on a non-real-time kernel** (Path B) end the same way as risk 1.

---

## 2. Change list

Files: `teleop_sm_fr3.py` (ResiP repo), Polymetis in `~/fr3_test/fairo` (outside the repo), and the test doc. **`src/real/teleop_sm.py` and `rdt` stay untouched.**

### Risk 1: no automatic restart
| # | File | Change | Effect |
|---|---|---|---|
| 1a | `fairo/…/franka_panda_client.cpp` | Turn off automatic error recovery: `RECOVERY_MAX_TRIES 3 → 0`, then `make franka_panda_client` | After any stop, the robot client exits instead of retrying. The robot stays stopped until the server is deliberately restarted |
| 1b | `teleop_sm_fr3.py` | On a lost server connection, print "robot stopped, restart server" and end cleanly without saving | No traceback; nothing keeps sending commands |
| 1c | Test doc §9.3 | Stop procedure: stop button → Ctrl+C terminal 4 (teleop) → Ctrl+C terminal 1 (server) → release button → clear error in Desk → restart | |

**Trade-off:** Path B timing stops also need a manual restart.

### Risk 2: keyboard
| # | File | Change | Effect |
|---|---|---|---|
| 2a | `teleop_sm_fr3.py` | Cap the keyboard rotation per step at **one key press** (0.13 rad), however many repeat events arrived | A held key turns at most 1.3 rad/s instead of ~4 rad/s |
| 2b | `teleop_sm_fr3.py` | New option `--no_keyboard_rotation`: `i k j l u o` are ignored, while `t`/`n` still work | For the first sessions, the keyboard can't move the robot |
| 2c | Test doc §9 | Note: tap keys, don't hold them; don't type in other windows during teleop | |

### Risk 3: speeds
| # | File | Change | Effect |
|---|---|---|---|
| 3a | `teleop_sm_fr3.py` | At startup, print the **real** maximum speeds, e.g. `max EE speed 0.27 m/s, max rotation 0.80 rad/s` | You see what the settings mean |
| 3b | Test doc §9.1 | Correct the explanation (×1.8 and ×4); first-session values `--max_pos_speed 0.15 --max_rot_speed 0.2` | |
| — | — | `sm_dpos_scalar` / `sm_drot_scalar` stay as the authors set them | Recorded data matches the original teleop settings |

### Risk 4: diff-IK protections
A small subclass of `DiffIKWrapper` in `teleop_sm_fr3.py` overrides `update_desired_ee_pose`; `rdt` itself isn't edited.

| # | Change | Effect |
|---|---|---|
| 4a | **Damped** least squares instead of plain `lstsq`: `Δq = Jᵀ(JJᵀ + λ²I)⁻¹ · error`, with λ = 0.05 | Near singular poses, joint jumps stay bounded. Elsewhere it's practically identical |
| 4b | **Clamp the joint step**: at most 0.15 rad per joint per 0.1 s (1.5 rad/s, below the 2.51 rad/s safety limit), scaling the whole step down evenly so the direction is kept | No sudden large joint motions; normal teleop motions are unchanged |
| 4c | **Keep targets inside the joint limits**: clip `q*` to the test doc's §5.1 limits minus 0.25 rad, so it stays outside the zone where Polymetis' safety springs act | No pulling into hard limits, so no stop/recover loops |
| 4d | Option `--no_ik_limits`: use the original, unprotected diff-IK | For comparison with the authors' behavior |

The protections only act in exceptional cases, so teleop feels the same in the middle of the workspace. The **recorded actions** are still the SpaceMouse-commanded target poses, not the clamped joint targets.

If the IK should stay identical to the authors' for now, leave out 4a and 4b, but keep 4c.

### Risk 5: collision thresholds
| # | File | Change | Effect |
|---|---|---|---|
| 5a | `fairo/polymetis/polymetis/conf/robot_client/franka_hardware.yaml` | `collision_behavior`: `lower_torque`/`upper_torque` → `[20, 20, 18, 18, 16, 14, 12]`; `lower_force`/`upper_force` → `[20, 20, 20, 25, 25, 25]` (Franka's `communication_test` values) | Collisions are detected about 2× earlier, including at the wrist |
| 5b | Test doc §5.1 | Explain the change and how to undo it | |

**Trade-off:** very fast motions may trigger a collision reflex by mistake. That's a stop, not a danger; if it happens often, raise the values slightly. A server restart applies the change.

### Risk 6: smaller points
| # | File | Change | Effect |
|---|---|---|---|
| 6a | `teleop_sm_fr3.py` | Button check becomes `(is_button_pressed(0) or is_button_pressed(1)) and last_grip_step > 10` | One press means one gripper toggle, for both buttons |
| 6b | `teleop_sm_fr3.py` | Before moving home: print current and home joint angles plus the move duration, check home lies within the §5.1 limits, and **wait for Enter** | You confirm the path is clear before the arm moves |
| 6c | Test doc | Note that Polymetis' robot model is the Panda's (see §3) | Information only, no code change |
| 6d | Test doc §9.3 | Path B timing stops follow the stop procedure from 1c | |

### Summary by file
| File | Changes |
|---|---|
| `src/real/teleop_sm_fr3.py` | 1b, 2a, 2b, 3a, 4a–4d, 6a, 6b |
| `fairo/…/franka_panda_client.cpp` (+ rebuild) | 1a |
| `fairo/…/conf/robot_client/franka_hardware.yaml` | 5a |
| `docs/FR3_ORIGINAL_CODE_TELEOP_TEST.md` | 1c, 2c, 3b, 5b, 6c, 6d; new options in the §8.2 table; a safety section at the start of §9 |
| `src/real/teleop_sm.py`, `rdt` | No changes |

### Testing after the changes
1. `python -m src.real.teleop_sm_fr3 --help` and `python -m py_compile` (no robot needed).
2. **1a:** start the server, press the stop button. The robot client must exit, and the robot must **not** resume when the button is released.
3. **6b:** start teleop. It must wait for Enter before moving to home.
4. **2a/2b:** with `--no_keyboard_rotation`, the rotation keys do nothing. Without it, a held key turns the gripper slowly and steadily.
5. **4a–4c:** teleop slowly toward the edge of the workspace and toward a joint limit (for example by twisting toward joint 7's limit). The motion must slow or stop at the limit, without a Polymetis safety stop.
6. **5a:** a gentle push against the arm while it's held triggers the collision reflex earlier than before.

### Recommended first-run command after the changes
```bash
python -m src.real.teleop_sm_fr3 -p 6000 --furniture one_leg --save_dir scratch/fr3_teleop_nocam \
       --max_pos_speed 0.15 --max_rot_speed 0.2 --no_cameras --no_gripper --no_keyboard_rotation
```
Keep a hand on the stop button. After pressing it, stop teleop and the server before releasing it.

---

## 3. Background: Polymetis' Panda robot model on the FR3 (risk 6c)

Polymetis loads its robot model from `conf/robot_model/franka_panda.yaml` → `data/franka_panda/panda_arm.urdf`. It was compared with the FR3 URDF shipped with libfranka 0.18 (`libfranka/test/fr3.urdf`).

| Part of the model | Panda vs FR3 | Used during teleop for |
|---|---|---|
| Joint origins, axes, link lengths, flange offset (0.107 m) | **Identical**, number for number | End-effector pose (`get_ee_pose`), Jacobian for the diff-IK |
| Link masses and centers of mass | **Different**, e.g. link 1: 4.97 vs 2.93 kg; link 2: 0.65 vs 2.94 kg | Feedforward torques (Coriolis and inertia; gravity is handled by the FR3 itself) |
| Joint speed limits | Panda: 2.18–2.61 rad/s; FR3: 2.62–5.26 rad/s | Duration of `move_to_joint_positions` (the move to home) |
| Joint position limits | Panda's (joint 6 down to −0.02) | Only Polymetis' IK solver, which teleop doesn't use. Safety limits come from `franka_hardware.yaml`, already set to FR3 values |

**Effect of keeping the Panda model:**
- **Kinematics:** poses, Jacobian and recorded `ee_pos`/`ee_quat` are exactly right.
- **Dynamics:** feedforward torques are slightly off, but small at teleop speeds and corrected by the stiff PD controller, so at most a slight tracking error.
- **Move to home:** slower than the FR3 allows, which is on the safe side.
- **Simulation:** ResiP's simulator also uses the Panda model, so both use the same kinematic model.

**If switching to an FR3 model later** (for example, if tracking quality matters):
1. Get the URDF from the connected robot with libfranka 0.18's `robot.getRobotModel()`. The test-folder `fr3.urdf` uses `fp3_` joint names and limits slightly different from the FR3 datasheet.
2. Add `conf/robot_model/franka_fr3.yaml` (a copy of `franka_panda.yaml` with the new URDF path, `ee_link_name` set to the FR3 flange link, and the FR3 limits).
3. Launch with `launch_robot.py … robot_model=franka_fr3`.
4. Check that `get_ee_pose()` still matches Desk, then compare tracking with the sim-replay test.

Risks of switching: the env's pinocchio 2.5 may not parse the newer URDF, and the move to home becomes up to 2× faster at the wrist.
