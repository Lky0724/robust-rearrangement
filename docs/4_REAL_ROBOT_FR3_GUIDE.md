# Real-Robot Guide: Franka Research 3

How to run stage 4 (teleop collection of D_real) and stage 5b (π_real rollout) of the ResiP pipeline on a **Franka Research 3 (FR3)**. The authors used a Franka **Panda** with Polymetis and their `rdt` tools; Polymetis doesn't support the FR3 out of the box, so this guide replaces the robot layer. To first try the original stack on the FR3 unchanged, see [FR3_ORIGINAL_CODE_TELEOP_TEST.md](FR3_ORIGINAL_CODE_TELEOP_TEST.md). The policy, data format and processing stay the same.

**§1 is the big picture:** a diagram of how the original code drives the robot, the same diagram for the FR3 plan, and a comparison of what to change. §2–7 give the details. **§8 is the step-by-step procedure**, from code changes to a real rollout.

Related docs: [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) (stages 4–5), [DATA_AND_POLICY_IO.md](DATA_AND_POLICY_IO.md) (pickle/zarr fields), [DATA_GUIDE.md](DATA_GUIDE.md) (visualizing demos).

---

## 1. Workflow: original code vs. the FR3 plan

### 1.1 How the original code drives the Panda

The paper (Sec. III-B) describes the stack: 10 Hz target end-effector poses → differential IK → joint targets → a 1 kHz PD (joint impedance) controller. Teleop and rollout share everything below the script layer.

```
 ═══ COMMAND PATH ═════════════════════════════════════════════════════════════════════════════════

        TELEOP (collect D_real)                            ROLLOUT (run π_real)
  SpaceMouse ─► rdt.spacemouse.Spacemouse          WandB run ─► src/behavior (get_actor), weights
  keyboard   ─► rdt.common.KeyboardInterface       obs ─► actor.action(obs)  →  [xyz, rot6d, grip]
              │ (success/fail keys, screw rotation)            │ (absolute TIP pose, action chunks)
              ▼                                                ▼
  src/real/teleop_sm.py                            src/real/minimal.py
    ObsActHelper.get_action():                       SimpleDiffIKFrankaEnv.step():
      integrate SpaceMouse Δ → target TIP pose         action2pose_mat → target TIP pose
      gripper button → toggle                          sign flip of grip → toggle (+1.5 s sleep)
    main loop: precise_wait → strict 10 Hz           main loop: inference time + sleep(0.025) ≈ 10 Hz
              └──────────────────────────┬─────────────────────┘
                                         ▼
  convert_tip2wrist()  (defined in both scripts)       TIP pose → FLANGE pose (0.1034 m, 45° about z)
                                         ▼
  rdt.polymetis_robot_utils.interfaces.diffik.DiffIKWrapper   (subclass of polymetis.RobotInterface)
    move_to_joint_positions(home) → start_joint_impedance(Kq, Kqd) → update_desired_ee_pose(T, dt)
    diff-IK: q* = q + J⁺·e, ONCE per 10 Hz command, client side → update_desired_joint_positions(q*)
  polymetis.GripperInterface: grasp(speed, force) / goto(width, speed, force)
                                         │ gRPC to the Polymetis server at 173.16.0.1
                                         ▼
  Polymetis server (fairo fork, libfranka 0.9.2 — furniture-bench/docker/server.Dockerfile)
    joint impedance @ 1 kHz:  τ = Kq (q* − q) − Kqd q̇      runs on a real-time-kernel machine
                                         │ FCI (libfranka, 1 kHz)
                                         ▼
  Franka Panda + Franka Hand

 ═══ SENSING PATH (back into the scripts each step) ═════════════════════════════════════════════════

  2× RealSense D435 ─► pyrealsense2 + rdt.image.factory / rdt.config      (serials: src/real/serials.py)
                     ─► color_image1 = wrist, color_image2 = front  (minimal.py: resize / resize_crop)
  Polymetis state    ─► RobotInterface.get_ee_pose / get_joint_positions / get_joint_velocities
                        / robot_model.compute_jacobian ─► rdt PolymetisHelper ─► convert_wrist2tip()
                     ─► robot_state: tip pos, quat xyzw, (EE velocity = J·q̇), gripper width
  meshcat            ◄─ rdt.common.mc_util.meshcat_frame_show (target / current frames)

 ═══ OUTPUT ═══════════════════════════════════════════════════════════════════════════════════════

  teleop:  <save_dir>/<timestamp>.pkl ─► src/data_processing/process_pickles.py -d real ─► D_real .zarr
  rollout: robot motion; optional --log-ci pickles (CollectInferHelper in minimal.py)
```

### 1.2 How the code should drive the FR3 (our plan)

The same structure, with the robot layer swapped out. Labels: **[same]** unchanged, **[edit]** small edit, **[new]** new file, **[swap]** existing in-repo module replacing an `rdt` one.

```
 ═══ COMMAND PATH ═════════════════════════════════════════════════════════════════════════════════

        TELEOP (collect D_real)                            ROLLOUT (run π_real)
  SpaceMouse ─► furniture_bench/device/spacemouse/  WandB run ─► src/behavior (get_actor)   [same]
                spacemouse_shared_memory.py  [swap]  obs ─► actor.action(obs)                [same]
  keyboard   ─► src/data_collection/keyboard_interface.py + collect_enum.py   [swap + edit: RECORD]
              ▼                                                ▼
  src/real/teleop_sm.py              [edit]        src/real/minimal.py                     [edit]
    imports + server IP only;                        imports + server IP + PAUSE_HOLD→PAUSE;
    control logic unchanged                          optional: restore strict 10 Hz
              └──────────────────────────┬─────────────────────┘
                                         ▼
  convert_tip2wrist()                                                                       [same]
                                         ▼
  src/real/fr3/client.py                                                                     [new]
    FR3Robot   (imported as DiffIKWrapper):    same methods; update_desired_ee_pose → "set_target"
    FR3Gripper (imported as GripperInterface): grasp / goto → "grasp" / "move_gripper"
                                         │ zmq REQ/REP over LAN, 10 Hz (target pose ↓, state ↑)
                                         ▼
  src/real/fr3/server.py   on the real-time control PC                                       [new]
    diff-IK loop ~500 Hz toward the latest target
      (same formula as furniture-bench/furniture_bench/controllers/diffik.py, clamped per tick)
    panda-py controllers.JointPosition(Kq, Kqd): joint impedance @ 1 kHz
    libfranka.Gripper (calls run in a thread; they block)
                                         │ FCI (libfranka ≥ 0.10, matched to the FR3 system version)
                                         ▼
  Franka Research 3 + Franka Hand

 ═══ SENSING PATH ═════════════════════════════════════════════════════════════════════════════════

  2× RealSense D435 ─► src/real/fr3/realsense.py                                            [new]
                       (serials: src/real/serials.py [edit]) ─► color_image1 / color_image2 [same]
  server "state"     ─► libfranka Model: FLANGE pose + Jacobian, q, q̇, gripper width       [new]
                     ─► FR3Robot getters ─► src/real/fr3/utils.py PolymetisHelper [new]
                     ─► convert_wrist2tip() ─► robot_state                                   [same]
  meshcat            ◄─ src/real/fr3/utils.py meshcat_frame_show                             [new]

 ═══ OUTPUT ═══════════════════════════════════════════════════════════════════════════════════════

  identical: same pickle format ─► process_pickles.py ─► D_real .zarr                       [same]
```

### 1.3 Comparison and what to modify

| Layer | Original (Panda) | FR3 plan | Change | Where in §8 |
|---|---|---|---|---|
| SpaceMouse reader | `rdt.spacemouse.Spacemouse` | `furniture_bench/device/spacemouse/spacemouse_shared_memory.py` (same class, same constructor) | import swap | Step 5 |
| Keyboard + episode keys | `rdt.common.KeyboardInterface`, `rdt...CollectEnum` | `src/data_collection/keyboard_interface.py`, `collect_enum.py` | import swap; add `RECORD`, use `PAUSE` | Step 5 |
| Teleop script | `teleop_sm.py` | same file | imports + `franka_ip` → server IP | Step 5 |
| Rollout script | `minimal.py` | same file | imports (both blocks), `franka_ip`, `PAUSE_HOLD`; optional 10 Hz fix | Step 5 |
| Policy, normalizer, WandB loading | `src/behavior`, WandB | same | none | — |
| Tip ↔ flange frames | `convert_tip2wrist` / `convert_wrist2tip` | same | none (FR3 shares Panda kinematics and the Franka Hand) | — |
| Robot client API | `DiffIKWrapper` (rdt) on `polymetis.RobotInterface` | `FR3Robot` in `src/real/fr3/client.py` with the same method names | **new file** | Step 4 |
| Gripper client API | `polymetis.GripperInterface` | `FR3Gripper` in `client.py` | **new file** | Step 4 |
| Transport | gRPC (Polymetis) | zmq REQ/REP | new | Steps 3–4 |
| Diff-IK | `rdt` `DiffIKWrapper` (`lars-dev`): sim formula, evaluated **once per 10 Hz command** on the client | in `server.py`: same formula, ~500 Hz (like the sim's per-substep update), or once per command to replicate the original | **new** | Step 3 |
| Low-level controller | Polymetis joint impedance @ 1 kHz, `Kq`/`Kqd` | panda-py `JointPosition(stiffness=Kq, damping=Kqd)` @ 1 kHz | new; same gains to start, retune | Steps 3, 8 |
| FCI library | libfranka 0.9.2 (Panda only) | libfranka ≥ 0.10, matched to the FR3 system version | upgrade | Step 1 |
| Control machine | Polymetis server on a real-time machine | `server.py` on a PREEMPT_RT PC | new setup | Steps 1–2 |
| Cameras | pyrealsense2 via `rdt.image` / `rdt.config` | `src/real/fr3/realsense.py` | **new file**; set serials | Steps 4–5 |
| Pose / meshcat helpers | `rdt` `PolymetisHelper`, `mc_util` | `src/real/fr3/utils.py` | **new file** | Step 4 |
| Data format and processing | pickle → `process_pickles.py` → zarr | same | none | — |
| Python env | Polymetis conda env (py3.8, torch 1.13) | repo training env + pyrealsense2, meshcat, pynput, pyzmq, spnav; no Polymetis | new env | Step 6 |
| Sim cameras for D_synth | authors' intrinsics and poses in `isaac_lab_rerender.py` | your intrinsics and poses | edit | Steps 7, 10 |

**What to modify, in short:**
- **Write 4 new files** in `src/real/fr3/`: `server.py` (runs on the control PC; diff-IK + 1 kHz impedance + gripper via panda-py), `client.py` (Polymetis-compatible `FR3Robot` / `FR3Gripper`), `realsense.py`, `utils.py`.
- **Edit 6 existing files**:
  - `teleop_sm.py` and `minimal.py`: imports, IP and the pause key.
  - `collect_enum.py` and `keyboard_interface.py`: the `RECORD` key.
  - `serials.py`: your camera serials.
  - `isaac_lab_rerender.py`: your cameras.
- **Leave everything else alone:** the control logic in both scripts, the frame conversions, the action and observation format, `process_pickles.py`, and training.

The one place behavior can differ is **diff-IK and gains**. The original `DiffIKWrapper` uses the same formula as the sim, but evaluates it once per 10 Hz command, after which the impedance controller holds that fixed `q*`. The sim re-evaluates it at every substep. Recomputing at a high rate (the plan's default) tracks more like the sim. Evaluating once per command (set the server loop to act only on a new target) replicates the authors' real robot. Either way, use the **same** setting for teleop and rollout, and check tracking with the sim-replay test (Step 8e).

### 1.4 Original control parameters

| Piece | Where | Value |
|---|---|---|
| Robot IP | `franka_ip` in [teleop_sm.py:387](../src/real/teleop_sm.py#L387), [minimal.py:656](../src/real/minimal.py#L656) | `173.16.0.1` |
| Home pose (rad) | same files | `[-0.0931, 0.0382, 0.1488, -2.3811, -0.0090, 2.4947, 0.1204]` + uniform ±5° noise per joint |
| Joint stiffness `Kq` | `start_joint_impedance(..., adaptive=True)` | `[150, 120, 160, 100, 110, 100, 40]` |
| Joint damping `Kqd` | same | `[20, 20, 20, 20, 12, 12, 8]` |
| Tip ↔ flange | `convert_tip2wrist` / `convert_wrist2tip` | 0.1034 m along z, 45° about z |
| Gripper close / open | `gripper.grasp(0.07, 70)` / `gripper.goto(0.065 or 0.08, 0.05, 0.1)` | Polymetis signatures: `grasp(speed, force)` closes until contact; `goto(width, speed, force)`. Open width 0.065 in teleop, 0.08 in rollout |
| Cameras | [serials.py](../src/real/serials.py), pyrealsense2 | 2× D435 (wrist, front), RGB at 640×480 |

**The sim controller that real control should match** ([diffik.py](../furniture-bench/furniture_bench/controllers/diffik.py), [furniture_sim_env.py:800-945](../furniture-bench/furniture_bench/envs/furniture_sim_env.py#L800-L945)):

```
e_pos = p_goal − p_ee
e_rot = axis_angle(R_goal · R_eeᵀ)                     # world-frame rotation error
q̇*   = lstsq(J, [e_pos; e_rot] / 0.1)                  # pos_scalar = rot_scalar = 1
q*    = q + q̇* · 0.1
```

The sim **recomputes this at every physics substep** toward the same goal. An IsaacGym position drive then tracks `q*` with stiffness 1000 and damping 200. The authors' real robot (`rdt` `DiffIKWrapper`) evaluated it only **once per 10 Hz command** (`q* = q + J⁺·e`, with `dt` cancelling) and let joint impedance hold `q*`. For tracking closer to sim, recompute it at a high rate toward the latest target; to replicate the original, evaluate once per command.

---

## 2. Why the code doesn't run on an FR3 as-is

1. **libfranka is too old.** The repo pins libfranka **0.9.2** ([server.Dockerfile](../furniture-bench/docker/server.Dockerfile), Polymetis fork `minoring/fairo@furniture`). FR3 needs libfranka **≥ 0.10**, and newer FR3 system versions need newer releases. Check Franka's compatibility matrix for your system version.
2. **Polymetis is no longer maintained.** It can be rebuilt against a newer libfranka, but the API changed after 0.9, so expect to patch it.
3. **`rdt` lives in a separate repo.** `rdt` means "real deployment tools", from MIT's Improbable AI lab: `github.com/anthonysimeonov/improbable_rdt`. It's public, but only its **`lars-dev`** branch has every module the scripts import. ⚠️ [setup.txt](../src/real/setup.txt) says `pip install rdt`, but `rdt` on PyPI is an unrelated package (SDV's Reversible Data Transforms). Don't install it. You can keep using `rdt`'s SpaceMouse, keyboard and camera helpers with this plan; only `DiffIKWrapper` has to go, because it is built on Polymetis.

**FCI is required either way.** The Franka Control Interface (FCI) is the only route to 1 kHz external control, and Polymetis itself is an FCI client. The only decision is which FCI client sits on top of it.

---

## 3. Replacing the robot layer

### 3.1 Pick an FCI client

| Option | Notes |
|---|---|
| **panda-py** (recommended) | Python bindings for libfranka with in-process 1 kHz controllers (joint-position impedance, Cartesian impedance), FK/Jacobian and the gripper. Closest to the Polymetis API. Install a build that matches your libfranka version |
| **franka_ros2** | Franka's official stack with first-class FR3 support. Use its joint impedance controller. Adds ROS 2 overhead |
| franky | libfranka wrapper focused on motion generation; usable, less direct for impedance-plus-target streaming |
| deoxys | NUC server + zmq. Its main controllers are OSC (Cartesian), which track differently from the sim's diff-IK |
| Polymetis from source | Keeps the code unchanged, but you'd have to port an unmaintained project to a newer libfranka |

### 3.2 What each `rdt` import becomes

| `rdt` import | Used for | Replacement |
|---|---|---|
| `DiffIKWrapper` | Polymetis `RobotInterface` + `update_desired_ee_pose(pose_mat, dt)` | **Your adapter (§3.3).** This is the only substantial piece |
| `Spacemouse` | SpaceMouse reader | In repo: [furniture_bench/device/spacemouse/spacemouse_shared_memory.py](../furniture-bench/furniture_bench/device/spacemouse/spacemouse_shared_memory.py) (same constructor: `Spacemouse(shm_manager=..., deadzone=...)`). Needs `spnav` + `spacenavd`, see [src/real/readme.md](../src/real/readme.md) |
| `KeyboardInterface`, `CollectEnum` | Success/fail keys; rotation keys for screwing (`keyboard_action[3:7]`, a quat) | In repo: [src/data_collection/keyboard_interface.py](../src/data_collection/keyboard_interface.py) + [collect_enum.py](../src/data_collection/collect_enum.py). `get_action()` returns `(action, enum)` in the same shape. Keys: `t` success, `n` fail, `i/k/j/l/u/o` rotate. ⚠️ `minimal.py` uses `CollectEnum.RECORD` and `PAUSE_HOLD`, which the local enum doesn't have. Add them or drop that logging |
| `enable_single_realsense`, `get_default_multi_realsense_cfg` | Start one pyrealsense2 pipeline per serial; width/height/fps | About 10 lines of pyrealsense2: `rs.config().enable_device(serial)`, color (+ depth) at 640×480 |
| `PolymetisHelper` | `polypose2mat` / `mat2polypose` | scipy `Rotation`, a few lines |
| `mc_util.meshcat_frame_show` | Draws target/current frames in meshcat | Optional; call `meshcat` directly or delete |

### 3.3 Adapter interface

The client classes in `src/real/fr3/client.py` (code in §8 Step 4) expose what [teleop_sm.py](../src/real/teleop_sm.py) and [minimal.py](../src/real/minimal.py) call, so both scripts only change their imports:

```python
class FR3Robot:
    def get_ee_pose(self) -> tuple[Tensor, Tensor]        # flange pos (3,), quat xyzw (4,)
    def get_joint_positions(self) -> Tensor               # (7,)
    def get_joint_velocities(self) -> Tensor              # (7,)
    robot_model.compute_jacobian(q) -> Tensor             # (6, 7), flange frame, base coords
    def move_to_joint_positions(self, q: Tensor)          # blocking, smooth
    def start_joint_impedance(self, Kq, Kqd, adaptive=True)
    def update_desired_ee_pose(self, pose_mat, dt=0.1)    # flange target → diff-IK (§1.4) → q*

class FR3Gripper:
    def grasp(self, speed, force, blocking=False)        # Polymetis order: close until contact
    def goto(self, width, speed, force, blocking=False)
    def get_state(self) -> obj with .width
```

Keep these unchanged, or real data stops matching sim:
- **Frames:** policy and pickle poses are **fingertip** poses. Keep `convert_tip2wrist` / `convert_wrist2tip` as they are. FR3 has the same kinematic (DH) parameters and the same Franka Hand as the Panda, so the offset and the sim model still apply.
- **Diff-IK:** the same error terms as the sim controller (§1.4), recomputed at a high rate. Use damped least squares if you get near singularities.
- **Per-step clamp:** limit `‖q* − q‖` per control tick, or interpolate the target, so 10 Hz target jumps don't trip FR3 reflexes.

---

## 4. Machine and robot setup

### 4.1 Layout

```
RT control PC (PREEMPT_RT kernel) ──direct Ethernet── FR3 Control (FCI)
   runs: FCI client, 1 kHz joint impedance, diff-IK, small zmq/gRPC server
            ▲
            │ LAN @ 10 Hz: target pose + gripper cmd ↓, robot state ↑
GPU workstation: RealSense ×2 (USB 3), SpaceMouse, teleop_sm.py / minimal.py, π_real inference
```

- FCI needs a real-time kernel. NVIDIA drivers on PREEMPT_RT are painful, which is why the controller usually runs on a separate small PC.
- Teleop needs no GPU, so it can run entirely on the RT PC if that's simpler. π_real inference needs the GPU box.
- With the split, the policy env doesn't need the Python 3.8 / torch 1.13 env that Polymetis required ([polymetis_conda_env.yml](../src/real/polymetis_conda_env.yml)).

### 4.2 FR3 bring-up in Desk

1. Update the FR3 system version (if needed) and install the matching libfranka on the RT PC.
2. In Desk, unlock the joints and **activate FCI**. Keep the external enabling device and user stop within reach.
3. Run libfranka's `communication_test` and `echo_robot_state`. Packet loss in `communication_test` means the RT setup or network isn't good enough yet.

### 4.3 FR3 vs. Panda differences that matter here

- **Joint limits differ** (especially joints 4, 6 and 7), and FR3 joint velocity limits depend on joint position. The home pose above appears to be inside FR3 limits, but check it against the FR3 datasheet before the first `move_to_joint_positions`.
- **Reflexes:** stiff impedance chasing step targets can trigger velocity or torque reflex errors. Clamp per tick (§3.3), and add automatic error recovery between episodes.
- **Gains:** `Kq` / `Kqd` were tuned on a Panda. Start with them, then retune so real tracking of a 10 Hz target sequence looks like sim. Plotting commanded vs. reached poses for a replayed sim trajectory is a quick way to check.

---

## 5. Cameras and scene

### 5.1 Cameras

- Two RealSense D435: **wrist** (`color_image1`) and **front** (`color_image2`). Set the serials in [serials.py](../src/real/serials.py). Swapping them silently degrades the policy.
- **Mount both rigidly before collecting D_real, and don't move them until rollouts are done.** D_real and deployment must share the same viewpoint.
- **Calibration:** the policy uses raw RGB only, with no depth or projection, so D_real itself needs no calibration. D_synth does. The Isaac re-renderer has to match your cameras:
  - **Intrinsics:** read the factory values at 640×480 (`profile.as_video_stream_profile().get_intrinsics()`) and put them into `front_435_camera_matrix` / `wrist_435_camera_matrix` ([isaac_lab_rerender.py:136-149](../src/sim2real/isaac_lab_rerender.py#L136-L149)). No checkerboard is needed.
  - **Front pose** in the robot base frame: currently `[0.82, -0.065, 0.8]`, euler XYZ `[0, 68.5, 90]`° ([isaac_lab_rerender.py:735-741](../src/sim2real/isaac_lab_rerender.py#L735-L741)).
  - **Wrist offset** from `panda_hand`: currently `[-0.05, 0.0, 0.04]` (same area of the file).
  - Measure roughly (tape or hand-eye calibration), then refine by eye: render a frame at a known robot pose and compare it with the real frame side by side, as the authors did in [07_as_align_inspect_cameras.ipynb](../notebooks/07_as_align_inspect_cameras.ipynb). Camera-pose randomization during rendering (`-dr`) covers the remaining error.

### 5.2 Scene

3D-print the FurnitureBench square-table top, one leg and the U-shaped fixture. Place them in the robot base frame to roughly match sim, using [one_leg.png](../src/assets/calibration/one_leg.png) and [setup_front.png](../src/assets/calibration/setup_front.png) as references.

---

## 6. Teleop data collection (D_real)

```bash
meshcat-server &                      # note the port it prints
python -m src.real.teleop_sm -p 6000 --furniture one_leg \
  --save_dir $DATA_DIR_RAW/raw/diffik/real/one_leg_real/teleop/low/success
```

- **One demo per launch**, written only when you end the episode with the success key. There is no failure folder; review demos afterwards and delete bad ones.
- **Controls:** SpaceMouse for translation and rotation (`--max_pos_speed 0.3`, `--max_rot_speed 0.7`, scaled by `sm_dpos_scalar = 1.8` and `sm_drot_scalar = 4.0`), a SpaceMouse button to toggle the gripper, and keyboard rotation keys for screwing (local-frame rotation).
- **Timing:** strict 10 Hz via `precise_wait`. Steps with no input are not recorded (`action_taken`), except for the 15 steps after a gripper toggle.

What gets recorded (the format `process_pickles` expects; keep it if you rewrite the script):

| Field | Content |
|---|---|
| `observations[t].color_image1` / `color_image2` | wrist / front RGB, full resolution |
| `observations[t].robot_state` | `ee_pos`, `ee_quat` (xyzw, **tip** frame), `ee_pos_vel`, `ee_ori_vel` (J·q̇), `gripper_width` |
| `actions[t]` | delta between consecutive **teleop targets** (not measured poses): xyz, quat xyzw as a right-multiplied rotation, ±1 grasp flag |
| `furniture`, `success=True`, `args`, `metadata` | task name; teleop scalars and gains |

Then process it into D_real:

```bash
python -m src.data_processing.process_pickles -c diffik -d real -f one_leg_real -s teleop -r low -o success
```

Checks:
- View a few demos as recorded ([DATA_GUIDE.md](DATA_GUIDE.md) §9). Confirm the image content and the camera order.
- Coverage: real action xyz should fall inside the sim distribution (notebook `07_real_vs_sim_state_action_space`, paper Fig. 19).
- The paper used 10–40 real demos.

---

## 7. π_real rollout

After co-training (IMPLEMENTATION_GUIDE.md §5a; keep WandB online, because `minimal.py` loads weights from WandB):

```bash
python -m src.real.minimal -p 6000 --save_dir <eval-log-dir> --run-id <proj>/<id>        # dry run: meshcat only
python -m src.real.minimal -p 6000 --save_dir <eval-log-dir> --run-id <proj>/<id> -ex    # moves the robot
```

- The policy outputs an **absolute** tip pose `[xyz, rot6d, grip]` (`control_mode: pos`). `step()` converts it to the flange frame and calls `update_desired_ee_pose`.
- The gripper is a **toggle**: it fires when the sign flips, |grip| > 0.019 and at least 20 steps have passed since the last toggle, and then the loop **sleeps 1.5 s**.
- ⚠️ **Rollout timing is not a strict 10 Hz.** The `precise_wait` calls in `minimal.py` are commented out. The rate comes from inference time plus `time.sleep(0.025)` between actions within a chunk, while teleop ran at a strict 10 Hz. Consider re-enabling `precise_wait` so rollouts match the demo timing.
- `-w <substring>` picks the checkpoint (default `best`, i.e. `best_test_loss`).
- **Evaluation protocol (paper Table II):** 10 trials from a fixed grid of initial part poses, recording success per stage (corner / grasp / insert / screw). Reference: Real+Sim with 40 demos completed 5–6/10; Real-only completed 2–3/10.

---

## 8. Step-by-step procedure

The whole path from code changes to a real rollout, in order. It uses **panda-py** on a **separate real-time PC** (§3.1, §4.1), with a small zmq server between that PC and the workstation. The client side mimics the Polymetis API, so `teleop_sm.py` and `minimal.py` only need import changes.

| # | Step | Where | Pass criterion |
|---|---|---|---|
| 0 | Gather information and hardware | — | FR3 system version and libfranka version known |
| 1 | Prepare the RT control PC | RT PC | PREEMPT_RT kernel; libfranka examples build |
| 2 | Prepare the robot in Desk | Robot | `communication_test` shows no packet loss |
| 3 | Write the FR3 server | RT PC | Server starts and answers `state` |
| 4 | Write the client adapter and helpers | Repo | Imports work on the workstation |
| 5 | Patch the repo | Repo | `teleop_sm.py` / `minimal.py` import without `rdt` or `polymetis` |
| 6 | Set up the workstation env | Workstation | Both cameras and the SpaceMouse are visible |
| 7 | Mount cameras, build the scene | Lab | Views are fixed; intrinsics recorded |
| 8 | Bring-up tests on the robot | Both | Tracking looks like sim; gripper works |
| 9 | Collect and process D_real | Both | Zarr builds; demos look right |
| 10 | Build D_synth with your cameras, co-train | Cluster | π_real checkpoint on WandB |
| 11 | Roll out π_real | Both | Dry run is sane, then evaluate |

Code blocks in steps 3–4 are **sketches**: the panda-py / libfranka binding names follow their docs as I understand them, but check them against the version you install.

### Step 0. Gather information and hardware

1. In Desk, note the FR3 **system version**. Use Franka's compatibility matrix to pick the matching **libfranka** version, then a **panda-py** build for that libfranka version.
2. Hardware:
   - FR3 with Franka Hand, plus the external enabling device.
   - RT control PC (Ubuntu 22.04, two Ethernet ports: one for the robot, one for the LAN).
   - GPU workstation.
   - 2× RealSense D435 with USB 3 cables (an active extension for the wrist camera), and a rigid wrist mount. Sim assumes the wrist camera sits about `[-0.05, 0, 0.04]` m from `panda_hand`.
   - 3Dconnexion SpaceMouse.
   - 3D-printed square-table top, one leg and the fixture (§5.2).

### Step 1. Prepare the RT control PC

1. **Real-time kernel:** on Ubuntu 22.04, `sudo pro enable realtime-kernel` (Ubuntu Pro), or build PREEMPT_RT as described in Franka's docs. Check that `uname -a` shows `PREEMPT_RT`.
2. **Real-time permissions** (from Franka's setup docs):
   ```bash
   sudo addgroup realtime && sudo usermod -a -G realtime $USER
   sudo tee -a /etc/security/limits.conf <<'EOF'
   @realtime soft rtprio 99
   @realtime soft priority 99
   @realtime soft memlock 102400
   @realtime hard rtprio 99
   @realtime hard priority 99
   @realtime hard memlock 102400
   EOF
   ```
   Log out and back in. Set the CPU governor to `performance`.
3. **Network:** cable the robot port directly to the FR3 Control (no switch), and give the PC a static IP on the robot's subnet. Use the second port for the LAN to the workstation.
4. **libfranka:** install the version from Step 0 (Franka's .deb or a source build), including the examples.
5. **Python:** `pip install panda-python pyzmq numpy scipy`, choosing the panda-py build that matches your libfranka.

### Step 2. Prepare the robot in Desk

1. Open Desk (`https://<robot-ip>`). Set the end effector to **Franka Hand**, so the load used for gravity compensation is correct.
2. Unlock the joints and **activate FCI**. Keep the enabling device in hand from here on.
3. On the RT PC:
   ```bash
   ./examples/communication_test <robot-ip>    # must report no (or ~zero) lost packets
   ./examples/echo_robot_state <robot-ip>
   ```

### Step 3. Write the FR3 server (RT PC)

Create `src/real/fr3/server.py`. It owns the robot and the gripper, runs diff-IK toward the latest target at a high rate (§1.4), and answers a few commands over zmq.

```python
# src/real/fr3/server.py — runs on the RT PC
import threading
import numpy as np, zmq, panda_py
from panda_py import controllers, libfranka
from scipy.spatial.transform import Rotation as R

ROBOT_IP, PORT = "<fr3-ip>", 5555
panda = panda_py.Panda(ROBOT_IP)
gripper = libfranka.Gripper(ROBOT_IP)
model = panda.get_model()

lock = threading.Lock()
target = None   # 4x4 FLANGE pose, set by the client at 10 Hz
ctrl = None

def flange_pose_and_jacobian(s):
    # Use the flange frame explicitly. O_T_EE includes the Desk end-effector
    # transform (Franka Hand TCP), but the repo code expects the flange.
    T = np.array(model.pose(libfranka.Frame.kFlange, s)).reshape(4, 4, order="F")
    J = np.array(model.zero_jacobian(libfranka.Frame.kFlange, s)).reshape(6, 7, order="F")
    return T, J

def control_loop(hz=500, max_dq=0.1):
    ctx = panda.create_context(frequency=hz)
    while ctx.ok():
        with lock:
            T_goal, c = target, ctrl
        if T_goal is None or c is None:
            continue
        s = panda.get_state()
        T_ee, J = flange_pose_and_jacobian(s)
        e = np.r_[T_goal[:3, 3] - T_ee[:3, 3],
                  R.from_matrix(T_goal[:3, :3] @ T_ee[:3, :3].T).as_rotvec()]
        dq = np.linalg.lstsq(J, e, rcond=None)[0]       # same as the sim: q* = q + J⁺ e
        c.set_control(np.array(s.q) + np.clip(dq, -max_dq, max_dq))

def gripper_async(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()   # libfranka gripper calls block

sock = zmq.Context().socket(zmq.REP)
sock.bind(f"tcp://*:{PORT}")
threading.Thread(target=control_loop, daemon=True).start()
while True:
    cmd, kw = sock.recv_pyobj()
    reply = None
    if cmd == "state":
        s, g = panda.get_state(), gripper.read_once()
        T, J = flange_pose_and_jacobian(s)
        reply = dict(q=np.array(s.q), dq=np.array(s.dq), T_flange=T, J=J, width=g.width)
    elif cmd == "move_to_joints":          # only while no controller is running
        panda.move_to_joint_position(kw["q"])
    elif cmd == "start_impedance":
        with lock:
            target = None
            ctrl = controllers.JointPosition(stiffness=kw["Kq"], damping=kw["Kqd"])
        panda.start_controller(ctrl)
    elif cmd == "set_target":
        with lock:
            target = kw["T"]
    elif cmd == "grasp":                   # close until contact; wide epsilons = always "succeeds"
        gripper_async(gripper.grasp, 0.0, kw["speed"], kw["force"], 0.08, 0.08)
    elif cmd == "move_gripper":
        gripper_async(gripper.move, kw["width"], kw["speed"])
    sock.send_pyobj(reply)
```

Notes:
- The diff-IK loop runs in Python at ~500 Hz. panda-py's C++ controller interpolates and tracks at 1 kHz.
- `max_dq` caps how far `q*` can sit from `q`, which bounds the impedance torque. 0.1 rad is a starting point; lower it during bring-up.
- Add error handling: catch libfranka control exceptions, call `panda.recover()`, and report the error to the client instead of crashing.
- Add `stop_controller` and a `home` command that stops the controller, moves, and restarts it. That way the arm can be re-homed between episodes without restarting the server.

### Step 4. Write the client adapter and helpers (repo, used on the workstation)

Create the package `src/real/fr3/` (with `__init__.py`) containing:

**`client.py`**, a drop-in for `DiffIKWrapper` and `GripperInterface`:
```python
# src/real/fr3/client.py
from types import SimpleNamespace
import numpy as np, torch, zmq
from scipy.spatial.transform import Rotation as R

def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)

class _Conn:
    def __init__(self, ip, port=5555):
        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.connect(f"tcp://{ip}:{port}")
    def call(self, cmd, **kw):
        self.sock.send_pyobj((cmd, kw))
        return self.sock.recv_pyobj()

class _Model:
    def __init__(self, conn):
        self.conn = conn
    def compute_jacobian(self, q):          # Jacobian at the current state; q is ignored
        return _t(self.conn.call("state")["J"])

class FR3Robot:                             # replaces rdt DiffIKWrapper
    def __init__(self, ip_address, port=5555):
        self.conn = _Conn(ip_address, port)
        self.robot_model = _Model(self.conn)
    def get_ee_pose(self):                  # flange pos, quat xyzw (like Polymetis)
        T = self.conn.call("state")["T_flange"]
        return _t(T[:3, 3]), _t(R.from_matrix(T[:3, :3]).as_quat())
    def get_joint_positions(self):
        return _t(self.conn.call("state")["q"])
    def get_joint_velocities(self):
        return _t(self.conn.call("state")["dq"])
    def move_to_joint_positions(self, q):
        self.conn.call("move_to_joints", q=np.asarray(q, dtype=np.float64))
    def start_joint_impedance(self, Kq, Kqd, adaptive=True):
        self.conn.call("start_impedance", Kq=np.asarray(Kq), Kqd=np.asarray(Kqd))
    def update_desired_ee_pose(self, pose_mat, dt=0.1):
        self.conn.call("set_target", T=np.asarray(pose_mat, dtype=np.float64))

class FR3Gripper:                           # replaces polymetis GripperInterface
    def __init__(self, ip_address, port=5555):
        self.conn = _Conn(ip_address, port)   # second REQ socket; the REP server serves both
    def grasp(self, speed, force, blocking=False):
        self.conn.call("grasp", speed=speed, force=force)
    def goto(self, width, speed, force, blocking=False):
        self.conn.call("move_gripper", width=width, speed=speed)
    def get_state(self):
        return SimpleNamespace(width=self.conn.call("state")["width"])
```

**`utils.py`**, replacing `PolymetisHelper` and `mc_util`:
```python
# src/real/fr3/utils.py
import numpy as np, torch, meshcat.geometry as g
from scipy.spatial.transform import Rotation as R

class PolymetisHelper:
    @staticmethod
    def polypose2mat(polypose):
        pos, quat_xyzw = polypose
        T = np.eye(4)
        T[:3, :3] = R.from_quat(np.asarray(quat_xyzw)).as_matrix()
        T[:3, 3] = np.asarray(pos)
        return T
    @staticmethod
    def mat2polypose(T):
        T = np.asarray(T)
        return (torch.tensor(T[:3, 3], dtype=torch.float32),
                torch.tensor(R.from_matrix(T[:3, :3]).as_quat(), dtype=torch.float32))

def meshcat_frame_show(vis, name, T, length=0.1):
    vis[name].set_object(g.triad(length))
    vis[name].set_transform(np.asarray(T, dtype=np.float64))
```

**`realsense.py`**, replacing the `rdt.image` / `rdt.config` helpers:
```python
# src/real/fr3/realsense.py
from types import SimpleNamespace
import pyrealsense2 as rs

def get_default_multi_realsense_cfg():
    return SimpleNamespace(WIDTH=640, HEIGHT=480, FRAME_RATE=30)

def enable_single_realsense(serial, ctx, width, height, fps):
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)  # rgb8, as rdt used
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)   # scripts align depth
    pipe = rs.pipeline(ctx)
    pipe.start(cfg)
    return pipe

def enable_realsense_devices(serials, ctx, width, height, fps):
    # minimal.py iterates `for device, pipe in ...` in serial order: [wrist, front]
    return [(s, enable_single_realsense(s, ctx, width, height, fps)) for s in serials]
```

### Step 5. Patch the repo

| File | Change |
|---|---|
| [collect_enum.py](../src/data_collection/collect_enum.py) | Add `RECORD = 13` (used by `minimal.py`'s optional logging) |
| [keyboard_interface.py](../src/data_collection/keyboard_interface.py) | In `on_press`, bind a free key (e.g. `g`) to `CollectEnum.RECORD` |
| [serials.py](../src/real/serials.py) | Your wrist and front serials (`rs-enumerate-devices`) |
| [teleop_sm.py](../src/real/teleop_sm.py) | Swap the imports (below). Set `franka_ip` (line 387) to the **RT PC's IP** |
| [minimal.py](../src/real/minimal.py) | Swap the imports in **both** duplicated import blocks (lines 15–43) and drop `lcm` / `get_realsense_rgbd_subscribers`. Set `franka_ip` (line 656). Replace `CollectEnum.PAUSE_HOLD` with `CollectEnum.PAUSE` (line 824; key `p`). Optionally restore strict 10 Hz timing (§7) |
| [isaac_lab_rerender.py](../src/sim2real/isaac_lab_rerender.py) | Your intrinsics and camera poses (Step 7, §5.1) |

Import swap for both scripts. The aliases keep every other line unchanged:

```python
# remove
from polymetis import GripperInterface, RobotInterface
from rdt.spacemouse.spacemouse_shared_memory import Spacemouse
from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.interfaces.diffik import DiffIKWrapper
from rdt.common import mc_util
from rdt.common.keyboard_interface import KeyboardInterface
from rdt.common.demo_util import CollectEnum
from rdt.image.factory import enable_single_realsense            # teleop_sm.py
from rdt.image.factory import enable_realsense_devices           # minimal.py
from rdt.image.factory import get_realsense_rgbd_subscribers     # minimal.py
import lcm                                                       # minimal.py

# add
from furniture_bench.device.spacemouse.spacemouse_shared_memory import Spacemouse   # teleop only
from src.data_collection.keyboard_interface import KeyboardInterface
from src.data_collection.collect_enum import CollectEnum
from src.real.fr3.client import FR3Robot as DiffIKWrapper, FR3Gripper as GripperInterface
from src.real.fr3.utils import PolymetisHelper
from src.real.fr3 import utils as mc_util
from src.real.fr3.realsense import (
    get_default_multi_realsense_cfg, enable_single_realsense, enable_realsense_devices,
)
```

The local `KeyboardInterface.get_action()` returns `(action, enum)`, with the rotation quat at `action[3:7]`, and resets the enum after each read. That is the shape `teleop_sm.py` expects. Its translation keys (`w/s/a/d/q/e`) and `z` are ignored by teleop.

### Step 6. Set up the workstation env

1. Start from the repo's training env, since `minimal.py` imports `src.behavior`, `pytorch3d`, `wandb` and `furniture_bench` (`pip install -e furniture-bench`). Polymetis is not needed.
2. Add:
   ```bash
   pip install pyrealsense2 meshcat pynput pyzmq
   sudo apt install libspnav-dev spacenavd && sudo systemctl start spacenavd
   pip install git+https://github.com/cheng-chi/spnav
   ```
3. Check: `rs-enumerate-devices` lists both serials on **USB 3**, and the SpaceMouse moves `spnav` output.
4. `pynput` (keyboard) needs a desktop session. It won't see keys over plain SSH.

### Step 7. Mount the cameras and build the scene

1. Mount the wrist and front cameras **rigidly** (§5.1), roughly where sim puts them. From now until the rollouts are finished, don't touch them.
2. Place the fixture and parts per §5.2.
3. Record both cameras' intrinsics at 640×480 and write them into `isaac_lab_rerender.py`. Measure the front camera pose and the wrist offset roughly; you'll refine them in Step 10.

### Step 8. Bring-up tests on the robot

Start the server on the RT PC (`python -m src.real.fr3.server`), then run short scripts from the workstation. Do each step with the enabling device in hand.

| Test | How | Pass criterion |
|---|---|---|
| a. State | Print `get_ee_pose()`; draw flange and tip frames in meshcat | Flange pose matches Desk; tip frame sits between the fingers, z pointing out |
| b. Home | Check the home pose against FR3 limits, then `move_to_joint_positions(home)` | Smooth move, no errors |
| c. Hold | `start_joint_impedance(Kq, Kqd)`, then push the arm by hand | Compliant, returns, no reflex errors |
| d. Small steps | From the current tip pose, send +1 cm in x every 0.1 s for 10 steps via `update_desired_ee_pose(convert_tip2wrist(T))` | Arm follows; no overshoot or jerks |
| e. Sim replay (no parts) | Take `robot_state` tip poses from a sim demo pickle, shift them to start at the current pose, send them as targets at 10 Hz, and plot commanded vs. reached | Lag and error similar to sim; tune `Kq`/`Kqd`/`max_dq` if not |
| f. Gripper | `grasp(0.07, 70)`, then `goto(0.065, 0.05, 0.1)` | Closes until contact; opens |

### Step 9. Collect and process D_real

1. `meshcat-server &`, then run `teleop_sm.py` as in §6 (one launch per demo, `t` to save, `n` to discard).
2. After the first demo, process it and look at it before collecting more:
   ```bash
   python -m src.data_processing.process_pickles -c diffik -d real -f one_leg_real -s teleop -r low -o success
   ```
   Check the camera order, that the colors look natural, and that actions are small and smooth. Use the viewers in [DATA_GUIDE.md](DATA_GUIDE.md) §9.
3. Collect 10–40 demos, varying the initial part poses as in the evaluation grid. Re-process at the end and run the coverage check (§6).

### Step 10. Build D_synth with your cameras and co-train

1. Align the sim cameras to yours (§5.1). Render one frame at a known robot pose and compare it with the real frame, adjusting the front pose and wrist offset until the table, fixture and gripper line up.
2. Re-render the sim rollouts and demos, process them into zarrs, and co-train π_real: IMPLEMENTATION_GUIDE.md stages 3b–3c and 5a. Keep WandB online.

### Step 11. Roll out π_real

1. Reset the scene to a grid pose, and start the server and `meshcat-server`.
2. **Dry run:** `python -m src.real.minimal -p 6000 --save_dir <eval-log-dir> --run-id <proj>/<id>`. Targets in meshcat should be plausible, and the loop rate stable.
3. **Execute:** add `-ex`. Keep the enabling device in hand. `p` pauses the policy.
4. Run 10 trials from the fixed grid, recording per-stage success (§7). Re-home between trials (Step 3 notes).

---

## 9. Unverified points

These come from reading the code and general Franka knowledge. Check them on your setup:
- The minimum libfranka version for your FR3 system version, and whether your chosen client (e.g. panda-py) has a build for it.
- That the home pose is inside FR3 joint limits.
- The exact panda-py / libfranka binding names used in the Step 3 sketch (`create_context`, `controllers.JointPosition(stiffness, damping)`, `get_model().pose / zero_jacobian`, `Gripper.grasp / move / read_once`), and whether `read_once()` is safe while a blocking gripper command runs in another thread. If it isn't, cache the gripper state in a background thread.
- Resolved: `rdt` streams color as **rgb8** (`lars-dev` `image/factory.py`), matching the sim, and its `DiffIKWrapper` evaluates diff-IK once per command (§1.4).
