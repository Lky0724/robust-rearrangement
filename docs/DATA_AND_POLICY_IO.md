# ResiP Data and Policy I/O Reference

What each stage of the ResiP pipeline reads and writes, and what each policy takes in and puts out: modalities, shapes and coordinate frames. Dimensions are for `one_leg` and were read from the code (not the paper). For the commands and the file paths of each stage, see `IMPLEMENTATION_GUIDE.md`.

Every policy outputs the same thing: a 10-D target end-effector pose plus a gripper command. Only what each policy observes changes.

---

## 1. Shared vectors

| Vector | Dim | Layout (index ranges) | Frame and notes |
|---|---|---|---|
| **robot_state** | 16 | `[0:3]` EE position · `[3:9]` EE rotation (6D) · `[9:12]` linear velocity · `[12:15]` angular velocity · `[15]` gripper width | Robot base frame. The env and robot emit **14-D** (quaternion instead of 6D); the actor converts it (`proprioceptive_quat_xyzw_to_rot_6d`). |
| **parts_poses** | 42 | 6 objects × `[pos(3), quat_xyzw(4)]`: tabletop, leg1, leg2, leg3, leg4, fixture front | AprilTag/world frame (`parts_poses_in_robot_frame=false`), **not** the robot base frame. The RL env appends the fixture pose (`furniture_rl_sim_env.py:1332`). Other tasks: (parts + 1) × 7, e.g. 28 for lamp and round_table. |
| **action** | 10 | `[0:3]` target EE position · `[3:9]` target EE rotation (6D) · `[9]` gripper (+1 close / −1 open) | **Absolute** pose in the robot base frame, at 10 Hz. Diff-IK turns it into joint targets. |

All three are min-max normalized to [−1, 1] before entering a network, then clamped to ±3.

---

## 2. The three policies

| | **π_base** (state diffusion) | **π_res** (ResiP residual) | **π_real** (RGB diffusion) |
|---|---|---|---|
| **Sees** | robot_state 16 + parts_poses 42 = **58-D** | 58-D state + π_base's current action 10 = **68-D** | robot_state 16 + wrist RGB 3×224×224 + front RGB 3×224×224 |
| **Outputs** | **32×10** action chunk; first 8 steps executed | **10-D** Gaussian correction × 0.1, added to the base action every step | **32×10** action chunk; first 8 steps executed |
| **Network** | Conditional 1D UNet, down dims [256, 512, 1024], ~66M params | Actor MLP 68→256→256→10 (ReLU), fixed log-std −1; critic MLP 68→256→256→1 | Two ResNet18s (R3M weights) → 512 → Linear → 128 each; 16 + 128 + 128 = **272-D** condition for a transformer (8 layers, 4 heads, width 256) |
| **Trained on** | D_sim: ~50 teleop demos | Online sim rollouts; reward 1 per successful assembly; PPO | D_real (10–40 demos) ∪ D_synth (~400 rendered rollouts) |
| **Loss** | Diffusion noise prediction (MSE) | PPO clipped objective + value loss | Diffusion noise prediction (+ optional sim/real feature-alignment loss) |
| **Runs in** | Sim only | Sim only (asserts state observations) | Sim evaluation and the real robot |

The diffusion policies sample with DDIM: 16 steps by default, 4 inside the RL loop (`residual_ppo.py:208`). Each new chunk starts from the previous chunk's unused actions, re-noised, instead of from pure noise.

---

## 3. Stage by stage

### Stage 1: D_sim → π_base

**Teleop pickle**: one file per episode of T steps, at `$DATA_DIR_RAW/raw/diffik/sim/one_leg/teleop/<rand>/success/*.pkl`.

| Key | Shape / type | Content |
|---|---|---|
| `observations[t].robot_state` | dict | `ee_pos` 3, `ee_quat` 4, `ee_pos_vel` 3, `ee_ori_vel` 3, `gripper_width` 1, plus joint positions, velocities and torques (7 each) and finger positions. Only the first five are used later. |
| `observations[t].parts_poses` | (42,) float | Pose vector above |
| `observations[t].color_image1` / `color_image2` | (240, 320, 3) uint8 | Wrist / front RGB. The sim renders at 1280×720; the collector shrinks the images before saving. |
| `actions` | T × 8 | **Delta** actions: Δpos 3, Δquat 4, gripper 1 |
| `rewards`, `skills` | T | Sparse reward and skill markers |
| `success`, `furniture` | scalar | Episode label |

There are T+1 observations; the frame after the final action is saved too.

**Processed zarr**: `$DATA_DIR_PROCESSED/processed/diffik/sim/one_leg/teleop/<rand>/success.zarr`. N = total frames, E = episodes.

| Key | Shape | Note |
|---|---|---|
| `robot_state` | (N, 16) f32 | Quaternion converted to 6D |
| `parts_poses` | (N, 42) f32 | |
| `color_image1`, `color_image2` | (N, 240, 320, 3) u8 | Wrist resized; front resized and cropped |
| `action/pos` | (N, 10) f32 | Absolute pose rebuilt from the deltas; **used for training** |
| `action/delta` | (N, 10) f32 | Deltas, converted to 6D |
| `reward`, `skill`, `augment_states` | (N,) | |
| `episode_ends` | (E,) | End index of each episode |
| `task`, `success`, `pickle_file` | (E,) | Per-episode metadata |

**Training sample** (`StateDataset`, already normalized):
- `obs`: (1, 58). Current frame only; the observation horizon is 1.
- `action`: (32, 10). The next 32 absolute actions, padded with the last action at the episode end.
- Images are in the zarr but **not loaded**.

**Training step:** add noise ε with shape (B, 32, 10) at a random diffusion step k. The UNet takes (noisy chunk, k, 58-D condition) and predicts ε; the loss is the MSE.

**Output:** a checkpoint `.pt` with the UNet weights, the normalizer's min/max for `robot_state`, `parts_poses` and `action`, and the full config.

### Stage 2: π_res (residual RL)

**Environment interface**, per step, for each of `n_envs` parallel envs:

| | Shape | Content |
|---|---|---|
| Env → policy | dict: `robot_state` (n_envs, 14), `parts_poses` (n_envs, 42) | Exact simulator state, clamped to ±3 by the wrapper |
| Policy → env | (n_envs, 10) | Final absolute action (base + residual, de-normalized) |
| Reward | (n_envs,) | 1 when the tabletop and leg4 reach the assembled pose, else 0. Normalized with running statistics and clipped to ±5. |
| Done / truncated | (n_envs,) bool | Success, or `num_env_steps` (700) reached |

**Inside one step** (`residual_ppo.py:347-371`):
1. π_base returns the next normalized action (10) from its queue. A new 32-step chunk is sampled every 8 steps.
2. The state is converted to 58-D and normalized, then concatenated with that action into a 68-D residual input.
3. The actor outputs a mean correction (10). During training a sample is drawn with σ = e⁻¹; during evaluation the mean is used.
4. Final normalized action = base + 0.1 × correction. It is de-normalized with **π_base's** normalizer and sent to the env.

**PPO buffers per iteration** (S = 700):
- `obs` (S, n_envs, 68)
- `actions` (S, n_envs, 10)
- `logprobs`, `rewards`, `values`, `dones`: (S, n_envs) each

**Input:** π_base's checkpoint; no dataset is read. **Output:** a checkpoint holding π_base and the residual's actor and critic. The combined policy maps 58-D state to a 10-D action at every step.

### Stage 3: expert rollouts → D_synth

**Rollout pickle** (`evaluate_model --save-rollouts`): the same format as the teleop pickle, except:
- **Images are 2×2 placeholders**; only states are recorded.
- The 10-D absolute actions are converted back to **8-D deltas** when saving (`src/data_collection/io.py`).
- The file is trimmed at the success step.

**Re-render** (`isaac_lab_rerender.py`):
- **Input:** one rollout pickle. It uses the robot joint positions from each frame's `robot_state` dict, plus `parts_poses`.
- **Output:** a new pickle with the same states and actions and real RGB rendered at **640×480** with domain randomization.

**Processing:** images are resized to 240×320, giving the same zarr schema as stage 1. `parts_poses` is kept but never read by π_real.

### Stage 4: D_real

**Real teleop pickle** (`teleop_sm.py`):
- `robot_state` dict from Polymetis (pose, velocities, gripper width)
- Wrist and front RGB at the RealSense stream resolution. It comes from the `rdt` camera config, typically 640×480; check yours.
- 8-D delta actions; `success=True`
- **No `parts_poses`**
- T observations for T actions (no extra final frame)

After `process_pickles -d real`: the stage-1 zarr schema, with empty `parts_poses` and 240×320 images.

### Stage 5: π_real

**Training sample** (`ImageDataset`, mixing real and sim zarrs):

| Key | Shape | Note |
|---|---|---|
| `robot_state` | (1, 16) | Normalized with a normalizer fitted on the **combined** real + sim data |
| `color_image1` | (1, 3, 240, 320) uint8 | Wrist |
| `color_image2` | (1, 3, 240, 320) uint8 | Front |
| `action` | (32, 10) | Normalized absolute poses |
| `domain` | (1,) | 0 = sim, 1 = real; only used by the optional feature-alignment loss |

**Inside the policy** (`src/behavior/base.py:427`):
1. Each image is transformed to 224×224:
   - Wrist: color jitter and blur, then resize.
   - Front: color jitter and blur, then a random 224×224 crop. At inference, a center crop.
2. Each image goes through its ResNet18 (512), then Linear (128), then LayerNorm.
3. The result is concatenated with robot_state: 16 + 128 + 128 = 272-D.
4. The transformer diffusion model is conditioned on it and predicts the noise on a 32×10 chunk.

**Output:** a checkpoint with both encoders, the projections, the transformer and the combined normalizer.

**Deployment** (`src/real/minimal.py`, 10 Hz):
- **Reads:** RealSense RGB, resized and cropped to 240×320 as in processing, then center-cropped (front) or resized (wrist) to 224×224. Also the 14-D robot state from Polymetis, converted to 16-D.
- **Every 8 steps:** encoders plus 16 DDIM steps produce a (32, 10) chunk; the first 8 actions are queued.
- **Every step:** pops one 10-D action, converts it to a 4×4 target pose, moves it from the gripper-tip frame to the wrist frame, and sends it to Polymetis diff-IK. The gripper command is ±1.

---

## 4. Dimension mismatches that fail silently

1. **The 58 depends on the task.** `parts_poses` is (parts + 1) × 7, so a π_base from one task can't serve as the base for another; the residual input size won't match.
2. **The fixture pose is inside `parts_poses`.** Anything that consumes part poses must expect 6 objects for `one_leg`, not 5.
3. **Two different frames.** `parts_poses` are in the AprilTag/world frame and the end-effector pose is in the robot base frame. This only matters for state policies, but keep it in mind when adding state features.
4. **Camera order.** Wrist is always `color_image1` and front always `color_image2`, in the pickles, the zarrs and `minimal.py`. They get different transforms, so swapping them degrades the policy without any error.
