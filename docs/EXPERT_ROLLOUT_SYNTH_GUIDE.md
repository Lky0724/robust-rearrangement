# Expert Rollout and Synthetic Data Guide (Stage 3, D_synth → D'_synth)

How to turn the trained ResiP policy (π_base + π_res) into the synthetic image dataset that Stage 5 co-trains on: what each step does, what it writes, which code has to be fixed first, and how to check the output before moving on.

Notation used here:

| Name | Paper | What it is | Where it lives |
|---|---|---|---|
| **D_synth** | D_synth | Successful ResiP rollouts, **state only** (2×2 placeholder images) | `$RAW/diffik/sim/one_leg/rollout/med/success/*.pkl` |
| **D'_synth** | D_synth-render | The same trajectories re-rendered in Isaac Sim with realistic, domain-randomized wrist and front images | `$RAW/diffik/sim/one_leg_render_rppo/rollout/med/success/*.pkl` → `$PROC/…/success*.zarr` |

`$RAW` = `$DATA_DIR_RAW/raw`, `$PROC` = `$DATA_DIR_PROCESSED/processed`, as in the Implementation Guide.

Numbers in the examples come from one run of the 3a command with a med-randomness ResiP checkpoint on an RTX 3000 Ada laptop GPU (8 GB), 60 GB RAM. That run produced **425 successful episodes**.

---

## 1. The idea in one paragraph

π_res can't leave the simulator: it observes exact part poses (`parts_poses`, 42-D), which a real robot doesn't have. So it serves as a **teacher**. Stage 3 records the teacher solving the task a few hundred times (D_synth). Those recordings hold states and actions, but no usable images, because the fast IsaacGym rollout doesn't render. A second simulator, Isaac Sim, then **replays** each trajectory frame by frame (it sets joint and part poses directly, with no physics) and renders photorealistic wrist and front images with randomized lighting, colors and camera pose (D'_synth). In Stage 5, an RGB **student** policy π_real is trained by plain BC on D'_synth together with a few real demos. No learning happens in Stage 3; it only produces data.

```
 π_res ckpt ──3a──► D_synth: state-only .pkl ──3d──► D'_synth: .pkl with images ──3e──► .zarr ──► Stage 5
             IsaacGym, resip env (py3.8)     Isaac Sim + Isaac Lab, separate env (py3.10)   resip env
```

---

## 2. The steps at a glance

| Step | What | Env | Reads | Writes | Status in this repo |
|---|---|---|---|---|---|
| **3a** | Roll out ResiP, keep successes | `resip` | med ResiP ckpt | D_synth pickles | Works as is |
| **3b** | Fix two bugs in the processing and re-render code | — | — | code | **Required** (§4) |
| **3c** | Install Isaac Sim + Isaac Lab | new `isaaclab` env | — | — | Manual setup (§5) |
| **3d** | Match the render cameras to your real cameras | `isaaclab` | camera intrinsics and poses | code constants | Required for your hardware (§6) |
| **3e** | Re-render every D_synth episode | `isaaclab` | D_synth pickles | D'_synth pickles | Slow; run in a resumable loop (§7) |
| **3f** | Process D'_synth into zarr(s) | `resip` | D'_synth pickles | zarr(s) | Needs chunking on 60 GB RAM (§8) |
| (opt.) | Re-render the teleop demos too | `isaaclab` | raw teleop pickles | rendered demo zarr | §9 |

---

## 3. Step 3a: generate expert rollouts (D_synth)

### 3.1 Command

```bash
python -m src.eval.evaluate_model --wt-path <med rppo ckpt> -f one_leg --randomness med \
  --n-envs 256 --n-rollouts 100000 --max-rollout-steps 750 \
  --action-type pos --observation-space state \
  --save-rollouts --break-on-n-success --stop-after-n-success 400
```

Use the **med** checkpoint. The paper generates D_synth at med randomness because its wider spread of initial part poses gives the student more coverage (paper §II-D, App. III; Fig. 19 compares the sim and real action distributions). The authors' rendered set `one_leg_render_rppo_1` has 345 episodes; the paper says "approximately 400".

### 3.2 What a run does

```
┌─ Setup ── load ckpt (π_base + π_res + normalizer); DDIM steps forced to 4
│           build 256 IsaacGym envs at med randomness (one-time)
│
├─ Round r = 1, 2, … (each round = all 256 envs in parallel)
│     reset all envs → new random part poses and obstacle position
│     for step in 1..750:
│         base  = π_base chunk (re-planned every 8 steps)         # stochastic: diffusion noise
│         act   = base + 0.1 × mean(π_res(state ⊕ base))          # residual is deterministic (no sampling)
│         env.step(act)
│     success = env reward sum == 1 (one_leg)
│     n_success += successes of this round
│     if n_success ≥ 400: stop                                    # checked only AFTER a full round
│
└─ Save ── every success of every round → one pickle each
```

Consequences:

- **You get more than 400.** The stop check runs only between rounds, and the whole last round is kept. With a success rate around 83%, round 1 gave fewer than 400 and round 2 brought the total to 425 (of 512). To get exactly N, keep the overshoot and drop extras in 3e/3f with `--max-files`, or delete files.
- **`--n-rollouts` is only an upper bound** (it sets the progress-bar length, 100000 // 256 = 390 rounds).
- **Re-running appends.** File names are timestamps, so a second run adds files to the same folder. To keep runs apart, use `--save-rollouts-suffix <name>` (writes `…/rollout/med/<name>/success/`) or clear the folder first.
- **The episodes are diverse even though the residual is deterministic:** the initial state is random, and π_base samples its chunks from noise.

### 3.3 What gets saved

One pickle per successful env at `$RAW/diffik/sim/one_leg/rollout/med/success/<ISO timestamp>.pkl`:

| Key | Content |
|---|---|
| `observations[t].robot_state` | dict: `ee_pos`, `ee_quat`, `ee_pos_vel`, `ee_ori_vel`, `gripper_width`, **`joint_positions` (7)**, `joint_velocities` (7), `joint_torques` (9), **`gripper_finger_1_pos`, `gripper_finger_2_pos`** |
| `observations[t].parts_poses` | (42,): 6 objects × (pos, quat xyzw), AprilTag frame |
| `observations[t].color_image1/2` | (2, 2, 3) zeros: placeholders |
| `actions` | T × 8 **deltas**: the 10-D absolute action converted to Δpos (target − EE pos), Δquat (EE quat⁻¹ · target quat), and gripper (continuous, e.g. −0.98) |
| `rewards` | T values; a single 1 at the success step |
| `success`, `task`, `action_type` | `True`, `one_leg`, `pos` |

- **Length:** each episode is cut at **success step + 30** (`n_steps_padding=30` in `calculate_success_rate`), so it ends with 30 steps of the assembled state. T actions and T + 1 observations.
- **The joint and finger positions are what 3e replays.** Isaac Sim never sees the actions.

Example run:

| Quantity | Value |
|---|---|
| Episodes | 425 |
| Total steps | 200,695 |
| Episode length (mean / min / max) | 472 / 398 / 743 |
| Success step (mean / min / max) | 442 / 368 / 713 |
| Size on disk | 168 MB |

No episode succeeded after step 720, so `--max-rollout-steps 750` loses nothing compared with longer limits. It does keep a few that a 700-step limit would cut off (the latest success was at 713).

### 3.4 Console output

```
Performing rollouts (one_leg): round 2/390, success: 425/512 (83.0%): …
Current number of success 425 greater than breaking threshold 400. Breaking
Checking if we should save rollouts (rollout_save_dir: …/raw/diffik/sim/one_leg/rollout/med)
Saving rollouts, have image observations: False (will make dummy video if False)
Success rate: 83.01% (425/512)
Not writing to wandb
```

`have image observations: False` is expected with `--observation-space state`.

### 3.5 Check D_synth before rendering it

```bash
python scripts/visualize_data.py info $RAW/diffik/sim/one_leg/rollout/med/success --max-files 50
python scripts/visualize_data.py plot $RAW/diffik/sim/one_leg/rollout/med/success --max-files 50 --episodes 0 1 2
python scripts/replay_in_sim.py …     # DATA_GUIDE §9.3: watch an episode, since there are no images
```

- `info` warns about gripper actions other than ±1. That's expected: the policy's gripper output is continuous, and processing takes its sign.
- In the `plot` overview, the initial part positions should be spread over the med-randomness area, and the EE paths should look smooth.

**Sanity facts already verified on the example data:** the EE reaches the commanded target within about 2 steps (angle to the target 0.137 rad at t, 0.039 rad at t + 2). So the stored deltas are real tracking leads, not a frame or quaternion-convention mismatch. The leads are large, though: mean rotation lead 0.138 rad, position lead up to 12.8 cm. That matters in §4.1.

---

## 4. Step 3b: required code fixes

Two bugs make the D'_synth zarr wrong or make processing crash. Fix both before running 3e/3f.

### 4.1 Rotation clipping scales the whole episode (`src/data_processing/utils.py`)

`process_pickles.py` clips every delta action: position to ±2.5 cm per axis, rotation to 0.35 rad. The rotation clip is buggy:

```python
magnitude = np.linalg.norm(delta_rotvec)        # Frobenius norm of the whole (T, 3) array, not per step
if magnitude > clip_mag:
    delta_rotvec = scale_factor * delta_rotvec  # scales EVERY step of the episode
```

On the example D_synth, this scales **every rotation delta in every episode by 0.071–0.098**. `action/pos` is rebuilt from these deltas, so its rotation targets would sit ~11× closer to the current EE orientation than the teacher commanded. The student would learn to barely rotate. It also affects teleop data: 6 of the 8 local teleop pickles in `$RAW/…/one_leg/teleop/low/success` would be scaled.

Fix (per-step clip):

```python
def clip_quat_xyzw_magnitude(delta_quat_xyzw: np.ndarray, clip_mag=0.35) -> np.ndarray:
    """
    Clips the rotation magnitude of each delta quaternion
    """
    assert delta_quat_xyzw.shape[-1] == 4

    delta_rotvec = R.from_quat(delta_quat_xyzw).as_rotvec()

    magnitude = np.linalg.norm(delta_rotvec, axis=-1, keepdims=True)
    scale_factor = np.minimum(1.0, clip_mag / np.maximum(magnitude, 1e-8))

    return R.from_rotvec(scale_factor * delta_rotvec).as_quat()
```

With the fix, the example data has 2.3% of steps rotation-clipped and 7.1% position-clipped. Every episode has at least one clipped step.

**Keep or drop the clipping?** The clipped targets still point in the teacher's direction; they only shorten the lead. Keeping the clip (recommended) makes D'_synth consistent with D_real, which goes through the same processing, and caps how far ahead the real-robot policy commands. To keep the teacher's actions exactly, add a flag that skips lines 132–139 of `process_pickles.py` for rollout data, and note it in the zarr attrs.

### 4.2 The Isaac Lab re-renderer saves mismatched image sizes (`src/sim2real/isaac_lab_rerender.py`)

The save block (around line 1119) stores the wrist image at 480×640 and the front image **upsampled to 720×960** (the script's own `resize` targets `th, tw = 720, 960`), plus a third `color_image3`. `process_pickles.py` asserts that both cameras have the same shape, so processing fails with `Color images have different shapes`. At full resolution the pickles would also take ~370 GB for 425 episodes.

The older `isaac_sim_raytrace.py`, which the authors used for their released data, saves both cameras at 240×320, the zarr resolution. Do the same:

1. In the script's `resize` (line ~341), set `th, tw = 240, 320`. The 640×480 render is 4:3, so this matches what `resize`/`resize_crop` in processing would do.
2. Replace the save block:

```python
        if args_cli.save:
            # Both cameras at 240x320, as in isaac_sim_raytrace.py and the zarr schema
            wrist_image = resize(wrist_camera.data.output["rgb"][0, :, :, :3]).cpu().numpy()
            front_image = resize(camera.data.output["rgb"][0, :, :, :3]).cpu().numpy()

            new_obs = dict(
                color_image1=wrist_image,
                color_image2=front_image,
                robot_state=obs["robot_state"],
                parts_poses=obs["parts_poses"],
            )
            episode_data["observations"].append(new_obs)
```

3. Next to `episode_data["actions"] = data["actions"]` (line ~892), also carry the rewards over:

```python
    if "rewards" in data:
        episode_data["rewards"] = data["rewards"]
```

At 240×320, a 472-step episode is ~220 MB, so ~93 GB for 425 episodes (779 GB free on the example machine).

### 4.3 Recommended: make re-rendering resumable

Each call renders one episode, picked by `-i`. The script **shuffles** the file list with seed 0 (lines 117–119) and names its output by the current time. So you can't tell which input a given output came from, and you can't skip finished episodes after a crash. Change two lines:

```python
FILES = sorted(list(DEMO_DIR.rglob("*.pkl")) + list(DEMO_DIR.rglob("*.pkl.xz")))
# (delete the np.random.seed(0) / np.random.shuffle(FILES) lines)
...
        pkl_path = demo_save_dir / Path(FILE).name     # was: f"{datetime.now()...}.pkl"
```

Then `-i` is the index in sorted order, each output keeps its input's name, and the loop in §7.2 can skip episodes that are already done. The output file is only written at the end of an episode, so a crashed episode leaves nothing behind and gets redone.

---

## 5. Step 3c: set up Isaac Sim + Isaac Lab

`isaac_lab_rerender.py` imports `omni.isaac.lab.*` and `omni.isaac.lab_assets.FRANKA_PANDA_CFG`. That is the **Isaac Lab 1.x** namespace on **Isaac Sim 4.x** (the readme says 4.1). Isaac Lab 2.x renamed the package to `isaaclab` and needs Isaac Sim 4.5+, so the script won't import there without porting. It needs its own conda env with Python 3.10; the `resip` env (Python 3.8, IsaacGym) can't host it.

```bash
conda create -n isaaclab python=3.10 -y
conda activate isaaclab
# Follow the Isaac Lab 1.x pip-installation page for the exact pins, e.g. for Isaac Lab 1.4 / Isaac Sim 4.2:
#   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
#   pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 \
#       isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
#   git clone -b v1.4.1 https://github.com/isaac-sim/IsaacLab.git && (cd IsaacLab && ./isaaclab.sh --install)

# What the re-render script imports from this repo
pip install gym numba scipy opencv-python imageio matplotlib ipython ipdb tqdm

# Every session, from the ResiP repo root
export PYTHONPATH=$PWD:$PWD/furniture-bench
export RARL_SOURCE_DIR=$PWD/src            # the script loads USD assets from $RARL_SOURCE_DIR/sim2real/assets/...
```

- Don't `pip install -e .` the ResiP package into this env: its dependency pins target Python 3.8. `PYTHONPATH` is enough. `furniture_bench` only needs `gym` and `numba` here; its `isaacgym` import is wrapped in a try/except.
- **Risks on the example machine:** Isaac Sim 4.x officially supports Ubuntu 20.04/22.04 (this machine runs 24.04), and an 8 GB RTX GPU is at its minimum spec. Run the smoke test in §7.1 before planning around it.
- If 3e can't run locally, run it on another machine or in the cloud. It only needs the D_synth pickles, the repo and the USD assets in `src/sim2real/assets`, and it writes pickles you copy back.

---

## 6. Step 3d: match the cameras to your real setup

The student is trained on rendered images and then sees real ones, so the rendered views must look like your real cameras. Everything is hard-coded in `isaac_lab_rerender.py`:

| What | Where | Current value | Set it to |
|---|---|---|---|
| Front intrinsics | `front_435_camera_matrix` (lines 136–142) | the authors' D435 at 640×480 | your front RealSense's factory intrinsics at 640×480 (`profile.as_video_stream_profile().get_intrinsics()`) |
| Wrist intrinsics | `wrist_435_camera_matrix` (lines 143–149) | the authors' D435 | your wrist camera's |
| Front camera pose | lines 735–741 | pos `[0.82, -0.065, 0.8]`, euler XYZ `[0, 68.5, 90]`° | your front camera's pose |
| Wrist camera pose | lines 744–751 | local to `panda_hand`: pos `[-0.05, 0, 0.04]`, euler XYZ `[0, 200, -90]`° | your wrist mount |
| Part color base | env var `PART_COLOR_BASE` | `white` | `white` or `black`, like your printed parts |

**Frames:** the front pose is in the Isaac **world/table frame**, not the robot base frame. The script places the robot base at `(-0.3, 0, 0.415)` in that frame (`franka_from_origin_mat`), so the current front camera sits at `(1.12, -0.065, 0.385)` relative to the robot base. Convert your measured pose the same way.

Procedure:

1. Put the real robot at a known joint configuration and take a front and wrist image.
2. Render the first frame of one episode without `--save` (§7.1) with the robot at a similar pose. Compare side by side, as the authors did in `notebooks/07_as_align_inspect_cameras.ipynb` / `07_la_sim_real_visual_align.ipynb`.
3. Adjust the pose until the table edge, fixture and gripper fingers line up. The camera-pose jitter of `-dr` (σ = 2.5 mm, 0.5° front; 0.75 mm, 0.25° wrist) only covers small residual errors.

The render uses the Isaac Lab **Panda** model. For a Franka Research 3 the kinematics and the Franka Hand are the same; only the arm links look slightly different in the front view.

---

## 7. Step 3e: re-render D_synth → D'_synth

### 7.1 Smoke test: one episode, no saving

```bash
LOAD=$DATA_DIR_RAW/raw/diffik/sim/one_leg/rollout/med/success
python -m src.sim2real.isaac_lab_rerender --headless --enable_cameras -i 0 --sub-steps 3 \
  --load-dir $LOAD --furniture one_leg --num-parts 5 \
  -dr rand --part-random full --table-random full
```

- Look for `[INFO]: Setup complete...`, then `Frames per second: …` lines, then `Time taken: <s>`. **The time for this one episode × 425 is your rendering budget**; it also includes Isaac Sim startup, which repeats for every episode.
- `--enable_cameras` is needed for camera sensors in headless mode in Isaac Lab 1.x. If your AppLauncher version doesn't know the flag, drop it.
- Then repeat with `--save --save-dir <scratch dir>` and inspect the pickle (`notebooks/09_as_rerender_pkl_viz.ipynb`, or `python scripts/visualize_data.py video <scratch dir>`).

### 7.2 Full run

After the §4.3 change, run this from the repo root in the `isaaclab` env:

```bash
LOAD=$DATA_DIR_RAW/raw/diffik/sim/one_leg/rollout/med/success
SAVE=$DATA_DIR_RAW/raw/diffik/sim/one_leg_render_rppo/rollout/med/success
mkdir -p $SAVE
mapfile -t FILES < <(ls $LOAD/*.pkl | LC_ALL=C sort)
for i in "${!FILES[@]}"; do
  name=$(basename "${FILES[$i]}")
  [ -f "$SAVE/$name" ] && continue                      # already rendered
  python -m src.sim2real.isaac_lab_rerender --headless --enable_cameras -i $i --sub-steps 3 \
    --load-dir $LOAD --furniture one_leg --num-parts 5 --save --save-dir $SAVE \
    -dr rand --part-random full --table-random full \
    || echo "FAILED $i $name" >> $SAVE/../rerender_failures.txt
done
```

Re-running the same loop resumes where it stopped. `LC_ALL=C sort` matches Python's `sorted()` order for these ISO-timestamp names.

### 7.3 What one call does

```
┌─ Launch Isaac Sim (AppLauncher), load table room, background, fixture walls,
│  5 parts (tabletop + 4 legs, USD from src/sim2real/assets), Panda, front + wrist cameras
├─ Place robot (joint positions + finger positions of frame 0), parts (parts_poses, AprilTag → sim frame),
│  fixture (last 7 values of parts_poses)
├─ For every observation t (T + 1 frames):
│     if -dr: lights on/off, light color and intensity, part color, table color, camera-pose jitter
│     write joint and part poses of frame t directly (kinematic replay, no physics)
│     sim.step(); render 1 + --sub-steps times (extra renders let the ray tracer converge)
│     store wrist and front RGB + the original robot_state
└─ Save one pickle: the same actions, success=True, new images
```

- **Domain randomization is re-sampled every frame**, not once per episode. `--random-frame-freq` is parsed but unused. With `obs_horizon=1` each frame is a separate training sample, so this adds diversity and doesn't break training. Camera jitter is drawn around the nominal pose each time, so it doesn't drift.
- `-dr` must come before the `rand` subcommand, and the `rand` options after it; the script prints `Rand key: …` lines with the values it actually uses. Without `rand`, `-dr` still randomizes, using `base` part and table ranges.
- `--part-random full` draws part colors from gray (0.5) to white; `base` stays near white. `--table-random full` draws table colors from 0.5–1.0 per channel; `base` stays close to the nominal color.

### 7.4 Check a sample before rendering everything

Open 3–5 rendered episodes and compare them with real frames:

- Wrist is `color_image1` and front is `color_image2`, as in D_real.
- The gripper fingers are visible in the wrist view, and the parts and fixture are where the robot interacts with them. A wrong AprilTag-to-sim transform shows up as parts floating next to the gripper.
- Colors and brightness span what your real scene shows.

This is the step the Implementation Guide calls the highest risk in the pipeline. Its pass criterion is "Rendered frames look like your real camera views".

---

## 8. Step 3f: process D'_synth into zarr

`process_pickles.py` holds every frame of every episode in RAM before writing. 425 episodes × ~473 frames × 2 cameras × 230 KB ≈ **92 GB**, more than the 60 GB machine has. Process in chunks of 100 episodes (~33 GB peak each) with `--offset/--max-files`, one zarr per chunk:

```bash
for k in 0 1 2 3 4; do
  python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg_render_rppo -s rollout -r med -o success \
    --offset $((k*100)) --max-files 100 --output-suffix part_$k --n-cpus 8
done
```

- **Reads:** `$RAW/diffik/sim/one_leg_render_rppo/rollout/med/success/*.pkl`, in sorted order, so the chunks don't overlap.
- **Writes:** `$PROC/diffik/sim/one_leg_render_rppo/rollout/med/success/part_{0..4}.zarr`, the Stage 1 schema. Images are already 240×320, so they're stored as is. `action/pos` is rebuilt from the (now correctly clipped) deltas, and the gripper action becomes ±1.
- Only with the §4.1 fix. Re-process anything you processed before it with `--overwrite`.

**Loading the chunks in Stage 5.** `get_processed_paths` only finds `…/success.zarr`, not `…/success/part_k.zarr`, so the list-valued `task=… demo_source=…` command of the Implementation Guide (5a) won't pick up the chunks. Pass all zarrs explicitly instead (paths relative to `$PROC`):

```bash
data.data_paths_override='[diffik/real/one_leg_real/teleop/low/success.zarr,diffik/sim/one_leg_render_rppo/rollout/med/success/part_0.zarr,…,diffik/sim/one_leg_render_rppo/rollout/med/success/part_4.zarr]'
```

Then check the `Using data from …` line at startup.

**Optional: a state-only D_synth zarr.** You only need it to measure the distillation gap with a state student (paper Fig. 1, right). Process D_synth the same way (`-f one_leg -s rollout -r med`). Its 2×2 placeholders get upsampled to 240×320 blank frames, so RAM use is the same as above: chunk it too.

---

## 9. Optional: re-render the teleop demos

The Implementation Guide's co-training command also loads `one_leg_render_demos`: the human teleop demos, re-rendered the same way. The released zarrs contain no joint positions, so you need the **raw** teleop pickles: your own, or the authors' from the bucket's `data/raw/diffik/sim/one_leg/teleop/med/success/` (DATA_GUIDE §3). Run §7.2 with:

```bash
LOAD=$DATA_DIR_RAW/raw/diffik/sim/one_leg/teleop/med/success
SAVE=$DATA_DIR_RAW/raw/diffik/sim/one_leg_render_demos/teleop/med/success
```

Then process with `-f one_leg_render_demos -s teleop -r med`. The authors rendered 25 demos (`one_leg_render_demos_*`).

---

## 10. Checks before Stage 5

| Check | How | Pass |
|---|---|---|
| D_synth count and spread | `visualize_data.py info/plot` on the rollout folder | ~400 episodes; initial part poses spread over the med area |
| Clip fix applied | in a processed zarr, the mean angle between the `action/pos` rotation and the `robot_state` rotation | On the order of 0.1 rad for rollouts, not ~0.01 |
| Rendered images | `visualize_data.py video` on a few D'_synth pickles or a zarr; side by side with real frames | Same framing, gripper visible, plausible colors |
| Camera order | the same videos | wrist left / front right, as for D_real |
| Zarr contents | `visualize_data.py info` on each `part_k.zarr` | `color_image1/2` (N, 240, 320, 3); episodes sum to the pickle count |
| Coverage | `notebooks/07_real_vs_sim_state_action_space.ipynb` (Fig. 19) once D_real exists | Sim action xyz covers the real ones |

---

## 11. Parameter reference

### 11.1 `src.eval.evaluate_model` (3a)

| Parameter | Value in 3a | What it does |
|---|---|---|
| `--wt-path` | med ResiP `.pt` | Local checkpoint (π_base + π_res + normalizer + config). Use either this or `--run-id <proj>/<id>` (+ `--wt-type`). |
| `-f / --task` | `one_leg` | Env task. Also the task folder of the output path. |
| `--randomness` | `med` | Initial part and obstacle randomization, and the randomness folder of the output path. `low`, `med`, `high`. |
| `--n-envs` | 256 | Parallel envs per round; also the granularity of the stop check (§3.2). Lower it if GPU memory is tight. |
| `--n-rollouts` | 100000 | Upper bound; rounds = `n_rollouts // n_envs`. |
| `--max-rollout-steps` | 750 | Episode step limit. The default is the task's timeout from `src/common/tasks.py`. |
| `--action-type` | `pos` | Must equal the checkpoint's `control.control_mode` (asserted). |
| `--observation-space` | `state` | `state` = no rendering (2×2 placeholders). `image` renders IsaacGym cameras, slowly; that isn't needed, since 3e renders anyway. |
| `--save-rollouts` | on | Save pickles to `$RAW/diffik/sim/<task>/rollout/<randomness>/[<suffix>/]success/`. |
| `--save-failures` | off | Also save failures to `…/failure/`. Not useful for distillation. |
| `--save-rollouts-suffix` | `""` | Extra folder level to keep runs apart (§3.2). |
| `--break-on-n-success` / `--stop-after-n-success` | on / 400 | Stop after the round in which the success count reaches N. |
| `--compress-pickles` | off | Write `.pkl.xz`. The re-renderer and processing both read it. |
| `--gpu` | 0 | CUDA device. |
| `--visualize` | off | Open the IsaacGym viewer (use few envs). |
| `--wandb` | off | Write the success rate into the wandb run summary (`--wt-path` writes `<ckpt>.summary.json` instead). |

### 11.2 `src.sim2real.isaac_lab_rerender` (3e)

| Parameter | Value | What it does |
|---|---|---|
| `-i / --demo-index` | 0…N−1 | Which pickle to render: index into the sorted file list after §4.3 (into a seed-0 shuffled list before). |
| `--load-dir` | D_synth folder | Searched recursively for `*.pkl` / `*.pkl.xz`. |
| `--save`, `--save-dir` | on, D'_synth folder | Write the rendered pickle. Without `--save`, nothing is written. |
| `--furniture` | `one_leg` | Key into `part_config_render.py` (which USD parts to load). |
| `--num-parts` | 5 | Parts to place from `parts_poses`: tabletop + 4 legs. The 6th pose (fixture) is placed separately. |
| `-sub / --sub-steps` | 3 | Extra `sim.render()` calls per frame, so the ray-traced image converges. Higher = cleaner and slower. |
| `-dr / --domain-rand` | on | Per-frame lighting, color and camera-pose randomization (§7.3). |
| `rand --part-random` | `full` | `base` (near white) or `full` (0.5–1.0 gray to white). Inverted ranges when `PART_COLOR_BASE=black`. |
| `rand --table-random` | `full` | `base` (tiny Gaussian around the nominal color) or `full` (0.5–1.0 per channel). |
| `rand --different-part-colors` | off | A different random color per part instead of one shared color. |
| `rand --random-frame-freq` | — | ⚠️ Parsed, not used. |
| `--headless`, `--enable_cameras` | on | Isaac Lab AppLauncher flags: no window; cameras enabled in headless mode. |
| `--render-viewer` | off | Also write the viewport view to `src/sim2real/output/` (debugging). |
| `-fw / --flipped-wrist-camera` | — | ⚠️ Parsed, not used. |
| `PART_COLOR_BASE` (env var) | `white` | Base part color family: `white` or `black`. |
| `RARL_SOURCE_DIR` (env var) | `$PWD/src` | Required: root of the USD asset path. |

### 11.3 `src.data_processing.process_pickles` (3f)

| Parameter | Value | What it does |
|---|---|---|
| `-c -d -f -s -r -o` | `diffik sim one_leg_render_rppo rollout med success` | Spell out the raw folder `$RAW/<c>/<d>/<f>/<s>/<r>/[<suffix>/]<o>/` and the output path. `-f` accepts any name. |
| `--suffix` | — | Raw subfolder between randomness and outcome (matches `--save-rollouts-suffix`). |
| `--offset`, `--max-files` | `k*100`, 100 | Slice of the sorted file list, for chunking (§8) or for using exactly N episodes. |
| `--output-suffix` | `part_k` | Output becomes `…/success/part_k.zarr` instead of `…/success.zarr`. |
| `--randomize-order`, `--random-seed` | off | Shuffle before slicing; can't be combined with `--offset`. |
| `--n-cpus` | 8 | Threads for loading pickles (no effect on peak RAM). |
| `--overwrite` | — | Replace an existing zarr. |
| `--chunk-size` | 1000 | Zarr chunk length along time. |

---

## 12. Known issues, in one place

1. **Whole-episode rotation scaling in processing** (§4.1). Fix before processing anything.
2. **Mismatched image sizes in the Isaac Lab re-renderer** (§4.2). Processing crashes, and full-resolution pickles are ~4× larger.
3. **Shuffled `-i` and timestamp output names** (§4.3). Not resumable or traceable until changed.
4. **Processing loads everything into RAM** (§8). Chunk with `--offset/--max-files/--output-suffix`, and point Stage 5 at the chunks with `data.data_paths_override`.
5. **Overshoot past `--stop-after-n-success`** (§3.2). Expected; trim if you need exactly N.
6. **The Isaac Lab script is the authors' in-development version.** Their released rendered data came from `isaac_sim_raytrace.py` (Isaac Sim 2022.2.1 / Orbit, no longer installable). Expect to debug the first episode by hand.
