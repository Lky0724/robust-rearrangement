# ResiP Implementation Guide

A working guide for reproducing the pipeline of Fig. 3 in *From Imitation to Refinement – Residual RL for Precise Assembly* (Ankile et al.) with this codebase: sim demos → BC base policy → residual RL → synthetic rendered data → real demos → co-trained real-world policy.

---

## 1. Task choice: start with `one_leg` (low randomness)

`one_leg` is the only task this repo supports end to end, and the only one the paper takes to the real world (Table II). The co-training configs (`src/config/experiment/image/real_ol_cotrain*.yaml`) and example scripts (`scripts/ant_local/`, `slurm/bc/cannon/real_ol_demo_scaling/`) exist only for it.

| | one_leg | peg_hole | mug_rack | lamp | round_table |
|---|---|---|---|---|---|
| Episode length | ~500 | ~200 | ~150 | ~600 | ~700 |
| Rewards / insertions / screws | 1 / 1 / 1 | 1 / 1 / 0 | 1 / 0 / 0 | 2 / 1 / 1 | 2 / 2 / 2 |
| DP → ResiP success, low rand. (%) | 54 → 98 | 5 → 99 | 26 → 88 | 7 → 97 | 12 → 94 |
| Re-render config | ✅ | ❌ | ✅ | ✅ | ✅ |
| Real-world configs/scripts | ✅ | ❌ | ❌ | ❌ | ❌ |
| Physical build | 3D-printable FurnitureBench parts | 0.2 mm clearance | scanned mug | rolling bulb | two precise insertions |

`factory_peg_hole` has the shortest episodes (200 steps), so it's the quickest task for testing the RL stage in sim. It is a poor real-world target.

**Strategy:** first run the whole pipeline with the released data and checkpoints (S3 bucket, see the README), then replace each stage's output with your own. This lets you debug the plumbing separately from the learning.

---

## 2. How the codebase fits together

```
                      furniture-bench/ (submodule: IsaacGym sim, task definitions, rewards)
                                   │  src/gym/__init__.py  get_env() / get_rl_env()
                                   ▼
 ① sim teleop ──► raw .pkl ──► process_pickles ──► .zarr ──► train/bc.py ──► π_base ckpt
   data_collection/          data_processing/               dataset/ + behavior/ + models/
                                                                               │
 ② train/residual_ppo.py  (frozen π_base + ResidualPolicy MLP, PPO) ◄──────────┘ ──► π_res ckpt
                                                                               │
 ③ eval/evaluate_model.py --save-rollouts ──► raw .pkl (state only) ◄──────────┘
        └──► sim2real/isaac_*_re*.py (Isaac Sim, separate env) ──► raw .pkl WITH images ──► process_pickles ──► .zarr (D_synth)
 ④ real/teleop_sm.py (Franka + RealSense) ──► raw .pkl ──► process_pickles -d real ──► .zarr (D_real)
 ⑤ train/bc.py +experiment=image/... (loads D_synth + D_real together) ──► π_real ──► real/minimal.py
```

Two conventions connect every stage:

1. **One pickle format.** Every stage reads and writes the `Trajectory` format in `src/common/types.py`: `observations[{robot_state, color_image1 (wrist), color_image2 (front), parts_poses}]`, plus `actions`, `rewards` and `success`.
2. **Directory names.** Data is found by path, and the path is built from config fields (`src/common/files.py`):
   - Raw: `$DATA_DIR_RAW/raw/<controller>/<domain>/<task>/<source>/<randomness>/success/*.pkl`
   - Processed: `$DATA_DIR_PROCESSED/processed/<controller>/<domain>/<task>/<source>/<randomness>/success.zarr`

   A training config field can hold a list, e.g. `task='[a,b]'` or `environment='[real,sim]'`. `get_processed_paths` then globs every combination and loads each zarr that exists. Co-training mixes real and sim data this way.

### Data conventions

| Item | Format |
|---|---|
| Action (training) | 10-D **absolute** EE pose in the robot base frame: pos(3) + rot6d(6) + gripper(1, +1 close / −1 open) |
| Action (raw pkl) | 8-D **delta** (pos + quat + gripper). `src/data_collection/io.py` converts absolute actions to delta when saving. `process_pickles.py` converts them back and stores both `action/pos` and `action/delta`. `control.control_mode=pos` picks which one training uses. |
| robot_state | The env emits 14-D (pos, quat, lin vel, ang vel, gripper width). The actor converts it to 16-D with rot6d. |
| parts_poses | 7 values (pos + quat) per object. Used by state policies only. |
| Images | Stored at 240×320 (wrist resized; front resized and cropped), and brought to 224×224 at inference |
| Normalizer | Min-max to [−1, 1], fitted on the training data and saved inside the checkpoint (`normalizer.*`). π_res reuses π_base's normalizer; π_real fits its own. |

---

## 3. Module map

### Simulation
| Path | Role |
|---|---|
| `furniture-bench/furniture_bench/envs/furniture_sim_env.py` | `FurnitureSimEnv`, the original env, used for teleop |
| `furniture-bench/furniture_bench/envs/furniture_rl_sim_env.py` | `FurnitureRLSimEnv`: fast vectorized env with force-based part randomization, obstacle randomization, perturbations and batched sparse reward (`_reward`, +1 per newly assembled part pair) |
| `furniture-bench/furniture_bench/furniture/`, `config.py` | Task definitions. `mug_rack` and `factory_peg_hole` are defined only in `config.py` (scanned URDFs + `assembly.json`) |
| `src/gym/__init__.py` | `get_env()` builds the teleop env and `get_rl_env()` builds the env for everything else. `april_tags=False` uses the tag-free assets in `src/assets` |
| `src/gym/env_rl_wrapper.py` | `RLPolicyEnvWrapper`: reward normalization and clipping, reset on success |
| `src/gym/mj_dual_franka_env.py` | MuJoCo env for the bimanual task |
| `src/controllers/` | Differential IK (EE target → joint targets) |

### Stage 1: demos → π_base
| Path | Role |
|---|---|
| `src/data_collection/teleop.py`, `data_collector_sm.py` | SpaceMouse teleop in sim, writing raw pickles |
| `src/data_processing/process_pickles.py` | Raw pickles → one zarr: 6D rotations, image resizing, absolute actions rebuilt from deltas |
| `src/dataset/dataset.py` | `StateDataset` and `ImageDataset`: slice (1 obs, 32-action) windows and fit the normalizer |
| `src/dataset/zarr.py` | `combine_zarr_datasets`: concatenates zarrs and labels each episode `domain` (sim=0, real=1) |
| `src/dataset/augmentation.py` | Image augmentations (paper Tables XI and XII) |
| `src/behavior/base.py` | `Actor` interface. `action(obs)` keeps a queue: sample a 32-step chunk, keep the first 8, return one per call. This queue is the open-loop chunking the paper criticizes |
| `src/behavior/diffusion.py` | Diffusion policy (DDPM training, DDIM inference) |
| `src/behavior/{mlp,rnn,idql}.py` | Baselines |
| `src/models/` | Networks: `unet.py`, `transformer.py`, `vision.py` (ResNet, R3M, DINO, …), `residual.py`, `value.py` |
| `src/train/bc.py` | BC trainer for state and image policies. Uses Hydra (`src/config/base.yaml` + `+experiment=` overlay), optionally runs sim rollouts during training, and saves `best_success_rate`, `best_test_loss` and `last` checkpoints to WandB. `bc_ddp.py` is the multi-GPU version |

### Stage 2: ResiP
| Path | Role |
|---|---|
| `src/behavior/residual_diffusion.py` | `ResidualDiffusionPolicy`, which acts at every step: π_base supplies the next action from its queue; the residual MLP sees `[normalized state (clamped ±3), base action]` and outputs a Gaussian correction × 0.1; final action = base + correction. `load_base_state_dict()` loads π_base's weights and normalizer. It asserts `observation_type == "state"`, so the residual exists only in sim |
| `src/models/residual.py` | `ResidualPolicy`: 2×256 actor and 2×256 critic, fixed log-std −1 |
| `src/train/residual_ppo.py` | PPO loop. Each iteration resets all envs, rolls `num_env_steps` steps in each of `num_envs` envs, then runs GAE and 50 PPO epochs with target-KL early stopping. Every 5th iteration is a deterministic evaluation, and the best is saved as `actor_chkpt_best_success_rate.pt` |
| `src/config/base_residual_rl.yaml` | Defaults, matching paper Tables VI and IX except `total_timesteps` (1B vs 500M) |
| `src/common/config_util.py` | Merges π_base's saved config into the RL config |
| `src/train/{ppo,dagger,vas,vas_eval,residual_ppo_w_bc}.py` | Baselines (PPO-C, DP-DAgger, IDQL, residual + BC loss); not needed |

### Stage 3: expert rollouts → D_synth
| Path | Role |
|---|---|
| `src/eval/evaluate_model.py` | Loads a policy (`--run-id` from WandB or `--wt-path` from a local file) and evaluates it. `--save-rollouts --break-on-n-success --stop-after-n-success N` keeps rolling out until N successes and saves them |
| `src/eval/rollout.py` | Rollout loop and `save_raw_rollout` calls. State-only rollouts get 2×2 placeholder images |
| `src/sim2real/convert_parts_to_usd.py` | OBJ → USD meshes |
| `src/sim2real/part_config_render.py` | Per-task parts and assets |
| `src/sim2real/isaac_lab_rerender.py` | Replays one pickle (`-i`) in Isaac Lab, renders front and wrist cameras with domain randomization, writes a pickle with images. `isaac_sim_raytrace.py` is the older Isaac Sim 2022 version |
| `src/real2sim/` | Importing scanned objects (only needed for `mug_rack`) |

### Stages 4–5: real robot
| Path | Role |
|---|---|
| `src/real/teleop_sm.py` | SpaceMouse teleop on a Franka (Polymetis) with two RealSense cameras; writes the same pickle format |
| `src/real/minimal.py` | Deployment. `SimpleDiffIKFrankaEnv` builds observations, calls `actor.action(obs)` at 10 Hz and sends EE targets. **It moves the robot only with `-ex`**; without it, it only drives the meshcat visualization. It converts between the gripper-tip frame (sim and policies) and the wrist frame (Polymetis) |
| `src/real/serials.py` | RealSense serial numbers |
| `src/config/experiment/image/real_ol_cotrain.yaml` | Co-training config (transformer diffusion, R3M ResNet18). `actor.confusion_loss_beta` optionally aligns sim and real features; the authors' final runs used 0 |

### Supporting code
- `src/common/geometry.py`: all rotation conversions (quat ↔ 6D, xyzw order). Use these; don't write your own.
- `src/eval/eval_utils.py`: loads checkpoints from WandB, or from a local path through `LocalCheckpointWrapper`.
- `slurm/`: the most current record of the commands the authors ran. Check it before `scripts/ant_local/`, which is partly stale.
- `notebooks/` worth opening: `07_*align_inspect_cameras` (sim/real camera alignment), `07_real_vs_sim_state_action_space` (Fig. 19 coverage check), `11_la_cotraining`, `14_la_combine_zarrs`, `15_la_visualize_image_augmentation`.
- Safe to ignore: `models/`, `reports/`, `references/`, `src/features/`, `src/baseline/`, `src/codecs/`, `sweeps/`.

---

## 4. Step-by-step plan

Run all commands from the repo root. Paths below use two shorthands:
- `$RAW` = `$DATA_DIR_RAW/raw` (raw pickles, one file per episode)
- `$PROC` = `$DATA_DIR_PROCESSED/processed` (processed zarr datasets, one per data folder)

### Stage 0: Setup
Install everything the README lists: conda env (Python 3.8), IsaacGym Preview 4, `pip install -e furniture-bench`, `pip install -e .`, SpaceMouse drivers. Then:

```bash
export DATA_DIR_RAW=/path/to/data          # may be the same directory as below
export DATA_DIR_PROCESSED=/path/to/data
export WANDB_ENTITY=<your-wandb-entity>
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --scripted
```
- **What it does:** the three variables tell every script where data lives and where to log runs. The last line is a smoke test: it opens the IsaacGym viewer and runs a scripted `one_leg` assembly.
- **Reads:** furniture assets in `furniture-bench/furniture_bench/assets`.
- **Writes:** nothing.

**Compute:** state-based BC training runs on a single consumer GPU. The RL stage (the paper used 1024 parallel envs and up to 500M env steps) and Isaac Sim rendering need a larger GPU or a cluster.

### Stage 1: D_sim → π_base

**1a. Get the demos: download them…**
```bash
python scripts/download_data.py --task one_leg
```
- **What it does:** downloads the authors' 50 teleop demos per randomness level, already processed into zarr.
- **Reads:** the public S3 bucket `s3://iai-robust-rearrangement/data/processed/diffik/sim/one_leg/teleop/`.
- **Writes:** `$PROC/diffik/sim/one_leg/teleop/<randomness>/success.zarr` for each level. If `DATA_DIR_PROCESSED` is unset, it writes under `./data` instead.

**1b. …or collect your own**
```bash
python -m src.data_collection.teleop --furniture one_leg --randomness low --ctrl-mode diffik --num-demos 50
```
- **What it does:** opens the sim viewer. You drive the robot with the SpaceMouse and use the keyboard to mark each episode as success, failure or reset; the key bindings print at startup. It exits after `--num-demos` saved demos.
- **Reads:** SpaceMouse and keyboard input.
- **Writes:** one pickle per demo at `$RAW/diffik/sim/one_leg/teleop/low/success/<timestamp>.pkl`.
  - Failures go to `…/failure/`, but only with `--save-failure`.
  - `--sample-perturbations` writes to `low_perturb/` instead of `low/`.

```bash
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s teleop -r low -o success
```
- **What it does:** merges every pickle in one folder into a single zarr. Along the way it converts rotations to 6D, rebuilds absolute-pose actions from the stored deltas, and resizes images to 240×320.
  - The flags spell out the folder path: `-c` controller, `-d` domain, `-f` task, `-s` source, `-r` randomness, `-o` outcome.
- **Reads:** `$RAW/diffik/sim/one_leg/teleop/low/success/*.pkl`
- **Writes:** `$PROC/diffik/sim/one_leg/teleop/low/success.zarr`. It won't replace an existing zarr unless you pass `--overwrite`.

Repeat 1b with `--randomness med` (and `-r med` when processing); stage 3 needs a med-randomness policy.

**1c. Train π_base**
```bash
python -m src.train.bc +experiment=state/diff_unet task=one_leg randomness=low \
  training.num_epochs=400 lr_scheduler.warmup_steps=500 \
  actor.inference_steps=4 rollout.max_steps=700 \
  wandb.entity=<you> dryrun=false
```
- **What it does:** trains the state-based diffusion policy (UNet, predicts 32 actions, executes 8).
  - Every 10 epochs, once test loss falls below 0.1, it runs 256 parallel sim episodes to measure success rate (`rollout.*` config). On a small GPU, lower this with `rollout.num_envs=64`.
  - `dryrun=true` gives a quick debug run: no WandB, less data, short epochs. It still runs a full rollout after every epoch, so each epoch takes minutes. Stop it once one rollout has finished.
- **Overrides that match the paper (Tables III–IV):** without them, the experiment config trains for 5000 epochs (5M steps).

  | Override | Paper | Config default |
  |---|---|---|
  | `training.num_epochs=400` | 400k gradient steps (1 epoch = 1000 steps) | 5000 epochs = 5M steps |
  | `lr_scheduler.warmup_steps=500` | 500 warmup steps | 10,000 |
  | `actor.inference_steps=4` | 4 DDIM inference steps | 16 |
  | `rollout.max_steps=700` | 700-step `one_leg` episodes | 1000 |

  - With `num_epochs=400`, the cosine LR schedule decays fully over the run.
  - The paper reports no wall-clock time. Estimate it as 400k ÷ the steps/s shown in the inner `Training` progress bar, plus about 40 rollouts.
- **Reads:** `$PROC/diffik/sim/one_leg/teleop/low/success.zarr`. The path is built from `task`, `randomness`, `demo_source=teleop`, `environment=sim` and `control.controller=diffik`, and printed at startup as `Using data from …`.
- **Writes:**
  - Checkpoints `actor_chkpt_best_success_rate.pt`, `actor_chkpt_best_test_loss.pt` and `actor_chkpt_last.pt`. A numbered checkpoint is saved every 500 epochs, so none is written in a 400-epoch run. They go to `models/<wandb-run-name>/` inside the Hydra run directory, `outputs/<date>/<time>/` (or `$RUN_OUTPUT_DIR/<date>/<time>/`).
  - Each checkpoint holds the weights, the normalizer and the config.
  - The same files are uploaded to the WandB run, in project `one_leg-state-low`.

**1d. Evaluate π_base**
```bash
python -m src.eval.evaluate_model --run-id <proj>/<id> --n-envs 128 --n-rollouts 1024 -f one_leg \
  --max-rollout-steps 700 --action-type pos --observation-space state --randomness low \
  --wt-type best_success_rate
```
- **What it does:** runs 1024 episodes, 128 at a time, and prints `Success rate: X% (n/N)`. The target is about 50% at low randomness. `--visualize` opens the viewer.
- **Reads:** one of:
  - a checkpoint from a WandB run: `--run-id`, with `--wt-type` selecting the file;
  - a local file: `--wt-path <file.pt>`, which replaces both flags. Use this for the released `checkpoints/bc/one_leg/low/actor_chkpt.pt` to compare with the paper.
- **Writes:** nothing by default. `--wandb` writes the result into the WandB run's summary.

### Stage 2: residual RL

**2a. Train π_res**
```bash
python -m src.train.residual_ppo base_policy.wt_path=<bc ckpt> \
  env.task=one_leg env.randomness=low num_env_steps=700 num_envs=1024 \
  total_timesteps=500000000 debug=false
```
- **What it does:** freezes π_base and trains the small residual MLP with PPO in `num_envs` parallel envs, using only the sparse task-success reward.
  - It logs success rate every iteration, and runs a deterministic evaluation every 5th iteration.
  - `batch_size = num_env_steps × num_envs`; lower `num_envs` if you run out of GPU memory. This shrinks each PPO batch below the paper's 700 × 1024.
- **Overrides that match the paper (Tables VI and IX):** `src/config/base_residual_rl.yaml` already matches the paper on everything except run length.

  | Override | Paper | Config default |
  |---|---|---|
  | `total_timesteps=500000000` | 500M env steps | 1B |
  | `num_envs=1024` | 1024 parallel envs | same |
  | `num_env_steps=700` | 700-step `one_leg` episodes | same |

  - Already matching (checked against the config): update epochs (50), mini-batches (1), actor/critic LR (3e-4 / 5e-3, cosine), value-loss coefficient (1.0), residual scale (0.1), initial log-std (−1), discount (0.999), GAE λ (0.95), clip ε (0.2), target KL (0.1), max grad norm (1.0), advantage normalization (on), critic (2×256, ReLU, last-layer bias 0.25).
  - **Base-policy DDIM steps are always 4**, matching Table IV. `residual_ppo.py` sets `agent.inference_steps = 4` after loading π_base, so the value stored in the BC checkpoint (16 in the released `checkpoints/bc/one_leg/{low,med}`) is ignored. `evaluate_model.py` does the same. `actor.inference_steps=…` on this command line has no effect either, because `merge_base_bc_config_with_root_config` replaces `cfg.actor` with the checkpoint's.
  - The paper's sim ran at about 4000 env steps/s across 1024 envs, so 500M steps is about 35 h of rollout at that speed, before PPO updates and evaluations. The one-step residual reached about 85% success in roughly 75M steps (App. "Effect of fully versus partially closed-loop policies"). You can stop once `best_success_rate` levels off.
- **The released residual checkpoints were trained with different settings.** The configs saved in `checkpoints/rppo/one_leg/{low,med}/actor_chkpt.pt` differ from both the paper tables and the config defaults. The command above follows the paper. To reproduce the released runs instead, add the overrides in the last column.

  | Setting | Paper | Config default | Released rppo checkpoints (override) |
  |---|---|---|---|
  | Initial log-std | −1.0 (Table IX) | −1.0, fixed | `actor.residual_policy.init_logstd=-0.9` |
  | Learned log-std | not listed | `learn_std: false` | `actor.residual_policy.learn_std=true` |
  | Entropy coefficient | not listed | 0.0 | `ent_coef=0.001` |
  | Reward normalization | not listed | `normalize_reward: true` | `normalize_reward=false` |
  | Env steps (budget) | 500M (Table VI) | 1B | low: 500M; med: `total_timesteps=1000000000` |
  | Env steps when the best checkpoint was saved | — | — | low: 244 iterations ≈ 175M; med: 932 iterations ≈ 668M (scheduler `last_epoch` × 716,800; matches Figs. 26 and 27b) |
- **Reads:** the π_base checkpoint, from exactly one of:
  - a local file: `base_policy.wt_path`;
  - a WandB run: `base_policy.wandb_id=<proj>/<id>`, with `base_policy.wt_type`.
  
  π_base's config and normalizer come from the checkpoint. No dataset is read.
- **Writes:** `models/<wandb-run-name>/actor_chkpt_best_success_rate.pt`, relative to the directory you launched from (this script doesn't switch into a Hydra run directory). It is also uploaded to WandB project `one_leg-residual-rl`.
  - The checkpoint contains π_base and the residual together, i.e. the full ResiP policy used from here on.

**2b. Evaluate:** use the 1d command with `--wt-path models/<run>/actor_chkpt_best_success_rate.pt`.
- Targets: ≈98% at low randomness, ≈76% at med.
- The released `checkpoints/rppo/one_leg/{low,med}` weights let you compare.
- Train the **med**-randomness residual as well; stage 3 depends on its wider coverage.

### Stage 3: expert rollouts → D_synth

**3a. Generate expert trajectories**
```bash
python -m src.eval.evaluate_model --wt-path <med rppo ckpt> -f one_leg --randomness med \
  --n-envs 256 --n-rollouts 100000 --max-rollout-steps 750 \
  --action-type pos --observation-space state \
  --save-rollouts --break-on-n-success --stop-after-n-success 400
```
- **What it does:** rolls out the ResiP policy, which sees exact object poses, and saves every successful episode. It stops at 400 successes, about what the paper used; `--n-rollouts` is only an upper bound.
- **Reads:** the med residual checkpoint.
- **Writes:** `$RAW/diffik/sim/one_leg/rollout/med/success/<timestamp>.pkl`, containing robot states, part poses, actions (stored as deltas) and rewards.
  - Images are 2×2 placeholders; 3b fills them in.
  - `--save-rollouts-suffix X` inserts `X/` before `success/`. `--save-failures` also keeps the failures.

**3b. Re-render with images.** This runs in a separate Python 3.10 env with Isaac Sim and Isaac Lab; setup is in `src/sim2real/readme.md`.
```bash
LOAD=$DATA_DIR_RAW/raw/diffik/sim/one_leg/rollout/med/success
SAVE=$DATA_DIR_RAW/raw/diffik/sim/one_leg_render_rppo/rollout/med/success
N=$(ls $LOAD/*.pkl | wc -l)
for i in $(seq 0 $((N-1))); do
  python -m src.sim2real.isaac_lab_rerender --headless -i $i --sub-steps 3 \
    --load-dir $LOAD --furniture one_leg --num-parts 5 --save --save-dir $SAVE \
    -dr rand --part-random full --table-random full
done
```
- **What it does:** replays trajectory number `i` (in sorted file order) in Isaac Sim: it sets the robot and part poses frame by frame and renders the wrist and front cameras.
  - With `-dr`, it randomizes lighting, camera pose, and part and table colors.
  - `rand …` is a subcommand, so it must come last.
  - `--num-parts 5` is the tabletop plus four legs.
  - The `one_leg` USD meshes already exist in `src/sim2real/assets`, so `convert_parts_to_usd.py -f <task>` is only needed for new tasks.
- **Reads:** a single pickle from `$LOAD`, the USD assets, and `part_config_render.py`.
- **Writes:** one new pickle per run in `$SAVE`, with the same trajectory and real rendered images in `color_image1` (wrist) and `color_image2` (front).
- **Before running:**
  - Set the camera intrinsics (`isaac_lab_rerender.py:136`) and camera poses to match your RealSenses.
  - Re-render your teleop demos too: point `LOAD` at `…/one_leg/teleop/med/success` and `SAVE` at e.g. `…/one_leg_render_demos/teleop/med/success`.
- ⚠️ **Highest risk in the pipeline.** This script is still in development and its DR flags don't parse reliably; if DR doesn't take effect, toggle it by hand in the script. Check a few outputs visually (notebook `09_as_rerender_pkl_viz`). The older `isaac_sim_raytrace.py` needs Isaac Sim 2022.2.1 / Orbit, which can no longer be installed.

**3c. Process into D_synth**
```bash
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg_render_rppo -s rollout -r med -o success
```
- **What it does:** the same conversion as in 1b, now with real images.
- **Reads:** `$RAW/diffik/sim/one_leg_render_rppo/rollout/med/success/*.pkl`
- **Writes:** `$PROC/diffik/sim/one_leg_render_rppo/rollout/med/success.zarr`
- Do the same for the re-rendered teleop demos: `-f one_leg_render_demos -s teleop`.

### Stage 4: D_real (10–40 demos)

**Hardware:**
- Franka Panda with Polymetis
- Front and wrist RealSense D435 (set their serials in `src/real/serials.py`)
- SpaceMouse and `meshcat-server`
- A replacement for the lab-internal `rdt` tools the real-robot scripts import

**Scene:** 3D-print the FurnitureBench square-table top, one leg and the U-shaped fixture. Place them in the robot base frame to roughly match the sim, using `src/assets/calibration/one_leg.png` and `setup_front.png` as references.

**4a. Collect real demos**
```bash
meshcat-server &    # note the port it prints
python -m src.real.teleop_sm -p 6000 --furniture one_leg \
  --save_dir $DATA_DIR_RAW/raw/diffik/real/one_leg_real/teleop/low/success
```
- **What it does:** records **one demo per launch**. It streams both cameras and sends your SpaceMouse commands to the Franka at 10 Hz; `-p` is the meshcat port used for visualization.
  - The pickle is written only when you end the episode with the success key, so rerun the command for each demo.
- **Reads:** SpaceMouse input, the two RealSense streams and the robot state.
- **Writes:** `<save_dir>/<timestamp>.pkl`, containing full-resolution wrist and front images, robot state and delta actions.
  - Pointing `--save_dir` at the raw-data layout, as above, means 4b finds the files without any moving.
  - `low` here is only a folder label.

**4b. Process into D_real**
```bash
python -m src.data_processing.process_pickles -c diffik -d real -f one_leg_real -s teleop -r low -o success
```
- **Reads:** `$RAW/diffik/real/one_leg_real/teleop/low/success/*.pkl`
- **Writes:** `$PROC/diffik/real/one_leg_real/teleop/low/success.zarr`. Images are resized and cropped to 240×320, the same way as the sim data.

**Coverage check:** plot real and sim action xyz on top of each other (Fig. 19, notebook `07_real_vs_sim_state_action_space`). Sim actions should cover the real ones; if they don't, widen the stage-3 randomization before training.

### Stage 5: co-train π_real and deploy

**5a. Co-train**
```bash
python -m src.train.bc +experiment=image/real_ol_cotrain actor/diffusion_model=transformer \
  task='[one_leg_real,one_leg_render_rppo,one_leg_render_demos]' environment='[real,sim]' \
  demo_source='[teleop,rollout]' randomness='[low,med]' \
  training.batch_size=256 lr_scheduler.warmup_steps=1000 lr_scheduler.encoder_warmup_steps=5000 \
  wandb.entity=<you> wandb.project=<your-project> dryrun=false
```
- **What it does:** trains the RGB policy: two ResNet18 encoders (R3M weights) and a transformer diffusion backbone, on real and synthetic data mixed together. The observation is robot state plus the two camera images; no part poses.
- **Overrides that match the paper (Table X):**

  | Override | Paper | Config default (`real_ol_cotrain.yaml`) |
  |---|---|---|
  | `lr_scheduler.warmup_steps=1000` | 1000 policy warmup steps | 2000 |
  | `lr_scheduler.encoder_warmup_steps=5000` | 5000 encoder warmup steps | 50,000 |

  - Table X labels both warmup rows "Policy scheduler warmup steps"; the second is read here as the encoder's.
  - Already matching: batch size 256 (kept explicit in the command), 500k gradient steps (5000 epochs × 100 steps), policy/encoder LR (1e-4 / 1e-5, cosine), weight decay (1e-3), transformer (8 layers, 4 heads, 256-d embedding, attention dropout 0.3, causal), R3M ResNet18 encoders with 128-d projection.
  - Batch 256 with two image encoders may not fit a small GPU. A smaller batch runs, but it no longer matches the paper.
  - **Not reachable by overrides:** the camera augmentations are hard-coded in `src/common/vision.py`. They use Gaussian-blur σ ∈ (0.01, 2.0) where Tables XI–XII say (0.01, 1.2), and they have no random erasing where Table XI says p = 0.2 on the front camera. The config also turns on `feature_layernorm` and `front_camera_dropout: 0.1`, which the paper doesn't list. The authors' final script (`slurm/bc/cannon/real_ol_demo_scaling/7_ol_cotrain.sh`) additionally passes `training.clip_grad_norm=true`.
- **Reads:** every `$PROC/diffik/<environment>/<task>/<demo_source>/<randomness>/success.zarr` that exists across all combinations of the listed values. Combinations with no folder are skipped. Here that is:
  - `real/one_leg_real/teleop/low`
  - `sim/one_leg_render_rppo/rollout/med`
  - `sim/one_leg_render_demos/teleop/med`
  
  Check the `Using data from …` line at startup to confirm which zarrs were loaded.
- **Writes:** the same layout as 1c, `outputs/<date>/<time>/models/<run>/`. Image training doesn't run sim evaluations, so there is no `best_success_rate` checkpoint: you get `actor_chkpt_best_test_loss.pt`, `actor_chkpt_last.pt`, and a numbered `actor_chkpt_<epoch>.pt` every 250 epochs. Files are also uploaded to WandB.
  - **Keep WandB online, or `wandb sync` offline runs:** the deployment script loads weights from WandB only.
- **Real-only baseline:** the same command with `task=one_leg_real environment=real demo_source=teleop randomness=low`.

**5b. Deploy**
```bash
python -m src.real.minimal -p 6000 --save_dir <eval-log-dir> --run-id <proj>/<id>        # dry run
python -m src.real.minimal -p 6000 --save_dir <eval-log-dir> --run-id <proj>/<id> -ex    # moves the robot
```
- **What it does:** runs π_real on the robot at 10 Hz: it reads the cameras and robot state, calls the policy, and sends end-effector targets through diff-IK.
  - Without `-ex`, it computes actions and shows them in meshcat only, so it's safe for checking timing and frames.
  - `-w <substring>` picks which checkpoint file to load (default `best`, i.e. `best_test_loss`).
- **Reads:** the checkpoint and config from the WandB run, plus the live cameras and robot state.
- **Writes:** episode logs under `<eval-log-dir>/<datetime>/`.
- **Evaluation protocol:** run 10 trials from a fixed grid of initial part poses and record success per stage (corner / grasp / insert / screw), as in Table II. For reference, the paper's 40 Real+Sim policy completed 5–6/10, versus 2–3/10 for Real-only.

---

## 5. Checkpoints

| After stage | Pass criterion |
|---|---|
| 1 | BC reaches 40–55% in sim (low randomness) |
| 2 | ResiP reaches ≥90% in sim (low and med) |
| 3 | Rendered frames look like your real camera views |
| 4 | Real actions fall inside the sim action distribution |
| 5 | Real+Sim beats Real-only on the same trials |

---

## 6. Known gotchas

1. **`furniture` → `task` rename.** Commit `af3d1f6` renamed the config key, but `image/base.yaml`, `real_ol_cotrain.yaml` and `scripts/ant_local/*.sh` still use `furniture=`. Pass `task='[...]'` instead, as in `slurm/bc/cannon/real_ol_demo_scaling/7_ol_cotrain.sh`.
2. **WandB entity.** The experiment configs hard-code `wandb.entity: robust-assembly`. Override it with `wandb.entity=<you>` or use `wandb.mode=offline`.
3. **Action clipping in processing.** `process_pickles.py` (lines 128–133) clips each step's position change to ±2.5 cm and its rotation change to magnitude 0.35. That is harmless for teleop data, but check how often it triggers on stage-3 rollouts, because it silently changes the expert's actions.
4. **Real teleop saves one demo per launch, and only on success.** `teleop_sm.py` labels the pickle `success=True` and writes it only when you end the episode with the success key. There is no failure folder and no multi-demo loop; review demos afterwards and delete any you're unhappy with.
5. **Hard-coded hardware details:** camera serials in `serials.py` and D435 intrinsics in `isaac_lab_rerender.py:136`. Set them to your hardware.
6. **Lab-internal dependency.** The real-robot scripts import the authors' `rdt` package (camera streaming, Polymetis helpers, diff-IK). Expect to replace those calls with your own drivers.
7. **Two normalizers.** π_base and π_res share the normalizer fitted on D_sim; π_real fits a new one on the combined data. Different action ranges between them are expected.
8. **Removed evaluation flags.** `scripts/ant_local/*` pass `--controller diffik --use-new-env` to `evaluate_model.py`. Those flags no longer exist, and argparse exits with "unrecognized arguments". Drop them.
9. **Deployment loads from WandB only.** `src/real/minimal.py` fetches weights through the WandB API (`--run-id`, `-w`) and has no local-path option. Keep π_real's run online, or `wandb sync` it before deploying.
