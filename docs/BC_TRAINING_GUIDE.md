# BC Training Guide (Stage 1, π_base)

How `src/train/bc.py` trains the state-based diffusion base policy from teleop demos: what a run does, what the console output means, and which metrics to watch.

The training command is the paper-consistent one from the Implementation Guide (Stage 1c). Its values match paper Tables III and IV; §7 explains each parameter.

```bash
python -m src.train.bc +experiment=state/diff_unet task=one_leg randomness=low \
  training.num_epochs=400 lr_scheduler.warmup_steps=500 \
  actor.inference_steps=4 rollout.max_steps=700 \
  wandb.entity=<you> dryrun=false
```

Numbers in the examples come from running this command on the authors' 50 `one_leg` low-randomness demos (`scripts/download_data.py --task one_leg`) on an RTX 3000 Ada laptop GPU (8 GB). The run was stopped after 16 epochs; later-epoch numbers come from the released checkpoint (§7.9).

---

## 1. The idea in one paragraph

π_base is a **diffusion policy**. Given the current observation (16-D robot state + 42-D part poses for `one_leg`), it generates a **chunk of the next 32 actions** (each a 10-D absolute end-effector pose: position 3 + 6D rotation 6 + gripper 1). At run time only the first 8 are executed before it plans again. Training is plain supervised learning on the demos: take a demo's true 32-action chunk, add random Gaussian noise, and train a 1D UNet to predict that noise, conditioned on the observation:

```
noisy_chunk = √ᾱ_t · demo_chunk + √(1−ᾱ_t) · noise     # t ~ U{0..99}, random per sample
loss        = MSE( UNet(noisy_chunk, t, obs), noise )
```

The simulator is never used for learning. It only appears in **rollouts**, which periodically test the current weights and pick which checkpoint is saved as best.

---

## 2. What a run looks like

### 2.1 Setup (once, at start)

1. Hydra composes the config: `src/config/base.yaml` + `experiment/state/diff_unet.yaml` (which pulls in `experiment/state/base.yaml`) + your overrides. Hydra then **changes into** the run directory `outputs/<date>/<time>/` (or `$RUN_OUTPUT_DIR/<date>/<time>/`), so every relative output path lands there.
2. Load the demos from `$DATA_DIR_PROCESSED/processed/diffik/sim/one_leg/teleop/low/success.zarr` (printed as `Using data from …`). `StateDataset` fits a min-max normalizer to [−1, 1] and cuts every episode into overlapping (1 obs, 32 actions) windows.
3. Split windows 95/5 into train and test with `random_split`.
4. Build the actor (66M-parameter UNet), copy the normalizer into it, and create one AdamW optimizer (lr 1e-4, weight decay 1e-6) with a cosine LR schedule.
5. Start a wandb run and create `models/<wandb-run-name>/` for checkpoints.

| Quantity | Formula | Example value |
|---|---|---|
| Demos / frames | from the zarr | 50 episodes, 23,371 frames (mean length 467) |
| Windows | frames − 50 × (32 − 1 − 7 padding) | 22,171 |
| Train / test windows | 95% / 5% | 21,062 / 1,109 |
| Batches per pass over train data | 21,062 / 256 | 83 |
| Steps per epoch | `training.steps_per_epoch` | 1000 |
| Total gradient steps | `1000 × num_epochs` | 400,000 (= LR schedule length) |
| Test batches per validation | `round(1000 × 0.05)` | 50 |

The split is over **windows, not episodes**, so almost every test window has overlapping neighbours from the same demo in the training set. Test loss is therefore optimistic. The `num_episodes_train/test` values (47/2) in the wandb summary are just `50 × 0.95` and `50 × 0.05`; they don't describe the actual split.

### 2.2 The main loop: what happens in which epoch

An "epoch" here is a fixed 1000 gradient steps, not one pass over the data. `epoch_idx` counts from 0, and every periodic action fires when `(epoch_idx + 1)` is a multiple of its interval:

| Epochs (`epoch_idx`) | Train 1000 steps | Validation | Rollout | `best_*` checkpoints | Action-sample MSE | `last` checkpoint |
|---|---|---|---|---|---|---|
| 0–8 | yes | no | no | no | no | saved |
| **9**, 19, 29, … | yes | **yes** (`eval_every: 10`) | **yes**, if test loss < 0.1 | saved if improved | no | saved |
| **99**, 199, 299, 399 | yes | yes | yes | saved if improved | **yes** (`sample_every: 100`) | saved |

Two consequences:

- **Only 40 rollouts in a 400-epoch run.** The success-rate curve has one point every 10k gradient steps.
- **The rollout gate is the test loss.** It must be below `rollout.loss_threshold` (0.1). Diffusion noise-prediction loss drops below that within the first epoch or two (example: 0.0085 at epoch 9), so in practice every validation epoch also runs a rollout.

### 2.3 Inside one training step

```
┌─ 1. Batch ── 256 windows from the train split                     (already normalized)
│     obs:    (256, 1, 58)   robot_state 16 + parts_poses 42
│     action: (256, 32, 10)  the next 32 demo actions
│
├─ 2. Corrupt ── noise ~ N(0, I), t ~ U{0..99} per sample
│     noisy = scheduler.add_noise(action, noise, t)      # DDPM, squaredcos_cap_v2
│
├─ 3. Predict ── noise_pred = UNet(noisy, t, global_cond = obs flattened)
│
├─ 4. Loss ── MSE(noise_pred, noise), mean over batch, horizon and action dims
│
└─ 5. Update ── backward → grad-norm computed (not clipped by default, see §7.4)
                → AdamW step → LR scheduler step (per batch)
```

Validation runs steps 1–4 on 50 test batches with no gradient.

### 2.4 Inside one rollout (every 10 epochs)

```
┌─ 1. Build env (first time only) ── IsaacGym, 256 envs; builds gymtorch, prints env info
├─ 2. Reset ── parts placed with low randomness
├─ 3. Run 700 steps; each step, per env:
│       if the action queue is empty:                        # every 8 steps
│           chunk = DDIM denoise, 4 steps (actor.inference_steps), 32 actions
│           queue = first 8 actions (un-normalized)
│       env.step(queue.popleft())                            # diff-IK tracks the EE target
├─ 4. Success ── env's rewards sum to n_parts_assemble (1 for one_leg)
└─ 5. Log success_rate = n_success / 256; save best_success_rate checkpoint if it improved
```

No gradients and no data from the rollout go back into training.

Sampling detail: `DiffusionPolicy._normalized_action` doesn't start from pure noise. It **warm-starts** from the unexecuted tail of the previous chunk (all zeros on the first call), noised to diffusion step 50, and then runs the full DDIM schedule on it. `actor.reset()` doesn't clear this buffer, so the first chunk of each rollout is warm-started from the last chunk of the previous one.

### 2.5 How long a run takes

| Phase | Time (example) |
|---|---|
| Training epoch (1000 steps) | ~141 s early in the run (~7 steps/s); ~161–180 s later (5.3–5.6 steps/s) |
| Validation (50 forward-only batches) | a few seconds |
| Rollout (256 envs × 700 steps, 4 DDIM steps) | ~93 s, plus a one-time env build on the first rollout |

Full run: `400 × ~150–180 s + 40 × ~95 s ≈ 17–21 h`. You don't need all of it: stop once `success_rate` plateaus (§4, §6). The paper reports no wall-clock time for BC.

---

## 3. Reading the console output

Everything is printed through nested tqdm bars, which overwrite each other in the terminal. Garbled or half-overwritten lines are a display artifact, not an error. A cleaned-up excerpt from the example run:

```
Using data from [PosixPath('…/processed/diffik/sim/one_leg/teleop/low/success.zarr')]
Splitting dataset into 21062 train and 1109 test samples.
Run name: golden-wave-1
Job started at: 2026-09-24 14:51
Epoch (one_leg, state):   2%| | 9/400 [21:11<15:20:31, 141.26s/it, best_success_rate=0, loss=0.00827, …]
Training: 100%|██████████| 1000/1000 [02:41<00:00,  5.56it/s, loss=0.00782]
Validation: 100%|██████████| 50/50 […]
Building extension module gymtorch...
Making DiffIK controller with pos_scalar: 1.0, rot_scalar: 1.0
Observation keys: ['robot_state/ee_pos', …, 'parts_poses']
Performing rollouts (one_leg): round 1/1, success: 33/256 (12.9%): 100%|█| 700/700 [01:33<00:00]
Checking if we should save rollouts (rollout_save_dir: None)
```

| Line / field | Meaning |
|---|---|
| `Using data from …` | The zarr(s) actually loaded. Check this first if the dataset size looks wrong. |
| `Splitting dataset into …` | Train/test **windows** (§2.1). |
| `Epoch (one_leg, state) 9/400` | Outer bar: epochs done / `num_epochs`. `s/it` is seconds per epoch, averaged over the run, including rollouts. The ETA is based on it. |
| Epoch bar postfix | `loss` = mean train loss of the last epoch; `test_loss` = last validation loss (`inf` until the first validation); `best_success_rate` = best rollout SR so far; `stopper_counter` = early-stopper counter (§4.2); `time` = wall clock. |
| `Training … 1000/1000 5.56it/s, loss=…` | Inner bar: gradient steps in this epoch. `it/s` is your real training throughput; `loss` is the current batch's loss. |
| `Validation … 50/50` | Test-loss batches (only every 10 epochs). |
| `Building extension module gymtorch…`, `Making DiffIK controller…`, `Observation keys…`, `Sim steps…`, `Max force magnitude…` | One-time IsaacGym env construction before the first rollout. |
| `Performing rollouts (one_leg): round 1/1, success: 33/256 (12.9%)` | Rollout progress: sim steps out of `rollout.max_steps`, successes so far. Rounds = `rollout.count // rollout.num_envs` (1 by default). |
| `Checking if we should save rollouts (rollout_save_dir: None)` | End of the rollout. `None` means rollouts aren't written to disk (`rollout.save_rollouts: false`). |
| `UserWarning: Plan failed with a cudnnException … CUDNN_STATUS_NOT_SUPPORTED` | cuDNN falling back to another conv1d algorithm at startup. Harmless. |
| `wandb: WARNING Symlinked 1 file into the W&B run directory` | A checkpoint was saved and queued for upload. Harmless. |

---

## 4. Important metrics

### 4.1 Tier 1: is the policy getting better?

| Metric (wandb key) | What it tells you | What you want |
|---|---|---|
| **`success_rate`** | Fraction of the 256 rollout envs that assembled the part. **The main metric**, and the only one that measures the policy as it is deployed. | Rising, then plateauing. Paper target for `one_leg` low: ~54% (Table I). |
| `best_success_rate` | Best SR so far, i.e. the SR of `actor_chkpt_best_success_rate.pt`. | Monotone non-decreasing. |
| `n_success` / `n_rollouts` | The counts behind SR. | With 256 envs, one SR has a standard error of ~3 percentage points at 50%, so treat differences under ~5 points as noise. |
| `test_epoch_loss` | Noise-prediction MSE on held-out windows. | Falling, then flat. |
| `epoch_loss` (= `train_bc_loss`) | Mean training loss over the epoch. | Falling. |

Example values:

| Epoch | `epoch_loss` | `test_epoch_loss` | `success_rate` |
|---|---|---|---|
| 0 | 0.197 | — | — |
| 1 | 0.0255 | — | — |
| 4 | 0.0138 | — | — |
| 9 | 0.0083 | 0.0085 | 12.9% (33/256) |
| 14 | 0.0064 | — | — |
| Released ckpt, epoch 59 (batch 1024) | — | 0.0024 | ≈ 49% (512 envs) |

**Why loss isn't used to pick π_base:** the loss measures one denoising step at a random noise level on in-distribution windows. It says nothing about compounding errors once the policy drives the robot into states the demos never visited, which is exactly where precise insertion fails. Loss keeps falling long after SR has plateaued. Use `actor_chkpt_best_success_rate.pt`.

**wandb plotting quirk:** the rollout metrics are logged with `wandb.log(...)` without a `step`, so they attach to the **previous** logged step. In the example, epoch 9's `success_rate` sits at `_step` 9000 (epoch 8's row), and it also overwrites that row's `epoch` field with 9. Plot SR against `_step` and remember it's shifted 1000 steps early.

### 4.2 Tier 2: is optimization healthy?

| Metric | What it tells you | Healthy | Warning sign |
|---|---|---|---|
| `train_grad_norm` | Gradient norm of the **last** batch in the epoch (not a mean). | Falls from ~1.5 in epoch 0 to ~0.1–0.3 (example: 0.14 at epoch 14). | Spikes or steady growth: LR too high. Consider `training.clip_grad_norm=true`. |
| `actor_lr` | Current LR. | Linear warmup over 500 steps, then cosine decay from 1e-4 to 0 at step 400k. | Barely below 1e-4 late in the run: `num_epochs` is much larger than the run you're actually doing. |
| `test_epoch_loss` − `epoch_loss` | Overfitting gap. | Small (0.0085 vs 0.0083 at epoch 9). | Test loss rising while train loss falls. The window-level split (§2.1) hides much of this, so trust SR more. |
| `action_sample/{train,val}_action_mse_error{,_pos,_rot,_width}` | MSE of a full DDIM sample against the demo chunk, in un-normalized units, split into position, 6D rotation and gripper. Logged only on epochs 99, 199, 299, 399. | Falling; `val` close to `train`. | `val` ≫ `train`. |
| `early_stopper/{ema_loss,best_loss,counter}` | Smoothed test loss and evaluations since it last improved. | Informational: `experiment/state/base.yaml` sets `patience: inf`, so it never stops the run. | — |

The early-stopper `counter` is already 1 after the first evaluation: it initializes `best_loss` to the first EMA value and then counts "not strictly better" as no improvement.

`epoch_mean_return` is always 0 in this setup: returns are only computed from saved rollout data, so it's meaningful only with `rollout.save_rollouts=true`.

### 4.3 Tier 3: throughput

| Metric | Note |
|---|---|
| `Training` bar `it/s` | Use it to estimate remaining time: `(400,000 − global_step) / (it/s)`, plus ~95 s per remaining rollout. |
| `system.gpu.0.gpu` (wandb System tab) | ~97% median in the example run: the GPU is the bottleneck, not data loading. |
| GPU memory | ~2.1 GB while training, up to ~3.4 GB once the rollout env exists (example run). Plenty of headroom on 8 GB. |
| `system.gpu.0.powerWatts`, `smClock` | In the example run the GPU sat at **~35 W and ~1.3 GHz** (max clock 3.1 GHz) at 59 °C: power-capped, not thermally limited. See §6. |

---

## 5. Checkpoints

All files go to `outputs/<date>/<time>/models/<run>/` (the Hydra run directory) and are also uploaded to the wandb run's Files tab.

| File | When it's written | Selected by |
|---|---|---|
| `actor_chkpt_best_success_rate.pt` | After a rollout whose SR beats the best so far. Overwritten each time. | **Use this one as π_base** for Stage 2. |
| `actor_chkpt_best_test_loss.pt` | After a validation whose test loss beats the best so far. | Rarely the best policy (§4.1). |
| `actor_chkpt_last.pt` | End of every epoch. | Resuming (`wandb.continue_run_id`). |
| `actor_chkpt_<epoch>.pt` | Every `checkpoint_interval` epochs (500 in the experiment config). | None written in a 400-epoch run. |

Each checkpoint is **~793 MB**: the model and normalizer (`model_state_dict`, ~264 MB for 66M float32 parameters), the AdamW state (2 × the parameters), the LR-scheduler state, `epoch`, `global_step` and the full resolved `config`. The optimizer state is what makes resuming exact. The residual stage only needs `model_state_dict` and `config`.

Caveats:

- **Stale metadata fields.** The save dict is built at the start of the evaluation block, before `best_test_loss` and `best_success_rate` are updated. So a `best_*` file stores the *previous* best in those fields (e.g. `best_success_rate: 0` in the first one), even though its weights are the new best. `epoch` and `global_step` are correct.
- **With EMA on (`training.ema.use=true`)**, validation and rollouts use the EMA weights, but the checkpoints store the raw weights. EMA is off by default.

To resume a stopped run, rerun the command with `wandb.continue_run_id=<run id>`. It downloads `actor_chkpt_last.pt` from wandb, restores that run's config (your other overrides are replaced), and continues from the saved epoch with the optimizer and LR schedule intact.

---

## 6. Making training faster

The GPU is compute-bound (~97% utilization) and data loading is essentially free (see the last row below), so the gains come from a faster GPU step or less total work:

| Change | Effect | Caveat |
|---|---|---|
| Put the laptop on AC power and its performance power mode | The example GPU ran capped at ~35 W and ~1.3 GHz of 3.1 GHz. Raising the cap is the cheapest possible speedup. | Laptop/vendor-specific. Check `nvidia-smi -q -d POWER,CLOCK` while training. |
| Stop by hand once `success_rate` plateaus, keeping `num_epochs=400` | The released `one_leg` low checkpoint was saved at epoch 59 = 60k steps (at batch 1024, §7.9). | Keeps the paper's LR schedule. Lowering `num_epochs` instead also shortens the cosine decay. |
| `rollout.every=20` or `rollout.num_envs=128` | Halves rollout time (~6% of the run). | Fewer or noisier SR points for picking the best checkpoint. |
| bf16 autocast in the training step | Likely the largest per-step gain on an Ada GPU. | Needs a code change: `training.mixed_precision` exists in the config but nothing reads it. |
| Rent a datacenter GPU | Single-GPU job; a few hours instead of ~17–21 h. | — |

What **not** to change for speed:

- **`training.batch_size`**: halving it roughly doubles `it/s` but halves the data per step. It's an optimization setting (paper: 256), not a speed knob.
- **`data.dataloader_workers`**: no effect. `FixedStepsDataloader` wraps the loader in `itertools.cycle`, which caches the first pass (83 batches) and **replays those same batches in the same order** for the rest of the 1000-step epoch. Only ~83 batches per epoch are actually loaded. It reshuffles only when the next epoch starts. The docstring's "random order every time" doesn't hold.
- **`training.steps_per_epoch`**: changes what an "epoch" means (and with it `eval_every`, `rollout.every` and the LR-schedule length), not the speed.

---

## 7. Parameter reference

Parameters are Hydra overrides of `src/config/base.yaml` and the files it pulls in: `training.yaml`, `data.yaml`, `regularization.yaml`, `early_stopper.yaml`, `actor/diffusion.yaml` (+ `actor/base_actor.yaml`, `actor/diffusion_model/unet.yaml`), `rollout/*.yaml`, then the experiment overlay `experiment/state/{diff_unet,base}.yaml`. Pass them as `key=value` after the module name. In the tables:

- **Default** is the value after the experiment overlay, i.e. what you get without that override.
- **Paper** gives the value from paper Tables III and IV where the paper lists one. A dash means the paper doesn't list it.
- ⚠️ marks a parameter that does nothing in this setup or is overridden.

### 7.1 The command, parameter by parameter

```bash
python -m src.train.bc +experiment=state/diff_unet task=one_leg randomness=low \
  training.num_epochs=400 lr_scheduler.warmup_steps=500 \
  actor.inference_steps=4 rollout.max_steps=700 \
  wandb.entity=<you> dryrun=false
```

| Parameter | Value | What it does |
|---|---|---|
| `python -m src.train.bc` | — | Runs the BC trainer. Run it from the repo root; Hydra then switches into `outputs/<date>/<time>/`. |
| `+experiment=state/diff_unet` | — | Experiment overlay: state observations, diffusion actor with the UNet backbone, batch 256, 1000 steps/epoch, rollouts on, `patience: inf`, wandb project `${task}-state-${randomness}`. The `+` is required because `experiment` isn't in the base defaults list. |
| `task` | `one_leg` | Which task's demos to load (path component) and which env to roll out in. Also names the wandb project. |
| `randomness` | `low` | Which demo folder to load (`…/teleop/low/…`) and the rollout env's initial-state randomization. |
| `training.num_epochs` | 400 | Number of 1000-step epochs → 400k gradient steps (paper: max 400k). Also the cosine-schedule length. Default 5000. |
| `lr_scheduler.warmup_steps` | 500 | Linear LR warmup, in gradient steps. Paper: 500. Default 10,000. |
| `actor.inference_steps` | 4 | DDIM denoising steps per plan, used in rollouts and action-sample metrics only (training always uses the 100-step DDPM noise schedule). Paper: 4. Default 16. |
| `rollout.max_steps` | 700 | Rollout episode length. Paper: 700 for `one_leg`. Default 1000. |
| `wandb.entity` | `<you>` | Required in practice: the experiment config sets `robust-assembly`, the authors' team, which your account can't log to. |
| `dryrun` | `false` | `true` makes a debug run: 5 demos, 10 steps/epoch, no workers, wandb off, and evaluation + rollout after **every** epoch with no loss threshold. It doesn't lower `num_epochs`, so stop it by hand after one rollout. |

### 7.2 Data

| Parameter | Default | Paper | What it does |
|---|---|---|---|
| `environment` | `sim` | — | Path component: `sim` or `real`. A list (e.g. `'[real,sim]'`) loads every existing combination. |
| `demo_source` | `teleop` | — | Path component: `teleop` or `rollout`. |
| `demo_outcome` | `success` | — | Path component: which outcome folder to load. |
| `data.suffix` | `null` | — | Optional extra path component. |
| `data.data_paths_override` | `null` | — | Explicit list of zarr paths, relative to `$DATA_DIR_PROCESSED/processed`, bypassing the path built from the fields above. |
| `data.data_subset` | `null` | — | Load only the first N episodes. `dryrun` sets 5. |
| `data.test_split` | 0.05 | — | Fraction of **windows** held out for validation (§2.1). |
| `data.dataloader_workers` | 16 | — | DataLoader worker processes. Barely matters (§6). |
| `obs_horizon` | 1 | 1 | Observation steps per sample. Top-level; `data.*` and `actor.*` copy it. |
| `pred_horizon` | 32 | 32 | Actions predicted per chunk. |
| `action_horizon` | 8 | 8 | Actions executed per chunk before replanning. Also sets end-of-episode padding. |
| `predict_past_actions` | `false` | — | `true` makes the chunk start at the oldest observation instead of the current one. Irrelevant with `obs_horizon=1`. |
| `data.pad_after` | `true` | — | Adds windows near each episode's end, padded by repeating the last frame (7 extra windows per episode). |
| `data.include_future_obs` | `false` | — | Return observations for the whole window instead of only the first `obs_horizon`. The diffusion actor only uses the first anyway. |
| `data.normalization` | `min_max` | [−1, 1] | ⚠️ Not read. The normalizer is always min-max to [−1, 1]. |
| `data.augment_image`, `data.load_into_memory`, `data.minority_class_power` | — | — | ⚠️ Image datasets only. |

### 7.3 Model (`actor.*`)

| Parameter | Default | Paper | What it does |
|---|---|---|---|
| `actor.name` | `diffusion` | — | Actor class: `diffusion`, `mlp`, `residual_diffusion`, `attentionpool_diffusion`. Selected by `experiment/state/diff_unet`. |
| `actor.diffusion_model.down_dims` | [256, 512, 1024] | same | UNet channel widths per level. Sets model size (66M params). |
| `actor.diffusion_model.diffusion_step_embed_dim` | 256 | 256 | Size of the noise-level (t) embedding. |
| `actor.diffusion_model.kernel_size` | 5 | 5 | Conv1d kernel size along the 32-step time axis. |
| `actor.diffusion_model.n_groups` | 8 | 8 | GroupNorm groups. |
| `actor.num_diffusion_iters` | 100 | 100 | DDPM training noise steps (t ∈ 0..99). |
| `actor.beta_schedule` | `squaredcos_cap_v2` | — | Noise schedule (cosine). |
| `actor.prediction_type` | `epsilon` | — | The UNet predicts the added noise. |
| `actor.clip_sample` | `true` | — | Clip denoised samples to [−1, 1] (the normalized action range). |
| `actor.inference_steps` | 16 | 4 | See 7.1. The residual stage hard-codes 4 regardless. |
| `actor.loss_fn` | `MSELoss` | — | `MSELoss` or `L1Loss` on the noise. |
| `actor.include_proprioceptive_pos` / `_ori` | `true` / `true` | — | `false` zeros the EE position (dims 0–2) / 6D rotation (dims 3–8) in the observation. Ablation only. |
| `actor.projection_dim`, `actor.confusion_loss_*`, `actor.rescale_loss_for_domain` | — | — | ⚠️ Image / co-training options; no effect for state. |

### 7.4 Optimization

| Parameter | Default | Paper | What it does |
|---|---|---|---|
| `training.batch_size` | 256 | 256 | Windows per gradient step. |
| `training.steps_per_epoch` | 1000 | — | Gradient steps per "epoch". `-1` = one real pass over the data. |
| `training.num_epochs` | 5000 | 400k steps | See 7.1. |
| `training.actor_lr` | 1e-4 | 1e-4 | Peak LR (AdamW). |
| `regularization.weight_decay` | 1e-6 | 1e-6 | AdamW weight decay. |
| `lr_scheduler.name` | `cosine` | Cosine | Schedule over `steps_per_epoch × num_epochs` steps, stepped every batch. |
| `lr_scheduler.warmup_steps` | 10,000 | 500 | See 7.1. |
| `training.clip_grad_norm` | `false` | — | Sets the clip threshold as `1 + 1000 × (1 − clip_grad_norm)`: `true` → clip at 1.0, `false` → 1001, i.e. effectively no clipping. The norm is logged either way. |
| `training.ema.use` / `.decay` / `.switch` | `false` / 0.999 / `false` | — | EMA of the weights, used for validation and rollouts. `switch=true` copies the EMA weights into the model every epoch. See §5 for the checkpoint caveat. |
| `training.encoder_lr`, `lr_scheduler.encoder_warmup_steps` | — | — | ⚠️ Image encoders only. |
| `regularization.*` (except `weight_decay`) | 0 / `false` | — | ⚠️ `state_noise`, `proprioception_dropout`, feature and camera dropout etc. are only read for image observations; they have no effect for state. |
| `training.mixed_precision` | `false` | — | ⚠️ Not read anywhere; training is always FP32. |
| `training.clip_sample` | `true` | — | ⚠️ Not read; `actor.clip_sample` is the one used. |

### 7.5 Validation and rollouts

| Parameter | Default | What it does |
|---|---|---|
| `training.eval_every` | 10 | Validate every N epochs. Rollouts, `best_*` checkpoints, the early stopper and action-sample metrics all happen only inside validation epochs. |
| `training.sample_every` | 100 | Log `action_sample/*` every N epochs (checked only on validation epochs). |
| `rollout.rollouts` | `true` | Run sim rollouts at all. `false` gives pure supervised training with no SR and no `best_success_rate` checkpoint. |
| `rollout.every` | 10 | Rollout every N epochs, checked only on validation epochs, so keep it a multiple of `eval_every`. |
| `rollout.loss_threshold` | 0.1 | Skip rollouts until test loss is below this. |
| `rollout.num_envs` | 256 | Parallel IsaacGym envs per rollout round. Lower it (e.g. 64) if GPU memory is tight. |
| `rollout.count` | `${rollout.num_envs}` | Total rollouts per evaluation; runs `count // num_envs` rounds. |
| `rollout.max_steps` | 1000 | See 7.1. |
| `rollout.task` / `rollout.randomness` | `${task}` / `${randomness}` | Env to evaluate in. Can differ from the training data, e.g. evaluate a low-data policy at `med`. |
| `rollout.save_rollouts` / `rollout.save_failures` | `false` / `false` | Write rollouts as raw pickles under `$DATA_DIR_RAW/raw/diffik/sim/<task>/rollout/<randomness>/`. `save_failures` also keeps failed ones. |
| `rollout.parts_poses_in_robot_frame` | `false` | Express part poses in the robot frame in the env. Must match how the training data was processed. |
| `rollout.n_parts_assemble` | `null` | ⚠️ Not passed through by `bc.py`; the env's own value is used (1 for `one_leg`). |
| `discount` | 0.999 | Only used for `epoch_mean_return`, which is 0 unless rollouts are saved (§4.2). |

### 7.6 Checkpointing and early stopping

| Parameter | Default | What it does |
|---|---|---|
| `training.store_best_success_rate_model` | `true` | Write `actor_chkpt_best_success_rate.pt`. |
| `training.store_best_test_loss_model` | `true` | Write `actor_chkpt_best_test_loss.pt`. |
| `training.store_last_model` | `true` | Write `actor_chkpt_last.pt` every epoch. |
| `training.checkpoint_interval` | 500 | Write `actor_chkpt_<epoch>.pt` every N epochs (only on validation epochs). `-1` = off. ~793 MB each. |
| `training.model_save_dir` | `models` | Checkpoint directory, relative to the Hydra run directory. |
| `early_stopper.patience` | `inf` | Validations without EMA-loss improvement before stopping. `inf` = never. |
| `early_stopper.smooth_factor` | 0.9 | EMA factor for the smoothed test loss. |
| `early_stopper.metric` | `val_loss` | ⚠️ Not read; it's always test loss. |

### 7.7 Logging, resuming and runtime

| Parameter | Default | What it does |
|---|---|---|
| `wandb.entity` | `robust-assembly` | See 7.1. |
| `wandb.project` | `${task}-state-${randomness}` | e.g. `one_leg-state-low`. |
| `wandb.mode` | `online` | `online`, `offline` (sync later with `wandb sync`) or `disabled`. |
| `wandb.name` / `wandb.notes` | `null` | Run name (random, e.g. `golden-wave-1`, if null) and notes. The name is also the checkpoint folder name. |
| `wandb.continue_run_id` | `null` | Resume a run in `wandb.project` (§5). If the run doesn't exist, a fresh run starts. |
| `wandb.osh_sync_interval` | 25 | In offline mode, trigger `wandb-osh` sync every N epochs. |
| `wandb.watch_model` | `false` | Log weight and gradient histograms every 1000 steps. |
| `training.load_checkpoint_run_id` | `null` | ⚠️ Meant to initialize weights from a wandb run (`<entity>/<project>/<id>`). It takes the **first** `.pt` file listed in that run, whichever it is, and passes the whole file to `load_state_dict`. That only works for old checkpoints that are a bare state dict; current checkpoints (§5) nest the weights under `model_state_dict` and fail to load. Use `wandb.continue_run_id` to resume instead. |
| `seed` | `null` | Random seed; `null` draws one and stores it in the run config. |
| `training.gpu_id` | 0 | CUDA device for training and rollouts. |
| `RUN_OUTPUT_DIR` (env var) | `./outputs` | Root of the Hydra run directories. |

### 7.8 Settings kept at paper values

Already matching Tables III–IV without overrides: 10-D absolute-pose actions with 6D rotation, 16-D proprioception, batch 256, peak LR 1e-4 with cosine decay, weight decay 1e-6, UNet dims [256, 512, 1024], embed dim 256, kernel 5, 8 groups, 66M parameters, horizons 1 / 32 / 8, 100 DDPM steps.

### 7.9 Settings of the released BC checkpoint

The config saved in `checkpoints/bc/one_leg/low/actor_chkpt.pt` differs from both the paper table and the command above:

| Setting | Released checkpoint | Paper / command above |
|---|---|---|
| `training.batch_size` | **1024** | 256 |
| `lr_scheduler.warmup_steps` | 2000 | 500 |
| `training.num_epochs` | 10,000 (saved at epoch 59 = 60k steps) | 400 |
| `actor.inference_steps` | 16 | 4 |
| `rollout.num_envs` / `rollout.count` | 512 / 512 | 256 / 256 |
| `randomness` | `[low, low_perturb]` | `low` |

The `low_perturb` entry added no data: the checkpoint reports the same 50 episodes and 22,171 windows as `low` alone. The checkpoint's stored metadata is `best_success_rate: 0.494`, `best_test_loss: 0.0024` (these are the previous-best values, see §5). 60k steps at batch 1024 covers as many samples as 240k steps at batch 256.

To reproduce that run instead of the paper table:

```bash
python -m src.train.bc +experiment=state/diff_unet task=one_leg randomness=low \
  training.batch_size=1024 lr_scheduler.warmup_steps=2000 training.num_epochs=10000 \
  rollout.max_steps=700 rollout.num_envs=512 \
  wandb.entity=<you> dryrun=false
```

and stop it by hand once `success_rate` plateaus. Activation memory grows with batch size, so check `nvidia-smi` during the first epoch; the example run used ~2.1 GB at batch 256.
