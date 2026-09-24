# Residual PPO Training Guide (Stage 2, π_res)

How `src/train/residual_ppo.py` trains the residual policy on top of a frozen diffusion base policy: what a run does, what the console output means, and which metrics to watch.

The training command is the paper-consistent one from the Implementation Guide (Stage 2a). Its values match paper Tables VI and IX; §7 explains each parameter.

```bash
python -m src.train.residual_ppo base_policy.wt_path=<bc ckpt> \
  env.task=one_leg env.randomness=low num_env_steps=700 num_envs=1024 \
  total_timesteps=500000000 debug=false
```

Numbers in the examples are from running this command with `<bc ckpt>` = `checkpoints/bc/one_leg/low/actor_chkpt.pt` (the released low-randomness BC policy) on an RTX 3000 Ada laptop GPU (8 GB).

---

## 1. The idea in one paragraph

The base policy π_base (diffusion, trained with BC in Stage 1) is **frozen**. A small MLP, the residual policy π_res (2×256 actor, 2×256 critic), looks at the observation **plus the base action** and outputs a correction:

```
action = base_action + 0.1 × residual_action      # action_scale: 0.1, in normalized action space
```

Only π_res (actor + critic) is trained, with PPO, from sparse task rewards in IsaacGym. The base policy supplies the "roughly right" motion; the residual learns the small corrections that make precise insertion succeed.

---

## 2. What a run looks like

### 2.1 Setup (once, at start)

1. Load π_base from `base_policy.wt_path` (or `base_policy.wandb_id`). The checkpoint's own config **overrides** `cfg.actor` (`src/common/config_util.py`), so base-policy settings can't be changed from the command line. The script then **hard-codes 4 DDIM inference steps** for the diffusion base (`agent.inference_steps = 4` in `residual_ppo.py`), whatever the checkpoint says. That matches paper Table IV.
2. Build 1024 parallel IsaacGym envs, the residual policy, and two Adam optimizers with cosine LR schedules (actor 3e-4, critic 5e-3).
3. Derive the run size from the config:

| Quantity | Formula | Example value |
|---|---|---|
| Steps per iteration | `data_collection_steps = num_env_steps` | 700 |
| Batch size | `700 × num_envs` | 716,800 transitions |
| Training iterations | `total_timesteps // batch_size` | 697 |
| Minibatch size | `batch_size // num_minibatches` (default 1) | 716,800 (full batch) |

4. Create `models/<run_name>/` for checkpoints and start a wandb run.

### 2.2 The main loop: two kinds of iterations

The loop runs `while global_step < total_timesteps`. Each iteration is either **eval** or **train**:

```
eval_mode = (iteration - eval_first) % eval_interval == 0
```

With the defaults (`eval_first: true`, `eval_interval: 5`):

| Iteration | Mode | Residual actions | Weights updated? | `global_step` advances? | Checkpoint |
|---|---|---|---|---|---|
| 1 | **eval** | mean (deterministic) | no | no | saved if SR beats best |
| 2–5 | **train** | sampled (std ≈ 0.37) | **yes**, PPO update | yes | no |
| 6 | **eval** | mean | no | no | saved if SR beats best |
| 7–10 | **train** | sampled | **yes** | yes | no |
| … | … | … | … | … | … |

Two consequences:

- **The policy is updated on every training iteration.** Eval only decides when `actor_chkpt_best_success_rate.pt` is written.
- **Eval iterations are extra.** The `697` in `Iteration: 2/697` counts training iterations only. Evals add about `697 / 4 ≈ 175` more iterations of rollout time on top.

### 2.3 Inside one training iteration

```
┌─ 1. Reset ── all envs reset (reset_every_iteration: true)
│
├─ 2. Rollout ── 700 steps × 1024 envs                              (~190 s, ~92% of time)
│     for each step:
│       base_action   = π_base(obs)            # frozen diffusion, 4 DDIM steps per replan (every 8 steps)
│       residual, logp, value = π_res(obs ⊕ base_action)
│       action        = base_action + 0.1 × residual
│       obs, reward, done = env.step(action)   # reward normalized by running std, clipped at 5
│       store obs, residual, logp, value, reward, done
│
├─ 3. Rollout stats ── SR, STS, MSEL printed (see §3)
│
├─ 4. Advantages ── GAE (discount 0.999, λ 0.95) on the 700 × 1024 batch
│
├─ 5. PPO update ── up to 50 epochs over the full batch                (~16 s, ~8%)
│     loss = clipped policy loss (clip 0.2) + 1.0 × value loss
│     grad clip 1.0; stop early if approx_kl > target_kl (0.1)
│
└─ 6. Log to wandb, step LR schedulers, optional periodic checkpoint
```

An **eval iteration** does steps 1–3 with mean residual actions, saves the checkpoint if SR improved, logs `eval/success_rate`, and skips steps 4–6.

### 2.4 How long a run takes

| Phase | Time |
|---|---|
| Rollout (700 × 1024 at ~3750 steps/s) | ~190 s |
| PPO update | ~16 s |
| One training iteration | ~207 s |
| One eval iteration | ~190 s |

Full example run: `697 × 207 s + ~175 × 190 s ≈ 47 h`. In practice you don't need all of it: the authors' `one_leg` low checkpoint was saved at ~175M steps. See §6.

---

## 3. Reading the console output

```
Evaluation success rate improved. Model saved to models/<run>/actor_chkpt_best_success_rate.pt
Iteration: 2/697
Run name: 1790257927__residual_diffusion_ppo__2375644001
Eval mode: False
env_step=100, global_step=103424, mean_reward=0.0 fps=5415.11
...
env_step=600, global_step=615424, mean_reward=0.595703125 fps=3740.16
SR: 13.2812%, SPS: 3750.23, STS: 10.0795%, MSEL: 531.25
Policy update: 100%|██████████| 50/50 [00:16<00:00,  3.04it/s]
Iteration 2/697, global step 716800, SPS 3452
```

| Field | Meaning |
|---|---|
| `Evaluation success rate improved…` | The previous eval iteration beat the best eval SR so far, and the checkpoint was overwritten. |
| `Eval mode` | `True` = deterministic evaluation, no training. `False` = collect data and update. |
| `env_step` | Step within the current 700-step rollout, printed every 100 steps. |
| `global_step` | Total training transitions so far (`num_envs` × steps). Doesn't advance during eval, which is why iteration 2 starts at 101 × 1024 = 103,424. |
| `mean_reward` | Reward summed so far this rollout, averaged over envs. Rewards are divided by a running std, so one success shows up as roughly 4–5, not 1. Treat it as a progress signal, not a success rate. |
| `fps` | Env steps per second so far in this rollout. It drops over the rollout as more envs are in contact-heavy physics. |
| `SR` | Success rate: fraction of envs that collected ≥ `n_parts_to_assemble` positive rewards (1 for `one_leg`). 13.28% ≈ 136 of 1024 envs. |
| `SPS` (rollout line) | Env steps per second for this rollout only. |
| `STS` | Success timesteps share: fraction of all collected timesteps that belong to successful episodes (up to the success step). 136 × 531.25 / (700 × 1024) ≈ 10.08%. |
| `MSEL` | Mean success episode length: average steps a successful env needed, out of 700. |
| `Policy update 50/50` | PPO epochs. Fewer than 50 means early stopping on KL. |
| `SPS` (iteration line) | Cumulative steps per second over all training iterations, including update time. |

If you see the same block printed several times with identical `global_step` and `fps` values, that's terminal scrollback being copied twice, not the run restarting.

---

## 4. Important metrics

### 4.1 Tier 1: is the policy getting better?

| Metric (wandb key) | What it tells you | What you want |
|---|---|---|
| **`eval/success_rate`** | SR of the deterministic policy, which is what you deploy. **The main metric.** | Rising, then plateauing. Stop training when it flattens. |
| `eval/best_eval_success_rate` | Best eval SR so far, i.e. the saved checkpoint's SR. | Monotone non-decreasing. |
| `charts/success_rate` | SR during training rollouts, with noisy actions. | Rising. Usually below eval SR. Only compare it with other training SRs. |
| `charts/mean_success_episode_length` (MSEL) | How fast successful episodes finish. | Falling over training. |
| `charts/success_timesteps_share` (STS) | How much of each batch comes from successful behavior, i.e. how much learning signal PPO gets. | Rising. Very low STS (< ~1%) means very sparse signal. |

**Why training SR isn't used for checkpointing:** training rollouts sample residual actions with noise (`init_logstd: -1`, std ≈ 0.37), while eval uses the mean action that you actually deploy. The two numbers measure different policies.

### 4.2 Tier 2: is PPO healthy?

These ranges are common PPO rules of thumb, not values derived from this repo.

| Metric | What it tells you | Healthy | Warning sign |
|---|---|---|---|
| `losses/approx_kl` | How far each update moves the policy. | ~0.01–0.05 | Often > 0.1: early stopping every iteration, updates too aggressive. Lower `learning_rate_actor`. |
| `losses/clipfrac` | Fraction of samples hitting the PPO clip. | ~0.05–0.3 | > 0.4: updates too large. |
| `losses/explained_variance` | How well the critic predicts returns. | Rises toward 0.5–1 | Stays ≤ 0 long after start: critic isn't learning. Negative early on is normal. |
| `losses/value_loss` | Critic fit error. | Decreasing or stable | Growing steadily. |
| `charts/action_norm_mean` | Size of the residual's position component (normalized units, before the 0.1 scale). | Small, grows slowly | Grows large: the residual is overriding the base policy instead of correcting it. |
| `values/mean_logstd` | Residual action noise. | Constant at −1 with `learn_std: false` | Changes when `learn_std` is false: config problem. |
| `training/learning_rate_actor` | Current LR on the cosine schedule. | Warmup over 5 iterations, then cosine decay to the end of `num_iterations` | — |

### 4.3 Tier 3: throughput

| Metric | Note |
|---|---|
| `training/SPS` | Cumulative training steps per second. Use it to estimate remaining time: `(total_timesteps − global_step) / SPS`, plus eval overhead. |
| GPU memory | Check with `nvidia-smi`, not `torch.cuda.memory_allocated()`. The latter misses PhysX and PyTorch's cache. The example run uses ~7.2 of 8 GB at 1024 envs. |

---

## 5. Checkpoints

| File | When it's written | Contents |
|---|---|---|
| `models/<run>/actor_chkpt_best_success_rate.pt` | After an eval iteration whose SR beats the best so far. Overwritten each time. | Full agent (base + residual), optimizers, LR schedulers, config, SR, STS, iteration. |
| `models/<run>/actor_chkpt_<iteration>.pt` | Every `checkpoint_interval` iterations. Off by default (`-1`). | Same, minus STS. |

Each checkpoint is **~266 MB** because it includes the frozen diffusion base policy. Don't set `checkpoint_interval=1` (~230 GB over a full run). `checkpoint_interval=25` gives ~28 snapshots (~7.5 GB), which is a reasonable safety net against a performance collapse between evals.

A better policy between two evals isn't lost for learning: those weights keep training. Only the exact snapshot can be lost. PPO changes the policy in small steps (`clip_coef: 0.2`, `target_kl: 0.1`), so this rarely matters over a 4-iteration gap.

---

## 6. Making training faster

The rollout takes ~92% of the time and the GPU is compute-bound (~98% utilization), so the gains come from doing less work:

| Change | Effect | Caveat |
|---|---|---|
| Stop early by hand once `eval/success_rate` flattens, keeping `total_timesteps=500000000` | The released `one_leg` low checkpoint was saved after 244 iterations ≈ **175M** steps; med after 932 ≈ **668M** (scheduler `last_epoch` × 716,800; matches paper Figs. 26 and 27b). For low that's ~14–16 h instead of ~47 h. | Keeps the same cosine LR schedule as the authors' run. Lowering `total_timesteps` would shorten the run too, but it makes the LR decay faster than theirs. |
| `eval_interval=10` (or 20) | Cuts eval overhead from ~20% to ~10% (or ~5%). | Fewer chances to catch a peak for the best checkpoint. |
| `wandb.mode=offline` | Minor. | Sync afterwards with `wandb sync`. |

The base policy already runs with only 4 DDIM steps (hard-coded, see §2.1), so there's little left to gain from cutting diffusion inference.

What **not** to change for speed:

- **`num_env_steps`**: successful episodes take ~530 steps (MSEL), so going much below 700 truncates episodes that would succeed.
- **`num_envs`**: on a compute-bound GPU, more envs raise SPS only modestly (roughly 10–30%). They also halve the number of PPO updates for the same `total_timesteps`, and the full-batch update may run out of memory. If you want to test it, compare rollout SPS at `num_envs=1536` against ~3750 and switch only if it's ≥ 25% higher with memory below ~7.8 GB.
- **`update_epochs`**: only ~8% of the time, and it changes how PPO learns.

---

## 7. Parameter reference

All parameters are Hydra overrides of `src/config/base_residual_rl.yaml` (plus `src/config/actor/residual_diffusion.yaml` for `actor.residual_policy.*`). Pass them as `key=value` after the module name. In the tables:

- **Paper** gives the value from paper Tables VI and IX, where the paper lists one. A dash means the paper doesn't list it.
- ⚠️ marks a parameter whose value in the repo does nothing or is overridden.

### 7.1 The command, parameter by parameter

```bash
python -m src.train.residual_ppo base_policy.wt_path=<bc ckpt> \
  env.task=one_leg env.randomness=low num_env_steps=700 num_envs=1024 \
  total_timesteps=500000000 debug=false
```

| Parameter | Value | What it does |
|---|---|---|
| `python -m src.train.residual_ppo` | — | Runs the residual PPO trainer. Run it from the repo root. Checkpoints go to `./models/<run_name>/`, relative to where you launch. |
| `base_policy.wt_path` | `<bc ckpt>`: path to a BC `.pt`, e.g. the released `checkpoints/bc/one_leg/low/actor_chkpt.pt` or your own from Implementation Guide 1c | The frozen base policy π_base. The file must contain `config` (the BC run's config, which replaces `cfg.actor`) and the weights with the normalizer. Use either this or `base_policy.wandb_id`. |
| `env.task` | `one_leg` | Task to train on. Sets the IsaacGym assets, the reward (+1 per newly assembled part pair) and `n_parts_to_assemble` (1 for `one_leg`), which defines success. Must match the task π_base was trained on. |
| `env.randomness` | `low` | Initial part-pose randomization: `low` or `med` in the paper. Must match the level π_base was trained on, because the normalizer and the demos come from that level. |
| `num_env_steps` | 700 | Episode length and rollout length. Each iteration steps every env 700 times, and the env wrapper truncates at 700. Paper: 700 for `one_leg`, 1000 for lamp and round table. It also sets `data_collection_steps`, and through that `batch_size`. |
| `num_envs` | 1024 | Number of parallel IsaacGym envs. It sets throughput, GPU memory use and `batch_size = num_env_steps × num_envs` (716,800). Paper: 1024. |
| `total_timesteps` | 500,000,000 | Training budget in env steps, counting training iterations only. It sets `num_iterations = total_timesteps // batch_size` (697), which is also the length of the cosine LR schedule. Paper: 500M (a cap). The released low checkpoint peaked at ~175M. |
| `debug` | `false` | Already the default; the command keeps it explicit. `true` disables wandb logging entirely. |

The command sets no wandb entity: with `WANDB_ENTITY` exported (Implementation Guide, Stage 0), runs log there. To pick one per run, add `wandb.entity=<you>` (see 7.5).

### 7.2 Base policy

| Parameter | Default | What it does |
|---|---|---|
| `base_policy.wt_path` | `null` | Local BC checkpoint (see 7.1). |
| `base_policy.wandb_id` | `null` | Alternative to `wt_path`: `<project>/<run_id>` of a BC wandb run. It takes precedence over `wt_path` if both are set. |
| `base_policy.wt_type` | `best_success_rate` | Which checkpoint file to fetch from the wandb run: `best_success_rate`, `best_test_loss` or `last`. Only used with `wandb_id`. |
| Diffusion inference steps | 4, hard-coded ⚠️ | Not a config key. `residual_ppo.py` sets `agent.inference_steps = 4` after loading, overriding the checkpoint's value. Paper Table IV: 4. |
| `observation_type` | `state` | Must be `state`: `ResidualDiffusionPolicy` asserts it. The residual stage is state-only. |

### 7.3 Environment and control

| Parameter | Default | Paper | What it does |
|---|---|---|---|
| `env.task` | `one_leg` | — | See 7.1. |
| `env.randomness` | `low` | — | See 7.1. |
| `control.controller` | `diffik` | — | Low-level controller that turns end-effector targets into joint commands. Must match π_base. |
| `control.control_mode` | `pos` | Absolute EE pose | `pos` = absolute pose actions, `delta` = relative. Must match π_base. |
| `control.act_rot_repr` | `rot_6d` | 6D | Rotation representation in the 10-D action. Must match π_base. |
| `sample_perturbations` | `false` | — | `true` applies random forces and torques to parts every step during training. It's a robustness ablation (paper Table XIV), not part of the main recipe. |
| `reset_every_iteration` | `true` | — | Reset all envs at the start of every iteration, so each rollout is 700 steps of fresh episodes. Eval iterations always reset. |
| `truncation_as_done` | `true` | — | Treats hitting step 700 as a terminal state for GAE (no bootstrapping past the time limit). |
| `reset_on_success` / `reset_on_failure` | `true` / `false` | — | ⚠️ No effect: the env wrapper stores them but never reads them. Envs don't auto-reset mid-rollout; a successful env just stays assembled until the next iteration's reset. |
| `headless` | `true` | — | `false` opens the IsaacGym viewer. That's very slow at 1024 envs, so use it only for debugging with few envs. |
| `gpu_id` | 0 | — | CUDA device used for simulation, rendering and the networks. |

### 7.4 Rollout, batch and iteration sizing

| Parameter | Default | Paper | What it does |
|---|---|---|---|
| `num_envs` | 1024 | 1024 | See 7.1. |
| `num_env_steps` | 700 | 700 (`one_leg`) | See 7.1. |
| `data_collection_steps` | `${num_env_steps}` | — | Steps collected per iteration. Leave it tied to `num_env_steps`. |
| `total_timesteps` | 1,000,000,000 | 500M | See 7.1. The config default is 2× the paper's cap. |
| `batch_size` | derived | — | `data_collection_steps × num_envs`. Don't override it. |
| `num_minibatches` | 1 | 1 | Minibatches per PPO epoch. With 1, each epoch is a single gradient step on the full 716,800-sample batch. |
| `minibatch_size` | derived | — | `batch_size // num_minibatches`. |
| `num_iterations` | derived | — | `total_timesteps // batch_size`. The number of training iterations, and the LR schedule length. |

### 7.5 Evaluation, checkpointing and resuming

| Parameter | Default | What it does |
|---|---|---|
| `eval_interval` | 5 | Every `eval_interval`-th iteration is a deterministic evaluation: mean residual action, no update, `global_step` doesn't advance. |
| `eval_first` | `true` | Shifts the schedule so iteration 1 is an eval, which measures the untrained residual (≈ base policy SR). |
| `checkpoint_interval` | −1 | Save `actor_chkpt_<iteration>.pt` every N iterations. −1 = off. Each file is ~266 MB (see §5). The best-SR checkpoint is saved regardless. |
| `wandb.entity` | `null` | wandb user or team. `null` falls back to `WANDB_ENTITY` or your login's default entity. |
| `wandb.project` | `${env.task}-residual-rl` | wandb project, e.g. `one_leg-residual-rl`. |
| `wandb.mode` | `online` | `online`, `offline` (sync later with `wandb sync`) or `disabled`. |
| `wandb.notes` | `null` | Free-text notes attached to the run. |
| `wandb.continue_run_id` | `null` | Resume an existing run by id, in `wandb.project`. It reloads that run's config, most recent checkpoint, optimizer and LR-scheduler state, iteration count and best SR. If the run doesn't exist, a fresh run starts. |
| `seed` | `null` | Random seed. `null` draws a random one, which appears at the end of the run name (`…_ppo__<seed>`). |
| `torch_deterministic` | `false` | Sets `cudnn.deterministic`. The physics sim is not fully deterministic anyway. |

### 7.6 Residual policy (`actor.residual_policy.*`)

| Parameter | Default | Paper | What it does |
|---|---|---|---|
| `action_scale` | 0.1 | 0.1 | α in `action = base + α × residual`, in the normalized [−1, 1] action space. So the residual's σ = 1 corresponds to ±0.1. |
| `init_logstd` | −1.0 | −1.0 | Initial log std of the Gaussian exploration noise: std = e⁻¹ ≈ 0.37, i.e. ≈ 0.037 after scaling by α. |
| `learn_std` | `false` | — | Whether the log std is trained. With `false`, the noise stays fixed for the whole run. |
| `action_head_std` | 0.0 | — | Init gain of the actor's last layer, which has no bias. With 0 the initial residual mean is exactly 0, so training starts from pure π_base. |
| `actor_hidden_size` / `actor_num_layers` | 256 / 2 | — | Actor MLP. Its input is the normalized state (clamped to ±3) concatenated with the base action. |
| `critic_hidden_size` / `critic_num_layers` | 256 / 2 | 256 / 2 | Critic MLP, same input. |
| `actor_activation` / `critic_activation` | `ReLU` | ReLU (critic) | Hidden-layer activation. |
| `critic_last_layer_bias_const` | 0.25 | 0.25 | Initial bias of the value output. |
| `critic_last_layer_std` | 0.25 | — | Orthogonal-init gain of the value output layer. |
| `pretrained_wts` | `null` | — | Path to a previous residual checkpoint to warm-start the residual's weights. Optimizers and schedules start fresh. |

### 7.7 Optimization

| Parameter | Default | Paper | What it does |
|---|---|---|---|
| `learning_rate_actor` | 3e-4 | 3e-4 | Peak LR for the actor (AdamW, eps 1e-5, weight decay 1e-6; eps and weight decay are hard-coded). |
| `learning_rate_critic` | 5e-3 | 5e-3 | Peak LR for the critic. It's much higher than the actor's so the value function keeps up with a changing policy. |
| `optimizer_betas_actor` | [0.9, 0.999] | — | Adam betas for the actor. The critic uses the PyTorch defaults. |
| `lr_scheduler.name` | `cosine` | Cosine | Cosine decay over `num_iterations`. It steps once per **training** iteration. |
| `lr_scheduler.actor_warmup_steps` | 5 | — | Linear warmup for the actor LR, in iterations. |
| `lr_scheduler.critic_warmup_steps` | 0 | — | Warmup for the critic LR, in iterations. |
| `update_epochs` | 50 | 50 | PPO epochs per iteration. With 1 minibatch, that's 50 gradient steps per iteration. |
| `max_grad_norm` | 1.0 | 1.0 | Gradient norm clip on the residual policy. |

### 7.8 PPO objective

| Parameter | Default | Paper | What it does |
|---|---|---|---|
| `discount` | 0.999 | 0.999 | γ. Close to 1 because the only reward comes at the end of a ~500-step episode. |
| `gae_lambda` | 0.95 | 0.95 | λ for Generalized Advantage Estimation. |
| `norm_adv` | `true` | true | Normalize advantages to zero mean and unit std within each minibatch. |
| `clip_coef` | 0.2 | 0.2 | PPO ratio clip ε. |
| `clip_vloss` | `false` | — | Also clip the value-function update. Off. |
| `vf_coef` | 1.0 | 1.0 | Weight of the value loss in the total loss. |
| `ent_coef` | 0.0 | — | Entropy bonus. With `learn_std=false` the entropy is constant, so this has no effect unless `learn_std=true`. |
| `target_kl` | 0.1 | 0.1 | Stop the epoch loop for this iteration once approx-KL exceeds this. `null` disables it. |
| `n_iterations_train_only_value` | 0 | — | For the first N iterations, train only the critic (no policy loss). This warms up the value function. |
| `residual_l1` / `residual_l2` | 0.0 / 0.0 | — | Optional L1 and L2 penalties on the residual mean, pushing corrections toward zero. Off. |
| `base_bc.train_bc` | `false` | — | ⚠️ Not used by `residual_ppo.py`; it belongs to the `residual_ppo_w_bc.py` variant. |

### 7.9 Reward

| Parameter | Default | What it does |
|---|---|---|
| `normalize_reward` | `true` | Divide each reward by a running std, without subtracting the mean, so 0 stays 0. A sparse +1 becomes ≈ 4–5 early in training. |
| `clip_reward` | 5.0 | Clip the normalized reward to ±5. |

### 7.10 Settings of the released residual checkpoints

The configs saved in `checkpoints/rppo/one_leg/{low,med}/actor_chkpt.pt` differ from the defaults above on a few settings the paper doesn't list, plus one it does (init log std). To reproduce those runs instead of the paper tables, add:

```bash
actor.residual_policy.init_logstd=-0.9 actor.residual_policy.learn_std=true \
ent_coef=0.001 normalize_reward=false
```

The med run also used `total_timesteps=1000000000`. See the Implementation Guide, Stage 2, for the side-by-side table.
