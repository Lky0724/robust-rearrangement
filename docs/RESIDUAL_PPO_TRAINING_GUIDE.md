# Residual PPO Training Guide (Stage 2, π_res)

How `src/train/residual_ppo.py` trains the residual policy on top of a frozen diffusion base policy: what a run does, what the console output means, and which metrics to watch.

Numbers in the examples are from a real `one_leg` / low-randomness run on an RTX 3000 Ada laptop GPU (8 GB):

```bash
python -m src.train.residual_ppo \
    base_policy.wt_path=checkpoints/bc/one_leg/low/actor_chkpt.pt \
    env.task=one_leg env.randomness=low \
    num_env_steps=700 num_envs=1024 total_timesteps=500000000
```

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

1. Load π_base from `base_policy.wt_path` (or `base_policy.wandb_id`). The checkpoint's own config **overrides** `cfg.actor` (`src/common/config_util.py`), so base-policy settings such as `actor.inference_steps` can't be changed from the command line.
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
│       base_action   = π_base(obs)            # frozen diffusion; DDIM steps from the BC ckpt (16 released, 4 in paper)
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

Full example run: `697 × 207 s + ~175 × 190 s ≈ 47 h`. See §6 for how to shorten it.

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
| Lower `total_timesteps`, e.g. `150_000_000` | ~47 h → ~14 h. The cosine LR schedule shrinks to match. | Pick a budget near where eval SR plateaus. Better than killing a long run early, when the LR is still high. |
| `eval_interval=10` (or 20) | Cuts eval overhead from ~20% to ~10% (or ~5%). | Fewer chances to catch a peak for the best checkpoint. |
| Fewer diffusion `inference_steps` (16 → 4) | The released BC checkpoints use 16; the paper's base policy uses 4 (Table IV). Cuts base-policy inference time in the rollout. | Comes from the BC checkpoint, so the command line can't change it. Either train π_base with `actor.inference_steps=4` (Implementation Guide 1c), or change the code to set `base_cfg.actor.inference_steps` before the agent is built. Check the base SR at 4 steps first. |
| `wandb.mode=offline` | Minor. | Sync afterwards with `wandb sync`. |

What **not** to change for speed:

- **`num_env_steps`**: successful episodes take ~530 steps (MSEL), so going much below 700 truncates episodes that would succeed.
- **`num_envs`**: on a compute-bound GPU, more envs raise SPS only modestly (roughly 10–30%). They also halve the number of PPO updates for the same `total_timesteps`, and the full-batch update may run out of memory. If you want to test it, compare rollout SPS at `num_envs=1536` against ~3750 and switch only if it's ≥ 25% higher with memory below ~7.8 GB.
- **`update_epochs`**: only ~8% of the time, and it changes how PPO learns.

---

## 7. Key config reference (`src/config/base_residual_rl.yaml`)

| Key | Default | Role |
|---|---|---|
| `num_envs` | 1024 | Parallel sim envs |
| `num_env_steps` | 700 | Rollout length = max episode length |
| `total_timesteps` | 1e9 | Training budget, sets `num_iterations` and the LR schedule length |
| `eval_interval` / `eval_first` | 5 / true | Eval frequency; eval at iteration 1 |
| `checkpoint_interval` | −1 | Periodic snapshots (off) |
| `update_epochs` / `num_minibatches` | 50 / 1 | PPO epochs over the full batch |
| `learning_rate_actor` / `learning_rate_critic` | 3e-4 / 5e-3 | Cosine schedule, actor warmup 5 iterations |
| `discount` / `gae_lambda` | 0.999 / 0.95 | GAE |
| `clip_coef` / `target_kl` | 0.2 / 0.1 | PPO trust region, KL early stop |
| `normalize_reward` / `clip_reward` | true / 5.0 | Reward divided by running std, then clipped |
| `actor.residual_policy.action_scale` | 0.1 | Residual scale relative to the base action |
| `actor.residual_policy.init_logstd` / `learn_std` | −1.0 / false | Fixed exploration noise (std ≈ 0.37) |
| `reset_every_iteration` | true | Reset all envs at the start of each iteration |
