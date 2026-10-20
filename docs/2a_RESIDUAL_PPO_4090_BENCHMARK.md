# Residual PPO Throughput Benchmark: RTX 4090

How many parallel envs (`num_envs`) to use for residual PPO on an RTX 4090, measured on 2026-10-20. It supplements §6 ("Making training faster") of [2_RESIDUAL_PPO_TRAINING_GUIDE.md](2_RESIDUAL_PPO_TRAINING_GUIDE.md).

**Result:** use `num_envs=2048 num_minibatches=2`. It collects data 1.45× faster than the paper's 1024 envs and keeps the paper's minibatch size and gradient steps per sample.

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB, driver 575.57.08 |
| CPU | AMD Ryzen 9 7950X (16 cores, 32 threads) |
| Code | commit `514137d` |
| Base policy | `outputs/2026-10-19/16-24-23.528744/models/ol-med-own-51-bs1024/actor_chkpt_best_success_rate.pt` (own BC policy, `one_leg`, med randomness) |
| Task / randomness | `one_leg` / `med` |
| Episode length | `num_env_steps=1000` |
| Diffusion inference | 4 DDIM steps (hard-coded in `residual_ppo.py`) |

Command for each run, with `N` = 1024, 2048 and 4096:

```bash
LD_LIBRARY_PATH=$HOME/miniconda3/envs/resip/lib:$LD_LIBRARY_PATH \
python -m src.train.residual_ppo \
  base_policy.wt_path=outputs/2026-10-19/16-24-23.528744/models/ol-med-own-51-bs1024/actor_chkpt_best_success_rate.pt \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=$N \
  total_timesteps=500000000 debug=true wandb.mode=disabled
```

## 2. Method

- **What was measured:** the first iteration of each run. With `eval_first: true`, that is an eval rollout: 1000 steps in every env, mean residual action, no PPO update. A training rollout costs the same per step; only the sampled noise differs.
- **SPS:** the `SPS` value on the end-of-rollout `SR:` line = `1000 × num_envs / rollout wall time`.
- **GPU memory and utilization:** sampled from `nvidia-smi` every 2 s. Memory is the peak; utilization is the mean over non-idle samples.
- **Not measured:** the PPO update phase. It is about 8% of iteration time at 1024 envs (from the laptop run), and it grows with batch size (see §4).
- Each configuration ran once.

## 3. Results

| `num_envs` | Rollout SPS | vs 1024 | Peak GPU memory | GPU utilization | Rollout time (1000 steps) | Iterations in 500M steps |
|---|---|---|---|---|---|---|
| 1024 | 8,367 | 1.00× | 3.7 GB | ~92% | ~122 s | 488 |
| **2048** | **12,174** | **1.45×** | 4.5 GB | ~93% | ~168 s | 244 |
| 4096 | 14,657 | 1.75× | 6.3 GB | ~92% | ~280 s | 122 |

SPS within each rollout falls as more envs reach contact-heavy phases:

| `num_envs` | fps at step 100 | fps at step 900 |
|---|---|---|
| 1024 | 8,800 | 8,375 |
| 2048 | 13,706 | 12,195 |
| 4096 | 16,523 | 14,686 |

**Reference:** the RTX 3000 Ada laptop GPU (8 GB) ran ~3,750 SPS at 1024 envs, with the released low-randomness BC policy and 700-step episodes, using ~7.2 GB. That's a different policy and episode length, so the comparison is approximate: the 4090 is about 2.2× faster at the same `num_envs`.

**Base-policy success rate from the same evals** (untrained residual, i.e. the base policy alone, 1000-step episodes):

| `num_envs` (episodes) | Success rate | Mean successful episode length |
|---|---|---|
| 1024 | 12.6% | 844 steps |
| 2048 | 12.6% | 844 steps |
| 4096 | 11.0% | 842 steps |

This confirms the med base policy needs more than 700 steps to succeed, which is why `num_env_steps=1000`.

## 4. Interpretation

- **The GPU is compute-bound at every size** (~92% utilization), and memory is never the limit (6.3 GB of 24 GB at 4096 envs). Larger batches still help because the GPU processes them more efficiently.
- **Returns diminish quickly.** 1024 → 2048 gives +45%; 2048 → 4096 gives only +20% more.
- **Larger `num_envs` changes PPO.** `batch_size = num_env_steps × num_envs`. With `num_minibatches=1`, PPO still takes 50 gradient steps per iteration, so doubling `num_envs` halves the gradient steps per env step. Scaling `num_minibatches` with `num_envs` keeps the paper's ratio:

| Configuration | Gradient steps per 1M samples | Minibatch size |
|---|---|---|
| Paper: `num_envs=1024 num_minibatches=1` | 50 | 1.02M |
| `num_envs=2048 num_minibatches=1` | 25 | 2.05M |
| **`num_envs=2048 num_minibatches=2`** | **50** | **1.02M** |
| `num_envs=4096 num_minibatches=4` | 50 | 1.02M |

- **4096 envs is not worth it.** The extra 20% throughput comes with only 122 iterations in a 500M-step run, so only ~24 evals (one every ~25 min). That makes it harder to catch the best checkpoint and to tell when training has plateaued.

## 5. Recommendation

```bash
python -m src.train.residual_ppo \
  base_policy.wt_path=outputs/2026-10-19/16-24-23.528744/models/ol-med-own-51-bs1024/actor_chkpt_best_success_rate.pt \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 num_minibatches=2 \
  total_timesteps=500000000 debug=false \
  wandb.entity=kaiyuan-luo-agile-agile-robots wandb.name=ol-med-own-51-resip-2048
```

- About 3 min per iteration including the PPO update, so ~13 h for the full 500M budget.
- Watch `losses/approx_kl`. Two minibatches mean 100 gradient steps per iteration instead of 50, so the policy can move further per iteration. `target_kl=0.1` stops an iteration's update early when it does. If "Early stopping at epoch …" appears in most iterations, fall back to `num_minibatches=1`.
- To match the paper exactly, use `num_envs=1024 num_minibatches=1`. It still runs 2.2× faster than the laptop.

## 6. Pitfalls found while benchmarking

- **`ImportError: libpython3.8.so.1.0`** when the script is run without activating the conda env: IsaacGym needs the env's `lib/` on `LD_LIBRARY_PATH`. `conda activate resip` sets it; otherwise prefix the command as in §1.
- **Log files lag far behind when output is redirected.** Python block-buffers stdout written to a file, so `env_step=…` lines appear only in large chunks or at exit. Set `PYTHONUNBUFFERED=1` when redirecting output (e.g. `nohup … > log.txt`).
- **Short test runs still write checkpoints.** The iteration 1 eval always "improves" on the initial best of 0, so it saves a ~250 MB `actor_chkpt_best_success_rate.pt`. With `debug=true` the folder is `models/dummy-<id>/`. Delete these after a benchmark.

## 7. Reproducing

The script used, with `PYTHONUNBUFFERED=1` added so that the stop condition sees output in time. It stops each run at env step 500, before the eval finishes and saves a checkpoint, and writes the log and GPU samples to `<outdir>`:

```bash
#!/bin/bash
# Usage: bench.sh <num_envs> <outdir>
N=$1; OUT=$2
cd /home/kaiyuan.luo/ResiP
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> $OUT/gpu_$N.txt; sleep 2; done ) &
MON=$!
PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$HOME/miniconda3/envs/resip/lib:$LD_LIBRARY_PATH \
  ~/miniconda3/envs/resip/bin/python -m src.train.residual_ppo \
  base_policy.wt_path=outputs/2026-10-19/16-24-23.528744/models/ol-med-own-51-bs1024/actor_chkpt_best_success_rate.pt \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=$N \
  total_timesteps=500000000 debug=true wandb.mode=disabled > $OUT/log_$N.txt 2>&1 &
PID=$!
for i in $(seq 1 900); do
  grep -q "env_step=500," $OUT/log_$N.txt && break
  kill -0 $PID 2>/dev/null || break
  sleep 1
done
kill -INT $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null
kill $MON
```

With an early stop, SPS comes from the `fps=` value at `env_step=500` rather than the end-of-rollout `SPS`. That reads slightly high, since fps drops later in the rollout (see §3).
