# Residual PPO Problem Analysis: Reward Hacking, Force, Jerky Motion

The problems found when reproducing ResiP Stage 2 (residual PPO) with our own BC base policy: what causes them, what has been tried, and the plan. §5 is a chronological log of every training run and measurement, with the code changes made before it, the exact command, and the results. The diagnostics and config options are documented in the [training guide](2_RESIDUAL_PPO_TRAINING_GUIDE.md) (§3, §4.4, §7.3, §7.9).

**Setup.** `one_leg`, med randomness. Base policy: own diffusion BC checkpoint `ol-med-own-51-bs1024` (`action_horizon` 8, 4 inference steps), at `outputs/2026-10-19/16-24-23.528744/models/ol-med-own-51-bs1024/actor_chkpt_best_success_rate.pt`. Residual PPO: `num_envs=2048`, `num_env_steps=1000`, `num_minibatches=2`, `action_scale` 0.1, fixed `init_logstd` −1, `discount` 0.999. Sim arm joint PD stiffness / damping 1000 / 200 (paper setting).

**Runs at a glance** (details in §5):

| # | Run | Starts from | What's new | Outcome |
|---|---|---|---|---|
| R1 | `ol-med-own-51-resip-2048` (`dn4e5fbw`) | Base (no residual) | First residual run, paper setup | Eval SR 11.8% → 38.5%. Screws the leg in. The **clean** checkpoint. |
| R2 | `-cont` (`po2cp28i`) | R1 best | Continued training | Eval SR 37.5% → 70.8%, but **hacking** from ~iteration 150 (MSEL 783 → 416) |
| R3 | `-cont2` (`m5xcfe46`) | R2 best | Continued training | Eval SR 70.8% → 83.7%, MSEL 336. The **hacking** checkpoint. |
| M1 | Eval only | Base, R1, R3 | Torque and rotation diagnostics | Hacking: 3× torque, 0.8 vs 1.9 `turns`, MSEL 333 vs 789 |
| R4 | `-tq` (`vf76goxr`) | R1 best | Torque penalty | Eval SR 41.8% → 14.8%. Failed. |
| M2 | Eval and training | R1, R3 | Per-episode peak torque | Clean and hacking peaks overlap heavily |
| R5 | `-ff02` (`gml0v0a1`) | R1 best | Force-fail at torque ratio 0.2 | Eval SR 33.5% → 15.1%, hacking came back below the threshold. Failed. |
| M3 | Eval and training | Base, R1, R3 | Twist, target offset, smoothness | Twist separates screwing (1.19 turns) from forcing (0.26) completely |
| R6 | `-tw075` (`ajuv8htb`) | R1 best | Stricter success check: twist ≥ 0.75 turns | Eval SR 38.9% → ~64% (best 65.8%). **No hacking**: twist −1.19 turns throughout. Force and jitter grow slowly over training. |
| M4 | Offline, saved rollouts | Base, R1, R2, R3 | Smoothness of `evaluate_model` rollouts | Clean EE 1.8× less smooth than base in position. Target rows were wrong at first (the saved delta was used as the target); corrected 2026-10-31 |
| R7 | `-tw075-sm1` (`pz60k6mc`) | R1 best | 1a + smoothness loss on the policy mean (3g, weight 1.0) | Smoother than the base policy itself, and force about halved, but eval SR fell from 37.6% to ~29% (R6 at the same point: ~61%). Weight too strong. Crashed at iteration 388. |
| R8 | `-tw075-sm01` | R1 best | Same as R7, weight 0.1 | Next |

---

## 1. Problems

| # | Problem | Root cause |
|---|---|---|
| P1 | **Reward hacking**: the policy forces the leg straight down past the threads instead of screwing it in (~1.2 turns of twist during insertion) | The success check looks only at the final pose, not at how it was reached. It requires the leg's orientation to be within ~20° of the assembled one (`ori_bound` 0.94), so the twist can only be about 0.26 + k turns; forcing is k = 0 and skips one full turn of the thread. With γ = 0.999, finishing sooner pays more: forcing succeeds in ~333 steps, screwing in ~789, so a forced success is worth 0.999^333 ≈ 0.72 vs 0.999^789 ≈ 0.45 (1.58×). The simulated threads are shallow (~0.6 mm deep, SDF resolution 512) and can be pushed past. |
| P2 | **Force may not be executable on the real robot** | Sim joint stiffness is 1000, while the real robot's polymetis `Kq` is 40–160. The residual output is unbounded (no tanh or clamp), and the action is an absolute pose target, so the target can be placed far below the part and the force has no cap. Forcing places the target ~76 mm below the end effector, vs ~22 mm when screwing (M3). Without forcing, the offset still creeps up over training (R6: 22 → 26 mm p50). |
| P3 | **Jerky motion**: successful PPO rollouts move in stops and starts, even on simple free-space segments, unlike the base policy. Seen in videos made with `src/eval/evaluate_model.py --save-rollouts` and `scripts/visualize_data.py video`. | Confirmed in M4 and R6, mainly in position, and it grows over training. **The PPO targets zigzag**: the lag-1 autocorrelation of the target's per-step motion is +0.61 for the base (consecutive steps keep their direction), +0.18 for clean, −0.21 for R6 and about −0.3 for the hacking runs (the next step tends to reverse). R6's end effector is 3× less smooth than the base's. Each base chunk starts with a surge, which the residual amplifies, but every other step also moves 2–3× the base's, and in R6's training the ordinary steps grew much more than the chunk-start step (+66% vs +13%). The base's part stays constant; the residual's part grows (R6 eval: 2.8 → 8.6 mm). Likely causes: nothing in the objective rewards smoothness (sparse success reward, and the stiff PD drive hides target jitter, so it costs no successes); the residual learns strong feedback to recover from the 7–8 mm/step exploration noise, which overshoots without noise; γ = 0.999 rewards speed; nothing makes the residual's mean vary smoothly between neighboring states. Rare one-step 180° flips of a part's orientation in the observation add single jolts. Base videos were made the same way, so the 3× playback speed of `visualize_data.py video` (30 fps for 10 Hz data) doesn't explain the difference. In R7, a smoothness loss at weight 1.0 removed the zigzag but cost too much SR; a lower weight is next. |
| P4 | **Remaining failures**: eval SR plateaus at ~64% (R6) | All saved R6 failures happen while grasping or setting the leg into the hole; every episode that set the leg in finished screwing it. The base fails at the same two stages (16 of its 17 saved failures). |

---

## 2. What didn't work: constraining joint torque

| Approach | Result | Why |
|---|---|---|
| Torque penalty (R4) | Eval SR fell from 42% to 15% over 176 iterations; R2 from a similar start without the penalty rose | It penalized short spikes the policy can't control, and legitimate contact too, so the policy learned to back off from contact |
| Force-fail at torque ratio 0.2 (R5) | Eval SR fell from 33.5% to 15.1%, the training force-fail share rose from 40% to 48%, MSEL fell from 788 to 602 and `turns` from 1.88 to 1.28 | Most failures came from exploration noise (training failed 2–2.5× as many episodes as eval), which the policy can't avoid with a fixed noise level. The only learnable signal left was finishing sooner, so the policy **learned to force the leg in below the threshold**. |
| Picking a threshold from per-episode peaks (M2) | Clean and hacking distributions overlap heavily | Peaks come from incidental spikes. In eval, half of the hacking successes already peak below 0.2. |

**Conclusion.** Joint torque doesn't separate forcing from screwing, whether as a penalty or as a failure condition: the two overlap, noise spikes at stiffness 1000 dominate the real difference, and forcing works below any useful threshold. **What does separate them is the leg's twist during insertion** (M3), and to a lesser degree MSEL and `turns`. The torque options stay in the code, off by default and marked not recommended.

---

## 3. Options

### P1 Reward hacking

| Option | What | Assessment |
|---|---|---|
| **1a. Stricter success check: require enough twist** | Success = assembled pose **and** \|twist\| during insertion (within 3 cm of an assembled position) ≥ N turns. The reward stays sparse (1 on success), no extra reward terms. | **Implemented** as `safety.min_insert_twist_turns` and **verified in R6**: over 612 iterations the twist stayed at −1.19 turns and eval `twist fail` near 0, while R2 (same start, without 1a) began forcing at ~iteration 150. N = 0.75, midway between forcing (0.26) and screwing (1.19). Keep it on in every later run. |
| 1b. Filter rollouts after the fact | Keep only successful episodes with insertion twist ≥ 0.75 turns when saving D_synth | Still worth doing as a data safeguard. Computing the twist from saved pickles must skip the one-step orientation flips in `parts_poses` (R6 results), or legitimate successes look forced. |
| 1c. `discount=0.9999` | Shrinks the forcing advantage from 1.58× to 1.05× | Not needed after R6. |
| 1d. Make the threads harder to force | SDF resolution 1024, lower `max_depenetration_velocity`, higher friction | Not needed after R6. |
| 1e. Tighter assembled-pose tolerance | Rejects forced results if they end up tilted or not fully seated | Not needed after R6. |
| ~~Torque penalty / torque force-fail~~ | — | Tried, didn't work (§2). |

### P2 Force and real-robot executability

| Option | What | Assessment |
|---|---|---|
| 2a. Lower stiffness toward the real robot (300–500) | Smaller noise spikes; sim forces map better to the real robot; softer motion | Recommended. Doesn't cap force by itself: the policy can put the target further below the part. Check the base policy's SR at the new stiffness first (`env.stiffness`). |
| **2b. Cap the target-to-EE offset** | A cap on how far below the end effector the target may be, so the maximum push is bounded, a hard limit the policy can't learn around | **Recommended.** Simple, guaranteed, and the same limit can run on the real robot. From M3: legitimate insertion reaches up to ~45 mm (clean, training p99), and the base itself sets 16–30 mm, so ~50 mm leaves normal motion alone while cutting typical forcing (~76 mm). With 1a in place, this is for real-robot force, not against hacking. R6 ended at 47 mm eval p99, so a 50 mm cap barely binds yet; it stops further growth and guards the real robot. |
| 2c. Check joint velocity limits in sim | Confirm the URDF velocity limits are set on the arm's DOF properties | A quick check. |
| 2d. Wrist force sensor | Log the end-effector external force in N | A useful diagnostic, comparable with the real arm's collision thresholds. Needs a furniture-bench change. |

### P3 Jerky motion

| Option | What | Assessment |
|---|---|---|
| ~~3f. Measure `evaluate_model` rollouts~~ | — | Done (M4, corrected in R6): the jitter is real, mainly in position, and is mostly step-to-step back-and-forth, plus a surge at each chunk start. |
| **3g. Smoothness loss on the policy mean** | Add to the actor loss the squared second difference of the final target (base + residual mean) over consecutive steps of the rollout | **Implemented as `residual_smooth`. It works, but weight 1.0 is too strong** (R7): the motion became smoother than the base's and eval SR fell to ~29%. Weight 0.1 is next (R8). Targets the measured cause (the residual's zigzag) directly. Uses the mean, so exploration noise doesn't enter (unlike 3e); the second difference leaves constant-velocity motion alone; including the base lets the residual also soften the chunk-start surge. No reward change, no added lag, one coefficient. |
| 3a. Smooth the residual | Exponential smoothing, or a cap on the per-step change of the residual | If 3g isn't enough. Guaranteed smoothness, but adds lag, which can hurt the precise insertion; best with the previous residual added to the observation. |
| 3h. Less, or temporally correlated, exploration noise | Lower `init_logstd`, or noise correlated over steps | Weakens the pressure to learn strong feedback against white noise. A second-round experiment. R7's 21-point gap between training and eval SR adds a reason: with less noise, PPO has to learn the alignment in the mean, which is what gets deployed. Unproven. |
| 3j. Apply the smoothness loss only away from the precise phases | Skip the triplets where the leg is near the hole (and possibly around the grasp), so the policy keeps its fast corrections there | If no single weight gives both smoothness and SR. Needs a state-based mask; a loss change, not a reward change. |
| 3c. Blend at replanning | Blend the old and new chunk where they meet, or execute fewer steps per chunk (e.g. 4 of 8) | Lower priority than first thought: it only addresses the chunk-start surge, not the zigzag at every step. Changes the base's behavior, so check the base's SR first. |
| 3i. Find the orientation flips in the observation | A part's quaternion in `parts_poses` jumps 180° for one step (R6 results) | Rare, but each gives a single jolt (~21 mm target step in R6). Find the source first (read-only); low priority. |
| 3b. `residual_l2` > 0 | Keeps the residual small, so the motion stays close to the smooth base | No code change; weakens the correction. |
| 3d. Low-pass filter D_synth actions before distillation | Affects only the distilled policy | A fallback. |
| 3e. Penalize the action change between steps | Extra reward term | Not recommended: the sampled actions are mostly exploration noise (R6 training residual second difference 37 mm vs 8.5 mm in eval), the same failure mode as the torque penalty. |

### P4 Remaining failures

| Option | What | Assessment |
|---|---|---|
| 4a. Check again after 3g | Grasping and setting the leg in are precise positioning, where a zigzagging target may hurt | R7 shows no improvement: much smoother motion came with lower SR, and the saved failures are at the same stages (20 episodes only). Compare again after R8. |
| 4b. Train longer with a fresh LR schedule | R6's cosine LR was below ~15% of its peak in the last quarter | Not tried: R7 started from R1, not R6. Try it on the final recipe. |
| 4c. Log failure stages in training | Missed grasp / not lifted / not set into the hole / not finished, over all 2048 envs | Gives reliable shares; the saved rollouts have 2–4 per class. Needs code (the wrapper already has the part and gripper states). |
| 4d. More BC demos of grasping and alignment | The base fails at the same stages | If RL stops improving them. |

---

## 4. Plan

Principles: get the success check right (1a), regularize the policy rather than the reward (3g), put hard physical limits in the env (2a, 2b, 3a), and filter after the fact as a fallback (1b, 3d). No complex reward design. Change one thing per run, so each result can be attributed.

| Step | Content | Status |
|---|---|---|
| ~~R6~~ | Retrain from R1 with 1a only | Done: no hacking (§5) |
| ~~R7~~ | 3g (smoothness loss) at weight 1.0 with 1a, from R1 best | Done: smoother than the base and half the force, but eval SR ~29% (§5) |
| SR check | Evaluate R7's checkpoint 375 and R1 best with `evaluate_model` on 1024 episodes: 20 saved episodes show more successes than the training eval | **Next**; minutes, no training (§5, R8) |
| Wandb storage | ~~Upload only the best-SR checkpoint and keep the others local~~ (code done in `residual_ppo.py` and `bc.py`, §5). The 13 numbered R6 and R7 checkpoints already on wandb are kept (3.46 of 5.87 GB; all have local copies), so usage doesn't drop | Done; check R8's upload (§5) |
| **R8** | Same as R7 with `residual_smooth=0.1`, so it differs only in the weight. Aim: SR close to R6's with EE jitter below its starting level | After the two above; command in §5 |
| R9 | From scratch (no residual weights) with 1a and 3g at the weight R8 settles on: the from-scratch recipe | After R8. If the start is slow, try 3h next |
| Smoothness, round 2 | 3j (loss only away from the precise phases) or 3h, if no single weight gives both smoothness and SR | Conditional |
| Physical limits | 2b (offset cap ~50 mm), then 2a (stiffness), one at a time. R7's offset and torque fell with the jitter, so check how much is still needed | After smoothness |
| Remaining failures | 4c (failure-stage logging) any time; 4a after R8; 4d if RL stalls | |
| Observation flips | 3i: find the source | Any time, read-only |
| End-to-end pipeline | Roll out a smooth, non-hacking checkpoint, filter by insertion twist (1b), smooth the actions if needed (3d), save D_synth, distill | After R8; needs an `evaluate_model.py` change |

Notes:

- If a lower stiffness is adopted, `src/eval/evaluate_model.py` needs the same `stiffness` / `damping` parameters, since it still builds the env with 1000 / 200.
- Saved rollout `actions` are deltas (target − EE pose at that step, `src/data_collection/io.py`). Rebuild the target as EE + delta before analyzing it.
- `torch.quantile` in the diagnostics fails above 2^24 (~16.8M) elements, i.e. `num_env_steps × num_envs`. Current settings use ~2M.
- All options added so far are off by default. With the default config, training behaves as with the upstream code (same rewards, terminations and PD gains; SPS unchanged within noise).

---

## 5. Run log

Chronological. Each entry lists the code changes made since the previous entry, the command, and the results. "Eval" means `residual_ppo`'s deterministic eval (residual mean), which runs at iteration 1 and every 5 iterations; "training" means the sampled rollouts in between. Iteration counts include eval iterations, which don't add env steps, so a run can show more iterations than `total_timesteps / (num_envs × num_env_steps)`. Wandb ids refer to the local `wandb/run-*-<id>` directories. Measurements (M) used `debug=true`, ran only iterations 1 (eval) and 2 (training), and were stopped with Ctrl+C.

Common prefix of all commands:

```bash
BASE=outputs/2026-10-19/16-24-23.528744/models/ol-med-own-51-bs1024/actor_chkpt_best_success_rate.pt
```

### R1 · `ol-med-own-51-resip-2048` · 2026-10-20 · `dn4e5fbw`

**Code.** Upstream residual PPO; no changes to the RL code (git `514137d`).

```bash
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 num_minibatches=2 \
  total_timesteps=500000000 debug=false \
  wandb.entity=kaiyuan-luo-agile-agile-robots wandb.name=ol-med-own-51-resip-2048
```

**Results.** Stopped at iteration 295 (483M steps).

| Iteration | 1 | 51 | 101 | 151 | 201 | 251 | 271 (best) |
|---|---|---|---|---|---|---|---|
| Eval SR | 11.8% | 13.1% | 14.9% | 23.8% | 31.3% | 35.5% | 38.5% |
| Eval MSEL | 845 | 839 | 826 | 816 | 789 | 785 | — |

The leg is screwed in. `models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt` is the **clean** checkpoint used as the start of R2, R4, R5 and R6.

### R2 · `-cont` · 2026-10-21 · `po2cp28i`

**Code since R1.**

- `wandb.name` now names both the wandb run and `models/<name>` (`12d0773`).
- `actor.residual_policy.pretrained_load_optim`: `false` loads only the weights from `pretrained_wts`, with a fresh optimizer and LR schedule. It was in the working tree for this run and committed later in `7a7bcb5`.

```bash
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt \
  actor.residual_policy.pretrained_load_optim=false \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 num_minibatches=2 \
  total_timesteps=1000000000 debug=false \
  wandb.entity=kaiyuan-luo-agile-agile-robots wandb.name=ol-med-own-51-resip-2048-cont
```

**Results.** Stopped at iteration 562 (920M steps).

| Iteration | 1 | 51 | 101 | 151 | 201 | 301 | 401 | 501 | 541 (best) |
|---|---|---|---|---|---|---|---|---|---|
| Eval SR | 37.5% | 39.7% | 45.1% | 43.3% | 46.2% | 54.1% | 61.3% | 66.5% | 70.8% |
| Eval MSEL | 783 | 771 | 769 | 700 | 577 | 476 | 452 | 426 | — |

**Hacking appears** between iterations 100 and 200: MSEL drops from ~770 to ~580 while SR keeps rising. In simulation the leg is forced straight down, the arm shakes and the leg is sometimes thrown out.

### R3 · `-cont2` · 2026-10-23 · `m5xcfe46`

**Code since R2.** None.

```bash
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048-cont/actor_chkpt_best_success_rate.pt \
  actor.residual_policy.pretrained_load_optim=false \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 num_minibatches=2 \
  total_timesteps=1000000000 debug=false \
  wandb.entity=kaiyuan-luo-agile-agile-robots wandb.name=ol-med-own-51-resip-2048-cont2
```

**Results.** Finished: 612 iterations, 1.0B steps, 33.8 h.

| Iteration | 1 | 101 | 201 | 301 | 401 | 501 | 581 (best) | 601 |
|---|---|---|---|---|---|---|---|---|
| Eval SR | 70.8% | 71.5% | 74.8% | 77.8% | 78.4% | 80.0% | 83.7% | 81.8% |
| Eval MSEL | 414 | 404 | 386 | 370 | 355 | 346 | — | 336 |

Fully hacking. `models/ol-med-own-51-resip-2048-cont2/actor_chkpt_best_success_rate.pt` is the **hacking** checkpoint used in M1–M3.

### M1 · Torque and rotation diagnostics · 2026-10-28

**Code since R3.**

- `env_rl_wrapper.py`: per-step `info["torque_ratio"]` (max_j \|τ_j\| / limit_j over the 7 arm joints; limits 87 Nm for joints 1–4, 12 Nm for 5–7) and `info["part_rotation"]` (rotation of the inserted part since the previous step, any axis, unsigned). `info["task_reward"]` keeps the raw 0/1 reward.
- `env_rl_wrapper.py`: optional torque penalty, `reward -= coef × max(0, torque_ratio − threshold)` before normalization (`safety.torque_penalty_coef`, `safety.torque_threshold`; off by default).
- `residual_ppo.py`: SR, STS and MSEL from `task_reward` (identical when the penalty is off); prints and logs `TQ p99` and `turns` (unsigned rotation of the part from reset to success); eval MSEL logged.

```bash
# base
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 debug=true
# clean (R1)
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 debug=true
# hacking (R3)
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048-cont2/actor_chkpt_best_success_rate.pt \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 debug=true
```

**Results** (iteration-1 eval):

| Policy | SR | MSEL | `TQ p99` | `turns` |
|---|---|---|---|---|
| Base only | 11.3% | 856 | 0.031 | 1.84 |
| Clean (R1) | 40.3% | 789 | 0.042 | 1.88 |
| Hacking (R3) | 85.6% | 333 | 0.126 | 0.82 |

### R4 · `-tq` · 2026-10-29 · `vf76goxr`

**Code since M1.** None. Uses the torque penalty with `torque_threshold=0.06`, set from M1's eval `TQ p99` values.

This run also changed three settings besides the penalty: `eval_interval=10`, `lr_scheduler.name=constant_with_warmup`, `n_iterations_train_only_value=3`.

```bash
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt \
  actor.residual_policy.pretrained_load_optim=false \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 num_minibatches=2 \
  total_timesteps=1000000000 checkpoint_interval=25 eval_interval=10 \
  lr_scheduler.name=constant_with_warmup n_iterations_train_only_value=3 \
  safety.torque_threshold=0.06 safety.torque_penalty_coef=0.5 \
  wandb.name=ol-med-own-51-resip-2048-tq
```

**Results.** Stopped at iteration 176. The best checkpoint is iteration 1.

| Iteration | 1 | 41 | 81 | 121 | 161 | 171 |
|---|---|---|---|---|---|---|
| Eval SR | 41.8% | 31.6% | 29.0% | 22.4% | 18.6% | 14.8% |
| Eval MSEL | 785 | 791 | 793 | 786 | 775 | 793 |
| Eval `TQ p99` | 0.040 | 0.030 | 0.027 | 0.023 | 0.022 | 0.027 |

**Failed.** Training's `TQ p99` was already 0.057, so the threshold of 0.06 (taken from eval) penalized ordinary training rollouts. Contact torques peak during legitimate insertion too, so the policy learned to back off from contact: torque fell and SR fell with it.

### M2 · Per-episode peak torque · 2026-10-29

**Code since R4.** `residual_ppo.py`: for each successful episode, the peak `torque_ratio` up to the success step; logs p50/p90/p99 and the share above 0.1/0.15/0.2/0.3, printed as `Success peak TQ …`.

```bash
# clean (R1); same with pretrained_wts=models/ol-med-own-51-resip-2048-cont2/actor_chkpt_best_success_rate.pt for hacking
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt \
  actor.residual_policy.pretrained_load_optim=false \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 debug=true
```

**Results:**

| | SR | MSEL | `TQ p99` | `turns` | Peak p50/p90/p99 | Peak above 0.1 / 0.15 / 0.2 / 0.3 |
|---|---|---|---|---|---|---|
| Clean, eval | 38.6% | 781 | 0.039 | 1.88 | 0.111/0.265/0.824 | 57.3 / 29.5 / 16.6 / 7.6% |
| Clean, training | 32.9% | 787 | 0.058 | 2.62 | 0.175/0.426/0.988 | 86.4 / 60.5 / 41.5 / 21.5% |
| Hacking, eval | 84.4% | 332 | 0.125 | 0.81 | 0.201/0.518/1.494 | 92.2 / 71.6 / 50.4 / 26.4% |
| Hacking, training | 81.2% | 342 | 0.162 | 1.08 | 0.282/0.730/1.998 | 97.9 / 86.5 / 72.4 / 46.9% |

The distributions overlap heavily: legitimate episodes have short spikes too (grasp impacts, contact, wrist at a joint limit), and peaks reach 1.0–2.0. 0.2 gave the largest gap (41.5% vs 72.4% in training), so it was tried in R5.

### R5 · `-ff02` · 2026-10-29 · `gml0v0a1`

**Code since M2.**

- `env_rl_wrapper.py`: optional force-fail (`safety.terminate_torque_ratio`, off by default). Once `torque_ratio` exceeds it before success, the env is done for the rest of the rollout and gets no success reward; spikes after success are ignored. `residual_ppo.py` prints and logs `force fail`.
- `get_rl_env` and config: `env.stiffness` / `env.damping` (default 1000 / 200), passed to `FurnitureRLSimEnv`. Not changed in any run so far.

```bash
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt \
  actor.residual_policy.pretrained_load_optim=false \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 num_minibatches=2 \
  total_timesteps=1000000000 checkpoint_interval=25 \
  safety.terminate_torque_ratio=0.2 wandb.name=ol-med-own-51-resip-2048-ff02
```

**Results.** Stopped at iteration 193. The best checkpoint is iteration 1.

| Eval iteration | Eval SR | Eval MSEL | Eval `turns` | Eval `force fail` | Training `force fail` (mean over iterations) |
|---|---|---|---|---|---|
| 1 | 33.5% | 788 | 1.88 | 15.9% | 40.1% (2–11) |
| 61 | 24.5% | 770 | 1.87 | 18.0% | 41.6% (56–66) |
| 116 | 20.3% | 708 | 1.69 | 19.8% | 44.4% (111–121) |
| 191 | 15.1% | 602 | 1.28 | 22.4% | 47.6% (182–193) |

**Failed.** Training failed 2–2.5× as many episodes as eval, so most failures came from exploration noise, which the policy can't avoid; the training failure share never fell. The only learnable signal left was finishing sooner, so MSEL and `turns` moved toward the hacking values while successful episodes' peaks crept up to just below 0.2.

The code up to here was committed as **`a8a2a49`**.

### M3 · Twist, target offset, smoothness · 2026-10-30

**Code since R5.**

- `env_rl_wrapper.py`: per-step `info["part_twist"]`, the signed rotation (rad) of the inserted part about its long axis (leg mesh local y) since the previous step, relative to the base part (swing-twist decomposition of the relative quaternion). `info["part_rel_pos"]`, its position in the base part's frame. `info["ee_goal_offset"]`, the position target (clipped to the env's action bounds) minus the end-effector position read before the step. No furniture-bench change.
- `residual_ppo.py`: records whether the base started a new action chunk, the position target and the base's part of it. Prints and logs three new lines (training guide §3): insertion twist, target offset, and smoothness.
- Two bugs found on the first run and fixed. The chunk-start check used `len(agent.actions)`, but `ResidualDiffusionPolicy` sets the queue to `None` before its first prediction; it's now `not agent.actions`. The offset clipping indexed the env's action bounds as `(10,)`, but they are `(num_envs, 10)`; it's now `[..., :3]`.

**`part_twist` vs `turns`.** `turns` sums the unsigned per-step rotation about any axis in the world frame from reset, so jitter, grasping and transport add to it (clean: 2.6 in training vs 1.9 in eval). `part_twist` is signed, about the leg's own axis only, relative to the tabletop, and summed over insertion only (the steps up to success while the leg is within 3 cm of its final position).

Commands: the three commands of M1.

**Results** (iteration-1 eval; iteration-2 training in brackets):

| | Base only | Clean (R1) | Hacking (R3) |
|---|---|---|---|
| SR | 12.5% (2.0%) | 37.9% (33.1%) | 85.4% (79.9%) |
| MSEL | 839 (825) | 784 (790) | 330 (344) |
| Insertion twist p10/p50/p90 (turns) | −1.20/−1.19/−1.18 | −1.20/−1.19/−1.18 | −0.27/−0.26/−0.24 |
| Share with \|twist\| ≥ 1 turn | 98.8% (85.0%) | 99.2% (97.6%) | 0.2% (0.6%) |
| Insertion steps | 495 | 468 | 122 |
| Peak target below EE in insertion, p50/p90/p99 (mm) | 15.9/23.4/30.6 (25.8/32.9/37.3) | 21.7/30.4/37.4 (29.6/37.0/45.1) | 76.2/86.7/98.5 (77.3/94.1/108.7) |
| \|offset\| p99 up to success (mm) | 22 (33) | 30 (37) | 63 (67) |
| Target step at chunk start / other (mm)² | 13.4 / 3.3 | 13.3 / 3.9 | 20.9 / 10.3 |
| EE second difference (mm/step²)² | 1.66 (7.31) | 1.87 (7.67) | 4.35 (9.74) |
| Target second difference, base / residual part² | 5.29 / 0 | 5.09 / 2.58 (5.70 / 30.67) | 6.42 / 14.97 |

² Averaged over all running envs, including failed episodes. Superseded by M4: failed episodes jitter in place and hid most of the difference between base and clean.

- **Twist separates screwing from forcing completely**, with no overlap: ~1.19 vs ~0.26 turns. Because of the success check's orientation tolerance, the twist can only be about 0.26 + k turns; forcing skips one full turn.
- Forcing places the target ~76 mm below the end effector, 3.5× the clean policy's ~22 mm.
- Training rollouts are 4–6× less smooth than eval, almost all from exploration noise in the residual.

### R6 · `-tw075` · 2026-10-30 · `ajuv8htb`

**Code since M3.**

- `env_rl_wrapper.py`: optional stricter success check (`safety.min_insert_twist_turns`, off by default). The wrapper sums `part_twist` while the leg is within 3 cm of any of the env's `assembled_rel_poses` positions, which are the same poses the success check uses. On the step the env reports the leg as newly assembled, it fails the env if the summed \|twist\| is below the threshold: no success reward, and the env is already done. It raises an error if enabled on an env without part and EE states.
- `residual_ppo.py`: prints `twist fail` on the `SR:` line and logs `{charts,eval}/twist_fail_rate`. Twist failures count as failures in SR, STS, MSEL and the best-checkpoint choice.
- Tested with a fake env (real tensor shapes): screwing 1.19 turns near the assembled position keeps the reward; forcing with 0.26 turns fails; 1.19 turns in the air followed by 0.26 near the assembled position fails; 1.19 turns in the other direction passes; with the option off, all keep the reward.

```bash
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt \
  actor.residual_policy.pretrained_load_optim=false \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 num_minibatches=2 \
  total_timesteps=1000000000 checkpoint_interval=25 \
  safety.min_insert_twist_turns=0.75 wandb.name=ol-med-own-51-resip-2048-tw075
```

**Expected at iteration 1:** eval SR ~37% (under 1% of clean successes twist less than 0.75 turns), `twist fail` under 1%.

**Watch:**
- Good signs: SR rises; eval MSEL stays near ~780 rather than dropping toward ~330; insertion twist p50 stays near −1.19; `twist fail` stays low. Attempts at forcing raise `twist fail` briefly, but they earn nothing, so it should come back down.
- Stop if eval SR hasn't beaten the start after ~100 iterations, or if `twist fail` keeps rising while SR falls.

**Results.** Finished 2026-10-31 after 612 iterations (488 training, 36 h). Iteration 1 matched the expectation: eval SR 38.9%, MSEL 786, `twist fail` 0.34%. Best eval SR 65.8% at iteration 586 (`actor_chkpt_best_success_rate.pt`). The last 8 evals average 63.9%, so the best is ~2 points optimistic, being the maximum of noisy evals. This run uses the M3 smoothness metrics (all running envs), not the success-only ones.

**Hacking is gone.** Eval, 50-iteration means, against R2 (the same start and command without 1a):

| Iterations | 0–49 | 150–199 | 300–349 | 450–499 | 600–612 |
|---|---|---|---|---|---|
| R6 eval SR | 38.3% | 50.0% | 58.1% | 63.5% | 62.9% |
| R6 eval MSEL | 781 | 754 | 716 | 691 | 664 |
| R2 eval SR | 37.7% | 44.7% | 53.8% | 64.9% | 67.8%³ |
| R2 eval MSEL | 780 | 640 | 471 | 433 | 417³ |

³ R2's last iterations, 550–563.

- Insertion twist p50 −1.19 turns throughout; \|twist\| ≥ 1 turn in 99.8–100% of eval successes. Eval `twist fail` 0.03–0.12%, training 0.3–0.4%: attempts at forcing come from exploration noise, and they earn nothing.
- R2's MSEL fell to 417 as it learned to force. R6's fell 15% with the twist unchanged: it screws faster (insertion steps 466 → 399).

**Force and jitter grow slowly** (eval, iterations 0–49 → 600–612; the smoothness rows are M3 metrics over all running envs, so only their trend counts):

| | Start | End | Hacking (R3, M3) |
|---|---|---|---|
| Peak target below EE in insertion, p50 / p99 (mm) | 22.1 / 39.5 | 26.4 / 46.5 | 76.2 / 98.5 |
| Peak torque ratio p50, successful episodes | 0.11 | 0.14 | |
| EE second difference (mm) | 1.89 | 2.83 | 4.35 |
| Target second difference, residual part (mm) | 2.78 | 8.56 | 14.97 |
| Target step at chunk start / other (mm) | 13.4 / 3.9 | 15.2 / 6.5 | 20.9 / 10.3 |

- The base part's second difference stays at ~5.3 mm; all of the growth is the residual's. The ordinary steps grew 66%, the chunk-start step 13%.
- It still screws 1.19 turns, but faster, pushing harder, and more jerkily. γ = 0.999 pays ~12% more for finishing 117 steps sooner (0.999⁻¹¹⁷ ≈ 1.12), and nothing in the objective resists the jitter. Not hacking, but P2 and P3 get worse the longer it trains.

**Training vs eval SR.** Eval led by 2–4 points at first (the exploration noise costs successes), the two were equal at iterations 250–299, and training led from ~300 on (66.9% vs 62.9% at the end). The residual's mean is tuned for the noisy closed loop, while deployment uses the mean. No saved failure is stuck in the threads (below), so the noise more likely helps when setting the leg into the hole.

**SR plateau.** Eval SR stayed at ~63–64% from iteration ~450. The cosine LR schedule (over the 488 training iterations) was below ~15% of its peak in the last quarter, which accounts for part of it.

**Saved rollouts** (`evaluate_model`, best checkpoint, 20 episodes, commands below): 13 successes (65%, matching eval). The analysis uses ad-hoc scripts, not repo code.

Smoothness, successful episodes only. The target is rebuilt as EE + the saved delta (M4 had used the delta itself; see the correction there):

| | Base | R1 (clean) | R6 | R2 | R3 (hacking) |
|---|---|---|---|---|---|
| Successful episodes | 3 | 18 | 13 | 8 | 16 |
| Mean length (steps) | 889 | 806 | 702 | 436 | 355 |
| Target step / EE step (mm) | 2.51 / 1.95 | 3.92 / 2.50 | 6.13 / 3.32 | 7.87 / 4.06 | 11.64 / 5.54 |
| Target second difference (mm) | 2.05 | 4.52 | 8.57 | 11.38 | 17.52 |
| **EE second difference (mm)** | **0.73** | **1.30** | **2.16** | 2.71 | 4.11 |
| **Lag-1 autocorrelation of the target's per-step motion** | **+0.61** | +0.18 | **−0.21** | −0.37 | −0.32 |
| Same, of the EE's | 0.91 | 0.84 | 0.72 | 0.64 | 0.64 |
| Target second difference before the grasp (free-space reach, mm) | 3.45 | 6.75 | 10.11 | 13.84 | 21.31 |
| Mean target-to-EE offset (mm) | 5.5 | 7.7 | 9.8 | 12.9 | 18.3 |
| Target / EE rotation second difference (°) | 1.36 / 0.52 | 1.61 / 0.58 | 2.15 / 0.73 | 2.44 / 0.79 | 4.42 / 1.38 |

Target step (mm) by step index mod 8 (0 = the first step of a new base chunk):

| Phase | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Base | 3.43 | 2.43 | 2.42 | 2.34 | 2.40 | 2.23 | 2.45 | 2.40 |
| R1 (clean) | 6.82 | 4.59 | 3.65 | 3.30 | 3.31 | 3.22 | 3.24 | 3.23 |
| R6 | 11.20 | 7.09 | 6.12 | 5.37 | 5.19 | 4.92 | 4.62 | 4.55 |
| R3 (hacking) | 20.36 | 14.77 | 12.31 | 10.71 | 9.98 | 8.62 | 8.30 | 8.11 |

- **The PPO targets zigzag.** The base's consecutive target steps keep their direction (+0.61); R6's tend to reverse (−0.21). R6's target moves 6.1 mm per step while its EE moves 3.3 mm, and the PD drive filters the rest. R6's EE is 3× less smooth than the base's and 1.7× R1's.
- **Mostly not the chunk boundary.** R6's first step of a chunk is 2.3× its mid-chunk step, but every phase moves 2–3× the base's.
- The free-space reach before the grasp is as jerky (2.9× the base's), matching the videos.
- R6 lies between clean and hacking in every smoothness row.

Failure stages, all saved episodes (only 2–4 per class):

| | Base (20) | R1 (30) | R6 (20) |
|---|---|---|---|
| Success | 15% | 60% | 65% |
| Missed grasp (gripper closed on nothing) | 40% | 23% | 10% |
| Closed on the leg, not lifted | 5% | 0 | 10% |
| Lifted, never set into the hole | 35% | 17% | 15% |
| Set into the hole, not finished | 5% | 0 | 0 |

- Every R6 episode that set the leg into the hole finished screwing it. The remaining failures are precise positioning: grasping, and setting the leg into the hole. R6's three alignment failures carried the leg to 25–28 mm from the assembled position, never set it in (no twist), and drifted away.
- The base fails at the same two stages (16 of its 17 failures); RL has more than halved each.

**Orientation flips in the observation.** A part's quaternion in `parts_poses` occasionally jumps to a 180°-rotated one for a single step and back, with the position unchanged, e.g. the tabletop's (0, 0.707, −0.707, 0) → (0, 0.707, 0.707, 0).

| | Base | R1 | R6 |
|---|---|---|---|
| Flipped observations / steps | 5 / 19,667 (leg) | 36 / 26,502 (26 tabletop, 10 leg) | 5 / 16,123 (tabletop) |
| Episodes with any | 1 of 20 | 6 of 30 | 4 of 20 |
| Target step around a flip, median (all steps) (mm) | 4.7 (3.4) | 10.1 (3.4) | 21.1 (4.9) |

- The policy sees the flips and reacts: R6's target step around one is ~4× a normal step. Too rare to explain the average jitter, but a source of single jolts. The source isn't known yet (3i).
- The 1a check reads the sim's rigid-body states, not the observation, so it isn't affected (its twist distribution is tight, p10/p90 −1.20/−1.17). Offline twist from the pickles is: before skipping flipped steps, 3 of the 13 R6 successes looked like −0.26, −0.73 and −0.80 turns; with them skipped, all 13 are −1.22 to −1.40.

Rollout commands:

```bash
python -m src.eval.evaluate_model --wt-path models/ol-med-own-51-resip-2048-tw075/actor_chkpt_best_success_rate.pt \
  -f one_leg --randomness med --n-envs 20 --n-rollouts 20 --max-rollout-steps 1000 \
  --action-type pos --observation-space image --save-rollouts --save-failures \
  --save-rollouts-suffix ol-med-own-51-resip-2048_tw075_eval_vid/actor_chkpt_best_success_rate
D=data/raw/diffik/sim/one_leg/rollout/med/ol-med-own-51-resip-2048_tw075_eval_vid/actor_chkpt_best_success_rate
python scripts/visualize_data.py video $D/success --episodes 0 2 4 6 8
python scripts/visualize_data.py video $D/failure --episodes 0 1 2
python scripts/visualize_data.py plot $D/success --episodes 0 2 4 6 8
python scripts/visualize_data.py plot $D/failure --episodes 0 1 2
```

**Conclusion.** 1a works and stays on. The open problems are now P3 (growing jitter, mostly the residual's step-to-step zigzag), P2 (slowly growing offset and torque) and P4 (grasp and alignment failures). Next: 3g in R7 (§4).

### M4 · Smoothness of saved `evaluate_model` rollouts · 2026-10-30

The jerky videos were made with `src/eval/evaluate_model.py --save-rollouts` and `scripts/visualize_data.py video`, for the base as well. M4 analyzes the saved successful episodes offline (`data/raw/diffik/sim/one_leg/rollout/med/<run>_eval_vid/actor_chkpt_best_success_rate/success/*.pkl`, made 2026-10-21 to 2026-10-28): the target pose from `actions`, the end-effector pose from `robot_state` (`ee_pos`, `ee_quat`). Ad-hoc script; no repo code involved.

**Correction (2026-10-31).** The saved `actions` are deltas (target − EE pose at that step, `src/data_collection/io.py`), and the first version of M4 used them as the target. The target rows and the phase table below are recomputed with target = EE + delta; the EE rows were right. Main changes: the base's target was smoother and its chunk surge smaller than first reported, and most of the PPO policies' extra target motion is step-to-step back-and-forth at every phase, not the chunk surge (R6 results, which add R6 and a zigzag measure).

**Results** (successful episodes only; corrected):

| | Base | Clean (R1) | `-cont` (R2) | Hacking (R3) |
|---|---|---|---|---|
| Successful episodes (of saved) | 3 (of 20) | 18 (of 30) | 8 (of 10) | 16 (of 20) |
| Mean length (steps) | 889 | 806 | 436 | 355 |
| Target step (mm) | 2.51 | 3.92 | 7.87 | 11.64 |
| Target second difference (mm) | 2.05 | 4.52 | 11.38 | 17.52 |
| **EE second difference (mm)** | **0.73** | **1.30** | 2.71 | 4.11 |
| Target rotation second difference (°) | 1.36 | 1.61 | 2.44 | 4.42 |
| EE rotation second difference (°) | 0.52 | 0.58 | 0.79 | 1.38 |

Target step (mm) by step index mod 8 (0 = the first step of a new base chunk; corrected):

| Phase | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Base | 3.43 | 2.43 | 2.42 | 2.34 | 2.40 | 2.23 | 2.45 | 2.40 |
| Clean | 6.82 | 4.59 | 3.65 | 3.30 | 3.31 | 3.22 | 3.24 | 3.23 |
| Hacking | 20.36 | 14.77 | 12.31 | 10.71 | 9.98 | 8.62 | 8.30 | 8.11 |

- The jitter is real: the clean policy's end effector is 1.8× less smooth than the base's in position, the hacking policy's 5.6×. Rotation differs much less (+12% for clean).
- The clean target moves 1.6× as far per step as the base's (3.9 vs 2.5 mm), its EE only 1.3× (2.5 vs 2.0 mm): much of the extra target motion is back-and-forth that the PD drive filters.
- Each base chunk starts with a surge (base 3.4 mm vs ~2.4 mm on other steps), and the residual, which takes the base action as input, amplifies it (clean 6.8 vs ~3.3 mm). But the clean target also moves ~1.4× the base's at every other phase.
- M3's smoothness rows had hidden this, because they averaged over all running envs: failed episodes (88% of the base's) jitter in place. Caveat: only 3 successful base episodes.

**Code since M3** (not used by R6, which was already running). `residual_ppo.py`: the smoothness metrics now cover successful episodes up to the success step, and the target step is logged per chunk phase (`{charts,eval}/target_step_mm_phase{0..7}`) instead of chunk start vs other. Printed as `Success smoothness (mm/step) …`. Logging only. Tested on synthetic rollouts: a 3 mm jump at each chunk start shows up in phase 0 only, and a failed env's jitter is left out.

The code up to here was committed as **`87e5eb5`**.

### R7 · `-tw075-sm1` · 2026-10-31 · `pz60k6mc`

**Code since M4** (option 3g).

- `residual_ppo.py`: optional smoothness loss on the policy mean, `residual_smooth` (off by default).
  - Each iteration, it finds the rollout steps whose previous and next steps belong to the same episode (from `dones`).
  - Each minibatch samples 1/16 of a minibatch of them (64k triplets at the current settings) and recomputes the residual mean at the three steps. It adds the base action, which ends the residual policy's observation.
  - The loss is the mean squared second difference over position and rotation (9 of the 10 action dims; not the gripper), in residual units (normalized action / `action_scale`). It's weighted by `residual_smooth`, added to the actor loss, and logged as `losses/residual_smooth`.
  - Estimated cost: +1–3% per iteration. The update is ~7% of an iteration (R6: ~15 of ~218 s), and this adds an actor-only pass over ~19% as many samples as the main loss.
- `residual_ppo.py`: the all-env smoothness metrics logged up to R6 are back as `{charts,eval}/{target,ee,base_target,residual}_jerk_mm_allenv`, printed as `All-env smoothness (mm, as up to R6) …`, so R7's curves can be overlaid on R6's. Logging only.
- Tested with fake rollouts (ad-hoc script):
  - every sampled triplet is one env's consecutive steps, with no reset in between, and the gripper dim is dropped;
  - minimizing the loss alone makes a deliberately high-gain toy policy stop reacting to a noisy input (loss 0.47 → 0.0004);
  - a base that jumps every 8 steps is penalized even with a zero residual;
  - with the weight at 0, the loss is 0;
  - the stats block gives the same values as in M4's test, and also runs when no env succeeds.

**Why from R1, not from R6.** The same start and command as R6 except `residual_smooth`, so the difference can be attributed to 3g. It also tests preventing the jitter rather than removing it, which is what training from scratch needs. R6's jitter grew from the start (residual part 2.8 → 4.3 mm by iterations 150–199), so 150–200 iterations should show whether 3g works.

**Weight 1.0.** A rough, order-of-magnitude estimate: at R6's level of jitter, the smoothness loss's gradient on the residual mean is of the same order as the policy gradient's. It shrinks as the motion gets smoother.

```bash
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt \
  actor.residual_policy.pretrained_load_optim=false \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 num_minibatches=2 \
  total_timesteps=1000000000 checkpoint_interval=25 \
  safety.min_insert_twist_turns=0.75 residual_smooth=1.0 wandb.name=ol-med-own-51-resip-2048-tw075-sm1
```

**Expected at iteration 1** (eval, before any update): the same as R6's iteration 1, eval SR ~39% and all-env EE second difference ~1.9 mm, since the policy hasn't changed yet.

**Watch**, against R6 at the same iterations in wandb. R6 logged the all-env metrics without the `_allenv` suffix (its `eval/ee_jerk_mm` is R7's `eval/ee_jerk_mm_allenv`; R7's `eval/ee_jerk_mm` is success-only), so overlay R6's `eval/ee_jerk_mm` with R7's `eval/ee_jerk_mm_allenv`, and likewise for the residual:

| Metric (R7 / R6 key) | R6 at iterations 0–49 → 150–199 | Good sign for R7 |
|---|---|---|
| `eval/success_rate` | 38.3% → 50.0% | Close to R6; a few points lower is an acceptable price |
| `eval/ee_jerk_mm_allenv` / `eval/ee_jerk_mm` | 1.89 → 2.11 | Flat or falling |
| `eval/residual_jerk_mm_allenv` / `eval/residual_jerk_mm` | 2.78 → 4.30 | Flat or falling |
| `losses/residual_smooth` | — | Falling |
| Insertion twist, `twist fail` | −1.19 turns, ~0% | Unchanged |

Also note the success-only metrics (`eval/ee_jerk_mm`, `eval/target_step_mm_phase*`) as reference values for later runs, and `losses/approx_kl`: the extra loss also moves the mean, so early stops may come sooner.

**Adjust:**
- If eval SR stays more than ~5 points below R6's at the same iterations after ~50 iterations (average several evals; one eval varies by ±1 point), stop and retry with `residual_smooth=0.5`.
- If the jitter grows like R6's after ~100 iterations, stop and retry with `residual_smooth=2.0`.
- Otherwise, let it run to the end; it would become the best checkpoint so far.
- Afterwards, save rollouts with the R6 rollout command (R7 checkpoint, a new `--save-rollouts-suffix`) for the offline zigzag measure and the failure stages.

**Results.** Crashed at iteration 388 of 488 after ~22 h (2026-11-01 21:12, wandb state `crashed`). The log stops mid-rollout with no traceback, and the machine didn't reboot; the cause wasn't found. Iteration 1 matched the expectation: eval SR 37.6%, all-env EE second difference 1.90 mm. **3g removes the jitter, but weight 1.0 costs too much SR.** Eval SR never beat iteration 1, so `actor_chkpt_best_success_rate.pt` holds R1's weights; `actor_chkpt_375.pt` is the one that shows R7's behavior.

| Iterations | 0–49 | 50–99 | 150–199 | 300–349 | 350–386⁴ |
|---|---|---|---|---|---|
| R7 eval SR | 35.3% | 27.3% | 30.1% | 29.3% | 29.1% |
| R6 eval SR | 38.3% | 42.2% | 50.0% | 58.1% | 60.8% |
| R7 training SR | 33.3% | 29.9% | 41.2% | 49.8% | 50.4% |
| R6 training SR | 35.2% | 40.2% | 48.9% | 59.9% | 62.1% |
| R7 eval MSEL | 789 | 811 | 749 | 686 | 665 |
| R6 eval MSEL | 781 | 773 | 754 | 716 | 708 |

⁴ R6's column is 350–399.

The first adjust rule above was met by iterations 50–99 (27.3% vs 42.2%). The run kept going, and SR didn't recover.

Eval metrics; the second differences are over all running envs:

| | R7, iteration 1 | R7, 350–386 | R6, 350–399 |
|---|---|---|---|
| Target second difference (mm) | 6.61 | 1.17 | 9.44 |
| EE second difference (mm) | 1.90 | 0.37 | 2.38 |
| Base part / residual part (mm) | 5.15 / 2.62 | 6.85 / 6.63 | 5.05 / 6.18 |
| Peak target below EE in insertion, p50 / p99 (mm) | 22.1 / 37.8 | 15.1 / 22.9 | 24.4 / 41.6 |
| Peak torque ratio p50 / p99, successful episodes | 0.11 / 0.63 | 0.07 / 0.47 | 0.13 / 1.12 |
| Insertion steps | 466 | 386 | 428 |
| Insertion twist p50 (turns) | −1.19 | −1.19 | −1.19 |

- **The jitter fell far past the goal.** EE second difference fell 5× while R6's rose 1.3×. The loss includes the base, so the residual learned to cancel the base's jitter: the base and residual parts are each ~6.8 mm, and their sum is 1.2 mm.
- **Force fell too.** Offset and peak torque are roughly half of R6's at the same iterations. Twist stays at −1.19 turns, eval `twist fail` at 0.00%, and it inserts faster than R6, so this isn't forcing.
- **The mean no longer performs like the noisy policy.** Training SR reached 50.4% while eval stayed at 29.1%, a 21-point gap (R6: ~1 point at those iterations). PPO keeps improving the noisy policy, and the loss constrains only the mean. Likely, though not shown, the mean now makes too few of the small corrections alignment needs, and the noise supplies some of them.
- `losses/residual_smooth` fell from 0.063 (iterations 6–49) to 0.015 (150–199) and 0.010 (300–388), so the estimate behind weight 1.0 was too high. `losses/approx_kl` stayed far below `target_kl` (bucket means ≤ 0.02 vs 0.1).

**Saved rollouts** (`evaluate_model`, 20 episodes per checkpoint, in `…-tw075-sm1_eval_vid/actor_chkpt_375` and `…/actor_chkpt_best_success_rate`). Smoothness of successful episodes, measured as in R6:

| | Base | R1 (clean) | R6 | R7, checkpoint 375 |
|---|---|---|---|---|
| Successful episodes | 3 | 18 | 13 | 11 |
| Mean length (steps) | 889 | 806 | 702 | 694 |
| Target step / EE step (mm) | 2.51 / 1.95 | 3.92 / 2.50 | 6.13 / 3.32 | 2.87 / 2.58 |
| Target second difference (mm) | 2.05 | 4.52 | 8.57 | 1.50 |
| **EE second difference (mm)** | 0.73 | 1.30 | 2.16 | **0.64** |
| **Lag-1 autocorrelation of the target's per-step motion** | +0.61 | +0.18 | −0.21 | **+0.81** |
| Same, of the EE's | 0.91 | 0.84 | 0.72 | 0.95 |
| Target second difference before the grasp (free-space reach, mm) | 3.45 | 6.75 | 10.11 | 1.51 |
| Mean target-to-EE offset (mm) | 5.5 | 7.7 | 9.8 | 7.6 |
| Target / EE rotation second difference (°) | 1.36 / 0.52 | 1.61 / 0.58 | 2.15 / 0.73 | 0.38 / 0.17 |
| Target step, chunk start / mid-chunk (phase 4) (mm) | 3.43 / 2.40 | 6.82 / 3.31 | 11.20 / 5.19 | 3.95 / 2.72 |

Checkpoint 375 beats even the base in every second-difference and autocorrelation row, and the zigzag is gone. The videos agree: the motion is visibly smoother than in earlier runs, but SR is lower and it sometimes can't insert precisely.

Failure stages, all saved episodes:

| | R1 (30) | R1's weights, new sample (20)⁵ | R6 (20) | R7, checkpoint 375 (20) |
|---|---|---|---|---|
| Success | 60% | 45% | 65% | 55% |
| Missed grasp (gripper closed on nothing) | 23% | 15% | 10% | 15% |
| Closed on the leg, not lifted | 0 | 10% | 10% | 15% |
| Lifted, never set into the hole | 17% | 25% | 15% | 15% |
| Set into the hole, not finished | 0 | 5% | 0 | 0 |

⁵ R7's `actor_chkpt_best_success_rate.pt`.

Of checkpoint 375's 9 failures, 3 came within 25–32 mm of the assembled position and never set the leg in, which is the imprecise insertion in the videos (R6's were at 25–28 mm). One of those 3 is counted as a missed grasp above: its gripper reads closed for most of the episode, but the leg was lifted 231 mm. Another 5 never lifted the leg, and 1 lifted it but stayed far from the hole. There are too few episodes to compare shares between policies.

**Saved-rollout SR disagrees with the training eval (unresolved).** Checkpoint 375 succeeded in 11 of 20 episodes (95% interval 34–74%), against 29.2% at iteration 376, the first eval with those weights. R1's weights lean the same way: 27 of 50 across both samples (40–67%), against ~38%. R6 agrees (13 of 20, ~64%). Both evals use the policy mean and 4 DDIM steps, so neither explains it. What remains is the small sample, the env count (20 vs 2048), image vs state observations, and the wrappers. The run comparisons here all use the 2048-env training eval.

**Conclusion.** 3g is the right tool at the wrong weight. R7 also shows that smoothness and lower force can come together, with no sign of hacking. Next: settle the SR disagreement, then retry at a much lower weight (R8).

The code up to here was committed as **`36e79a1`**.

### R8 · `-tw075-sm01` · planned

**Code since R7** (wandb storage).

- `residual_ppo.py` and `bc.py` upload only the best-SR checkpoint to wandb. Every other checkpoint is still written locally. The training scripts this project doesn't use (`ppo`, `vas`, `dagger`, `residual_ppo_w_bc`, `bc_ddp`) are unchanged, so they still upload everything they save.
- `wandb.continue_run_id` prefers local files, since the others are no longer on wandb. `residual_ppo.py` loads the newest checkpoint in `models/<run>/`, and `bc.py` the newest `actor_chkpt_last.pt` under `outputs/*/*/models/<run>/`. Without them, both fall back to wandb ([BC guide](1_BC_TRAINING_GUIDE.md) §5, [training guide](2_RESIDUAL_PPO_TRAINING_GUIDE.md) §5.1).
- Side effect: image BC has no best-SR file, so it uploads no checkpoint, and `src/real/minimal.py` reads weights only from wandb, so π_real needs a manual upload before deployment ([implementation guide](IMPLEMENTATION_GUIDE.md) §6, gotcha 9).
- Found while checking this: all three BC runs logged repeated checkpoint upload errors (HTTP 400, `PROTOCOL_ERROR`), including for the best-SR file, which is probably why they have no `.pt` on wandb. The cause isn't known. So all 5.87 GB on wandb is from residual runs, and a BC checkpoint may not be on wandb even after this change.

**Check first** (a few minutes each, no training):

```bash
python -m src.eval.evaluate_model --wt-path models/ol-med-own-51-resip-2048-tw075-sm1/actor_chkpt_375.pt \
  -f one_leg --randomness med --n-envs 1024 --n-rollouts 1024 --max-rollout-steps 1000 \
  --action-type pos --observation-space state
python -m src.eval.evaluate_model --wt-path models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt \
  -f one_leg --randomness med --n-envs 1024 --n-rollouts 1024 --max-rollout-steps 1000 \
  --action-type pos --observation-space state
```

Each prints `Success rate: …`. If they read about 29% and 38%, the training eval holds, and the higher saved-rollout SR comes from the small sample or from running 20 envs with images. If they read much higher, `evaluate_model` and the training eval aren't comparable: find out why before judging R8 by SR.

**Wandb storage is still full** (§4). Numbered checkpoints no longer upload, but R8's best-SR file needs ~0.27 GB. After the first eval, check that it appears in the run's Files tab and that no numbered file does. The local copy is written whether or not the upload succeeds.

**Weight 0.1.** R7's EE jitter ended 5× below its start and below the base's, and SR had already dropped within 100 iterations, so there's room to ease off. R7's adjust rule said 0.5, but the overshoot is large enough that a 10× step tells us more. This is a judgment call, not a computed value.

```bash
python -m src.train.residual_ppo base_policy.wt_path=$BASE \
  actor.residual_policy.pretrained_wts=models/ol-med-own-51-resip-2048/actor_chkpt_best_success_rate.pt \
  actor.residual_policy.pretrained_load_optim=false \
  env.task=one_leg env.randomness=med num_env_steps=1000 num_envs=2048 num_minibatches=2 \
  total_timesteps=1000000000 checkpoint_interval=25 \
  safety.min_insert_twist_turns=0.75 residual_smooth=0.1 wandb.name=ol-med-own-51-resip-2048-tw075-sm01
```

**Expected at iteration 1:** the same as R6 and R7, eval SR ~38% and all-env EE second difference ~1.9 mm.

**Watch** (R6's keys have no `_allenv` suffix; see R7):

| Metric | R6, iterations 150–199 / 350–399 | R7, 150–199 / 350–386 | Good sign for R8 |
|---|---|---|---|
| Eval SR | 50.0% / 60.8% | 30.1% / 29.1% | Within ~5 points of R6 |
| All-env EE second difference (mm) | 2.11 / 2.38 | 0.57 / 0.37 | Below its ~1.9 start and not rising |
| All-env target second difference (mm) | 7.92 / 9.44 | 1.86 / 1.17 | Below its ~6.6 start and not rising |
| Training SR − eval SR (points) | −1.1 / +1.2 | +11.1 / +21.3 | Near 0 |

**Adjust:**
- If eval SR stays more than ~5 points below R6's over iterations 50–99 (R6: 42.2%), stop and retry with `residual_smooth=0.03`. R7's drop was clear by then (~6 h).
- If all-env EE jitter rises like R6's by iterations 150–199 (2.1 mm), stop and retry with `residual_smooth=0.3`.
- Otherwise, let it run to the end. Afterwards, save rollouts from the best checkpoint and compare them with the tables in R7. Aim for EE second difference at or below R1's 1.30 mm, positive target autocorrelation, and no more near-hole failures than R6.

**Results.** Not started.
