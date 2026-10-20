# RGB Residual on Top of π_student (Ideas, Not Implemented)

Ideas for recovering the precision that is lost when the state-based ResiP teacher is distilled into the RGB student policy. Nothing here is implemented yet. Code references are to the current repo and say what could be reused.

Paper references (§, Fig., Tab., App.) are to `docs/2025 Ankile ResiP From Imitation to Refinement - Residual RL for Precise Assembly.pdf`.

---

## 1. The problem

The paper's pipeline:

1. BC: train a chunked diffusion policy π_base on ~50 sim teleop demos (state input).
2. RL: train a per-timestep residual π_res with PPO on top of the frozen π_base, from privileged state (part poses). Teacher = π_base + π_res.
3. Distill: roll out the teacher, re-render the rollouts as RGB with visual randomization, and train π_student (an RGB diffusion policy) on ~400 synthetic + 10–40 real demos.
4. Deploy π_student alone on the real robot.

The teacher only produces data. It can't run on the real robot because it needs exact part poses.

**The distillation gap.** On one leg the teacher reaches 98%, but distilled students plateau at ~73% (RGB, ~1k trajectories) and ~78–80% (state, 10k–100k trajectories) (App. IV-A, Fig. 1 right). The student is again a chunked, offline-trained diffusion policy, which brings back the open-loop chunk execution and offline distribution shift that the residual fixed. The paper rules out the image modality as the main cause, since state and RGB students score about the same. It names online distillation as future work (App. VII-a).

**The idea.** Add a residual that runs every timestep on top of the frozen π_student at deployment:

```
a_t = a_student,t + α · π_res_rgb(o_t)        # α = 0.1, as in Tab. IX
```

**Why not just run residual PPO from pixels:**

- **Cost.** ResiP needed ~75M environment steps with rendering off at ~4000 steps/s (App. II-C, X-A). With cameras on, throughput drops a lot, and pixel RL usually needs more samples than state RL.
- **Precision from pixels.** The corrections are millimetre-scale alignment fixes during insertion, often while the gripper hides the contact.
- **Sim-to-real for the residual.** In the current pipeline nothing RL-trained runs on the real robot, and real demos anchor the student during co-training. A residual trained only in sim carries both the rendering gap and the contact-dynamics gap.
- **Base mismatch.** The residual learns to fix the student's *sim* errors. The real student, co-trained on real data, fails in different ways.

The four approaches below address these problems. Approaches 1 and 2 are training methods. Approaches 3 and 4 are design choices that apply to both.

---

## 2. Shared setup

- **Deploy-time policy:** frozen chunked π_student, plus a small MLP residual evaluated every step.
- **Residual input (default):** `o_t = [student vision features (front + wrist), proprio, a_student,t]`. This has the same structure as the current residual input `[nobs, base_naction]` in [residual_ppo.py:357](../src/train/residual_ppo.py#L357), with image features in place of part poses.
- **Reuse the student's frozen vision encoder** (ResNet18 + R3M weights, 128-d projection per camera, Tab. X), for three reasons:
  - It was co-trained on real images, so it already absorbs part of the visual sim-to-real gap.
  - The residual doesn't have to learn vision from scratch.
  - Rollout buffers can store ~1 KB of features per step instead of ~300 KB of images.

---

## 3. Approach 1: DAgger-distill the residual from the state-based teacher (recommended first)

**Idea.** The state-based teacher already knows the right correction in every state. Supervise the RGB residual with the teacher on the states that *student + RGB residual* actually visit. This is supervised learning with a label at every step, so there's no exploration or credit-assignment problem. App. IX-B shows DAgger is very sample-efficient on these tasks: after ~20k gradient steps it beats BC distillation runs that used >10k rollouts and 500k gradient steps.

**Loop:**

```
for iter:
    roll out N envs; at each step:
        obs   = {images, proprio, parts_poses}     # parts_poses used only by the teacher
        a_s   = student.action(obs)
        Δ_rgb = π_res_rgb(features(obs), proprio, a_s)
        label = teacher_correction(obs, a_s)       # see "Labels" below
        execute: teacher action with prob. β, else a_s + α·Δ_rgb
        store (features, proprio, a_s, label)
    aggregate into dataset D; train π_res_rgb with MSE on D
    decay β
```

**Already in the repo:**

| Need | Where |
|---|---|
| Image-mode env that still returns `parts_poses` for the teacher | [dagger.py:102-121](../src/train/dagger.py#L102-L121) (`observation_type: image`) |
| β mixing of teacher/student actions and β decay schedule | [dagger.py:307-369](../src/train/dagger.py#L307-L369), config in [base_dagger.yaml](../src/config/base_dagger.yaml) |
| Teacher correcting a given student action | `correct_student_action_only` flag ([dagger.py:352](../src/train/dagger.py#L352)) → [`ResidualDiffusionPolicy.correct_action`](../src/behavior/residual_diffusion.py#L232) |
| MSE loss for a scaled residual | [`ResidualPolicy.bc_loss`](../src/models/residual.py#L164) |

**Changes needed:**

- Train a residual MLP on top of a frozen student. Currently dagger.py retrains the whole student diffusion policy.
- **Keep failed trajectories.** dagger.py stores only successful envs ([dagger.py:415](../src/train/dagger.py#L415)). For a residual, the states just before failures are the most useful data.

**Labels:**

| | (a) Teacher's residual applied to the student's action | (b) Full teacher action − student action |
|---|---|---|
| Label | `teacher.correct_action(obs, a_s) − a_s` | `teacher.action(obs) − a_s` |
| Pro | Always small and local, matches the α = 0.1 scale | Uses the teacher's full behavior |
| Con | The teacher's residual was trained on its own base's actions, so `a_s` is slightly out of its distribution | Large whenever the student is in a different mode or chunk phase, so it stops being a local correction |

Start with (a). Clip labels to a small range, or drop samples where the label is large.

**Limits:**

- **Capped at the teacher** (~98% on one leg).
- **Privileged-information gap.** The teacher corrects using exact part poses. When the peg is hidden by the gripper, the RGB residual can't see what the teacher saw and learns an averaged correction. Approach 2 addresses this.

---

## 4. Approach 2: residual PPO with an asymmetric actor-critic

**Idea.** Only the actor runs on the real robot; the critic is only used in training. So give the critic privileged state and the actor RGB features:

- **Actor:** `π_res_rgb([image features, proprio, a_s])`, the same network as approach 1.
- **Critic:** `V([robot_state, parts_poses, a_s])`, the current residual PPO critic input.

With a sparse reward over ~700 steps, the hard part of PPO is the value estimate. A privileged critic makes that as easy as in the original ResiP, and the RGB actor follows those advantages. This is the standard way to make pixel PPO workable in sim.

**Code changes in [residual_ppo.py](../src/train/residual_ppo.py):**

- [`ResidualPolicy`](../src/models/residual.py#L67) currently feeds the same `nobs` to `actor_mean` and `critic`. Split it into `actor_obs` and `critic_obs`, and store both in the rollout buffer.
- `get_value` and the GAE bootstrap ([residual_ppo.py:474](../src/train/residual_ppo.py#L474)) use `critic_obs`. `get_action_and_value` takes both.
- Swap the base from the state diffusion policy to the vision student, with the env in image mode as in dagger.py.

**Practical issues:**

- **Memory.** Two 224×224×3 uint8 images are ~300 KB/step, so 1024 envs × 700 steps ≈ 216 GB per rollout. Two 128-d feature vectors are ~1 KB/step ≈ 0.7 GB. Keep the encoder frozen and store features. To fine-tune the encoder, cut env count or rollout length.
- **Throughput.** Rendering will cut the ~4000 steps/s a lot. Measure it before committing: ~75M steps may take days.
- **Visual randomization during RL.** Randomize colors, lighting and camera pose per episode. The actor now learns from rendered pixels directly, not from data re-rendered afterwards.
- **Warm start:**
  - Initialize the actor from approach 1.
  - Use a low initial log-std.
  - Warm up the critic for a few iterations with the actor frozen, so a random critic doesn't wreck the good initial actor.
- **Stay near the DAgger solution.** Keep α = 0.1 (approach 4). Optionally add a small MSE/KL penalty toward the approach-1 residual so PPO doesn't drift into sim-only tricks.

**What it adds over approach 1:**

- It optimizes success directly, so it can go past the teacher.
- It learns corrections from what the cameras can actually see, which fixes the privileged-information gap.

**Cost:** much more compute, and more risk of overfitting to sim contact physics.

---

## 5. Approach 3: use a smaller residual input

Two full RGB streams are the hardest input to learn from and to transfer. The base policy already handles the global motion; the residual only needs to see enough for local alignment. Smaller inputs to consider:

| Input | Pro | Con |
|---|---|---|
| Student encoder features, both cameras (default, §2) | No new model; encoder already co-trained on real images | Features were trained for the base's job, which may not encode mm-level alignment |
| Wrist camera + proprio only | Close-up view of the gripper/part contact; less background, so less to randomize | Loses global context (fine if the base handles it); wrist view can be blocked at the moment of contact |
| Proprio + estimated part poses (from a separate pose estimator) | Same input format as the existing state residual, which could then be reused or fine-tuned | Needs a pose estimator accurate to a few mm in the real world, under occlusion; the residual must be trained with realistic pose noise |
| Proprio + task keypoints (e.g. vector from leg tip to hole) | Very low-dimensional; directly encodes what the correction depends on | Task-specific; needs a keypoint detector and a definition per task |

Notes:

- With pose or keypoint inputs, add noise during training that matches the estimator's real error. Otherwise the residual will trust the inputs too much.
- Proprio alone (no vision) is a useful ablation. If it recovers part of the gap, the residual is mainly fixing chunk open-loop errors rather than perception errors.

---

## 6. Approach 4: keep the residual output small

Keep a small residual action scale, like α = 0.1 in Tab. IX ([residual.py:80](../src/models/residual.py#L80), applied at [residual_ppo.py:368](../src/train/residual_ppo.py#L368)). Note that α multiplies the residual in **normalized** action space. To know what 0.1 means in metres or radians, convert it with the action normalizer.

Why it matters:

- **Locality.** The residual can only nudge the base action. It can't take over the trajectory, so the base's long-horizon behavior is kept. The paper credits this for ResiP's stable training (§IV-B.1).
- **Bounded harm on the real robot.** If the residual transfers badly, the worst case is a small offset around the student's action, not an unsafe motion.
- **Consistent labels.** In approach 1, label option (a) matches this scale by construction. Label option (b) can exceed it, which is a reason to clip.

Deployment tips:

- **Ramp α on the real robot.** α can be lowered at deploy time without retraining, because a smaller α just applies part of the correction. Start at 0 (student only), then 0.05, then 0.1, comparing success at each step.
- **Clip the residual norm** per step, in addition to α, as a safety limit.
- **Log residual magnitudes** in sim and real. If real-world residuals are much larger than in sim, the residual is seeing out-of-distribution inputs.

---

## 7. Suggested order and checks

1. **Approach 1 in sim**, with the default input (§2) and α = 0.1. Compare student alone vs. student + RGB residual on the same eval seeds. The goal is to close most of the 73% → 98% gap. If it doesn't, find out why before going further: label choice, input features, or the occlusion/privileged-information gap.
2. **Ablate the input** (approach 3): proprio only, wrist only, both cameras.
3. **Approach 2**, warm-started from approach 1, only if the remaining failures look like occlusion or privileged-information failures.
4. **Real robot**, with the α ramp and norm clip from approach 4. Test on the same Part/Obs grids as Tab. II so results are comparable with the paper.

If it doesn't work in sim, it won't work in the real world.
