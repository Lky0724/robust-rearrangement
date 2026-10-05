# ResiP Data Guide

How the data for *From Imitation to Refinement – Residual RL for Precise Assembly* (Ankile et al.) is stored, where it lives, and how each stage of the pipeline reads it. Companion to [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

---

## 1. The essentials

- **Two formats, same trajectories.** A `.pkl` holds **one episode**, as recorded. A `.zarr` holds **a whole dataset**: many episodes concatenated and converted into training form. Training reads only zarrs.
- **Paths are the metadata.** Every file sits at `{raw|processed}/<controller>/<domain>/<task>/<source>/<randomness>/<outcome>`. Training configs choose data by these fields, not by file path.
- **Source.** Everything comes from the public S3 bucket `iai-robust-rearrangement` (us-east-2, no credentials needed; browse at `https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/index.html`). The bucket is about 2.1 TB in 2.45 M objects, so download only the folders you need (§8).
- **Environment variables.** The code resolves paths through `DATA_DIR_PROCESSED` and `DATA_DIR_RAW`, and fails if they are unset:
  ```bash
  export DATA_DIR_PROCESSED=/home/kaiyuan.luo/ResiP/data
  export DATA_DIR_RAW=/home/kaiyuan.luo/ResiP/data
  ```

---

## 2. Path grammar

### 2.1 Complete bucket tree

Built from a full listing of the bucket (2,451,550 objects, ~2.1 TB) taken on 2026-09-23. A `.zarr` counts as one entry; its internal chunk files aren't listed. For pickle folders, the count and total size are shown instead of file names. Pickles of about 100 MB or more contain camera images; the small ones are state-only.

```
iai-robust-rearrangement/
├── index.html                                    bucket browser page
├── packages/IsaacGym_Preview_4_Package.tar.gz    201 MB · IsaacGym installer
├── weights/                                      empty
├── videos/                                       1.4 GB · website media
│   ├── uncut_rollouts/
│   │   ├── one_leg_low/        uncut_1k_bc.mp4, uncut_1k_rppo.mp4
│   │   └── round_table_med/    uncut_1k_rt_med_bc.mp4, uncut_1k_rt_med_rppo.mp4
│   └── website/                1_main_video … 9_outro_video (.m4v / .png)
│       └── real/               3_40_real, 3_40_real_350_sim, 5_black_40_real_explanation,
│                               6_black_40_real_400_sim, 7_lamp_40_real_400_sim_1x_{1,2} (.mp4)
├── checkpoints/                                  3.9 GB · 16 × actor_chkpt.pt
│   ├── bc/                                       π_base
│   │   ├── one_leg/{low,med}/actor_chkpt.pt
│   │   ├── round_table/{low,med}/actor_chkpt.pt
│   │   ├── lamp/{low,med}/actor_chkpt.pt
│   │   ├── mug_rack/low/actor_chkpt.pt
│   │   └── factory_peg_hole/low/actor_chkpt.pt
│   └── rppo/                                     π_base + π_res; same 8 task/randomness folders as bc/
└── data/
    ├── processed/diffik/                         ── training-ready zarrs ──
    │   ├── sim/                                  223 GB
    │   │   ├── one_leg/
    │   │   │   ├── teleop/{low,med,high}/success.zarr                          2.2 / 2.3 / 2.8 GB · 50 eps each
    │   │   │   └── rollout/
    │   │   │       ├── low/success.zarr                                         6.2 GB · 150 eps
    │   │   │       ├── low/success/bc_low_{000,250,500,750}.zarr                52.0 GB · 4 × 250 files
    │   │   │       ├── low/success/rppo_low_{000,250,500,750}.zarr              44.1 GB · 4 × 250 eps
    │   │   │       ├── low/success/rppo_10_demos.zarr                           2.2 GB
    │   │   │       └── med/success.zarr                                         10.7 GB · 200 eps
    │   │   ├── one_leg_demo_scaling/teleop/low/success/demo_scaling.zarr        3.7 GB · 90 eps
    │   │   ├── one_leg_state_distill/rollout/low/success/
    │   │   │   └── rl_state_{0,1000,…,9000}.zarr                                5.9 GB · 10 × 1000 eps
    │   │   ├── one_leg_state_distill_large/rollout/low/success/
    │   │   │   └── rl_state_{0,1000,…,99000}.zarr                               58.5 GB · 100 × 1000 eps
    │   │   ├── round_table/
    │   │   │   ├── teleop/{low,med}/success.zarr                                3.2 / 3.2 GB
    │   │   │   └── rollout/{low,med}/success.zarr                               7.0 / 0.1 GB · med 100 eps
    │   │   ├── lamp/
    │   │   │   ├── teleop/{low,med}/success.zarr                                3.2 / 3.2 GB
    │   │   │   └── rollout/{low,med}/success.zarr                               8.3 / 0.1 GB
    │   │   ├── mug_rack/teleop/low/success.zarr                                 1.6 GB
    │   │   ├── mug_rack_handle/teleop/low/success.zarr                          2.0 GB
    │   │   └── factory_peg_hole/teleop/low/success.zarr                         0.9 GB
    │   ├── real/                                 37 GB
    │   │   ├── one_leg_full/teleop/low/
    │   │   │   ├── success.zarr                                                 10.3 GB · 50 eps
    │   │   │   └── one_leg_low_real.zarr.zip                                    7.4 GB · zipped zarr (contents not checked)
    │   │   ├── one_leg_full_new/teleop/low/success.zarr                         7.3 GB · 40 eps
    │   │   ├── lamp/teleop/low/success.zarr                                     9.2 GB
    │   │   └── mug_rack_handle/teleop/low/success.zarr                          3.3 GB
    │   ├── sim2real/                             136 GB · re-rendered with realistic images
    │   │   ├── one_leg_render_rppo_1/rollout/med/success.zarr                   35.0 GB · 345 eps
    │   │   ├── one_leg_render_rppo_brighter/rollout/med/success.zarr            30.9 GB
    │   │   ├── one_leg_render_rppo_black/rollout/med/success.zarr               9.7 GB
    │   │   ├── one_leg_render_demos_brighter/teleop/{med,med_perturb}/success.zarr   2.3 / 2.3 GB · med 25 eps
    │   │   ├── one_leg_render_demos_black/teleop/{med,med_perturb}/success.zarr      2.2 / 2.1 GB
    │   │   ├── lamp_render_rppo/rollout/{low,med}/success.zarr                  14.6 / 14.4 GB
    │   │   └── mug_rack_gym_rerender/rollout/low/success/
    │   │       └── mug_rack_rerender_{0,250,500,750}.zarr                       22.2 GB
    │   └── distillation/                         187 GB · 100 zarrs per task, ~1000 RL successes each
    │       ├── one_leg/rollout/low/success/rl_state_{0,1000,…,99000}.zarr       58.5 GB
    │       ├── round_table/rollout/low/success/rppo_{0…99}.zarr                 69.6 GB
    │       ├── lamp/rollout/low/success/rppo_{0…99}.zarr                        47.5 GB
    │       └── factory_peg_hole/rollout/low/success/rppo_{0…99}.zarr            10.9 GB
    ├── raw/diffik/                               ── one .pkl per episode ──
    │   ├── sim/                                  1.07 TB
    │   │   ├── one_leg/rollout/
    │   │   │   ├── low/success/                                   186 pkl · 37.5 GB
    │   │   │   ├── low/bc_unet/success/                          1001 pkl · 256 GB
    │   │   │   ├── low/rppo_low/success/                         1000 pkl · 214 GB
    │   │   │   ├── low/rppo_10_demos/success/                      57 pkl · 12.6 GB
    │   │   │   ├── low/side_by_side_bc/{success,failure}/     12 + 20 pkl · 9.5 GB
    │   │   │   ├── low/side_by_side_rl/{success,failure}/      31 + 1 pkl · 6.6 GB
    │   │   │   ├── med/bc_unet/success/                          1061 pkl · 277 GB
    │   │   │   ├── med/rppo_1/success/                            506 pkl · 130 GB
    │   │   │   └── med/rppo_med/success/                           93 pkl · 23.1 GB
    │   │   ├── lamp/
    │   │   │   ├── teleop/{low,med}/success/                  50 + 50 pkl · 27.1 GB
    │   │   │   ├── rollout/low/success/                           104 pkl · 36.3 GB
    │   │   │   ├── rollout/low/rppo/success/                   89,146 pkl · 28.1 GB
    │   │   │   └── rollout/med/success/                           134 pkl · 0.1 GB
    │   │   ├── mug_rack/teleop/low/success/                        50 pkl · 4.3 GB
    │   │   └── factory_peg_hole/teleop/low/success/                52 pkl · 3.9 GB
    │   ├── real/                                 260 GB
    │   │   ├── one_leg_full/teleop/low/success/                    50 pkl · 51.1 GB
    │   │   ├── one_leg_full_new/teleop/low/success/                40 pkl · 37.8 GB
    │   │   ├── one_leg_full_new_backup/2024-08-11T18:52:10.pkl      1 pkl · 0.9 GB
    │   │   ├── one_leg_simple/teleop/low/success/                  50 pkl · 28.4 GB
    │   │   ├── one_leg_insert/teleop/low/success/                 125 pkl · 59.6 GB
    │   │   ├── one_leg_corner_insert/teleop/{low,med}/success/ 8 + 15 pkl · 16.3 GB
    │   │   ├── one_leg_color/teleop/low/success/                    2 pkl · 1.6 GB
    │   │   ├── lamp/teleop/low/success/                            40 pkl · 49.8 GB
    │   │   ├── pick_cup/teleop/low/success/                        55 pkl · 6.5 GB
    │   │   └── place_shade/teleop/low/success/                     20 pkl · 7.8 GB
    │   └── distillation/                         64 GB
    │       ├── one_leg/rollout/low/success/                    10,336 pkl · 4.3 GB
    │       ├── round_table/rollout/low/success/rppo/success/  111,908 pkl · 51.3 GB
    │       └── factory_peg_hole/rollout/low/rppo/success/     122,980 pkl · 8.7 GB
    ├── coverage/                                 58.7 GB · rollouts saved while evaluating specific WandB runs
    │   ├── ol-state-dr-low-1/6i7hupje/raw/diffik/sim/one_leg/rollout/low/success/   34,550 pkl
    │   ├── ol-state-dr-med-1/9zjnzg4r/raw/diffik/sim/one_leg/rollout/med/
    │   │   ├── success/                                                              22,209 pkl
    │   │   └── failure/                                                               4,075 pkl
    │   ├── ol-state-dr-1/r9wm1uo6/raw/diffik/sim/one_leg/rollout/med/{success,failure}/   230 + 794 pkl
    │   ├── ol-rppo-dr-med-1/xeeg8wsc/raw/diffik/sim/one_leg/rollout/med/success/     56,595 pkl
    │   └── residual-ppo-dr-1/
    │       ├── 7mv6o4i9/raw/diffik/sim/one_leg/rollout/med/{success,failure}/   1,080 + 1,224 pkl
    │       └── fj7ggmg7/raw/diffik/sim/one_leg/rollout/med/{success,failure}/     537 + 487 pkl
    └── distill/                                  51.1 GB
        └── ol-rppo-dr-low-1/kzlx4y3f/raw/diffik/sim/one_leg/rollout/low/success/   124,374 pkl
```

What the tree shows:
- **Not every processed zarr has its raw pickles on S3.** There are no raw pickles for the `sim` teleop demos of `one_leg` or `round_table`, or for any `sim2real` render. Treat those zarrs as the only copy.
- **A few folders break the naming pattern.** `raw/distillation/round_table/…/success/rppo/success/` has `success` twice, and `raw/distillation/factory_peg_hole/…/low/rppo/success/` uses `rppo` as a suffix folder. `coverage/` and `distill/` prefix the normal `raw/…` path with `<wandb project>/<run id>/`.

### 2.2 Reading a path

#### Path shapes

Every path in the bucket has one of these six shapes:

| # | Shape | Example |
|---|---|---|
| A | `processed/<ctrl>/<domain>/<task>/<source>/<rand>/<outcome>.zarr` | `processed/diffik/sim/one_leg/teleop/low/success.zarr` |
| B | `processed/<ctrl>/<domain>/<task>/<source>/<rand>/<outcome>/<suffix>.zarr` | `processed/diffik/sim/one_leg/rollout/low/success/rppo_low_000.zarr` |
| C | `raw/<ctrl>/<domain>/<task>/<source>/<rand>/<outcome>/<timestamp>.pkl` | `raw/diffik/real/one_leg_full/teleop/low/success/2024-08-11T18:52:10.pkl` |
| D | `raw/<ctrl>/<domain>/<task>/<source>/<rand>/<suffix>/<outcome>/<timestamp>.pkl` | `raw/diffik/sim/one_leg/rollout/low/rppo_low/success/….pkl` |
| E | `{coverage,distill}/<wandb project>/<run id>/raw/…` followed by shape C | `coverage/ol-state-dr-med-1/9zjnzg4r/raw/diffik/sim/one_leg/rollout/med/failure/….pkl` |
| F | `checkpoints/<bc\|rppo>/<task>/<rand>/actor_chkpt.pt` | `checkpoints/rppo/one_leg/med/actor_chkpt.pt` |

All shapes except F are under `data/`. **The suffix sits in a different place in each tree:** after the outcome in processed paths (it becomes the zarr's name), before the outcome in raw paths. Three folders on S3 fit none of the shapes: `raw/diffik/distillation/round_table/rollout/low/success/rppo/success/`, `raw/diffik/distillation/factory_peg_hole/rollout/low/rppo/success/` (suffix but no outcome level before it) and `raw/diffik/real/one_leg_full_new_backup/` (a single pickle directly under the task).

#### Levels

| Level | Values on S3 | Meaning | Set by (training config / `process_pickles` flag) |
|---|---|---|---|
| stage | `raw`, `processed` | One `.pkl` per episode vs. a training-ready `.zarr` (§4) | Root from `$DATA_DIR_RAW` / `$DATA_DIR_PROCESSED` |
| controller | `diffik` | Low-level controller the actions were recorded with (differential IK). The code also accepts `osc`, which isn't on S3 | `control.controller` / `-c` |
| domain | `sim`, `real`, `sim2real`, `distillation` | `sim`: IsaacGym. `real`: Franka robot. `sim2real`: sim trajectories re-rendered in Isaac Sim. `distillation`: large RL rollout sets for scaling runs | `environment` / `-d` (accepts `sim`, `real`, `distillation`; `sim2real` can't be produced by the current script) |
| task | See the task-name table below | Task, optionally with a variant label | `task` / `-f` |
| source | `teleop`, `rollout` | `teleop`: human with a SpaceMouse. `rollout`: a trained policy. The code also accepts `scripted` and `augmentation`, which aren't on S3 | `demo_source` / `-s` |
| randomness | `low`, `med`, `high`, `med_perturb` | Initial-state spread; `_perturb` = recorded with random force perturbations. See the randomness table below | `randomness` / `-r` |
| outcome | `success`, `failure` | Whether the episode completed the task. Written automatically when saving ([io.py:78](src/data_collection/io.py#L78)). The code also accepts `partial_success` | `demo_outcome` (default `success`) / `-o` |
| suffix | e.g. `bc_unet`, `rppo_low`, `rppo_1`, `rppo_med`, `rppo_10_demos`, `side_by_side_{bc,rl}`, `rppo` (raw); `bc_low_000`, `rppo_low_250`, `rl_state_5000`, `rppo_42`, `demo_scaling`, `mug_rack_rerender_0` (zarr names) | A sub-dataset within one config combination. A trailing number is usually a start index: the first raw file (`--offset`) or first episode in that zarr | `suffix` / `--suffix` (raw folder), `--output-suffix` (zarr name) |
| file | `<timestamp>.pkl`, `success.zarr`, `<suffix>.zarr` | Pickles are named by save time: `2024-08-11T18:52:10.pkl` (teleop) or `2024-06-05T02:06:52.361045.pkl` (rollouts, with microseconds) | – |

The zarr's `.zattrs` repeats most of these levels. Its `domain` is only ever `sim` or `real`: `process_pickles` writes `real` for real data and `sim` for everything else, including distillation. So `sim2real` and `distillation` episodes count as sim (0) in co-training.

#### Task names

The task level is a label: only the base task corresponds to a simulator environment, and variants reuse it.

| Base task | Paper name | Variants on S3 | What a variant means |
|---|---|---|---|
| `one_leg` | one_leg | `one_leg_full`, `one_leg_full_new` (real) | Full task; 50 and 40 real demos |
| | | `one_leg_simple`, `one_leg_insert`, `one_leg_corner_insert`, `one_leg_color` (real, raw only) | Earlier or partial real versions of the task (inferred from names and slurm scripts) |
| | | `one_leg_demo_scaling` | 90 sim teleop demos for demo-count scaling |
| | | `one_leg_state_distill`, `one_leg_state_distill_large` | 10k / 100k RL rollouts for state-distillation scaling |
| | | `one_leg_render_rppo_{1,black,brighter}`, `one_leg_render_demos_{black,brighter}` | Re-rendered RL rollouts / teleop demos; suffix = render variant |
| `round_table` | round_table | – | |
| `lamp` | lamp | `lamp_render_rppo` | Re-rendered RL rollouts |
| `mug_rack` | mug-rack | `mug_rack_handle`, `mug_rack_gym_rerender` | Mug-rack variant, presumably grasping the handle (inferred from the name); IsaacGym re-render |
| `factory_peg_hole` | peg-in-hole | – | |
| `pick_cup`, `place_shade` | – | real only, raw only | Other real-robot tasks not in the paper |

#### Randomness levels

The same level name means different things in the two simulator environments:
- **RL env** (`FurnitureRLSimEnv`, [furniture_rl_sim_env.py:1692](../furniture-bench/furniture_bench/envs/furniture_rl_sim_env.py#L1692)) puts the parts at their nominal poses and pushes each one with a random horizontal force and twist at reset. It also offsets the fixture and adds noise to the arm's starting joints. It is used for **RL training, evaluation, rollouts and SpaceMouse demo collection**: [data_collector_sm.py](../src/data_collection/data_collector_sm.py) has created this env since 2024-05-15, and the released `one_leg` teleop demos were recorded on 2024-05-16 to 05-28, so they very likely share evaluation's initial-state distribution.
- **Original FurnitureBench env** (`FurnitureSimEnv`, [furniture_sim_env.py:1349](../furniture-bench/furniture_bench/envs/furniture_sim_env.py#L1349)) places each part uniformly at random around its nominal pose. It is used by `furniture_bench.scripts.run_sim_env` and the older scripted collector ([data_collector.py](../src/data_collection/data_collector.py)).

| Level | Original env: part position / yaw | RL env: fixture offset | RL env: max push force / torque | RL env: joint noise | In the paper |
|---|---|---|---|---|---|
| `low` | ±1.5 cm / ±15° | ±2 cm | 0.2 / 0.007 | ±5° | Yes |
| `med` | ±5 cm / ±45° | ±4 cm | 0.5 / 0.01 | ±10° | Yes |
| `high` | Predefined high-randomness poses | ±6 cm | 0.75 / 0.015 | ±13° | No |
| `<level>_perturb` | as `<level>`, plus random pushes during the demo | – | – | – | – |

Push forces are scaled per part (for `one_leg`, ×25 for the heavier tabletop, ×1 for the legs), so the resulting displacement has no fixed range. The paper describes the ranges as ±1.5 cm (low) and ±5 cm (med), with fixture offsets of ±2 / ±4 cm (§III-C).

`_perturb` folders hold teleop demos collected with `--sample-perturbations`: random forces are applied to parts while the operator acts ([teleop.py:81](src/data_collection/teleop.py#L81), [data_collector_sm.py:483](src/data_collection/data_collector_sm.py#L483)). On S3 the level only appears as `sim2real/one_leg_render_demos_*/teleop/med_perturb`. The `state/scaling/one_leg/low/*` configs also reference `sim/one_leg/teleop/low_perturb`, which isn't in the bucket.

#### Who writes which path

| Producer | Writes to |
|---|---|
| Sim teleop ([teleop.py](src/data_collection/teleop.py)) | `raw/<ctrl>/sim/<task>/teleop/<rand>[_perturb]/{success,failure}/` |
| Rollouts (`evaluate_model --save-rollouts [--save-rollouts-suffix S]`) | `raw/diffik/sim/<task>/rollout/<rand>/[S/]{success,failure}/` |
| Real teleop ([teleop_sm.py](src/real/teleop_sm.py)) and re-rendering ([isaac_lab_rerender.py](src/sim2real/isaac_lab_rerender.py)) | Whatever `--save_dir` / `--save-dir` you give. Point it at the matching `raw/…/success/` folder so processing can find it |
| `process_pickles -c C -d D -f F -s S -r R -o O [--suffix X] [--output-suffix Y]` | Reads `raw/C/D/F/S/R/[X/]O/*.pkl*`; writes `processed/C/D/F/S/R/O[/Y].zarr` |

#### How training selects paths

[get_processed_paths](src/common/files.py) builds one glob per combination of config values:
- A **list** value, e.g. `environment='[real,sim]'`, expands into one pattern per element.
- An **unset** (`null`) level becomes `**`, matching any depth.
- `suffix` is appended after the outcome, matching shape B.

Every zarr that matches is loaded. To load specific files instead, set `data.data_paths_override` to paths relative to `$DATA_DIR_PROCESSED/processed/`; this is how the scaling configs pick suffixed zarrs. When *writing*, [get_processed_path](src/common/files.py) joins a list value into one folder name, sorted and joined with `-` (e.g. `task=[a,b]` → `a-b/`).

Path helpers: [src/common/files.py](src/common/files.py) (`get_processed_path(s)`, `get_raw_paths`, `path_override`).

---

## 3. What is in the bucket

### Top level

| Branch | Contents | Used for |
|---|---|---|
| `checkpoints/bc/<task>/<low\|med>/actor_chkpt.pt` | π_base (diffusion policy weights + normalizer) | Base for residual RL; baseline evaluation |
| `checkpoints/rppo/<task>/<low\|med>/actor_chkpt.pt` | π_base + π_res | Evaluation; generating expert rollouts |
| `data/processed/` | Training-ready zarrs | `train/bc.py` |
| `data/raw/` | The per-episode pickles the zarrs were built from | Reprocessing only |
| `data/coverage/<wandb project>/<run id>/raw/…` | Large pools of saved rollouts from specific BC (`ol-state-dr-*`) and RL (`ol-rppo-*`, `residual-ppo-*`) runs; some include failures | State-coverage analysis |
| `data/distill/ol-rppo-dr-low-1/…` | ~124k raw successful rollouts of the `one_leg` low RL policy | Source of the large distillation zarrs |
| `packages/` | `IsaacGym_Preview_4_Package.tar.gz` | Installing the simulator (Stage 0 of the implementation guide) |
| `videos/` | Website videos and uncut 1k-rollout comparisons (BC vs. ResiP) | Reference only |
| `weights/`, `index.html` | Empty folder; bucket browser page | – |

Checkpoints exist for `one_leg`, `round_table` and `lamp` (low and med), and for `mug_rack` and `factory_peg_hole` (low only).

### `data/processed/diffik/`

**`sim/`**: simulation data (IsaacGym images, or 2×2 placeholders in state-only rollouts)

| Path | Meaning |
|---|---|
| `<task>/teleop/{low,med}` | **D_sim**: the 50 human demos per task that π_base is trained on. `one_leg` also has `high`, which the paper doesn't use |
| `<task>/rollout/{low,med}/success.zarr` | Successful rollouts of a trained policy (state only) |
| `one_leg/rollout/low/success/bc_low_{000,250,500,750}.zarr` | BC policy rollouts in batches of 250 files; the number is the starting file index |
| `one_leg/rollout/low/success/rppo_low_{000,…,750}.zarr` | The same from the residual RL policy (250 eps each). Used for the BC-vs-RL data-scaling runs |
| `one_leg/rollout/low/success/rppo_10_demos.zarr` | Rollouts of an RL policy whose base was trained on only 10 demos |
| `one_leg_demo_scaling/…/demo_scaling.zarr` | 90 teleop demos for the demo-count scaling runs |
| `one_leg_state_distill/…/rl_state_{0…9000}.zarr` | 10 × 1000 RL rollouts (10k in total). The number is the starting episode index |
| `one_leg_state_distill_large/…/rl_state_{0…99000}.zarr` | 100 × 1000 RL rollouts (100k): the 100k point in Fig. 1 (right) |
| `mug_rack_handle`, `lamp`, `round_table`, `factory_peg_hole` | Other tasks |

**`real/`**: real-robot teleop demos (**D_real**)

| Path | Meaning |
|---|---|
| `one_leg_full/teleop/low` | 50 demos of the full task. The folder also holds `one_leg_low_real.zarr.zip` (7.4 GB), a zipped zarr whose contents haven't been checked |
| `one_leg_full_new/teleop/low` | 40 demos: the paper's "40 Real" |
| `lamp`, `mug_rack_handle` | Other tasks |

`raw/diffik/real/` also holds unprocessed sub-task sets: `one_leg_insert`, `one_leg_corner_insert`, `one_leg_simple`, `one_leg_color`, `pick_cup` and `place_shade`. These are earlier or partial versions of the task.

**`sim2real/`**: sim trajectories re-rendered in Isaac Sim with realistic images (**D_synth-render**)

| Path | Meaning |
|---|---|
| `one_leg_render_rppo_{1,black,brighter}/rollout/med` | Re-rendered RL rollouts. The suffix is the render variant: `brighter` (lighting), `black` (black parts, Fig. 11) or `1` (the first batch) |
| `one_leg_render_demos_{black,brighter}/teleop/{med,med_perturb}` | The teleop demos re-rendered the same way |
| `lamp_render_rppo`, `mug_rack_gym_rerender` | Other tasks |

**`distillation/`**: large rollout sets for the scaling experiments. `<task>/rollout/low/success/` holds 100 zarrs of about 1000 RL successes each, for `one_leg`, `lamp`, `round_table` and `factory_peg_hole`. For `one_leg` they are named `rl_state_<start index>`; for the other tasks, `rppo_0…99`. `distillation/one_leg/rl_state_0` has the same episode and step counts as `sim/one_leg_state_distill_large/.../rl_state_0`, so it is probably a second copy of the same data.

### `one_leg` datasets for the full pipeline (~50 GB)

| Zarr (under `processed/diffik/`) | Episodes | Mean length | Size | Role |
|---|---|---|---|---|
| `sim/one_leg/teleop/low` | 50 | 467 | 2.2 GB | D_sim |
| `sim/one_leg/teleop/med` | 50 | 482 | 2.3 GB | D_sim |
| `sim/one_leg/rollout/low` | 150 | 436 | 6.2 GB | RL rollouts (state only) |
| `sim/one_leg/rollout/med` | 200 | 557 | 10.7 GB | RL rollouts (state only) |
| `real/one_leg_full_new/teleop/low` | 40 | 513 | 7.3 GB | D_real |
| `real/one_leg_full/teleop/low` | 50 | 554 | 10.3 GB | D_real (alternative) |
| `sim2real/one_leg_render_rppo_brighter/rollout/med` | – | – | 30.9 GB | D_synth (the one the authors' co-training script uses) |
| `sim2real/one_leg_render_rppo_1/rollout/med` | 345 | 541 | – | D_synth (the paper's "~400") |
| `sim2real/one_leg_render_demos_brighter/teleop/{med,med_perturb}` | 25 (med) | 491 | 2.3 + 2.3 GB | Rendered demos |

---

## 4. File formats

### `.pkl`: one episode, raw

A Python pickle of a plain `dict`. The whole file loads into memory at once. Read it with `unpickle_data` ([src/visualization/render_mp4.py](src/visualization/render_mp4.py)), which also handles `.pkl.gz` and `.pkl.xz`. Schema (`Trajectory` in [src/common/types.py](src/common/types.py)):

```python
{
  "observations": [                        # T+1 dicts (sim); T dicts (real)
    {
      "robot_state": {                     # nested dict of raw robot readings
        "ee_pos": (3,), "ee_quat": (4,),               # gripper pose, quaternion xyzw
        "ee_pos_vel": (3,), "ee_ori_vel": (3,),
        "gripper_width": (1,),
        "joint_positions": (7,), "joint_velocities": (7,), "joint_torques": (9,),
        "gripper_finger_1_pos": (1,), "gripper_finger_2_pos": (1,),
      },
      "color_image1": uint8 (H, W, 3),     # wrist camera; 2×2 placeholder in state-only rollouts
      "color_image2": uint8 (H, W, 3),     # front camera
      "parts_poses":  float (42,),         # 6 objects × (xyz + quat xyzw)
    }, ...
  ],
  "actions":  float (T, 8),                # DELTA: Δpos(3) + Δquat xyzw(4) + gripper(1)
  "rewards":  float (T,),                  # sparse: +1 when a part pair is assembled
  "success":  bool,
  "furniture": "one_leg",                  # task name (legacy key; newer files may use "task")
  "action_type": "pos",
}
```

Pickles are written by sim teleop ([src/data_collection/](src/data_collection/)), policy rollouts ([src/eval/rollout.py](src/eval/rollout.py)) and real teleop ([src/real/teleop_sm.py](src/real/teleop_sm.py)). The Isaac Sim re-render script reads and writes them too.

### `.zarr`: many episodes, ready to train

A chunked, compressed store of N-dimensional arrays (zarr v2). On disk it is a **directory**:

```
success.zarr/
├── .zgroup, .zattrs            # JSON: group marker + metadata (n_episodes, n_timesteps, randomness, domain, …)
├── robot_state/
│   ├── .zarray                 # JSON: shape [23371,16], dtype <f4, chunks [1000,16], blosc-lz4
│   └── 0.0  1.0 … 23.0         # compressed chunks of 1000 timesteps; name = chunk position
├── color_image1/0.0.0.0 …      # each chunk = 1000 frames × 240×320×3 (~30 MB compressed)
├── action/pos/, action/delta/  # sub-groups are nested folders
└── episode_ends/, success/, task/, …
```

Readers decompress only the chunks a slice touches. If a single chunk file is missing, reading that array fails.

**Arrays.** All episodes are concatenated along one time axis:

| Array | Shape | Contents |
|---|---|---|
| `robot_state` | (N, 16) float32 | EE pos (3) + rot6d (6) + lin vel (3) + ang vel (3) + gripper width (1) |
| `parts_poses` | (N, 42) float32 | 6 × (pos + quat xyzw). For `one_leg`: table top, legs 1–4, U-fixture (`obstacle_front`). Only the top and leg 4 are assembled |
| `action/pos` | (N, 10) float32 | **Training target**: absolute EE pose in the robot base frame, pos (3) + rot6d (6), plus gripper (+1 close / −1 open) |
| `action/delta` | (N, 10) float32 | The same action as a delta, in rot6d |
| `color_image1` | (N, 240, 320, 3) uint8 | Wrist camera |
| `color_image2` | (N, 240, 320, 3) uint8 | Front camera (resized and cropped) |
| `reward`, `skill`, `augment_states` | (N,) | `reward` marks the step where the simulator detected assembly during teleop (1 or NaN, see §10.5); the other two are all zeros |
| `episode_ends` | (E,) uint32 | Cumulative end index: episode *i* is rows `[ends[i-1], ends[i])` |
| `success`, `task`, `pickle_file` | (E,) | Per-episode flag, task name, source pickle path |

Reading one episode:

```python
import zarr
z = zarr.open("data/processed/diffik/sim/one_leg/teleop/low/success.zarr", "r")
print(dict(z.attrs))                       # n_episodes, n_timesteps, randomness, …
ends = z["episode_ends"][:]
i = 0
s, e = (0 if i == 0 else ends[i - 1]), ends[i]
states  = z["robot_state"][s:e]            # (T_i, 16)
actions = z["action/pos"][s:e]             # (T_i, 10)
wrist   = z["color_image1"][s:e]           # (T_i, 240, 320, 3); only these chunks are decompressed
```

---

## 5. Converting `.pkl` to `.zarr`

```bash
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s teleop -r low -o success
#   optional: --suffix <raw subdir> --output-suffix <zarr name> --offset K --max-files M
```

[src/data_processing/process_pickles.py](src/data_processing/process_pickles.py) does the following:

| | `.pkl` | `.zarr` |
|---|---|---|
| Scope | 1 episode | All matching episodes, concatenated, plus `episode_ends` |
| Robot state | Nested dict incl. joint data | 16-D vector with rot6d; joint data dropped |
| Rotation | Quaternion (xyzw) | 6D rotation |
| Action | 8-D delta | 10-D absolute (`action/pos` = state + delta) and 10-D delta |
| Action clipping | – | Δpos clipped per axis to ±2.5 cm; Δrot scaled down per *episode* (a bug, see §11); gripper → sign |
| Images | Any size | 240×320 (wrist resized; front resized and cropped) |
| Length | T+1 observations (sim) | Last observation dropped so it lines up with T actions |

Conventions are shared across stages. Rotations are converted in [src/common/geometry.py](src/common/geometry.py), with quaternions in xyzw order. Images are resized again to 224×224 inside the model.

---

## 6. How each pipeline stage uses the data

The stages follow Fig. 3 of the paper:

| Stage | Reads | Writes | Bucket equivalent |
|---|---|---|---|
| ① BC → π_base | `sim/<task>/teleop/{low,med}` zarr (state policy: `robot_state` ⊕ `parts_poses`) | checkpoint | `checkpoints/bc/…` |
| ② Residual RL → π_res | **No dataset.** Loads π_base (weights + normalizer) and learns from the simulator's sparse reward | checkpoint | `checkpoints/rppo/…` |
| ③a Expert rollouts | π_res checkpoint | raw state-only `.pkl` → `sim/<task>/rollout/…` | `sim/one_leg/rollout/…`, `distillation/…`, `distill/…` |
| ③b Re-render | Rollout pickles, replayed in Isaac Sim with domain randomization | raw `.pkl` with images → zarr | `sim2real/one_leg_render_*` |
| ④ Real demos | Franka + 2 RealSense via teleop | raw `.pkl` → `real/<task>/teleop/…` | `real/one_leg_full*` |
| ⑤ Co-train π_real | D_real + D_synth zarrs (image policy: 2 images + `robot_state`) | checkpoint | – |

The data enters RL only through π_base's normalizer. π_res reuses it, while π_real fits a new one on the combined real + synthetic data.

---

## 7. How training loads zarrs

1. **Choose the data.** Config fields `task`, `environment` (= domain), `demo_source`, `randomness` and `controller` may each be a list. [get_processed_paths](src/common/files.py) globs every combination and keeps the zarrs that exist, e.g.:
   ```bash
   python -m src.train.bc +experiment=image/real_ol_cotrain \
     task='[one_leg_full_new,one_leg_render_demos_brighter,one_leg_render_rppo_brighter]' \
     environment='[real,sim]' demo_source='[teleop,rollout]' randomness='[low,med,med_perturb]' …
   ```
   You can also list paths relative to `$DATA_DIR_PROCESSED/processed/` in `data.data_paths_override` (see the `state/scaling/*` configs). Suffixed zarrs have to be selected this way.
2. **Combine.** [combine_zarr_datasets](src/dataset/zarr.py) concatenates the zarrs, shifts `episode_ends` to match, and tags each episode with `domain` from the zarr's `.zattrs` (sim = 0, real = 1).
3. **Windows.** [StateDataset / ImageDataset](src/dataset/dataset.py) cut samples of 1 observation step + 32 action steps (`pred_horizon`) from `action/<control_mode>`, where `control_mode` defaults to `pos`. At run time the policy executes 8 of those 32 actions (`action_horizon`).
4. **Normalize.** A min-max normalizer to [−1, 1] is fitted on the loaded data and saved inside the checkpoint (`normalizer.*`).

---

## 8. Downloading

- **Teleop demos only:** `python scripts/download_data.py --task one_leg`. This fetches `processed/diffik/sim/<task>/teleop/` (low, med and, for `one_leg`, high) into `$DATA_DIR_PROCESSED/processed/…`.
- **Checkpoints:** `python scripts/download_checkpoints.py` fetches `checkpoints/{bc,rppo}/one_leg/{low,med}/actor_chkpt.pt` (1.6 GB) into `./checkpoints/…`. Narrow it with `--task`, `--kind` and `--randomness`, get all 16 with `--task all`, and preview with `--dry-run`. Files already present with the right size are skipped.
- **Anything else:** use the AWS CLI without credentials. Put `sim2real` zarrs under `sim/` so the co-training config can find them (§11, item 1):
  ```bash
  B=s3://iai-robust-rearrangement
  D=$DATA_DIR_PROCESSED/processed/diffik
  aws s3 sync --no-sign-request $B/data/processed/diffik/real/one_leg_full_new $D/real/one_leg_full_new
  aws s3 sync --no-sign-request $B/data/processed/diffik/sim2real/one_leg_render_rppo_brighter $D/sim/one_leg_render_rppo_brighter
  aws s3 sync --no-sign-request $B/checkpoints/rppo/one_leg ./checkpoints/rppo/one_leg
  ```
- **Check a download.** For each array, compare the number of chunk files with `ceil(shape[0] / chunks[0])` from its `.zarray`, and check that `episode_ends[-1] == n_timesteps`. An interrupted download leaves a zarr that opens but fails partway through reading.

**Local copy as of 2026-09-23:** `sim/one_leg/teleop/{low,med,high}` matches S3 byte for byte (236, 245 and 281 files). Nothing else has been downloaded.

---

## 9. Visualizing data and evaluating checkpoints

### 9.1 Which tool for which data

Both raw (`.pkl`) and processed (`.zarr`) data can be viewed **as recorded** (the stored camera images and trajectories) or **re-simulated** (the recorded actions played again in IsaacGym, watched live or re-rendered to video).

| | As recorded: videos and plots | Re-simulated in IsaacGym |
|---|---|---|
| **Raw `.pkl`** (one file or a folder) | [scripts/visualize_data.py](../scripts/visualize_data.py) (§9.2) | [scripts/replay_in_sim.py](../scripts/replay_in_sim.py), exact start state (§9.3) |
| **Processed `.zarr`** | [scripts/visualize_data.py](../scripts/visualize_data.py) (§9.2) | [scripts/replay_in_sim.py](../scripts/replay_in_sim.py), approximate start state (§9.3) |

Use the recorded view to check what was captured (image content, camera framing, action/state consistency, coverage). Use the re-simulated view when there are no images (state-only rollouts), or to check that the recorded actions still move the robot the same way.

**Setup.** Run both scripts with the `resip` env's Python from the repo root. The simulator tools also need the env's `lib/` on the library path; without it IsaacGym fails with `libpython3.8.so.1.0: cannot open shared object file`:
```bash
conda activate resip
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### 9.2 Recorded data: videos and plots

```bash
python scripts/visualize_data.py info  <data>                      # metadata, episode stats, sanity checks
python scripts/visualize_data.py video <data> --episodes 0 5 12    # wrist | front mp4 per episode
python scripts/visualize_data.py plot  <data> --episodes 0         # per-episode time series + dataset overview
```

**Parameters** (`python scripts/visualize_data.py <command> <data> [options]`)

| Parameter | Default | Applies to | Meaning |
|---|---|---|---|
| `command` | required | – | `info`, `video` or `plot` (see the next table) |
| `data` | required | all | What to load: a `.zarr` directory, a single `.pkl` / `.pkl.gz` / `.pkl.xz`, or a folder of pickles (e.g. `raw/.../success/`). A folder is treated as a dataset with one episode per file, in sorted file-name order |
| `--episodes N [N ...]` | `0` | `video`, `plot` | Episode indices to render. For a zarr, the index into `episode_ends`; for a pickle folder, the file's position in sorted order; for a single pickle, only `0` exists. `plot` always draws the overview from **all** loaded episodes; this only picks the time-series figures |
| `--max-files N` | all | pickle folders | Load at most the first N pickles. Pickles load fully into memory, so use this on large folders |
| `--fps N` | `30` | `video` | Playback frame rate. The data is recorded at 10 Hz, so 30 plays at 3× speed and 10 plays in real time |
| `--out-dir DIR` | `viz` | `video`, `plot` | Output folder, created if missing. Files are named after the dataset: the randomness folder for `…/<level>/success.zarr` (e.g. `low_ep003.mp4`), otherwise the file or folder name |

| Command | Output |
|---|---|
| `info` | Metadata (zarr `.zattrs` or pickle count), array shapes, episode lengths, success count. Checks for NaNs, gripper actions other than ±1, and the gap between commanded and actual position |
| `video` | `<name>_ep###.mp4`: wrist and front cameras side by side, with step, gripper command and gripper width drawn on each frame. Zarrs only decompress the chunks of the requested episodes |
| `plot` | `<name>_ep###_timeseries.png`: EE x/y/z against the commanded target, gripper width against the open/close command. `<name>_overview.png`: EE paths (top-down and side), initial part positions, episode-length histogram |

- **Both formats are treated the same.** A pickle's 8-D delta actions are turned into absolute position targets (EE position + delta), as `process_pickles.py` does, so the plots are directly comparable with zarr plots. A folder of pickles is loaded as a dataset, one episode per file in sorted order.
- **State-only rollouts** (all of `coverage/` and `distill/`, and any rollout saved with `--observation-space state`) have 2×2 placeholder images. `video` reports this instead of writing blank files; use `plot` or §9.3 for those.
- **Pickle size.** Pickles with images are large (68 MB for a 146-step `factory_peg_hole` demo, up to ~1 GB for real demos) and load fully into memory; use `--max-files` on folders.

Tested on `one_leg/teleop/low` (zarr, 50 episodes), a state-only `one_leg` RL rollout pickle and a `factory_peg_hole` teleop pickle with images.

### 9.3 Re-simulating an episode in IsaacGym

```bash
python scripts/replay_in_sim.py <episode.pkl>              --record replay.mp4   # raw pickle
python scripts/replay_in_sim.py <success.zarr> --episode 3 --record replay.mp4   # processed zarr
python scripts/replay_in_sim.py <success.zarr> --episode 3 --viewer              # watch live
```

**Parameters** (`python scripts/replay_in_sim.py <path> [options]`)

| Parameter | Default | Applies to | Meaning |
|---|---|---|---|
| `path` | required | all | A raw `.pkl` episode (`.pkl.gz` / `.pkl.xz` also work) or a processed `.zarr` directory. A folder of pickles isn't accepted; pass one file |
| `--episode N` | `0` | zarr | Which episode of the zarr to replay (index into `episode_ends`) |
| `--task NAME` | from the data | all | Simulator task to load. By default it is read from the zarr's `task` array or the pickle's `task`/`furniture` key. Variant labels are mapped to their base task by prefix (`one_leg_full` → `one_leg`). Valid base tasks: `one_leg`, `round_table`, `lamp`, `mug_rack`, `factory_peg_hole` |
| `--viewer` | off | all | Open the IsaacGym viewer window and play the replay live. Needs a display; runs headless without it |
| `--record FILE` | off | all | Render both simulated cameras and save them side by side (wrist \| front, 240 px high) to this mp4. Turns on camera rendering, which slows the replay |
| `--fps N` | `30` | `--record` | Frame rate of the recorded mp4 (the simulation steps at 10 Hz) |
| `--settle-steps N` | `40` | zarr | Steps used to drive the arm from its default joints to the recorded start pose before the parts are placed. The start error is printed (`Start pose reached within … cm`); raise this if it isn't close to 0 |
| `--zarr-rotation state\|action` | `state` | zarr | Where orientation targets come from. `state`: the recorded orientation `--rotation-lead` steps ahead, which reproduces the real motion. `action`: `action/pos` as stored, which is under-rotated by the processing bug (§11) and useful only to show its effect |
| `--rotation-lead N` | `1` | zarr, `state` | How many steps ahead the orientation target looks. The controller lags its target by about a step; 2 made no consistent difference in tests |
| `--hold-steps N` | `20` | all | Extra steps that repeat the final action, so the controller can settle and a final insertion or screw can register. Not counted in the tracking error |
| `--gpu N` | `0` | all | CUDA device for the simulation and rendering |
| `--seed N` | `0` | all | Seed for Python, NumPy and torch. The simulator isn't bit-for-bit reproducible, so step-by-step tracking still varies slightly between runs |

The randomness level passed to the environment is taken from the zarr's `.zattrs` (pickles use `low`). It doesn't change the replay, because the start state is overwritten with the recorded one.

The script prints how closely the simulated end effector follows the recording, and whether the task reward fired:
```
Replayed 505 steps (+20 hold) of rollout.pkl (one_leg)
  EE tracking vs. recording: mean 0.07 cm, max 1.01 cm, final 0.00 cm
  Reward collected: 1  (assembled ✓)
```

**How the start state is restored**

| Input | Start state | Actions replayed |
|---|---|---|
| `.pkl` | **Exact.** Robot joint angles and part poses from the first observation (`env.reset_to`) | Absolute targets built from the recorded EE pose + each delta |
| `.zarr` | **Approximate.** Zarrs don't store joint angles, so the arm is reset to its default joints (with the env's reset noise switched off) and driven to the recorded start EE pose (`--settle-steps`, default 40). Only then are the parts placed at their recorded start poses | `action/pos` for position and gripper. Orientation targets come from the next recorded state (`--zarr-rotation state`, default), because the stored action rotations are scaled down by a processing bug (§11) |

Variant task names such as `one_leg_full` use the base task's environment; `--task` overrides it.

**What replay can and can't tell you.** Replay runs the recorded actions without feedback, so the robot's motion is reproduced closely but contacts can play out differently.

**Reset randomness is switched off.** A normal `env.reset()` adds random noise to the arm's joints and pushes the parts with random forces. The replay overwrites the start state anyway, but before this was switched off the noise still leaked through: a 7-joint arm reaches the same gripper pose with different elbow positions, and pushed parts can still be moving. The same episode then succeeded in only 1 of 4 identical runs. The script now resets with zero joint noise and zero pushes, and zeroes the parts' velocities when placing them.

| Test episode | EE tracking (mean / max) | Reward fired? |
|---|---|---|
| `one_leg` RL rollout pickle (recorded in the same RL env) | 0.07 / 1.01 cm | ✓ |
| `factory_peg_hole` teleop pickle | 0.01 / 0.25 cm | ✗: the peg stayed in the gripper instead of being released in the hole |
| `one_leg/teleop/low` zarr, episodes 0–3 | 0.04–0.14 / 0.5–1.0 cm | 2 of 4 (episodes 2 and 3; episode 3 in 3 of 3 repeats) |

- **Good for** watching any episode, including state-only rollouts, from both cameras; checking that recorded actions are physically consistent (large tracking errors point to bad data or a controller mismatch); and comparing replays side by side with the recorded video.
- **Not a pass/fail test for teleop data.** Two things about how teleop demos are recorded ([data_collector_sm.py](../src/data_collection/data_collector_sm.py)) make an exact replay unlikely to trip the reward:
  - **Success is the operator's judgment.** The episode ends when the operator presses `t`, not when the simulator's assembly check fires, so some demos stop just short of it: in the local `one_leg/teleop/low` zarr, 4 of 50 demos end with the leg outside the simulator's ±20° twist tolerance.
  - **Idle steps aren't recorded.** Steps without SpaceMouse or keyboard input are skipped (except the 8 steps after a gripper toggle), although the simulation kept running during them. A replay therefore gives parts less time to settle at grasps and insertions.

  Replays of RL rollouts, which record every step and end on the simulator's own success signal, did reproduce success.
- **Replaying deltas drifts.** Applying the pickle's delta actions to the *simulated* pose (as the upstream `run_sim_env --replay-path` does) compounds errors: 2.5 cm mean drift on the RL rollout above, against 0.07 cm with absolute targets.

The upstream alternative is `python -m furniture_bench.scripts.run_sim_env --furniture one_leg --randomness low --replay-path <episode>.pkl [--record] [--headless]` ([run_sim_env.py](../furniture-bench/furniture_bench/scripts/run_sim_env.py)). It resets to the pickle's first observation in the teleop environment and replays the deltas. It has not been run on this machine.

### 9.4 Notebooks

The [notebooks/](../notebooks/) folder has more specialized views. Most hard-code the authors' data paths, so edit those before running.

| Notebook | Shows |
|---|---|
| `17_la_visualize_rollouts` | Rollout videos and trajectories |
| `18_la_produce_uncut_videos`, `14_as_residual_viz_videos` | Long uncut BC-vs-ResiP videos; residual visualizations |
| `07_real_vs_sim_state_action_space` | Real vs. sim action coverage (paper Fig. 19) |
| `08_as_initial_coverage_quant_viz` | Initial-state coverage, from `coverage/` rollouts |
| `09_as_rerender_pkl_viz`, `06_as_isaac_sim_rerender_inspection` | Rendered (`sim2real`) data |
| `10_la_visualize_randomness_levls`, `15_as_randomness_figures2` | What low, med and high randomness look like |
| `15_la_visualize_image_augmentation` | Training image augmentations (paper Tables XI–XII) |
| `16_la_visualize_encoder_features` | Vision encoder features |
| `04_la_real_world_demos`, `07_*align_inspect_cameras` | Real demos; sim/real camera alignment |
| `11_as_bc_fail_rl_succ` | Episodes where BC fails and ResiP succeeds |
| [visualize_demos.py](../notebooks/visualize_demos.py) | Every episode of several zarrs to mp4 (front camera only) |

### 9.5 Evaluating checkpoints

[src/eval/evaluate_model.py](../src/eval/evaluate_model.py) loads a WandB run (`--run-id`) or a local file (`--wt-path`, through `LocalCheckpointWrapper` in the same file), rolls it out in `--n-envs` parallel simulator copies and prints the success rate.

```bash
# 1. Success rate. Compare with paper Table I (one_leg low: DP 54%, ResiP 98%; med: 26% / 76%)
python -m src.eval.evaluate_model --wt-path checkpoints/rppo/one_leg/low/actor_chkpt.pt \
    -f one_leg --randomness low --n-envs 32 --n-rollouts 256 --max-rollout-steps 700 \
    --action-type pos --observation-space state

# 2. Watch live in the IsaacGym viewer: same command with few envs
    ... --n-envs 4 --n-rollouts 4 --visualize

# 3. Record episodes with camera images (keep it to ONE round: --n-rollouts = --n-envs), then make mp4s with §9.2
python -m src.eval.evaluate_model --wt-path checkpoints/rppo/one_leg/low/actor_chkpt.pt \
    -f one_leg --randomness low --n-envs 32 --n-rollouts 32 --max-rollout-steps 700 \
    --action-type pos --observation-space image --save-rollouts --save-failures --save-rollouts-suffix eval_vid

# Turn the saved episodes into mp4s
python scripts/visualize_data.py video data/raw/diffik/sim/one_leg/rollout/low/eval_vid/success --episodes 0 1 2
python scripts/visualize_data.py video data/raw/diffik/sim/one_leg/rollout/low/eval_vid/failure --episodes 0 1 2
```

#### Parameters

Exactly one of `--wt-path`, `--run-id`, `--sweep-id` or `--project-id` must be given. `--task` is always required.

**What to evaluate**

| Parameter | Default | Meaning |
|---|---|---|
| `--wt-path FILE` | – | Local checkpoint (`.pt`). Its stored config rebuilds the policy, so `bc` and `rppo` files use the same command |
| `--run-id P/ID [P/ID ...]` | – | One or more WandB runs as `<project>/<run id>`; weights are downloaded from the run. Needs WandB login and access to the project |
| `--sweep-id ID` / `--project-id P` | – | Evaluate every run in a WandB sweep / project |
| `--wt-type NAME` | `best_success_rate` | WandB runs only: which saved weights to load: `best_success_rate`, `best_test_loss` or `last`. Ignored with `--wt-path` |
| `--run-state S [S ...]` | all | WandB runs only: keep only runs in these states: `running`, `finished`, `failed`, `crashed` |
| `--eval-top-k K` | all | WandB runs only: evaluate only the K runs with the highest stored success rate |

**Environment and rollouts**

| Parameter | Default | Meaning |
|---|---|---|
| `--task`, `-f NAME` | required | Simulator task: `one_leg`, `round_table`, `lamp`, `mug_rack`, `factory_peg_hole` (plus `square_table`, `desk`, `cabinet`, `bimanual_insertion`, which have no released checkpoints) |
| `--randomness LEVEL` | `low` | Initial-scene randomness: `low`, `med` or `high`. See *Randomness settings* below |
| `--n-envs N` | `1` | Parallel simulator copies. Bounded by GPU memory; about 16–64 on an 8 GB GPU in state mode, fewer with images |
| `--n-rollouts N` | `1` | Total episodes. Runs in rounds of `--n-envs`, and the count is **truncated** to `n_rollouts // n_envs` rounds, so use a multiple of `--n-envs`. A value below `--n-envs` runs nothing |
| `--max-rollout-steps N` | per-task timeout | Episode length limit in 10 Hz steps. Always set it (values in the table below); the default is longer than the paper's and inflates success |
| `--n-parts-assemble N` | all parts | How many part pairs must be assembled to count as success. Also sets the default timeout (1000 steps × N) |
| `--observation-space state\|image` | `state` | `image` also renders both cameras. The policy still gets the state it needs; use it to save rollouts with real images |
| `--action-type pos\|delta\|relative` | `pos` | Action format of the environment. Must match the checkpoint's `control.control_mode` (an assertion checks this); all released checkpoints use `pos`. `relative` is run as `pos` |
| `--april-tags` | off | Use the original FurnitureBench part meshes with AprilTag textures instead of the tag-free assets in `src/assets` |
| `--visualize` | off | Open the IsaacGym viewer (turns off headless mode). Keep `--n-envs` small |
| `--gpu N` | `0` | CUDA device for the policy and the simulator |
| `--verbose`, `-v` | off | Print the environment's setup output |

**Saving rollouts**

| Parameter | Default | Meaning |
|---|---|---|
| `--save-rollouts` | off | Save each episode as a raw pickle (not an mp4) under `$DATA_DIR_RAW/raw/diffik/sim/<task>/rollout/<randomness>/[<suffix>/]success/`. **Nothing is written until all `--n-rollouts` episodes have finished**: every episode, including its camera frames, is kept in RAM until then. With `--observation-space image` that is about 460 KB per env-step, so ~10 GB per round of 32 envs × 700 steps. 256 rollouts need ~80 GB and get the process killed on a 60 GB machine, leaving the folder empty. Record one or two rounds per run and repeat the command for more; each run adds new pickles |
| `--save-failures` | off | Also save failed episodes (to `…/failure/`). Needs `--save-rollouts` |
| `--save-rollouts-suffix S` | `""` | Extra folder level under the randomness folder, to keep evaluations apart |
| `--store-full-resolution-video` | off | Save images at full camera resolution instead of downsizing them to 240×320 |
| `--compress-pickles` | off | Save as `.pkl.xz` |
| `--break-on-n-success` + `--stop-after-n-success N` | off, `0` | Stop once at least N successes have been collected (checked after each round). Used to generate expert data |
| `--record-for-coverage` | off | Instead of full episodes, save only each episode's first state and outcome to `first_states.npz`, for coverage analysis. Needs `--save-rollouts` |

**Reporting results (WandB)**

| Parameter | Default | Meaning |
|---|---|---|
| `--wandb` | off | Write success rate, success count, rollout count and returns into the run's WandB summary. With `--wt-path`, writes them to `<checkpoint>.summary.json` next to the checkpoint instead |
| `--if-exists skip\|overwrite\|append\|error` | `error` | What to do when the run already has a stored success rate. `append` pools the new episodes with the stored ones. Irrelevant for `--wt-path` (no stored results) |
| `--max-rollouts N` | – | Skip runs that already have at least N stored rollouts |
| `--multitask` | off | Prefix the summary keys with the task name (`one_leg/success_rate`) |
| `--store-video-wandb` | off | Opens the WandB run for video logging (needs `--wandb`). As written, `evaluate_model` doesn't pass `save_rollouts_to_wandb` to the rollout code, so **no videos are uploaded** |
| `--continuous-mode`, `--continuous-interval S` | off, `60` | Keep polling the sweep/project every S seconds and evaluate new runs |
| `--prioritize-fewest-rollouts` | off | After each run, re-sort runs by stored rollout count and start again with the fewest |
| `--ignore-currently-evaluating-flag` | off | Evaluate runs even if another evaluator has marked them `currently_evaluating` |

**Not usable:** `--leaderboard` (an assertion rejects it), `--action-horizon` (parsed but never used). `--controller` and `--use-new-env`, which the authors' `scripts/ant_local/` and `slurm/` commands pass, no longer exist; the script stops with "unrecognized arguments". Remove them when reusing those commands.

**Fixed inside the script:** the diffusion policy samples with 4 DDIM steps (paper Table IV), and the environment uses 6D rotation actions.

#### Randomness settings

`--randomness` selects how much the initial scene varies in the RL env (§2.2). Only `low`, `med` and `high` are valid here. Folder labels such as `med_perturb` make the env fail with `Unknown initialization mode`.

| Level | Fixture offset | Part push at reset (max force / torque) | Arm joint noise | Paper Table I |
|---|---|---|---|---|
| `low` | ±2 cm | 0.2 / 0.007 | ±5° | All 6 tasks |
| `med` | ±4 cm | 0.5 / 0.01 | ±10° | `one_leg`, `round_table`, `lamp` |
| `high` | ±6 cm | 0.75 / 0.015 | ±13° | Not reported |

**Match the checkpoint to its level.** Each released checkpoint was trained at the level in its path (`checkpoints/<kind>/<task>/<level>/`), so evaluate it at that level to compare with the paper. Evaluating a `low` checkpoint at `med` is a valid robustness test, but has no paper number.

| Task | Levels with checkpoints | Paper success, DP → ResiP (low / med) | `--max-rollout-steps` |
|---|---|---|---|
| `one_leg` | low, med | 54 → 98 / 26 → 76 | 700 (paper); authors' eval scripts often use 750 |
| `round_table` | low, med | 12 → 94 / 4 → 77 | 1000 |
| `lamp` | low, med | 7 → 97 / 2 → 70 | 1000 |
| `mug_rack` | low | 26 → 88 / – | 400 |
| `factory_peg_hole` | low | 5 → 99 / – | 200; **required**, the default lookup has no entry for this task and fails |

Always pass `--max-rollout-steps`. Without it the timeout comes from [tasks.py](../src/common/tasks.py): 1000 steps per part (`one_leg` 1000; `round_table`, `lamp`, `mug_rack` 2000). That is longer than the paper's episodes, so success rates come out higher than Table I. The values above are the RL episode lengths from paper Table VI and the authors' slurm scripts.

**Perturbations during the episode (paper Fig. 9)** can't be enabled from the command line. [rollout.py:190](../src/eval/rollout.py#L190) hard-codes `env.step(action, sample_perturbations=False)`. Changing it to `True` makes each step push about 1% of parts with the level's force and torque limits. The paper reports that ResiP loses 12 points under perturbation and chunked policies 19–26. The RL training wrapper has its own switch (`sample_perturbations` in [env_rl_wrapper.py](../src/gym/env_rl_wrapper.py)).

**Demos and evaluation use the same randomization.** The SpaceMouse collector creates the same RL env with the same `--randomness` (§2.2), and the released `one_leg` demos were recorded after it switched to that env. Demos collected with the older scripted collector or `run_sim_env` use the original env's uniform placement instead.

#### Notes

- **Recording (3)** is the flag set the authors used to record BC-vs-RL videos ([9_ol_bc_vs_rppo_pert.sh](../scripts/ant_local/rollout_for_render/9_ol_bc_vs_rppo_pert.sh)). With `--observation-space image` the simulator renders both cameras, so saved pickles contain real images. Rollouts saved with `--observation-space state` have placeholder images; re-render them with §9.3.
- **The failures folder** shows fastest how a policy fails.
- **During training.** [bc.py](../src/train/bc.py) runs simulator rollouts every `training.eval_every` epochs, and [residual_ppo.py](../src/train/residual_ppo.py) evaluates every 5th iteration. Both log success rates (and videos where enabled) to WandB, so those curves are the main check while training.

**Before running on this machine:**
- **Checkpoints.** The four `one_leg` checkpoints are in `./checkpoints/`; get others with `scripts/download_checkpoints.py` (§8).
- **Library path.** Export `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` (§9.1), as for the other simulator tools.
- **8 GB GPU.** Use `--n-envs` of about 16–64 instead of the paper's 1024. Image-mode rollouts are slower and use more memory.
- **Status.** `evaluate_model --wt-path` on the released checkpoints hasn't been run on this machine yet.
- **meshcat** isn't installed in the `resip` env; only the real-robot scripts need it.

---

## 10. Collecting demonstrations

The repo has three collectors. Only the SpaceMouse simulation collector was used for the released demos.

| Collector | Command | Environment | Status |
|---|---|---|---|
| **Simulation, SpaceMouse** (+ keyboard) | `python -m src.data_collection.teleop` ([teleop.py](../src/data_collection/teleop.py) → [data_collector_sm.py](../src/data_collection/data_collector_sm.py)) | `FurnitureRLSimEnv`, the same env as RL and evaluation | Used for the released demos (since 2024-05-15). Not run on this machine yet |
| Simulation, scripted | `python -m src.data_collection.scripted` ([scripted.py](../src/data_collection/scripted.py)) | Original `FurnitureSimEnv`, OSC controller, FurnitureBench's hard-coded assembly script | **Doesn't start as written**: it calls `trajectory_save_dir(environment=…)` without `controller`, which don't match the function's arguments (§10.7) |
| Real robot, SpaceMouse | `python -m src.real.teleop_sm` ([teleop_sm.py](../src/real/teleop_sm.py)) | Franka + two RealSense cameras | Needs the authors' internal `rdt` package (§10.7) |

### 10.1 Setup

```bash
# SpaceMouse driver (from the README)
pip install numpy termcolor atomics scipy
pip install git+https://github.com/cheng-chi/spnav
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd

# every session
conda activate resip
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DATA_DIR_RAW=/home/kaiyuan.luo/ResiP/data      # where pickles are written
```

- **A display is required.** The collector always opens the IsaacGym viewer; you teleoperate by watching it.
- **The keyboard is read through `pynput`**, which needs an X session. The terminal also needs focus to receive Enter at the start of each episode.

### 10.2 Running

```bash
python -m src.data_collection.teleop --furniture one_leg --randomness low --ctrl-mode diffik --num-demos 50
```

| Parameter | Default | Meaning |
|---|---|---|
| `--furniture NAME` | required | Task to load, e.g. `one_leg`, `round_table`, `lamp`, `mug_rack`, `factory_peg_hole` (any key of FurnitureBench's `config["furniture"]`) |
| `--ctrl-mode diffik\|osc` | required | Low-level controller. The collector asserts `diffik`; pass `diffik` |
| `--randomness low\|med\|high` | `low` | Initial-scene randomness, in the RL env's sense (§2.2): fixture offset, pushes on the parts and arm joint noise at each reset |
| `--num-demos N` | `100` | Stop after N **successful** demos. Failures don't count |
| `--save-failure` | off | Also save episodes you end with `n`, to `failure/` |
| `--sample-perturbations` | off | While you are moving, randomly push about 1% of parts each step with the level's force limits. Saves to `<level>_perturb/` |
| `--resume-dir DIR` | – | Meant to continue from the **last** state of saved pickles in `DIR`, preloading their steps into the new episode. **Broken:** it reads the misspelled key `state["r/manuewards"]` and crashes with `KeyError` |
| `--no-ee-laser` | laser on | Hide the laser drawn from the gripper (a visual aid for alignment in the viewer; not recorded) |
| `--draw-marker` | off | Show AprilTag markers on the parts. Off matches the released data |
| `--gpu-id N` | `0` | GPU for simulation and rendering |

### 10.3 Controls and episode flow

| Input | Action |
|---|---|
| SpaceMouse motion | Moves the gripper in 6-DoF: translation up to 0.3 m/s × 1.8 and rotation up to 0.7 rad/s × 4, with a 0.05 dead zone. Each command is capped at 2.5 cm and 20° per step |
| SpaceMouse button (either) | Toggle gripper open/close |
| `w`/`s`, `a`/`d`, `q`/`e` | Move along −x/+x, −y/+y, −z/+z (robot base frame) |
| `j`/`l`, `i`/`k`, `u`/`o` | Rotate about x, y, z |
| `z` | Toggle gripper |
| `[` / `]` | Decrease / increase the keyboard step size |
| **`t`** | **End the episode as a success** and save it |
| `y` | Success, and also write the final relative pose of each assembled pair to a JSON next to the task's assembly file. Used to set up the reference poses of scanned tasks; works only for tasks with an `assembly_json_fname` in the config (`mug_rack`, `factory_peg_hole`, `factory_nut_bolt`). For others such as `one_leg` the pickle is saved, then the collector crashes on the missing key |
| `` ` `` | Mark a skill boundary at this step (stored as 1 in the pickle's `skills`) |
| `n` | End as a failure (saved only with `--save-failure`) |
| `b` | Undo: drop the last 10 recorded steps and reset the simulator to the state before them |
| `p` / `c` | Pause / resume recording. You can still move while paused, e.g. to reposition |

There is no quit key: the run ends after `--num-demos` successes, or stop it with Ctrl-C. Keys `r` and the digits are read by the keyboard interface but ignored by this collector.

**One episode:**
1. The scene resets to a new random layout. **Press Enter** in the terminal to start.
2. Wait about 2 s for the robot to settle; nothing is recorded before that.
3. Teleoperate the task. For `one_leg`: push the tabletop into the corner of the fixture, pick up the leg, insert it into a hole and screw it in.
4. Keep screwing until the terminal prints **`Part assembled!`**. That is the simulator's success check firing: the leg within 5 mm of fully seated, and its twist within about ±20° of its tightened angle.
5. Release the leg, move away, and press **`t`**.

### 10.4 Collecting good demos

**What the paper says.** Almost nothing about *how* the demos were collected: "50 demonstrations in simulation by teleoperating the robot" (§II-B), collected "using a SpaceMouse" (§III-A), at the same randomness levels later used for evaluation (App. II-B). It calls collecting 100k human demos "not feasible" (§VI), which is partly why the method leans on RL and synthetic data. The authors' earlier paper *Juicer: Data-efficient imitation learning for robotic assembly* (ref. [22]) targets few-demo learning on the same FurnitureBench tasks and is the better source for collection advice (not reviewed here).

**What the released demos look like** (measured from the `one_leg` teleop zarrs; a concrete target):

| | `low` | `med` |
|---|---|---|
| Demos | 50 | 50 |
| Length (steps at 10 Hz) | 392–579, mean 467 | mean 482 |
| Passed the simulator's check (§10.5) | 43 / 50 | 50 / 50 |

- **The same strategy every time:** push the tabletop into the fixture corner, grasp the leg, insert, then screw with several regrasp cycles (the repeated gripper open/close in `visualize_data.py plot`).
- **All finish well inside the 700-step episode limit** used by RL and evaluation.

#### What makes a demo useful for training

These follow from how the method works and from the paper's failure analysis; tips marked *(inference)* are not stated in the paper.

1. **Cover the initial-state distribution you'll evaluate on.** The residual policy only makes *local* corrections, and the paper attributes ResiP's lower `med` score (76% vs 98% at `low`) to scenes that "deviate too far" from what the base policy handles (§IV-A, §VI).
   - Collect with the same `--randomness` you'll evaluate at.
   - Don't reset or skip awkward layouts; those are the states the policy most needs to have seen.
   - Check coverage with `visualize_data.py plot`: the overview's initial-part-position panel should show a spread matching the level.
2. **Be consistent.** With 50 demos, one repeated strategy (same corner, similar grasp point on the leg, same push-then-grasp order) is easier to imitate than several. Two of the paper's failure modes come from the grasp: "bad grasp for insertion" (Fig. 7) and "unseen in-hand poses" (Fig. 10B). Grasp the leg the same way each time.
3. **Be precise at the bottleneck.** The most common base-policy failure is "pushing the leg down before achieving perfect alignment with the hole" (§IV-A). Line the leg up over the hole first, then lower it, then screw.
4. **End only on a real success.** Press `t` only after the terminal prints `Part assembled!` (§10.3), and remove demos without the success marker afterwards (§10.5).
5. **Stay inside the time limit.** Aim to finish in about 400–600 steps. A policy that imitates a slow demo may run out of the 700 steps *(inference)*.
6. **Quality matters more than quantity.**
   - **More demos give diminishing returns:** Fig. 1 shows BC saturating at about 80% even with 100k demos.
   - **The RL stage mainly needs a base policy that sometimes succeeds**, because the reward is sparse and exploration depends on it. The paper started from 54% (`low`) and 26% (`med`), and notes that going from 10% to 26% made fine-tuning "far easier" (App. VIII).
   - **Target:** 50 clean, consistent, fully assembled demos per level is the paper's recipe.

#### SpaceMouse tips

These follow from how the collector behaves (§10.3, §10.5):

- **Stop instead of pausing.** Steps with no input aren't recorded, so hesitating costs nothing. Avoid `p`: if you keep moving while paused, the recording jumps between two states.
- **Use `b` (undo) to recover from mistakes.** It rewinds the last 10 recorded steps and resets the simulator to match, so the demo stays consistent. A long fumbled recovery teaches the policy to fumble; small corrective adjustments near the hole are fine and arguably useful *(inference)*.
- **Screw with the keyboard.** `u`/`o` rotate by a fixed 0.2 rad (≈11°) about vertical per press, easier to do cleanly than twisting the cap. Keyboard input overrides the SpaceMouse for that step. Both rotate about the base frame's vertical, so a vertical leg turns about its own axis either way.
- **Keep the laser on** (the default; `--no-ee-laser` hides it). It shows where the gripper points, which helps line up over the hole, and isn't recorded.
- **Slow down near contact.** Full deflection moves up to 2.5 cm per step; light pressure near the hole gives finer, smoother motion.
- **Review each batch before training** with `visualize_data.py info`, `plot` and `video` (§10.6): look for outlier lengths, missing success markers and gripper chatter.

### 10.5 What gets recorded

**Files:** one pickle per episode at `$DATA_DIR_RAW/raw/diffik/sim/<task>/teleop/<randomness>[_perturb]/{success,failure}/<YYYY-MM-DDTHH:MM:SS>.pkl` (format in §4). Besides observations, actions, rewards and `success`, each pickle stores `skills`, `furniture`, `error` / `error_description`, and `metadata` with the teleop settings (frequency 10 Hz, command latency, dead zone, maximum speeds, SpaceMouse scalings).

**Which steps are kept:**
- **Only steps with input.** A step is recorded when the SpaceMouse or keyboard commands motion. Idle steps are skipped even though the simulation keeps running, except the 8 steps after a gripper toggle, so the gripper closing is captured.
- **Not the first ~2 s** after pressing Enter, and nothing while paused.
- **Undo removes steps.** `b` deletes the last 10 recorded steps from the episode.

**Success labels come from the operator.** The collector's env is created with `manual_done=True`, so the simulator never ends or labels an episode; pressing `t` does. The simulator's check still runs, and the step where it first fires is marked in `rewards`:

| Demos | Marker | Episodes with the marker | When the operator pressed `t` |
|---|---|---|---|
| `one_leg/teleop/med` (recorded 2024-05-16 to 05-17) | reward `1` | 50 / 50 | – |
| `one_leg/teleop/low` (2024-05-27 to 05-28) | `None` in the pickle, NaN in the zarr | 43 / 50 | 31–57 steps after the marker (median 40) |
| `one_leg/teleop/high` (2024-05-27) | NaN | 49 / 50 | – |

`None`/NaN appears because later env code returns `print("Part assembled!")` (i.e. `None`) from `_reward` when `manual_done` is set. The 7 `low` episodes and 1 `high` episode without a marker were saved as successes but never passed the simulator's check. **Press `t` only after `Part assembled!` has been printed.** To find unmarked episodes in a folder of pickles:

```python
from pathlib import Path
from src.visualization.render_mp4 import unpickle_data
for f in sorted(Path("data/raw/diffik/sim/one_leg/teleop/low/success").glob("*.pkl*")):
    r = unpickle_data(f)["rewards"]
    if not any(x is None or x == 1 for x in r):
        print("never passed the simulator's check:", f.name)
```

### 10.6 After collecting

```bash
# 1. Inspect: counts, lengths, trajectories, videos (§9.2)
python scripts/visualize_data.py info  data/raw/diffik/sim/one_leg/teleop/low/success
python scripts/visualize_data.py video data/raw/diffik/sim/one_leg/teleop/low/success --episodes 0 1 2 --max-files 5
# 2. Remove bad demos (e.g. those without the success marker above), then process into a zarr (§5)
python -m src.data_processing.process_pickles -c diffik -d sim -f one_leg -s teleop -r low -o success
# 3. Train (IMPLEMENTATION_GUIDE.md, Stage 1)
```

- **Processing shrinks rotation actions** (§11, item 4). This affects newly collected data too; fix the clip before processing if you want the operator's full rotations in `action/pos`.
- **The paper used 50 demos per task and randomness level** (§II-B).
- **Replaying a new demo in the simulator** (§9.3) is a quick check that it's physically consistent, but for the reasons above an exact replay rarely re-triggers the success check.

### 10.7 Other collectors

**Scripted (`scripted.py`).** Meant to generate demos without a human, using FurnitureBench's hard-coded assembly script in the original `FurnitureSimEnv` with the OSC controller. It fails before collecting anything: it calls `trajectory_save_dir(environment="sim", …)`, but the function takes `controller` (required) and `domain`, not `environment`. The authors' demos weren't collected this way.

**Real robot (`src/real/teleop_sm.py`).** Same SpaceMouse scheme on a real Franka (Polymetis) with two RealSense D435 cameras, with a meshcat visualization.

| Parameter | Default | Meaning |
|---|---|---|
| `--save_dir DIR` | required | Folder for the pickles. They're written directly into it, with no `success/` or `failure/` subfolder |
| `--furniture NAME` | required | Task name stored in the pickle |
| `-p`, `--port_vis N` | `6000` | meshcat visualization port |
| `--frequency N` | `10` | Control and recording rate (Hz) |
| `--command_latency S` | `0.01` | Time budget for sending each command |
| `--deadzone X` | `0.05` | SpaceMouse dead zone |
| `--max_pos_speed V` / `--max_rot_speed W` | `0.3` / `0.7` | Maximum speeds (m/s, rad/s) |
| `--use_lcm` | off | Use LCM for robot communication |

- **Every saved demo is marked as a success** ([teleop_sm.py:448](../src/real/teleop_sm.py#L448)), so review and delete bad ones yourself.
- **Before processing,** move the pickles to `$DATA_DIR_RAW/raw/diffik/real/<task>/teleop/<randomness>/success/`, then run `process_pickles -d real`.
- **Dependencies:** it imports the authors' lab-internal `rdt` package (camera streaming, Polymetis helpers), and meshcat, which isn't installed in the `resip` env. Camera serial numbers are set in [serials.py](../src/real/serials.py).

---

## 11. Known pitfalls

1. **Rendered data is filed under a domain the loader never searches.** On S3 it lives in `processed/diffik/sim2real/…`, but training globs only `processed/<controller>/<environment>/…` with `environment ∈ {sim, real}`. Copy the zarrs into `sim/<task>/…`, or list them with `data.data_paths_override`.
2. **Task names in IMPLEMENTATION_GUIDE.md don't exist on S3.** Stage 5 of IMPLEMENTATION_GUIDE.md uses `one_leg_real` and `one_leg_render_rppo`; on S3 they are `one_leg_full` / `one_leg_full_new` and `one_leg_render_rppo_{1,black,brighter}`. Also use `task=`, not the outdated `furniture=` key.
3. **The zarr `reward` field is a marker, not a clean reward.** In teleop data it marks the step where the simulator detected assembly, as **1** in demos recorded before about 2024-05-20 (`one_leg/teleop/med`) and as **NaN** in later ones (`low`, `high`), because the collector's env returns `None` for that step (§10.5). Episodes without the marker were labelled successful by the operator but never passed the simulator's check. `skill` and `augment_states` are all zeros in the released `one_leg` data. BC training doesn't use `reward`; `StateDataset` builds its own (1 at each episode end) when it needs one, and RL rewards come from the simulator.
4. **Processing scales down every episode's rotation actions.** `clip_quat_xyzw_magnitude` ([utils.py:103](../src/data_processing/utils.py#L103)) computes one norm over the *whole* (T, 4) array it is given, and `process_pickles.py` passes a whole episode. So whenever an episode's rotations add up to more than 0.35 rad, every rotation step in it is shrunk by the same factor. In all three released `one_leg` teleop zarrs, every episode's combined rotation is exactly 0.35 rad and no single step exceeds 2.1°. The recorded *states* are unaffected; only `action/pos` and `action/delta` orientations are. The code has behaved this way since April 2024 (commit `b534d7a`), so the released checkpoints were most likely trained on data like this; that isn't verified. If you reprocess data, clip per step instead (e.g. `np.linalg.norm(delta_rotvec, axis=-1, keepdims=True)`). The position clip (each axis to ±2.5 cm) is applied per element and is fine.
5. **State-only rollouts have 2×2 placeholder images.** This covers everything in `coverage/` and `distill/`, and any rollout saved with `--observation-space state`. Some processed rollout zarrs do have real IsaacGym images (e.g. `rppo_low_*`, `bc_low_*`). Check with `scripts/visualize_data.py info` before training an image policy.
6. **Real teleop marks every saved demo as a success.** Filter bad demos yourself before processing.
7. **Watch the size of large zarrs.** By default (`data.load_into_memory=true`, [bc.py:202](src/train/bc.py#L202)), `ImageDataset` copies all images into RAM, so a 30 GB render needs about that much memory. Set `data.load_into_memory=false` to keep images on disk and read them from the zarr per sample.
