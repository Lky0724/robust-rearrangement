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

Part placement comes from `FurnitureSimEnv` ([furniture_sim_env.py:1349](furniture-bench/furniture_bench/envs/furniture_sim_env.py#L1349)). Fixture offset, force perturbations and joint noise come from `FurnitureRLSimEnv` ([furniture_rl_sim_env.py:1692](furniture-bench/furniture_bench/envs/furniture_rl_sim_env.py#L1692)), the environment used for RL and evaluation.

| Level | Part position | Part yaw | Fixture offset | Max perturbation force / torque | Initial joint noise | In the paper |
|---|---|---|---|---|---|---|
| `low` | ±1.5 cm | ±15° | ±2 cm | 0.2 / 0.007 | ±5° | Yes |
| `med` | ±5 cm | ±45° | ±4 cm | 0.5 / 0.01 | ±10° | Yes |
| `high` | Predefined high-randomness poses | – | ±6 cm | 0.75 / 0.015 | ±13° | No |
| `<level>_perturb` | as `<level>` | | | | | – |

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
| `reward`, `skill`, `augment_states` | (N,) | Effectively unused (see §10) |
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
| Action clipping | – | Per-step Δpos clipped to ±2.5 cm, Δrot magnitude to 0.35; gripper → sign |
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
- **Anything else:** use the AWS CLI without credentials. Put `sim2real` zarrs under `sim/` so the co-training config can find them (§10, item 1):
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

These tools cover checking data quality (videos, open-loop replay in the simulator, plots) and checking model performance (success rate, a live viewer, recorded rollouts). All of them run on a single workstation GPU; this machine has an RTX 3000 Ada (8 GB) and a display (`DISPLAY=:1`) for the IsaacGym viewer.

### 9.1 Inspecting data

**Zarr episode → video.** `create_mp4` in [src/visualization/render_mp4.py](src/visualization/render_mp4.py) writes frames with imageio and ffmpeg. Tested on `one_leg/teleop/low`, episode 0: 401 frames, wrist | front side by side.

```python
import zarr, numpy as np
from src.visualization.render_mp4 import create_mp4

z = zarr.open("data/processed/diffik/sim/one_leg/teleop/low/success.zarr", "r")
ends = z["episode_ends"][:]
i = 0
s, e = (0 if i == 0 else ends[i - 1]), ends[i]
frames = np.concatenate([z["color_image1"][s:e], z["color_image2"][s:e]], axis=2)  # wrist | front
create_mp4(frames, f"ep{i}.mp4", fps=30)
```

- **Raw pickle → video:** `mp4_from_pickle(path, "out.mp4")` from the same file.
- **Every episode of several zarrs:** [notebooks/visualize_demos.py](notebooks/visualize_demos.py). It hard-codes the authors' cluster `DATA_DIR_PROCESSED`, so edit that first.
- **State-only rollouts** have 2×2 placeholder images, so their videos are blank. Replay them in the simulator instead.

**Replay a raw pickle in the simulator** with [run_sim_env.py](furniture-bench/furniture_bench/scripts/run_sim_env.py):

```bash
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --randomness low \
    --replay-path <episode>.pkl     # reset to the pickle's first observation, then replay its actions
#   --record    save a video      --headless    no viewer
```

- **Open-loop check.** Replay runs the recorded actions without feedback. If it still assembles, the actions are physically consistent. If it drifts, suspect action clipping (§5), a controller mismatch or noisy data.
- **Needs a raw pickle.** The pickle's actions are 8-D deltas with quaternions, which is the script's default rotation format (`--act-rot-repr quat`). `--file-path` also replays actions but skips the reset, so parts start at random poses; prefer `--replay-path`.
- **What can be replayed.** S3 has no raw pickles for the `one_leg` sim teleop demos (§2.1). Rollout, `coverage/`, `distill/` and real pickles can be replayed.

**Notebooks** ([notebooks/](notebooks/)). Most hard-code the authors' paths, so edit those before running.

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

### 9.2 Evaluating checkpoints

[src/eval/evaluate_model.py](src/eval/evaluate_model.py) loads a WandB run (`--run-id`) or a local file (`--wt-path`, through `LocalCheckpointWrapper` in [eval_utils.py](src/eval/eval_utils.py)), rolls it out in `--n-envs` parallel simulator copies and prints the success rate.

```bash
# 1. Success rate. Compare with paper Table I (one_leg low: DP 54%, ResiP 98%; med: 26% / 76%)
python -m src.eval.evaluate_model --wt-path checkpoints/rppo/one_leg/low/actor_chkpt.pt \
    -f one_leg --randomness low --n-envs 32 --n-rollouts 256 --max-rollout-steps 700 \
    --action-type pos --observation-space state

# 2. Watch live in the IsaacGym viewer: same command with few envs
    ... --n-envs 4 --n-rollouts 4 --visualize

# 3. Record videos: render cameras, save every episode as a pickle, then mp4_from_pickle on each
    ... --observation-space image --save-rollouts --save-failures --save-rollouts-suffix eval_vid
```

- **Recording (3)** is the flag set the authors used to record BC-vs-RL videos ([9_ol_bc_vs_rppo_pert.sh](scripts/ant_local/rollout_for_render/9_ol_bc_vs_rppo_pert.sh)). With `--observation-space image` the simulator renders both cameras, so saved pickles contain real images; a state policy still works because the observations still include `parts_poses`.
- **Save location.** Episodes go to `$DATA_DIR_RAW/raw/diffik/sim/<task>/rollout/<randomness>/eval_vid/{success,failure}/`. The failures show fastest how a policy fails.
- **Other useful flags:**

| Flag | Effect |
|---|---|
| `--wt-type best_success_rate\|best_test_loss\|last` | Which weights to load from a WandB run |
| `--store-video-wandb --wandb` | Send videos to WandB instead of saving pickles |
| `--store-full-resolution-video` | Don't downsize saved images |
| `--break-on-n-success --stop-after-n-success N` | Stop after N successes (used to generate expert data) |
| `--compress-pickles` | Save as `.pkl.xz` |

- **During training.** [bc.py](src/train/bc.py) runs simulator rollouts every `training.eval_every` epochs, and [residual_ppo.py](src/train/residual_ppo.py) evaluates every 5th iteration. Both log success rates (and videos where enabled) to WandB, so those curves are the main check while training.

**Before running on this machine:**
- **Download checkpoints first:** `checkpoints/{bc,rppo}/one_leg/{low,med}/actor_chkpt.pt` (§8). None are local yet.
- **8 GB GPU.** Use `--n-envs` of about 16–64 instead of the paper's 1024. Image-mode rollouts are slower and use more memory.
- **Status.** Only the zarr→video path has been tested here. `--wt-path` evaluation of the released checkpoints and `run_sim_env --replay-path` haven't been run on this machine yet.
- **meshcat** isn't installed in the `resip` env; only the real-robot scripts need it.

---

## 10. Known pitfalls

1. **Rendered data is filed under a domain the loader never searches.** On S3 it lives in `processed/diffik/sim2real/…`, but training globs only `processed/<controller>/<environment>/…` with `environment ∈ {sim, real}`. Copy the zarrs into `sim/<task>/…`, or list them with `data.data_paths_override`.
2. **Task names in IMPLEMENTATION_GUIDE.md don't exist on S3.** Stage 5 of IMPLEMENTATION_GUIDE.md uses `one_leg_real` and `one_leg_render_rppo`; on S3 they are `one_leg_full` / `one_leg_full_new` and `one_leg_render_rppo_{1,black,brighter}`. Also use `task=`, not the outdated `furniture=` key.
3. **`reward`, `skill` and `augment_states` in zarrs aren't meaningful.** In the `one_leg` teleop zarrs, `reward` is all zeros plus one stray NaN per episode, and the other two are all zeros. `StateDataset` builds its own reward (1 at each episode end) when it needs one. RL rewards come from the simulator.
4. **Processing clips actions.** Harmless for teleop, but check how often it triggers on policy rollouts.
5. **State-only rollouts have 2×2 placeholder images.** Don't train an image policy on `sim/*/rollout` zarrs; use the `sim2real` renders.
6. **Real teleop marks every saved demo as a success.** Filter bad demos yourself before processing.
7. **Watch the size of large zarrs.** By default (`data.load_into_memory=true`, [bc.py:202](src/train/bc.py#L202)), `ImageDataset` copies all images into RAM, so a 30 GB render needs about that much memory. Set `data.load_into_memory=false` to keep images on disk and read them from the zarr per sample.
