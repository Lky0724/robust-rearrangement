import os
from pathlib import Path

from ipdb import set_trace as bp


import random
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from src.behavior.diffusion import DiffusionPolicy
from src.behavior.residual_diffusion import ResidualDiffusionPolicy
from src.behavior.residual_mlp import ResidualMlpPolicy
from src.eval.eval_utils import get_model_from_api_or_cached
from diffusers.optimization import get_scheduler


from src.gym.env_rl_wrapper import RLPolicyEnvWrapper
from src.common.config_util import merge_base_bc_config_with_root_config
from src.gym.observation import DEFAULT_STATE_OBS

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import wandb
from wandb.apis.public.runs import Run
from wandb.errors.util import CommError

from src.gym import get_rl_env
import gymnasium as gym

# Register the eval resolver for omegaconf
OmegaConf.register_new_resolver("eval", eval)


@torch.no_grad()
def calculate_advantage(
    values: torch.Tensor,
    next_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    steps_per_iteration: int,
    discount: float,
    gae_lambda: float,
):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(steps_per_iteration)):
        if t == steps_per_iteration - 1:
            nextnonterminal = 1.0 - next_done.to(torch.float)
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1].to(torch.float)
            nextvalues = values[t + 1]

        delta = rewards[t] + discount * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = (
            delta + discount * gae_lambda * nextnonterminal * lastgaelam
        )
    returns = advantages + values
    return advantages, returns


@hydra.main(
    config_path="../config",
    config_name="base_residual_rl",
    version_base="1.2",
)
def main(cfg: DictConfig):

    OmegaConf.set_struct(cfg, False)

    if (job_id := os.environ.get("SLURM_JOB_ID")) is not None:
        cfg.slurm_job_id = job_id

    # Ensure exactly one of cfg.base_policy.wandb_id or cfg.base_policy.wt_path is set
    assert (
        sum(
            [
                cfg.base_policy.wandb_id is not None,
                cfg.base_policy.wt_path is not None,
            ]
        )
        == 1
    ), "Exactly one of base_policy.wandb_id or base_policy.wt_path must be set"

    run_state_dict = None

    # Check if we are continuing a run
    run_exists = False
    if cfg.wandb.continue_run_id is not None:
        try:
            run: Run = wandb.Api().run(
                f"{cfg.wandb.project}/{cfg.wandb.continue_run_id}"
            )
            run_exists = True
        except (ValueError, CommError):
            pass

    if run_exists:
        print(f"Continuing run {cfg.wandb.continue_run_id}, {run.name}")

        run_id = cfg.wandb.continue_run_id
        run_path = f"{cfg.wandb.project}/{run_id}"

        # Load the weights from the run
        cfg, wts = get_model_from_api_or_cached(
            run_path, "latest", wandb_mode=cfg.wandb.mode
        )

        # Only the best-SR checkpoint is on wandb; prefer the newest checkpoint on this machine
        local_wts = sorted(
            Path("models", run.name).glob("actor_chkpt_*.pt"), key=os.path.getmtime
        )
        if local_wts:
            wts = str(local_wts[-1])

        # Update the cfg.continue_run_id to the run_id
        cfg.wandb.continue_run_id = run_id

        base_cfg = cfg.base_policy
        merge_base_bc_config_with_root_config(cfg, base_cfg)

        print(f"Loading weights from {wts}")

        run_state_dict = torch.load(wts)

        # Set the best test loss and success rate to the one from the run
        try:
            best_eval_success_rate = run.summary["eval/best_eval_success_rate"]
        except KeyError:
            best_eval_success_rate = run.summary["eval/success_rate"]

        iteration = run.summary["iteration"]
        global_step = run.lastHistoryStep
        sps = run.summary.get("charts/SPS", run.summary.get("training/SPS", 0))
        training_cum_time = sps * global_step
        run_name = run.name

    else:
        global_step = 0
        iteration = 0
        best_eval_success_rate = 0.0
        training_cum_time = 0

        # Load the behavior cloning actor
        if cfg.base_policy.wandb_id is not None:
            base_cfg, base_wts = get_model_from_api_or_cached(
                cfg.base_policy.wandb_id,
                wt_type=cfg.base_policy.wt_type,
                wandb_mode=cfg.wandb.mode,
            )
        elif cfg.base_policy.wt_path is not None:
            base_wts = cfg.base_policy.wt_path
            base_cfg: DictConfig = OmegaConf.create(torch.load(base_wts)["config"])
        else:
            raise ValueError("No base policy provided")

        merge_base_bc_config_with_root_config(cfg, base_cfg)
        cfg.actor_name = f"residual_{cfg.base_policy.actor.name}"

        if cfg.seed is None:
            cfg.seed = random.randint(0, 2**32 - 1)

        # Use the given name for both the WandB run and models/<run_name>, else generate one
        run_name = (
            cfg.wandb.get("name", None)
            or f"{int(time.time())}__{cfg.actor_name}_ppo__{cfg.seed}"
        )

    if "task" not in cfg.env:
        cfg.env.task = "one_leg"

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    gpu_id = cfg.gpu_id
    device = torch.device(f"cuda:{gpu_id}")

    env: gym.Env = get_rl_env(
        gpu_id=gpu_id,
        act_rot_repr=cfg.control.act_rot_repr,
        action_type=cfg.control.control_mode,
        april_tags=False,
        concat_robot_state=True,
        ctrl_mode=cfg.control.controller,
        obs_keys=DEFAULT_STATE_OBS,
        task=cfg.env.task,
        compute_device_id=gpu_id,
        graphics_device_id=gpu_id,
        headless=cfg.headless,
        num_envs=cfg.num_envs,
        observation_space="state",
        randomness=cfg.env.randomness,
        max_env_steps=100_000_000,
        stiffness=cfg.env.get("stiffness", 1000.0),
        damping=cfg.env.get("damping", 200.0),
    )

    n_parts_to_assemble = env.n_parts_assemble

    if cfg.base_policy.actor.name == "diffusion":
        agent = ResidualDiffusionPolicy(device, base_cfg)
    elif cfg.base_policy.actor.name == "mlp":
        agent = ResidualMlpPolicy(device, base_cfg)
    else:
        raise ValueError(f"Unknown actor type: {cfg.base_policy.actor}")

    agent.to(device)
    agent.eval()

    # Set the inference steps of the actor
    if isinstance(agent, DiffusionPolicy):
        agent.inference_steps = 4

    env: RLPolicyEnvWrapper = RLPolicyEnvWrapper(
        env,
        max_env_steps=cfg.num_env_steps,
        normalize_reward=cfg.normalize_reward,
        reset_on_success=cfg.reset_on_success,
        reset_on_failure=cfg.reset_on_failure,
        reward_clip=cfg.clip_reward,
        sample_perturbations=cfg.sample_perturbations,
        torque_penalty_coef=cfg.get("safety", {}).get("torque_penalty_coef", 0.0),
        torque_threshold=cfg.get("safety", {}).get("torque_threshold", 0.5),
        terminate_torque_ratio=cfg.get("safety", {}).get("terminate_torque_ratio", None),
        min_insert_twist_turns=cfg.get("safety", {}).get("min_insert_twist_turns", None),
        min_held_twist_turns=cfg.get("safety", {}).get("min_held_twist_turns", None),
        max_insert_base_move_mm=cfg.get("safety", {}).get("max_insert_base_move_mm", None),
        device=device,
    )

    optimizer_actor = optim.AdamW(
        agent.actor_parameters,
        lr=cfg.learning_rate_actor,
        betas=cfg.get("optimizer_betas_actor", (0.9, 0.999)),
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_actor = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_actor,
        num_warmup_steps=cfg.lr_scheduler.actor_warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    optimizer_critic = optim.AdamW(
        agent.critic_parameters,
        lr=cfg.learning_rate_critic,
        eps=1e-5,
        weight_decay=1e-6,
    )

    lr_scheduler_critic = get_scheduler(
        name=cfg.lr_scheduler.name,
        optimizer=optimizer_critic,
        num_warmup_steps=cfg.lr_scheduler.critic_warmup_steps,
        num_training_steps=cfg.num_iterations,
    )

    if run_state_dict is not None:
        if "actor_logstd" in run_state_dict["model_state_dict"]:
            agent.residual_policy.load_state_dict(run_state_dict["model_state_dict"])
        else:
            agent.load_state_dict(run_state_dict["model_state_dict"])

        optimizer_actor.load_state_dict(run_state_dict["optimizer_actor_state_dict"])
        optimizer_critic.load_state_dict(run_state_dict["optimizer_critic_state_dict"])
        lr_scheduler_actor.load_state_dict(run_state_dict["scheduler_actor_state_dict"])
        lr_scheduler_critic.load_state_dict(
            run_state_dict["scheduler_critic_state_dict"]
        )
    else:
        agent.load_base_state_dict(base_wts)

    residual_policy = agent.residual_policy

    if (
        "pretrained_wts" in cfg.actor.residual_policy
        and cfg.actor.residual_policy.pretrained_wts
    ):
        print(
            f"Loading pretrained weights from {cfg.actor.residual_policy.pretrained_wts}"
        )
        run_state_dict = torch.load(cfg.actor.residual_policy.pretrained_wts)

        if "actor_logstd" in run_state_dict["model_state_dict"]:
            agent.residual_policy.load_state_dict(run_state_dict["model_state_dict"])
        else:
            agent.load_state_dict(run_state_dict["model_state_dict"])

        # Set pretrained_load_optim=false to warm-start the weights with a fresh
        # optimizer and LR schedule (the schedule length comes from this run's config)
        if cfg.actor.residual_policy.get("pretrained_load_optim", True):
            optimizer_actor.load_state_dict(
                run_state_dict["optimizer_actor_state_dict"]
            )
            optimizer_critic.load_state_dict(
                run_state_dict["optimizer_critic_state_dict"]
            )
            lr_scheduler_actor.load_state_dict(
                run_state_dict["scheduler_actor_state_dict"]
            )
            lr_scheduler_critic.load_state_dict(
                run_state_dict["scheduler_critic_state_dict"]
            )
        else:
            print("Using a fresh optimizer and LR schedule for the pretrained weights")

        # The pretrained weights include the log std, which replaces init_logstd
        pretrained_logstd = cfg.actor.residual_policy.get("pretrained_logstd", None)
        if pretrained_logstd is not None:
            with torch.no_grad():
                agent.residual_policy.actor_logstd.fill_(pretrained_logstd)
            print(f"Set the residual policy's log std to {pretrained_logstd}")

    steps_per_iteration = cfg.data_collection_steps

    print(f"Total timesteps: {cfg.total_timesteps}, batch size: {cfg.batch_size}")
    print(
        f"Mini-batch size: {cfg.minibatch_size}, num iterations: {cfg.num_iterations}"
    )

    print(OmegaConf.to_yaml(cfg, resolve=True))

    run = wandb.init(
        id=cfg.wandb.continue_run_id,
        resume=None if cfg.wandb.continue_run_id is None else "allow",
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        save_code=True,
        mode=cfg.wandb.mode if not cfg.debug else "disabled",
    )

    obs: torch.Tensor = torch.zeros(
        (
            steps_per_iteration,
            cfg.num_envs,
            residual_policy.obs_dim,
        )
    )
    actions = torch.zeros((steps_per_iteration, cfg.num_envs) + env.action_space.shape)
    logprobs = torch.zeros((steps_per_iteration, cfg.num_envs))
    rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    dones = torch.zeros((steps_per_iteration, cfg.num_envs))
    values = torch.zeros((steps_per_iteration, cfg.num_envs))

    # Raw task reward (before any penalty or normalization), used for success statistics
    task_rewards = torch.zeros((steps_per_iteration, cfg.num_envs))
    # Safety diagnostics: max joint torque / limit, and the inserted part's rotation (rad)
    torque_ratios = torch.zeros((steps_per_iteration, cfg.num_envs))
    part_rotations = torch.zeros((steps_per_iteration, cfg.num_envs))
    # Envs failed for exceeding safety.terminate_torque_ratio (latched)
    force_fails = torch.zeros((steps_per_iteration, cfg.num_envs), dtype=torch.bool)
    # Envs failed by safety.min_insert_twist_turns: assembled without enough twist (latched)
    twist_fails = torch.zeros((steps_per_iteration, cfg.num_envs), dtype=torch.bool)
    # ... by safety.min_held_twist_turns and safety.max_insert_base_move_mm (latched)
    held_twist_fails = torch.zeros_like(twist_fails)
    base_move_fails = torch.zeros_like(twist_fails)
    # Inserted part's twist so far while held (rad), and the base part's peak horizontal
    # move since the insertion started (m), as the env-side checks see them
    held_twists = torch.zeros((steps_per_iteration, cfg.num_envs))
    peak_base_moves = torch.zeros((steps_per_iteration, cfg.num_envs))
    # Motion diagnostics: inserted part's signed twist about its long axis (rad) and its
    # position in the base part's frame (m); target position, the base policy's target
    # and target - end-effector offset (m)
    part_twists = torch.zeros((steps_per_iteration, cfg.num_envs))
    part_rel_pos = torch.zeros((steps_per_iteration, cfg.num_envs, 3))
    target_pos = torch.zeros((steps_per_iteration, cfg.num_envs, 3))
    base_target_pos = torch.zeros((steps_per_iteration, cfg.num_envs, 3))
    ee_goal_offsets = torch.zeros((steps_per_iteration, cfg.num_envs, 3))
    # Steps where the base policy started a new action chunk
    chunk_starts = torch.zeros(steps_per_iteration, dtype=torch.bool)

    start_time = time.time()

    next_done = torch.zeros(cfg.num_envs)
    next_obs = env.reset()
    agent.reset()

    # Create model save dir
    model_save_dir: Path = Path("models") / wandb.run.name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    while global_step < cfg.total_timesteps:
        iteration += 1
        print(f"Iteration: {iteration}/{cfg.num_iterations}")
        print(f"Run name: {run_name}")
        iteration_start_time = time.time()

        # If eval first flag is set, we will evaluate the model before doing any training
        eval_mode = (iteration - int(cfg.eval_first)) % cfg.eval_interval == 0

        # Also reset the env to have more consistent results
        if eval_mode or cfg.reset_every_iteration:
            next_obs = env.reset()
            agent.reset()

        print(f"Eval mode: {eval_mode}")

        for step in range(0, steps_per_iteration):
            if not eval_mode:
                # Only count environment steps during training
                global_step += cfg.num_envs

            # Get the base normalized action (a new chunk is predicted when the queue is empty)
            # (same check as Actor.action; the queue is None before the first prediction)
            chunk_starts[step] = not agent.actions
            base_naction = agent.base_action_normalized(next_obs)

            # Process the obs for the residual policy
            next_nobs = agent.process_obs(next_obs)
            next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)

            dones[step] = next_done
            obs[step] = next_residual_nobs

            with torch.no_grad():
                residual_naction_samp, logprob, _, value, naction_mean = (
                    residual_policy.get_action_and_value(next_residual_nobs)
                )

            residual_naction = residual_naction_samp if not eval_mode else naction_mean
            naction = base_naction + residual_naction * residual_policy.action_scale

            action = agent.normalizer(naction, "action", forward=False)
            target_pos[step] = action[:, :3].cpu()
            base_target_pos[step] = agent.normalizer(
                base_naction, "action", forward=False
            )[:, :3].cpu()
            next_obs, reward, next_done, truncated, info = env.step(action)

            if cfg.truncation_as_done:
                next_done = next_done | truncated

            values[step] = value.flatten().cpu()
            actions[step] = residual_naction.cpu()
            logprobs[step] = logprob.cpu()
            rewards[step] = reward.view(-1).cpu()
            task_rewards[step] = info["task_reward"].view(-1).cpu()
            if "torque_ratio" in info:
                torque_ratios[step] = info["torque_ratio"].view(-1).cpu()
                part_rotations[step] = info["part_rotation"].view(-1).cpu()
                part_twists[step] = info["part_twist"].view(-1).cpu()
                part_rel_pos[step] = info["part_rel_pos"].cpu()
                ee_goal_offsets[step] = info["ee_goal_offset"].cpu()
            if "force_failed" in info:
                force_fails[step] = info["force_failed"].view(-1).cpu()
            if "twist_failed" in info:
                twist_fails[step] = info["twist_failed"].view(-1).cpu()
            if "held_twist" in info:
                held_twists[step] = info["held_twist"].view(-1).cpu()
                peak_base_moves[step] = info["peak_base_move"].view(-1).cpu()
            if "held_twist_failed" in info:
                held_twist_fails[step] = info["held_twist_failed"].view(-1).cpu()
            if "base_move_failed" in info:
                base_move_fails[step] = info["base_move_failed"].view(-1).cpu()
            next_done = next_done.view(-1).cpu()

            if step > 0 and (env_step := step * 1) % 100 == 0:
                print(
                    f"env_step={env_step}, global_step={global_step}, mean_reward={rewards[:step+1].sum(dim=0).mean().item()} fps={env_step * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
                )

        # Calculate the success rate
        # Use the raw task reward, since penalties and normalization change `rewards`
        # Env is successful if it received a reward more than or equal to n_parts_to_assemble
        env_success = (task_rewards > 0).sum(dim=0) >= n_parts_to_assemble
        success_rate = env_success.float().mean().item()

        # Safety diagnostics over all envs and steps of this rollout
        torque_ratio_mean = torque_ratios.mean().item()
        torque_ratio_p99 = torch.quantile(torque_ratios.flatten(), 0.99).item()
        force_fail_rate = force_fails.any(dim=0).float().mean().item()
        twist_fail_rate = twist_fails.any(dim=0).float().mean().item()
        held_twist_fail_rate = held_twist_fails.any(dim=0).float().mean().item()
        base_move_fail_rate = base_move_fails.any(dim=0).float().mean().item()
        success_part_turns = 0.0
        # Per-episode peak torque ratio of successful episodes (up to the success step),
        # used to pick a force threshold that legit episodes stay below
        peak_tq_thresholds = (0.1, 0.15, 0.2, 0.3)
        peak_tq_stats = {}
        # Twist and target offset during insertion in successful episodes
        twist_thresholds = (1.0, 1.5)
        held_twist_thresholds = (0.9,)
        base_move_thresholds_mm = (20, 30)
        insert_stats = {}

        def second_diff_mm(x):
            return (x[2:] - 2 * x[1:-1] + x[:-2]).norm(dim=-1) * 1000

        # Smoothness over the steps where the env is still running, including failed
        # episodes (second differences in mm, as logged up to R6, for comparing with those
        # runs). The success-only metrics below are the ones to judge a policy by.
        running = dones == 0
        d2_running = running[2:] & running[1:-1] & running[:-2]
        smooth_stats = {
            "target_jerk_mm_allenv": second_diff_mm(target_pos)[d2_running].mean().item(),
            "base_target_jerk_mm_allenv": second_diff_mm(base_target_pos)[d2_running]
            .mean()
            .item(),
            "residual_jerk_mm_allenv": second_diff_mm(target_pos - base_target_pos)[
                d2_running
            ]
            .mean()
            .item(),
            "ee_jerk_mm_allenv": second_diff_mm(target_pos - ee_goal_offsets)[d2_running]
            .mean()
            .item(),
        }

        if success_rate > 0:
            # Calculate the share of timesteps that come from successful trajectories that account for the success rate and the varying number of timesteps per trajectory
            # Count total timesteps in successful trajectories
            timesteps_in_success = task_rewards[:, env_success]

            # Find index of last reward in each trajectory
            # This has all timesteps including and after episode is done
            success_dones = timesteps_in_success.cumsum(dim=0) >= n_parts_to_assemble
            last_reward_idx = success_dones.int().argmax(dim=0)

            # Calculate the total number of timesteps in successful trajectories
            total_timesteps_in_success = (last_reward_idx + 1).sum().item()

            # Calculate the share of successful timesteps
            success_timesteps_share = total_timesteps_in_success / rewards.numel()

            # Mean successful episode length
            mean_success_episode_length = (
                total_timesteps_in_success / env_success.sum().item()
            )
            max_success_episode_length = last_reward_idx.max().item()

            # Rotation of the inserted part from reset up to the success step, in turns
            rotation_to_success = part_rotations[:, env_success].cumsum(dim=0)
            rotation_to_success = rotation_to_success.gather(
                0, last_reward_idx.unsqueeze(0)
            ).squeeze(0)
            success_part_turns = (rotation_to_success / (2 * np.pi)).mean().item()

            before_success = (
                torch.arange(steps_per_iteration).unsqueeze(1)
                <= last_reward_idx.unsqueeze(0)
            )
            peak_tq = (torque_ratios[:, env_success] * before_success).max(dim=0).values
            for q in (0.5, 0.9, 0.99):
                peak_tq_stats[f"peak_torque_ratio_p{int(q * 100)}"] = torch.quantile(
                    peak_tq, q
                ).item()
            for thr in peak_tq_thresholds:
                peak_tq_stats[f"peak_torque_ratio_above_{thr}"] = (
                    (peak_tq > thr).float().mean().item()
                )

            # Insertion: steps up to success where the part is within 3 cm of its final
            # position relative to the base part (the threads span the last ~2 cm)
            rel_pos = part_rel_pos[:, env_success]
            final_rel_pos = rel_pos.gather(
                0, last_reward_idx.view(1, -1, 1).expand(1, -1, 3)
            )
            inserting = before_success & (
                (rel_pos - final_rel_pos).norm(dim=-1) < 0.03
            )
            twists = part_twists[:, env_success]
            insert_turns = (twists * inserting).sum(dim=0) / (2 * np.pi)
            net_turns = (twists * before_success).sum(dim=0) / (2 * np.pi)
            for q in (0.1, 0.5, 0.9):
                insert_stats[f"insert_twist_turns_p{int(q * 100)}"] = torch.quantile(
                    insert_turns, q
                ).item()
            for thr in twist_thresholds:
                insert_stats[f"insert_twist_abs_above_{thr}"] = (
                    (insert_turns.abs() >= thr).float().mean().item()
                )
            insert_stats["net_twist_turns_p50"] = torch.quantile(net_turns, 0.5).item()
            insert_stats["insert_steps"] = inserting.sum(dim=0).float().mean().item()

            # Held twist and base part move at the success step, as the env-side checks
            # (safety.min_held_twist_turns, safety.max_insert_base_move_mm) see them
            if env.track_insertion:
                success_idx = last_reward_idx.unsqueeze(0)
                held_turns = (
                    held_twists[:, env_success].gather(0, success_idx).squeeze(0).abs()
                    / (2 * np.pi)
                )
                base_move_mm = (
                    peak_base_moves[:, env_success].gather(0, success_idx).squeeze(0) * 1000
                )
                for q in (0.1, 0.5, 0.9):
                    insert_stats[f"held_twist_turns_p{int(q * 100)}"] = torch.quantile(
                        held_turns, q
                    ).item()
                for thr in held_twist_thresholds:
                    insert_stats[f"held_twist_below_{thr}"] = (
                        (held_turns < thr).float().mean().item()
                    )
                for q in (0.5, 0.9, 0.99):
                    insert_stats[f"insert_base_move_mm_p{int(q * 100)}"] = torch.quantile(
                        base_move_mm, q
                    ).item()
                for thr in base_move_thresholds_mm:
                    insert_stats[f"insert_base_move_above_{thr}mm"] = (
                        (base_move_mm > thr).float().mean().item()
                    )

            # How far below the end effector the target was placed (positive = below)
            offsets = ee_goal_offsets[:, env_success]
            down_offset = -offsets[..., 2]
            peak_down_mm = (down_offset * inserting).max(dim=0).values * 1000
            for q in (0.5, 0.9, 0.99):
                insert_stats[f"insert_down_offset_mm_p{int(q * 100)}"] = torch.quantile(
                    peak_down_mm, q
                ).item()
            insert_stats["insert_steps_down_offset_above_5mm"] = (
                ((down_offset > 0.005) & inserting).sum(dim=0).float().mean().item()
            )
            insert_stats["offset_mm_p99"] = torch.quantile(
                offsets.norm(dim=-1)[before_success] * 1000, 0.99
            ).item()

            # Smoothness in mm per step: change of the target between steps, by the step's
            # position in the base's action chunk (0 = the base just started a new chunk),
            # and second differences (stop-and-go) of the target, its base and residual
            # parts, and the end-effector position. Failed episodes are left out, since
            # they can jitter in place and would mask the difference between policies.
            def step_change_mm(x):
                return (x[1:] - x[:-1]).norm(dim=-1) * 1000

            succ_target = target_pos[:, env_success]
            succ_base = base_target_pos[:, env_success]
            succ_ee = succ_target - offsets
            d1_valid = before_success[1:]
            d2_valid = before_success[2:]

            steps = torch.arange(steps_per_iteration)
            last_chunk_start = torch.cummax(
                torch.where(chunk_starts, steps, 0), dim=0
            ).values
            chunk_phase = (steps - last_chunk_start)[1:].unsqueeze(1)
            target_step = step_change_mm(succ_target)
            for k in range(int(chunk_phase.max()) + 1):
                smooth_stats[f"target_step_mm_phase{k}"] = (
                    target_step[(chunk_phase == k) & d1_valid].mean().item()
                )
            smooth_stats["target_jerk_mm"] = second_diff_mm(succ_target)[d2_valid].mean().item()
            smooth_stats["base_target_jerk_mm"] = (
                second_diff_mm(succ_base)[d2_valid].mean().item()
            )
            smooth_stats["residual_jerk_mm"] = (
                second_diff_mm(succ_target - succ_base)[d2_valid].mean().item()
            )
            smooth_stats["ee_jerk_mm"] = second_diff_mm(succ_ee)[d2_valid].mean().item()
        else:
            success_timesteps_share = 0
            mean_success_episode_length = 0
            max_success_episode_length = 0

        print(
            f"SR: {success_rate:.4%}, SPS: {steps_per_iteration * cfg.num_envs / (time.time() - iteration_start_time):.2f}"
            f", STS: {success_timesteps_share:.4%}, MSEL: {mean_success_episode_length:.2f}"
            f", TQ p99: {torque_ratio_p99:.3f}, turns: {success_part_turns:.2f}"
            f", force fail: {force_fail_rate:.2%}, twist fail: {twist_fail_rate:.2%}"
            f", held twist fail: {held_twist_fail_rate:.2%}"
            f", base move fail: {base_move_fail_rate:.2%}"
        )
        if peak_tq_stats:
            print(
                "Success peak TQ p50/p90/p99: "
                + "/".join(
                    f"{peak_tq_stats[f'peak_torque_ratio_p{q}']:.3f}" for q in (50, 90, 99)
                )
                + ", share above "
                + ", ".join(
                    f"{thr}: {peak_tq_stats[f'peak_torque_ratio_above_{thr}']:.1%}"
                    for thr in peak_tq_thresholds
                )
            )
        if insert_stats:
            print(
                "Success insertion (within 3 cm): twist turns p10/p50/p90: "
                + "/".join(
                    f"{insert_stats[f'insert_twist_turns_p{q}']:.2f}" for q in (10, 50, 90)
                )
                + ", |twist| >= "
                + ", ".join(
                    f"{thr}: {insert_stats[f'insert_twist_abs_above_{thr}']:.1%}"
                    for thr in twist_thresholds
                )
                + f", net from reset p50: {insert_stats['net_twist_turns_p50']:.2f}"
                + f", steps: {insert_stats['insert_steps']:.0f}"
            )
            print(
                "Success target below EE in insertion, peak mm p50/p90/p99: "
                + "/".join(
                    f"{insert_stats[f'insert_down_offset_mm_p{q}']:.1f}" for q in (50, 90, 99)
                )
                + f", steps > 5 mm: {insert_stats['insert_steps_down_offset_above_5mm']:.1f}"
                + f", |offset| p99 up to success: {insert_stats['offset_mm_p99']:.1f} mm"
            )
        if "held_twist_turns_p50" in insert_stats:
            print(
                "Success held twist turns p10/p50/p90: "
                + "/".join(
                    f"{insert_stats[f'held_twist_turns_p{q}']:.2f}" for q in (10, 50, 90)
                )
                + ", below "
                + ", ".join(
                    f"{thr}: {insert_stats[f'held_twist_below_{thr}']:.1%}"
                    for thr in held_twist_thresholds
                )
                + "; base part move in insertion, peak mm p50/p90/p99: "
                + "/".join(
                    f"{insert_stats[f'insert_base_move_mm_p{q}']:.1f}" for q in (50, 90, 99)
                )
                + ", above "
                + ", ".join(
                    f"{thr} mm: {insert_stats[f'insert_base_move_above_{thr}mm']:.1%}"
                    for thr in base_move_thresholds_mm
                )
            )
        print(
            "All-env smoothness (mm, as up to R6): 2nd diff target/EE: "
            + f"{smooth_stats['target_jerk_mm_allenv']:.2f}/{smooth_stats['ee_jerk_mm_allenv']:.2f}"
            + f", base/residual: {smooth_stats['base_target_jerk_mm_allenv']:.2f}"
            + f"/{smooth_stats['residual_jerk_mm_allenv']:.2f}"
        )
        if "ee_jerk_mm" in smooth_stats:
            print(
                "Success smoothness (mm/step): target step by chunk phase: "
                + " ".join(
                    f"{v:.2f}"
                    for k, v in smooth_stats.items()
                    if k.startswith("target_step_mm_phase")
                )
                + f", 2nd diff target/EE: {smooth_stats['target_jerk_mm']:.2f}/{smooth_stats['ee_jerk_mm']:.2f}"
                + f", base/residual: {smooth_stats['base_target_jerk_mm']:.2f}/{smooth_stats['residual_jerk_mm']:.2f}"
            )

        if eval_mode:
            # If we are in eval mode, we don't need to do any training, so log the result and continue

            # Save the model if the evaluation success rate improves
            if success_rate > best_eval_success_rate:
                best_eval_success_rate = success_rate
                model_path = str(model_save_dir / f"actor_chkpt_best_success_rate.pt")
                torch.save(
                    {
                        # Save the weights of the residual policy (base + residual)
                        "model_state_dict": agent.state_dict(),
                        "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                        "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                        "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                        "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "success_rate": success_rate,
                        "success_timesteps_share": success_timesteps_share,
                        "iteration": iteration,
                        "training_cum_time": training_cum_time,
                    },
                    model_path,
                )

                wandb.save(model_path)
                print(f"Evaluation success rate improved. Model saved to {model_path}")

            wandb.log(
                {
                    "eval/success_rate": success_rate,
                    "eval/best_eval_success_rate": best_eval_success_rate,
                    "eval/mean_success_episode_length": mean_success_episode_length,
                    "eval/torque_ratio_mean": torque_ratio_mean,
                    "eval/torque_ratio_p99": torque_ratio_p99,
                    "eval/success_leg_rotation_turns": success_part_turns,
                    **{f"eval/{k}": v for k, v in peak_tq_stats.items()},
                    **{f"eval/{k}": v for k, v in insert_stats.items()},
                    **{f"eval/{k}": v for k, v in smooth_stats.items()},
                    "eval/force_fail_rate": force_fail_rate,
                    "eval/twist_fail_rate": twist_fail_rate,
                    "eval/held_twist_fail_rate": held_twist_fail_rate,
                    "eval/base_move_fail_rate": base_move_fail_rate,
                    "iteration": iteration,
                },
                step=global_step,
            )
            # Start the data collection again
            # NOTE: We're not resetting here now, that happens before the next
            # iteration only if the reset_every_iteration flag is set
            continue

        b_obs = obs.reshape((-1, residual_policy.obs_dim))
        b_actions = actions.reshape((-1,) + env.action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)

        # Get the base normalized action
        # Process the obs for the residual policy
        base_naction = agent.base_action_normalized(next_obs)
        next_nobs = agent.process_obs(next_obs)
        next_residual_nobs = torch.cat([next_nobs, base_naction], dim=-1)
        next_value = residual_policy.get_value(next_residual_nobs).reshape(1, -1).cpu()

        # bootstrap value if not done
        advantages, returns = calculate_advantage(
            values,
            next_value,
            rewards,
            dones,
            next_done,
            steps_per_iteration,
            cfg.discount,
            cfg.gae_lambda,
        )

        b_advantages = advantages.reshape(-1).cpu()
        b_returns = returns.reshape(-1).cpu()

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []

        # For the smoothness loss: the middle steps of three consecutive steps within one
        # episode, as indices into b_obs (step t of env n is t * num_envs + n)
        if cfg.residual_smooth > 0:
            same_episode = (dones[1:-1] == 0) & (dones[2:] == 0)
            smooth_mid_inds = torch.nonzero(same_episode.flatten()).squeeze(1) + cfg.num_envs
        for epoch in trange(cfg.update_epochs, desc="Policy update"):
            early_stop = False

            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                # Get the minibatch and place it on the device
                mb_obs = b_obs[mb_inds].to(device)
                mb_actions = b_actions[mb_inds].to(device)
                mb_logprobs = b_logprobs[mb_inds].to(device)
                mb_advantages = b_advantages[mb_inds].to(device)
                mb_returns = b_returns[mb_inds].to(device)
                mb_values = b_values[mb_inds].to(device)

                # Calculate the loss
                _, newlogprob, entropy, newvalue, action_mean = (
                    residual_policy.get_action_and_value(mb_obs, mb_actions)
                )
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                policy_loss = 0

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean() * cfg.ent_coef

                ppo_loss = pg_loss - entropy_loss

                # Add the auxiliary regularization loss
                residual_l1_loss = torch.mean(torch.abs(action_mean))
                residual_l2_loss = torch.mean(torch.square(action_mean))

                # Smoothness loss: squared second difference of the target (base action +
                # residual mean) over three consecutive steps, for position and rotation
                # (not the gripper), in residual units (normalized action / action_scale).
                # It uses the mean, so the exploration noise doesn't enter.
                if cfg.residual_smooth > 0:
                    mid = smooth_mid_inds[
                        torch.randint(len(smooth_mid_inds), (cfg.minibatch_size // 16,))
                    ]
                    triplet_obs = b_obs[
                        torch.cat([mid - cfg.num_envs, mid, mid + cfg.num_envs])
                    ].to(device)
                    n_act = b_actions.shape[-1]
                    # The residual policy's observation ends with the normalized base action
                    triplet_target = (
                        triplet_obs[:, -n_act:] / residual_policy.action_scale
                        + residual_policy.actor_mean(triplet_obs)
                    )[:, :-1]
                    prev, cur, nxt = triplet_target.chunk(3)
                    residual_smooth_loss = torch.mean(torch.square(nxt - 2 * cur + prev))
                else:
                    residual_smooth_loss = torch.zeros((), device=device)

                # Normalize the losses so that each term has the same scale
                if iteration > cfg.n_iterations_train_only_value:

                    # Scale the losses using the calculated scaling factors
                    policy_loss += ppo_loss
                    policy_loss += cfg.residual_l1 * residual_l1_loss
                    policy_loss += cfg.residual_l2 * residual_l2_loss
                    policy_loss += cfg.residual_smooth * residual_smooth_loss

                # Total loss
                loss: torch.Tensor = policy_loss + v_loss * cfg.vf_coef

                optimizer_actor.zero_grad()
                optimizer_critic.zero_grad()

                loss.backward()
                nn.utils.clip_grad_norm_(
                    residual_policy.parameters(), cfg.max_grad_norm
                )

                optimizer_actor.step()
                optimizer_critic.step()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    print(
                        f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl:.4f} > {cfg.target_kl:.4f}"
                    )
                    early_stop = True
                    break

            if early_stop:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        action_norms = torch.norm(b_actions[:, :3], dim=-1).cpu()

        training_cum_time += time.time() - iteration_start_time
        sps = int(global_step / training_cum_time) if training_cum_time > 0 else 0

        wandb.log(
            {
                "training/learning_rate_actor": optimizer_actor.param_groups[0]["lr"],
                "training/learning_rate_critic": optimizer_critic.param_groups[0]["lr"],
                "training/SPS": sps,
                "charts/rewards": rewards.sum().item(),
                "charts/success_rate": success_rate,
                "charts/success_timesteps_share": success_timesteps_share,
                "charts/mean_success_episode_length": mean_success_episode_length,
                "charts/max_success_episode_length": max_success_episode_length,
                "charts/torque_ratio_mean": torque_ratio_mean,
                "charts/torque_ratio_p99": torque_ratio_p99,
                "charts/success_leg_rotation_turns": success_part_turns,
                **{f"charts/{k}": v for k, v in peak_tq_stats.items()},
                **{f"charts/{k}": v for k, v in insert_stats.items()},
                **{f"charts/{k}": v for k, v in smooth_stats.items()},
                "charts/force_fail_rate": force_fail_rate,
                "charts/twist_fail_rate": twist_fail_rate,
                "charts/held_twist_fail_rate": held_twist_fail_rate,
                "charts/base_move_fail_rate": base_move_fail_rate,
                "charts/action_norm_mean": action_norms.mean(),
                "charts/action_norm_std": action_norms.std(),
                "values/advantages": b_advantages.mean().item(),
                "values/returns": b_returns.mean().item(),
                "values/values": b_values.mean().item(),
                "values/mean_logstd": residual_policy.actor_logstd.mean().item(),
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/total_loss": loss.item(),
                "losses/entropy_loss": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "losses/explained_variance": explained_var,
                "losses/residual_l1": residual_l1_loss.item(),
                "losses/residual_l2": residual_l2_loss.item(),
                "losses/residual_smooth": residual_smooth_loss.item(),
                "histograms/values": wandb.Histogram(values),
                "histograms/returns": wandb.Histogram(b_returns),
                "histograms/advantages": wandb.Histogram(b_advantages),
                "histograms/logprobs": wandb.Histogram(logprobs),
                "histograms/rewards": wandb.Histogram(rewards),
                "histograms/action_norms": wandb.Histogram(action_norms),
            },
            step=global_step,
        )

        # Step the learning rate scheduler
        lr_scheduler_actor.step()
        lr_scheduler_critic.step()

        # Checkpoint every cfg.checkpoint_interval steps (kept locally, not uploaded to wandb)
        if cfg.checkpoint_interval > 0 and iteration % cfg.checkpoint_interval == 0:
            model_path = str(model_save_dir / f"actor_chkpt_{iteration}.pt")
            torch.save(
                {
                    "model_state_dict": agent.state_dict(),
                    "optimizer_actor_state_dict": optimizer_actor.state_dict(),
                    "optimizer_critic_state_dict": optimizer_critic.state_dict(),
                    "scheduler_actor_state_dict": lr_scheduler_actor.state_dict(),
                    "scheduler_critic_state_dict": lr_scheduler_critic.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "success_rate": success_rate,
                    "iteration": iteration,
                    "training_cum_time": training_cum_time,
                },
                model_path,
            )

            print(f"Model saved to {model_path}")

        # Print some stats at the end of the iteration
        print(
            f"Iteration {iteration}/{cfg.num_iterations}, global step {global_step}, SPS {sps}"
        )

    print(f"Training finished in {(time.time() - start_time):.2f}s")


if __name__ == "__main__":
    main()
