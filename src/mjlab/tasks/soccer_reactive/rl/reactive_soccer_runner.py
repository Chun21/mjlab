"""Paper-style runner wiring for reactive soccer."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from mjlab.tasks.soccer_reactive.models.history_encoder_actor_critic import (
  ReactiveSoccerActorCritic,
)
from mjlab.tasks.soccer_reactive.models.symmetry import SoccerSymmetry
from mjlab.tasks.soccer_reactive.rl.amp_discriminator import AmpDiscriminator
from mjlab.tasks.soccer_reactive.rl.amp_motion_loader import ReactiveSoccerAmpMotionLoader
from mjlab.tasks.soccer_reactive.rl.amp_normalizer import ReactiveSoccerAmpNormalizer
from mjlab.tasks.soccer_reactive.rl.reactive_soccer_ppo import ReactiveSoccerPPO


class ReactiveSoccerRunner:
  """Runner that integrates paper-style PPO, decoder targets and AMP."""

  GOAL_REWARD_TERMS = frozenset({"goal_scored", "ball_approach", "goal_progress"})
  EXTERNAL_AUX_REWARD_TERMS = frozenset({"amp_style"})

  def __init__(
    self,
    env,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
    **_: Any,
  ) -> None:
    self.env = env
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir
    self.current_learning_iteration = 0
    self.rank = int(os.environ.get("RANK", "0"))
    self.logger = SimpleNamespace(save_model=lambda *args, **kwargs: None)
    self.tb_log_dir = Path(log_dir) / "tensorboard" if log_dir is not None else None
    use_tensorboard = (
      self.tb_log_dir is not None
      and train_cfg.get("logger", "tensorboard") == "tensorboard"
      and self.rank == 0
    )
    self.tb_writer = SummaryWriter(log_dir=str(self.tb_log_dir)) if use_tensorboard else None
    if self.tb_writer is not None:
      print(f"[INFO] TensorBoard logging to: {self.tb_log_dir}")

    actor_current_dim = self._flat_dim(
      env.unwrapped.observation_manager.group_obs_dim["actor_current"]
    )
    actor_history_dim = env.unwrapped.observation_manager.group_obs_dim["actor_history"][-1]
    history_steps = env.unwrapped.observation_manager.group_obs_dim["actor_history"][0]
    critic_current_dim = self._flat_dim(
      env.unwrapped.observation_manager.group_obs_dim["critic_current"]
    )
    privileged_dim = self._flat_dim(
      env.unwrapped.observation_manager.group_obs_dim["critic_privileged"]
    )
    action_dim = env.unwrapped.action_manager.total_action_dim

    self.actor_critic = ReactiveSoccerActorCritic(
      actor_obs_dim=actor_current_dim,
      history_obs_dim=actor_history_dim,
      history_steps=history_steps,
      action_dim=action_dim,
      privileged_dim=privileged_dim,
      reconstruct_dim=privileged_dim,
    ).to(device)

    algorithm_cfg = train_cfg.get("algorithm", {})
    self.symmetry_helper = SoccerSymmetry.for_g1_reactive_soccer()
    self.alg = ReactiveSoccerPPO.construct_algorithm(
      device=device,
      actor_critic=self.actor_critic,
      action_dim=action_dim,
      learning_rate=float(algorithm_cfg.get("learning_rate", 1.0e-3)),
      gamma=float(algorithm_cfg.get("gamma", 0.995)),
      lam=float(algorithm_cfg.get("lam", 0.95)),
      clip_param=float(algorithm_cfg.get("clip_param", 0.2)),
      value_loss_coef=float(algorithm_cfg.get("value_loss_coef", 1.0)),
      entropy_coef=float(algorithm_cfg.get("entropy_coef", 0.01)),
      desired_kl=float(algorithm_cfg.get("desired_kl", 0.01)),
      num_learning_epochs=int(algorithm_cfg.get("num_learning_epochs", 5)),
      num_mini_batches=int(algorithm_cfg.get("num_mini_batches", 4)),
      reconstruction_coef=1.0,
      symmetry_coef=10.0,
      schedule=str(algorithm_cfg.get("schedule", "adaptive")),
      max_grad_norm=float(algorithm_cfg.get("max_grad_norm", 1.0)),
      symmetry_helper=self.symmetry_helper,
    )
    self.alg.initialize_storage(
      num_envs=env.num_envs,
      num_steps=train_cfg.get("num_steps_per_env", 4),
      actor_obs_shape=(actor_current_dim,),
      critic_obs_shape=(critic_current_dim,),
      action_shape=(action_dim,),
      reconstruction_shape=(privileged_dim,),
      actor_history_shape=tuple(
        env.unwrapped.observation_manager.group_obs_dim["actor_history"]
      ),
      privileged_shape=(privileged_dim,),
    )

    motion_root = Path("data/soccer_amp/motions_unified")
    self.amp_loader = ReactiveSoccerAmpMotionLoader(motion_root=motion_root, device=device)
    self.amp_normalizer = ReactiveSoccerAmpNormalizer(
      feature_dim=self.amp_loader.transition_dim,
      device=device,
    )
    self.amp_discriminator = AmpDiscriminator(
      input_dim=self.amp_loader.transition_dim,
      hidden_dims=(256, 256, 128),
    ).to(device)
    self.amp_optimizer = torch.optim.Adam(self.amp_discriminator.parameters(), lr=1.0e-3)
    self.amp_gradient_penalty_coef = 50.0
    self.amp_reward_weight = float(
      self.env.unwrapped.reward_manager.get_term_cfg("amp_style").weight
    )
    setattr(
      self.env.unwrapped,
      "_soccer_reactive_amp_style_reward",
      torch.zeros(self.env.num_envs, device=self.device),
    )

  def _apply_training_curriculum(self, iteration: int) -> None:
    curriculum = getattr(self.env.unwrapped.cfg, "training_curriculum", None)
    if curriculum is None:
      return
    if iteration < curriculum.phase1_end_iter:
      pose_range = curriculum.phase1_ball_pose_range
    elif iteration < curriculum.phase2_end_iter:
      pose_range = curriculum.phase2_ball_pose_range
    else:
      pose_range = curriculum.phase3_ball_pose_range
    self.env.unwrapped.event_manager.get_term_cfg("reset_ball").params["pose_range"] = pose_range
    self.env.unwrapped.event_manager.get_term_cfg("reset_ball_only").params["pose_range"] = pose_range

  @staticmethod
  def _flat_dim(shape: tuple[int, ...] | list[tuple[int, ...]]) -> int:
    if isinstance(shape, list):
      total = 0
      for term_shape in shape:
        prod = 1
        for dim in term_shape:
          prod *= dim
        total += prod
      return total
    prod = 1
    for dim in shape:
      prod *= dim
    return prod

  def _compose_log_dict(
    self,
    update_dict: dict[str, float],
    amp_reward: float,
  ) -> dict[str, float]:
    return {
      "surrogate": float(update_dict.get("surrogate", 0.0)),
      "goal_value": float(update_dict.get("goal_value", 0.0)),
      "aux_value": float(update_dict.get("aux_value", 0.0)),
      "reconstruction": float(update_dict.get("reconstruction", 0.0)),
      "symmetry": float(update_dict.get("symmetry", 0.0)),
      "amp_discriminator": float(update_dict.get("amp_discriminator", 0.0)),
      "amp_reward": float(amp_reward),
    }

  def add_git_repo_to_log(self, *_args: Any, **_kwargs: Any) -> None:
    return None

  def _write_tensorboard_scalars(
    self,
    iteration: int,
    log_dict: dict[str, float],
  ) -> None:
    if self.tb_writer is None:
      return
    for key in (
      "surrogate",
      "goal_value",
      "aux_value",
      "reconstruction",
      "symmetry",
      "amp_discriminator",
      "amp_reward",
    ):
      if key in log_dict:
        self.tb_writer.add_scalar(f"train/{key}", float(log_dict[key]), iteration)

  def _should_log_iteration(self, iteration: int, total_iterations: int) -> bool:
    if self.rank != 0:
      return False
    interval = int(self.cfg.get("stdout_log_interval", 1) or 1)
    if interval <= 0:
      interval = 1
    return iteration == 1 or iteration == total_iterations or iteration % interval == 0

  @staticmethod
  def _format_duration(duration_s: float) -> str:
    duration_s = max(float(duration_s), 0.0)
    if duration_s < 60.0:
      return f"{duration_s:.1f}s"
    total_seconds = int(round(duration_s))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
      return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"

  def _print_iteration_started(self, iteration: int, total_iterations: int) -> None:
    if not self._should_log_iteration(iteration, total_iterations):
      return
    rollout_steps = int(self.cfg.get("num_steps_per_env", 4))
    print(
      f"[TRAIN] iteration {iteration}/{total_iterations} started "
      f"(envs={self.env.num_envs}, rollout_steps={rollout_steps})",
      flush=True,
    )

  def _print_iteration_finished(
    self,
    iteration: int,
    total_iterations: int,
    log_dict: dict[str, float],
    episode_log_stats: dict[str, float],
    *,
    total_timesteps: int,
    iteration_time_s: float,
    total_time_s: float,
    eta_s: float,
  ) -> None:
    if not self._should_log_iteration(iteration, total_iterations):
      return
    separator = "-" * 80
    print(separator, flush=True)
    for key in sorted(episode_log_stats):
      print(f"Mean episode {key}: {episode_log_stats[key]:.4f}", flush=True)
    for key in (
      "surrogate",
      "goal_value",
      "aux_value",
      "reconstruction",
      "symmetry",
      "amp_discriminator",
      "amp_reward",
    ):
      if key in log_dict:
        print(f"Mean step {key}: {log_dict[key]:.4f}", flush=True)
    print(separator, flush=True)
    print(f"{'Total timesteps:':>40} {total_timesteps}", flush=True)
    print(
      f"{'Iteration time:':>40} {self._format_duration(iteration_time_s)}",
      flush=True,
    )
    print(
      f"{'Total time:':>40} {self._format_duration(total_time_s)}",
      flush=True,
    )
    print(f"{'ETA:':>40} {self._format_duration(eta_s)}", flush=True)

  @staticmethod
  def _accumulate_episode_log(
    totals: dict[str, float],
    counts: dict[str, int],
    extras: dict[str, Any] | None,
  ) -> None:
    if extras is None:
      return
    log_payload = extras.get("log")
    if not isinstance(log_payload, dict):
      return
    for key, value in log_payload.items():
      if isinstance(value, torch.Tensor):
        scalar_value = float(value.detach().float().mean().item())
      else:
        scalar_value = float(value)
      totals[key] = totals.get(key, 0.0) + scalar_value
      counts[key] = counts.get(key, 0) + 1

  @staticmethod
  def _mean_episode_log(
    totals: dict[str, float],
    counts: dict[str, int],
  ) -> dict[str, float]:
    return {
      key: totals[key] / max(counts.get(key, 1), 1)
      for key in totals
    }

  def _checkpoint_paths(self) -> list[Path]:
    if self.log_dir is None:
      return []
    return sorted(
      Path(self.log_dir).glob("model_*.pt"),
      key=lambda path: int(path.stem.split("_")[-1]),
    )

  def _prune_old_checkpoints(self) -> None:
    if self.log_dir is None:
      return
    keep_last = int(self.cfg.get("checkpoint_keep_last", 0) or 0)
    if keep_last <= 0:
      return
    checkpoint_paths = self._checkpoint_paths()
    stale_paths = checkpoint_paths[:-keep_last]
    for stale_path in stale_paths:
      stale_path.unlink(missing_ok=True)

  def _run_periodic_eval(self, iteration: int, checkpoint_path: Path | None = None) -> None:
    if self.log_dir is None or self.rank != 0:
      return
    eval_interval = int(self.cfg.get("eval_interval", 0) or 0)
    if eval_interval <= 0 or iteration % eval_interval != 0:
      return

    from mjlab.tasks.soccer_reactive.scripts.play import ReactiveSoccerEvalArgs, run_play

    eval_dir = Path(self.log_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    temp_checkpoint_path: Path | None = None
    if checkpoint_path is None:
      temp_checkpoint_path = eval_dir / f"model_eval_{iteration}.pt"
      self.save(str(temp_checkpoint_path))
      checkpoint_path = temp_checkpoint_path

    args = ReactiveSoccerEvalArgs(
      agent="trained",
      checkpoint_file=checkpoint_path,
      metrics_json=eval_dir / f"metrics_{iteration}.json",
      num_episodes=int(self.cfg.get("eval_num_episodes", 16) or 16),
      device=self.device,
      num_envs=int(self.cfg.get("eval_num_envs", min(self.env.num_envs, 32)) or min(self.env.num_envs, 32)),
    )
    run_play(args)
    if temp_checkpoint_path is not None:
      temp_checkpoint_path.unlink(missing_ok=True)

  def save(self, path: str, infos=None) -> None:
    state = {
      "iter": self.current_learning_iteration,
      "algo": self.alg.save(),
      "amp_discriminator": self.amp_discriminator.state_dict(),
      "amp_optimizer": self.amp_optimizer.state_dict(),
      "amp_normalizer": {
        "count": self.amp_normalizer.count,
        "mean": self.amp_normalizer.mean,
        "var": self.amp_normalizer.var,
      },
      "infos": infos or {},
    }
    torch.save(state, path)

  def load(
    self,
    path: str,
    load_cfg=None,
    strict: bool = True,
    map_location: str | None = None,
  ):
    del load_cfg, strict
    state = torch.load(path, map_location=map_location or self.device, weights_only=False)
    self.current_learning_iteration = state.get("iter", 0)
    self.alg.load(state.get("algo", {}))
    if "amp_discriminator" in state:
      self.amp_discriminator.load_state_dict(state["amp_discriminator"])
    if "amp_optimizer" in state:
      self.amp_optimizer.load_state_dict(state["amp_optimizer"])
    if "amp_normalizer" in state:
      normalizer_state = state["amp_normalizer"]
      self.amp_normalizer.count = normalizer_state["count"]
      self.amp_normalizer.mean = normalizer_state["mean"]
      self.amp_normalizer.var = normalizer_state["var"]
    return state.get("infos", {})

  def _tensor_dict_to_batch(
    self,
    obs: TensorDict | dict[str, torch.Tensor],
  ) -> dict[str, torch.Tensor]:
    if isinstance(obs, TensorDict):
      return {k: obs[k] for k in obs.keys()}
    return dict(obs)

  def get_inference_policy(
    self,
    device: str | None = None,
  ) -> Callable[[TensorDict | dict[str, torch.Tensor]], torch.Tensor]:
    policy_device = device or self.device

    def _policy(obs: TensorDict | dict[str, torch.Tensor]) -> torch.Tensor:
      batch = self._tensor_dict_to_batch(obs)
      model_batch = {
        "actor_current": batch["actor_current"].to(policy_device),
        "critic_current": batch["critic_current"].to(policy_device),
        "actor_history": batch["actor_history"].to(policy_device),
        "critic_privileged": batch["critic_privileged"].to(policy_device),
      }
      with torch.no_grad():
        outputs = self.actor_critic.forward_train(model_batch)
      return outputs["actions_mean"]

    return _policy

  def _split_reward_groups(self) -> tuple[torch.Tensor, torch.Tensor]:
    reward_manager = self.env.unwrapped.reward_manager
    term_names = list(reward_manager.active_terms)
    step_rewards = reward_manager._step_reward

    goal_indices = [
      idx for idx, name in enumerate(term_names) if name in self.GOAL_REWARD_TERMS
    ]
    aux_indices = [
      idx
      for idx, name in enumerate(term_names)
      if name not in self.GOAL_REWARD_TERMS and name not in self.EXTERNAL_AUX_REWARD_TERMS
    ]

    goal_reward = (
      step_rewards[:, goal_indices].sum(dim=1)
      if goal_indices
      else torch.zeros(self.env.num_envs, device=self.device)
    )
    aux_reward = (
      step_rewards[:, aux_indices].sum(dim=1)
      if aux_indices
      else torch.zeros(self.env.num_envs, device=self.device)
    )
    return goal_reward, aux_reward

  @staticmethod
  def _build_policy_amp_transition(
    current_amp_obs: torch.Tensor,
    next_amp_obs: torch.Tensor,
  ) -> torch.Tensor:
    joint_pos_t = current_amp_obs[:, :23]
    joint_vel_t = current_amp_obs[:, 23:46]
    joint_pos_t1 = next_amp_obs[:, :23]
    joint_vel_t1 = next_amp_obs[:, 23:46]
    return torch.cat((joint_pos_t, joint_vel_t, joint_pos_t1, joint_vel_t1), dim=-1)

  def _compute_amp_reward(self, policy_transition: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      normalized_transition = self.amp_normalizer.normalize(policy_transition)
      scores = self.amp_discriminator(normalized_transition)
      return AmpDiscriminator.adversarial_reward(scores)

  def _update_amp_discriminator(
    self,
    policy_transitions: list[torch.Tensor],
  ) -> dict[str, float]:
    if not policy_transitions:
      return {"amp_discriminator": 0.0, "amp_reward": 0.0}

    policy_batch = torch.cat(policy_transitions, dim=0)
    expert_batch = self.amp_loader.sample_transition_batch(batch_size=policy_batch.shape[0])
    self.amp_normalizer.update(torch.cat((expert_batch, policy_batch), dim=0))
    expert_norm = self.amp_normalizer.normalize(expert_batch)
    policy_norm = self.amp_normalizer.normalize(policy_batch)
    loss_dict = self.amp_discriminator.discriminator_loss(
      expert_norm,
      policy_norm,
      gradient_penalty_coef=self.amp_gradient_penalty_coef,
    )
    self.amp_optimizer.zero_grad(set_to_none=True)
    loss_dict["total_loss"].backward()
    self.amp_optimizer.step()
    amp_reward = AmpDiscriminator.adversarial_reward(loss_dict["policy_scores"]).mean()
    return {
      "amp_discriminator": float(loss_dict["total_loss"].detach().item()),
      "amp_reward": float(amp_reward.detach().item()),
    }

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True) -> None:
    del init_at_random_ep_len
    obs, _ = self.env.reset()
    learn_start_time = time.perf_counter()
    for _ in range(num_learning_iterations):
      iteration_start_time = time.perf_counter()
      next_iteration = self.current_learning_iteration + 1
      self._print_iteration_started(next_iteration, num_learning_iterations)
      self._apply_training_curriculum(self.current_learning_iteration)
      assert self.alg.storage is not None
      self.alg.storage.reset()
      rollout_steps = int(self.cfg.get("num_steps_per_env", 4))
      policy_transitions: list[torch.Tensor] = []
      amp_reward_running = 0.0
      episode_log_totals: dict[str, float] = {}
      episode_log_counts: dict[str, int] = {}

      for _rollout_idx in range(rollout_steps):
        batch = self._tensor_dict_to_batch(obs)
        action_dict = self.alg.act(
          {
            "actor_current": batch["actor_current"],
            "critic_current": batch["critic_current"],
            "actor_history": batch["actor_history"],
            "critic_privileged": batch["critic_privileged"],
          }
        )
        next_obs, rewards, dones, extras = self.env.step(action_dict["actions"])
        self._last_step_extras = extras
        self._accumulate_episode_log(episode_log_totals, episode_log_counts, extras)
        next_batch = self._tensor_dict_to_batch(next_obs)

        policy_transition = self._build_policy_amp_transition(
          batch["amp"],
          next_batch["amp"],
        )
        policy_transitions.append(policy_transition.detach())
        amp_reward = self._compute_amp_reward(policy_transition).squeeze(-1)
        amp_reward_running += float(amp_reward.mean().item())
        setattr(
          self.env.unwrapped,
          "_soccer_reactive_amp_style_reward",
          amp_reward.detach(),
        )

        goal_rewards, aux_rewards = self._split_reward_groups()
        aux_rewards = aux_rewards + self.amp_reward_weight * amp_reward
        self.alg.process_env_step(
          {
            "actor_current": batch["actor_current"],
            "critic_current": batch["critic_current"],
            "actor_history": batch["actor_history"],
            "critic_privileged": batch["critic_privileged"],
          },
          {"goal": goal_rewards, "aux": aux_rewards, "total": rewards},
          dones,
          extras,
        )
        obs = next_obs

      self.alg.compute_returns(self._tensor_dict_to_batch(obs))
      update_dict = self.alg.update()
      amp_update = self._update_amp_discriminator(policy_transitions)
      update_dict["amp_discriminator"] = amp_update["amp_discriminator"]
      self._latest_logs = self._compose_log_dict(
        update_dict,
        amp_reward=amp_update["amp_reward"]
        if policy_transitions
        else amp_reward_running / max(rollout_steps, 1),
      )
      self.current_learning_iteration += 1
      save_interval = int(self.cfg.get("save_interval", 0) or 0)
      if self.log_dir is not None:
        self._write_tensorboard_scalars(
          self.current_learning_iteration, self._latest_logs
        )
        saved_checkpoint_path: Path | None = None
        if save_interval > 0 and self.current_learning_iteration % save_interval == 0:
          saved_checkpoint_path = Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"
          self.save(str(saved_checkpoint_path))
          self._prune_old_checkpoints()
        self._run_periodic_eval(
          self.current_learning_iteration,
          checkpoint_path=saved_checkpoint_path,
        )
      total_timesteps = self.current_learning_iteration * self.env.num_envs * rollout_steps
      iteration_time_s = time.perf_counter() - iteration_start_time
      total_time_s = time.perf_counter() - learn_start_time
      average_iteration_time_s = total_time_s / max(self.current_learning_iteration, 1)
      remaining_iterations = max(num_learning_iterations - self.current_learning_iteration, 0)
      eta_s = average_iteration_time_s * remaining_iterations
      self._print_iteration_finished(
        self.current_learning_iteration,
        num_learning_iterations,
        self._latest_logs,
        self._mean_episode_log(episode_log_totals, episode_log_counts),
        total_timesteps=total_timesteps,
        iteration_time_s=iteration_time_s,
        total_time_s=total_time_s,
        eta_s=eta_s,
      )

  def close(self) -> None:
    if self.tb_writer is not None:
      self.tb_writer.flush()
      self.tb_writer.close()
