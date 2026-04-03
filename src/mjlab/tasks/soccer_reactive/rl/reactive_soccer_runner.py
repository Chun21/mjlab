"""Paper-style runner wiring for reactive soccer."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from mjlab.tasks.soccer_reactive.models.history_encoder_actor_critic import (
  ReactiveSoccerActorCritic,
)
from mjlab.tasks.soccer_reactive.models.symmetry import SoccerSymmetry
from mjlab.tasks.soccer_reactive.odometry_artifacts import (
  ODOMETRY_INFO_KEY,
  build_odometry_info_from_env,
)
from mjlab.tasks.soccer_reactive.rl.amp_discriminator import AmpDiscriminator
from mjlab.tasks.soccer_reactive.rl.amp_motion_loader import ReactiveSoccerAmpMotionLoader
from mjlab.tasks.soccer_reactive.rl.amp_normalizer import ReactiveSoccerAmpNormalizer
from mjlab.tasks.soccer_reactive.rl.reactive_soccer_ppo import ReactiveSoccerPPO


class ReactiveSoccerRunner:
  """Runner that integrates paper-style PPO, decoder targets and AMP."""

  GOAL_REWARD_TERMS = frozenset(
    {
      "goal_scored",
      "ball_approach",
      "goal_progress",
    }
  )
  EXTERNAL_AUX_REWARD_TERMS = frozenset({"amp_style"})
  OBS_CLIP = 100.0
  AMP_TRANSITION_CLIP = 10.0
  VALUE_ABS_WARN_THRESHOLD = 1.0e4
  RETURN_ABS_WARN_THRESHOLD = 1.0e4
  REDUCE_MAX_KEYS = frozenset(
    {
      "goal_value_abs_max",
      "aux_value_abs_max",
      "goal_return_abs_max",
      "aux_return_abs_max",
      "action_abs_max",
      "reward_abs_max",
      "amp_transition_abs_max",
    }
  )

  @staticmethod
  def _activation_from_cfg(name: str | None) -> type[nn.Module]:
    normalized = (name or "elu").strip().lower()
    mapping: dict[str, type[nn.Module]] = {
      "elu": nn.ELU,
      "relu": nn.ReLU,
      "tanh": nn.Tanh,
      "leaky_relu": nn.LeakyReLU,
      "silu": nn.SiLU,
    }
    return mapping.get(normalized, nn.ELU)

  @staticmethod
  def _unwrap_module(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)

  def __init__(
    self,
    env,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
    enable_distributed: bool | None = None,
    **_: Any,
  ) -> None:
    self.env = env
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir
    self.current_learning_iteration = 0
    detected_rank = int(os.environ.get("RANK", "0"))
    detected_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    auto_distributed = (
      dist.is_available() and dist.is_initialized() and detected_world_size > 1
    )
    self.distributed = (
      auto_distributed if enable_distributed is None else bool(enable_distributed)
    )
    if self.distributed:
      self.rank = dist.get_rank()
      self.world_size = dist.get_world_size()
    else:
      self.rank = 0
      self.world_size = 1
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

    actor_cfg = train_cfg.get("actor", {})
    critic_cfg = train_cfg.get("critic", {})
    distribution_cfg = actor_cfg.get("distribution_cfg", {})
    action_std = float(distribution_cfg.get("init_std", 0.35))
    actor_hidden_dims = tuple(actor_cfg.get("hidden_dims", (256, 256, 128)))
    critic_hidden_dims = tuple(critic_cfg.get("hidden_dims", (256, 256, 128)))
    activation = self._activation_from_cfg(actor_cfg.get("activation"))

    actor_critic_module = ReactiveSoccerActorCritic(
      actor_obs_dim=actor_current_dim,
      history_obs_dim=actor_history_dim,
      history_steps=history_steps,
      action_dim=action_dim,
      privileged_dim=privileged_dim,
      reconstruct_dim=privileged_dim,
      actor_hidden_dims=actor_hidden_dims,
      critic_hidden_dims=critic_hidden_dims,
      activation=activation,
    ).to(device)
    self.actor_critic = self._wrap_module_for_distributed(actor_critic_module)

    algorithm_cfg = train_cfg.get("algorithm", {})
    self.symmetry_helper = SoccerSymmetry.for_g1_reactive_soccer()
    self.alg = ReactiveSoccerPPO.construct_algorithm(
      device=device,
      actor_critic=self.actor_critic,
      action_dim=action_dim,
      action_std=action_std,
      learning_rate=float(algorithm_cfg.get("learning_rate", 1.0e-3)),
      gamma=float(algorithm_cfg.get("gamma", 0.995)),
      lam=float(algorithm_cfg.get("lam", 0.95)),
      clip_param=float(algorithm_cfg.get("clip_param", 0.2)),
      value_loss_coef=float(algorithm_cfg.get("value_loss_coef", 1.0)),
      use_clipped_value_loss=bool(algorithm_cfg.get("use_clipped_value_loss", True)),
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

    amp_cfg = train_cfg.get("amp", {})
    motion_root = Path(amp_cfg.get("motion_root", "data/soccer_amp/motions_unified"))
    self.amp_loader = ReactiveSoccerAmpMotionLoader(motion_root=motion_root, device=device)
    self.amp_normalizer = ReactiveSoccerAmpNormalizer(
      feature_dim=self.amp_loader.transition_dim,
      device=device,
    )
    amp_hidden_dims = tuple(
      amp_cfg.get("discriminator_hidden_dims", (256, 256, 128))
    )
    amp_discriminator_module = AmpDiscriminator(
      input_dim=self.amp_loader.transition_dim,
      hidden_dims=amp_hidden_dims,
    ).to(device)
    self.amp_discriminator = self._wrap_module_for_distributed(amp_discriminator_module)
    self.amp_optimizer = torch.optim.Adam(
      self.amp_discriminator.parameters(),
      lr=float(amp_cfg.get("discriminator_learning_rate", 3.0e-4)),
    )
    self.amp_gradient_penalty_coef = 50.0
    self.amp_reward_weight = float(
      self.env.unwrapped.reward_manager.get_term_cfg("amp_style").weight
    )
    self.amp_reward_schedule_initial = float(amp_cfg.get("reward_scale_initial", 1.5))
    self.amp_reward_schedule_final = float(amp_cfg.get("reward_scale_final", 1.0))
    self.amp_reward_schedule_decay_iters = max(
      int(amp_cfg.get("reward_scale_decay_iters", 500)),
      0,
    )
    setattr(
      self.env.unwrapped,
      "_soccer_reactive_amp_style_reward",
      torch.zeros(self.env.num_envs, device=self.device),
    )

  def _wrap_module_for_distributed(self, module: nn.Module) -> nn.Module:
    if not self.distributed:
      return module
    torch_device = torch.device(self.device)
    if torch_device.type == "cuda":
      return DistributedDataParallel(
        module,
        device_ids=[torch_device.index],
        output_device=torch_device.index,
      )
    return DistributedDataParallel(module)

  def _apply_training_curriculum(self, iteration: int) -> None:
    curriculum = getattr(self.env.unwrapped.cfg, "training_curriculum", None)
    if curriculum is None:
      return
    if iteration < curriculum.phase1_end_iter:
      phase_prefix = "phase1"
    elif iteration < curriculum.phase2_end_iter:
      phase_prefix = "phase2"
    else:
      phase_prefix = "phase3"

    pose_range = getattr(curriculum, f"{phase_prefix}_ball_pose_range")
    reset_from_motion_prob = float(
      getattr(
        curriculum,
        f"{phase_prefix}_reset_from_motion_prob",
        getattr(self.env.unwrapped.cfg, "reset_from_motion_prob", 0.0),
      )
    )
    ball_teleport_prob = float(
      getattr(curriculum, f"{phase_prefix}_ball_teleport_prob", 0.0)
    )
    ball_velocity_prob = float(
      getattr(curriculum, f"{phase_prefix}_ball_velocity_prob", 0.0)
    )
    robot_velocity_prob = float(
      getattr(curriculum, f"{phase_prefix}_robot_velocity_prob", 0.0)
    )
    robot_push_prob = float(getattr(curriculum, f"{phase_prefix}_robot_push_prob", 0.0))

    self.env.unwrapped.event_manager.get_term_cfg("reset_ball").params["pose_range"] = pose_range
    self.env.unwrapped.event_manager.get_term_cfg("reset_ball_only").params["pose_range"] = pose_range
    self.env.unwrapped.event_manager.get_term_cfg("motion_clip_reset").params[
      "probability"
    ] = reset_from_motion_prob
    self.env.unwrapped.event_manager.get_term_cfg("ball_teleport_disturbance").params[
      "pose_range"
    ] = pose_range
    self.env.unwrapped.event_manager.get_term_cfg("ball_teleport_disturbance").params[
      "probability"
    ] = ball_teleport_prob
    self.env.unwrapped.event_manager.get_term_cfg("ball_velocity_disturbance").params[
      "probability"
    ] = ball_velocity_prob
    self.env.unwrapped.event_manager.get_term_cfg("robot_velocity_disturbance").params[
      "probability"
    ] = robot_velocity_prob
    self.env.unwrapped.event_manager.get_term_cfg("robot_push_disturbance").params[
      "probability"
    ] = robot_push_prob
    self.env.unwrapped.cfg.reset_from_motion_prob = reset_from_motion_prob

  def _amp_reward_schedule_scale(self, iteration: int) -> float:
    if self.amp_reward_schedule_decay_iters <= 0:
      return self.amp_reward_schedule_final
    progress = min(
      max(float(iteration - 1), 0.0) / float(self.amp_reward_schedule_decay_iters),
      1.0,
    )
    return (
      self.amp_reward_schedule_initial
      + progress * (self.amp_reward_schedule_final - self.amp_reward_schedule_initial)
    )

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

  @classmethod
  def _sanitize_tensor(
    cls,
    tensor: torch.Tensor,
    *,
    clip: float | None = None,
  ) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
      tensor = torch.clamp(tensor, min=-clip, max=clip)
    return tensor

  def _distributed_reduce_scalar(
    self,
    value: float,
    *,
    reduce_op: str = "mean",
  ) -> float:
    if not self.distributed:
      return float(value)
    tensor = torch.tensor(float(value), device=self.device, dtype=torch.float32)
    if reduce_op == "max":
      dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
      return float(tensor.item())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if reduce_op == "sum":
      return float(tensor.item())
    return float((tensor / max(self.world_size, 1)).item())

  def _distributed_any(self, predicate: bool) -> bool:
    if not self.distributed:
      return bool(predicate)
    tensor = torch.tensor(
      1 if predicate else 0,
      device=self.device,
      dtype=torch.int32,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())

  def _distributed_reduce_log_dict(self, log_dict: dict[str, float]) -> dict[str, float]:
    if not self.distributed:
      return log_dict
    reduced: dict[str, float] = {}
    for key, value in log_dict.items():
      if key in self.REDUCE_MAX_KEYS or key.endswith("_max"):
        reduced[key] = self._distributed_reduce_scalar(value, reduce_op="max")
      elif key.endswith("_count") or key.startswith("skipped_nonfinite"):
        reduced[key] = self._distributed_reduce_scalar(value, reduce_op="sum")
      else:
        reduced[key] = self._distributed_reduce_scalar(value, reduce_op="mean")
    return reduced

  @staticmethod
  def _empty_amp_update() -> dict[str, float]:
    return {
      "amp_discriminator": 0.0,
      "amp_reward": 0.0,
      "amp/expert_score_mean": 0.0,
      "amp/policy_score_mean": 0.0,
      "amp/score_gap": 0.0,
      "amp/wasserstein_loss": 0.0,
      "amp/gradient_penalty": 0.0,
      "amp/expert_reward_mean": 0.0,
      "amp/policy_reward_mean": 0.0,
    }

  def _compose_log_dict(
    self,
    update_dict: dict[str, float],
    amp_reward: float,
    *,
    amp_reward_rollout: float | None = None,
    amp_schedule_scale: float | None = None,
    rollout_health: dict[str, float] | None = None,
    reward_diagnostics: dict[str, float] | None = None,
  ) -> dict[str, float]:
    log_dict = {
      "surrogate": float(update_dict.get("surrogate", 0.0)),
      "goal_value": float(update_dict.get("goal_value", 0.0)),
      "aux_value": float(update_dict.get("aux_value", 0.0)),
      "reconstruction": float(update_dict.get("reconstruction", 0.0)),
      "symmetry": float(update_dict.get("symmetry", 0.0)),
      "kl": float(update_dict.get("kl", 0.0)),
      "learning_rate": float(update_dict.get("learning_rate", self.alg.learning_rate)),
      "amp_discriminator": float(update_dict.get("amp_discriminator", 0.0)),
      "amp_reward": float(amp_reward),
      "amp/reward_update_mean": float(amp_reward),
      "amp/reward_rollout_mean": float(
        amp_reward if amp_reward_rollout is None else amp_reward_rollout
      ),
      "amp/reward_schedule_scale": float(
        1.0 if amp_schedule_scale is None else amp_schedule_scale
      ),
      "goal_value_abs_mean": float(update_dict.get("goal_value_abs_mean", 0.0)),
      "goal_value_abs_max": float(update_dict.get("goal_value_abs_max", 0.0)),
      "aux_value_abs_mean": float(update_dict.get("aux_value_abs_mean", 0.0)),
      "aux_value_abs_max": float(update_dict.get("aux_value_abs_max", 0.0)),
      "goal_return_abs_mean": float(update_dict.get("goal_return_abs_mean", 0.0)),
      "goal_return_abs_max": float(update_dict.get("goal_return_abs_max", 0.0)),
      "aux_return_abs_mean": float(update_dict.get("aux_return_abs_mean", 0.0)),
      "aux_return_abs_max": float(update_dict.get("aux_return_abs_max", 0.0)),
      "skipped_nonfinite_loss_updates": float(
        update_dict.get("skipped_nonfinite_loss_updates", 0.0)
      ),
      "skipped_nonfinite_grad_updates": float(
        update_dict.get("skipped_nonfinite_grad_updates", 0.0)
      ),
    }
    for key, value in update_dict.items():
      if key.startswith("amp/"):
        log_dict[key] = float(value)
    if rollout_health is not None:
      log_dict.update({key: float(value) for key, value in rollout_health.items()})
    if reward_diagnostics is not None:
      log_dict.update({key: float(value) for key, value in reward_diagnostics.items()})
    return log_dict

  @staticmethod
  def _count_nonfinite_tensor(tensor: torch.Tensor) -> float:
    return float((~torch.isfinite(tensor)).sum().item())

  def _count_nonfinite_in_batch(
    self,
    batch: TensorDict | dict[str, torch.Tensor],
  ) -> float:
    if isinstance(batch, TensorDict):
      values = [batch[key] for key in batch.keys()]
    else:
      values = list(batch.values())
    total = 0.0
    for value in values:
      if isinstance(value, torch.Tensor):
        total += self._count_nonfinite_tensor(value)
    return total

  def _weighted_step_rewards(self) -> tuple[list[str], torch.Tensor]:
    reward_manager = self.env.unwrapped.reward_manager
    term_names = list(reward_manager.active_terms)
    step_rewards = reward_manager._step_reward
    if getattr(reward_manager, "_scale_by_dt", False):
      step_rewards = step_rewards * float(self.env.unwrapped.step_dt)
    return term_names, step_rewards

  @staticmethod
  def _init_rollout_health() -> dict[str, float]:
    return {
      "obs_nonfinite_count": 0.0,
      "next_obs_nonfinite_count": 0.0,
      "reward_nonfinite_count": 0.0,
      "action_abs_mean": 0.0,
      "action_abs_max": 0.0,
      "reward_abs_mean": 0.0,
      "reward_abs_max": 0.0,
      "amp_transition_abs_max": 0.0,
    }

  def _init_reward_diagnostics(self) -> dict[str, float]:
    diagnostics = {
      "amp/reward_scaled_mean": 0.0,
      "amp/reward_scaled_abs_mean": 0.0,
      "reward_groups/goal_abs_mean": 0.0,
      "reward_groups/aux_abs_mean": 0.0,
      "reward_groups/amp_external_abs_mean": 0.0,
      "reward_groups/total_abs_mean": 0.0,
    }
    term_names, _ = self._weighted_step_rewards()
    for term_name in term_names:
      if term_name in self.EXTERNAL_AUX_REWARD_TERMS:
        continue
      diagnostics[f"reward_terms/{term_name}_abs_mean"] = 0.0
    diagnostics["reward_terms/amp_style_external_abs_mean"] = 0.0
    return diagnostics

  def _accumulate_reward_diagnostics(
    self,
    diagnostics: dict[str, float],
    *,
    amp_contribution: torch.Tensor,
  ) -> None:
    term_names, step_rewards = self._weighted_step_rewards()
    detached_rewards = step_rewards.detach()
    goal_abs_mean = 0.0
    aux_abs_mean = 0.0
    for idx, term_name in enumerate(term_names):
      if term_name in self.EXTERNAL_AUX_REWARD_TERMS:
        continue
      term_abs_mean = float(detached_rewards[:, idx].abs().mean().item())
      diagnostics[f"reward_terms/{term_name}_abs_mean"] += term_abs_mean
      if term_name in self.GOAL_REWARD_TERMS:
        goal_abs_mean += term_abs_mean
      else:
        aux_abs_mean += term_abs_mean
    amp_abs_mean = float(amp_contribution.detach().abs().mean().item())
    diagnostics["amp/reward_scaled_mean"] += float(amp_contribution.detach().mean().item())
    diagnostics["amp/reward_scaled_abs_mean"] += amp_abs_mean
    diagnostics["reward_terms/amp_style_external_abs_mean"] += amp_abs_mean
    diagnostics["reward_groups/goal_abs_mean"] += goal_abs_mean
    diagnostics["reward_groups/aux_abs_mean"] += aux_abs_mean
    diagnostics["reward_groups/amp_external_abs_mean"] += amp_abs_mean
    diagnostics["reward_groups/total_abs_mean"] += goal_abs_mean + aux_abs_mean + amp_abs_mean

  @staticmethod
  def _finalize_reward_diagnostics(
    diagnostics: dict[str, float],
    rollout_steps: int,
  ) -> dict[str, float]:
    divisor = max(int(rollout_steps), 1)
    finalized = {key: float(value) / divisor for key, value in diagnostics.items()}
    denom = max(finalized.get("reward_groups/total_abs_mean", 0.0), 1.0e-8)
    for key, value in tuple(finalized.items()):
      if key.startswith("reward_terms/") and key.endswith("_abs_mean"):
        share_key = key.removesuffix("_abs_mean") + "_abs_share"
        finalized[share_key] = value / denom
    for key in (
      "reward_groups/goal_abs_mean",
      "reward_groups/aux_abs_mean",
      "reward_groups/amp_external_abs_mean",
    ):
      finalized[key.removesuffix("_abs_mean") + "_abs_share"] = finalized[key] / denom
    return finalized

  def _finalize_rollout_health(
    self,
    health: dict[str, float],
    rollout_steps: int,
  ) -> dict[str, float]:
    divisor = max(int(rollout_steps), 1)
    return {
      "obs_nonfinite_count": health["obs_nonfinite_count"],
      "next_obs_nonfinite_count": health["next_obs_nonfinite_count"],
      "reward_nonfinite_count": health["reward_nonfinite_count"],
      "action_abs_mean": health["action_abs_mean"] / divisor,
      "action_abs_max": health["action_abs_max"],
      "reward_abs_mean": health["reward_abs_mean"] / divisor,
      "reward_abs_max": health["reward_abs_max"],
      "amp_transition_abs_max": health["amp_transition_abs_max"],
    }

  def _emit_health_warnings(
    self,
    iteration: int,
    log_dict: dict[str, float],
  ) -> None:
    warnings: list[str] = []
    if log_dict.get("obs_nonfinite_count", 0.0) > 0.0:
      warnings.append(
        f"obs_nonfinite_count={int(log_dict['obs_nonfinite_count'])}"
      )
    if log_dict.get("next_obs_nonfinite_count", 0.0) > 0.0:
      warnings.append(
        f"next_obs_nonfinite_count={int(log_dict['next_obs_nonfinite_count'])}"
      )
    if log_dict.get("reward_nonfinite_count", 0.0) > 0.0:
      warnings.append(
        f"reward_nonfinite_count={int(log_dict['reward_nonfinite_count'])}"
      )
    if log_dict.get("skipped_nonfinite_loss_updates", 0.0) > 0.0:
      warnings.append(
        "skipped_nonfinite_loss_updates="
        f"{int(log_dict['skipped_nonfinite_loss_updates'])}"
      )
    if log_dict.get("skipped_nonfinite_grad_updates", 0.0) > 0.0:
      warnings.append(
        "skipped_nonfinite_grad_updates="
        f"{int(log_dict['skipped_nonfinite_grad_updates'])}"
      )
    if (
      log_dict.get("goal_value_abs_max", 0.0) > self.VALUE_ABS_WARN_THRESHOLD
      or log_dict.get("aux_value_abs_max", 0.0) > self.VALUE_ABS_WARN_THRESHOLD
    ):
      warnings.append(
        "critic_value_abs_max="
        f"{max(log_dict.get('goal_value_abs_max', 0.0), log_dict.get('aux_value_abs_max', 0.0)):.2f}"
      )
    if (
      log_dict.get("goal_return_abs_max", 0.0) > self.RETURN_ABS_WARN_THRESHOLD
      or log_dict.get("aux_return_abs_max", 0.0) > self.RETURN_ABS_WARN_THRESHOLD
    ):
      warnings.append(
        "critic_return_abs_max="
        f"{max(log_dict.get('goal_return_abs_max', 0.0), log_dict.get('aux_return_abs_max', 0.0)):.2f}"
      )
    if warnings:
      print(
        f"[WARN] iteration {iteration} health monitor: " + ", ".join(warnings),
        flush=True,
      )

  def add_git_repo_to_log(self, *_args: Any, **_kwargs: Any) -> None:
    return None

  def _write_tensorboard_scalars(
    self,
    iteration: int,
    log_dict: dict[str, float],
  ) -> None:
    if self.tb_writer is None:
      return
    for key in sorted(log_dict):
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
      "kl",
      "learning_rate",
      "amp_discriminator",
      "amp_reward",
      "amp/reward_rollout_mean",
      "amp/reward_update_mean",
      "amp/reward_schedule_scale",
      "amp/expert_score_mean",
      "amp/policy_score_mean",
      "amp/score_gap",
      "amp/wasserstein_loss",
      "amp/gradient_penalty",
      "amp/expert_reward_mean",
      "amp/policy_reward_mean",
      "amp/reward_scaled_mean",
      "amp/reward_scaled_abs_mean",
      "goal_value_abs_mean",
      "goal_value_abs_max",
      "aux_value_abs_mean",
      "aux_value_abs_max",
      "goal_return_abs_mean",
      "goal_return_abs_max",
      "aux_return_abs_mean",
      "aux_return_abs_max",
      "skipped_nonfinite_loss_updates",
      "skipped_nonfinite_grad_updates",
      "obs_nonfinite_count",
      "next_obs_nonfinite_count",
      "reward_nonfinite_count",
      "action_abs_mean",
      "action_abs_max",
      "reward_abs_mean",
      "reward_abs_max",
      "amp_transition_abs_max",
      "reward_groups/goal_abs_mean",
      "reward_groups/aux_abs_mean",
      "reward_groups/amp_external_abs_mean",
      "reward_groups/total_abs_mean",
      "reward_groups/goal_abs_share",
      "reward_groups/aux_abs_share",
      "reward_groups/amp_external_abs_share",
    ):
      if key in log_dict:
        print(f"Mean step {key}: {log_dict[key]:.4f}", flush=True)
    self._emit_health_warnings(iteration, log_dict)
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
    if len(log_payload) == 0:
      return
    reset_count = int(extras.get("episode_reset_count", 0) or 0)
    for key, value in log_payload.items():
      if isinstance(value, torch.Tensor):
        scalar_value = float(value.detach().float().mean().item())
      else:
        scalar_value = float(value)
      if key.startswith("Episode_Termination/") and reset_count > 0:
        scalar_value /= float(reset_count)
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

  def _eval_due(self, iteration: int) -> bool:
    eval_interval = int(self.cfg.get("eval_interval", 0) or 0)
    return eval_interval > 0 and iteration % eval_interval == 0

  def _eval_sync_paths(self, iteration: int) -> tuple[Path, Path, Path]:
    assert self.log_dir is not None
    eval_dir = Path(self.log_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    stem = f"periodic_eval_{iteration}"
    return (
      eval_dir / f".{stem}.running",
      eval_dir / f".{stem}.done",
      eval_dir / f".{stem}.failed",
    )

  def _mark_eval_status(
    self,
    path: Path,
    content: str,
  ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

  def _wait_for_periodic_eval(self, iteration: int) -> None:
    if not self.distributed or self.rank == 0 or self.log_dir is None:
      return
    running_path, done_path, failed_path = self._eval_sync_paths(iteration)
    timeout_s = float(self.cfg.get("eval_sync_timeout_s", 7200.0) or 7200.0)
    poll_s = max(float(self.cfg.get("eval_sync_poll_s", 1.0) or 1.0), 0.1)
    start_time = time.perf_counter()
    announced_wait = False
    while True:
      if failed_path.exists():
        failure_reason = failed_path.read_text(encoding="utf-8").strip()
        raise RuntimeError(
          "Periodic evaluation failed on rank 0 "
          f"(iteration={iteration}): {failure_reason or 'unknown error'}"
        )
      if done_path.exists():
        return
      elapsed_s = time.perf_counter() - start_time
      if elapsed_s >= timeout_s:
        observed_state = (
          "running"
          if running_path.exists()
          else "not-started"
        )
        raise RuntimeError(
          "Timed out while waiting for rank 0 periodic evaluation to finish "
          f"(iteration={iteration}, state={observed_state}, timeout_s={timeout_s:.1f})."
        )
      if not announced_wait:
        print(
          "[INFO] Waiting for rank 0 periodic evaluation to finish "
          f"(iteration={iteration})",
          flush=True,
        )
        announced_wait = True
      time.sleep(poll_s)

  def _run_periodic_eval(self, iteration: int, checkpoint_path: Path | None = None) -> None:
    if self.log_dir is None or self.rank != 0:
      return
    if not self._eval_due(iteration):
      return

    from mjlab.tasks.soccer_reactive.scripts.play import ReactiveSoccerEvalArgs, run_play

    eval_dir = Path(self.log_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    running_path, done_path, failed_path = self._eval_sync_paths(iteration)
    running_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    failed_path.unlink(missing_ok=True)
    self._mark_eval_status(
      running_path,
      f"started_at={time.time():.6f}\nrank={self.rank}\niteration={iteration}\n",
    )
    temp_checkpoint_path: Path | None = None
    try:
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
        num_envs=int(
          self.cfg.get("eval_num_envs", min(self.env.num_envs, 32))
          or min(self.env.num_envs, 32)
        ),
      )
      run_play(args)
      self._mark_eval_status(
        done_path,
        f"finished_at={time.time():.6f}\nrank={self.rank}\niteration={iteration}\n",
      )
    except Exception:
      self._mark_eval_status(
        failed_path,
        traceback.format_exc(),
      )
      raise
    finally:
      running_path.unlink(missing_ok=True)
      if temp_checkpoint_path is not None:
        temp_checkpoint_path.unlink(missing_ok=True)

  def save(self, path: str, infos=None) -> None:
    if self.rank != 0:
      return
    merged_infos = dict(infos or {})
    merged_infos[ODOMETRY_INFO_KEY] = build_odometry_info_from_env(self.env.unwrapped)
    state = {
      "iter": self.current_learning_iteration,
      "algo": self.alg.save(),
      "amp_discriminator": self._unwrap_module(self.amp_discriminator).state_dict(),
      "amp_optimizer": self.amp_optimizer.state_dict(),
      "amp_normalizer": {
        "count": self.amp_normalizer.count,
        "mean": self.amp_normalizer.mean,
        "var": self.amp_normalizer.var,
      },
      "infos": merged_infos,
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
      self._unwrap_module(self.amp_discriminator).load_state_dict(state["amp_discriminator"])
    if "amp_optimizer" in state:
      self.amp_optimizer.load_state_dict(state["amp_optimizer"])
    if "amp_normalizer" in state:
      normalizer_state = state["amp_normalizer"]
      self.amp_normalizer.count = self._sanitize_tensor(normalizer_state["count"])
      self.amp_normalizer.mean = self._sanitize_tensor(normalizer_state["mean"])
      self.amp_normalizer.var = self._sanitize_tensor(normalizer_state["var"])
    return state.get("infos", {})

  def _tensor_dict_to_batch(
    self,
    obs: TensorDict | dict[str, torch.Tensor],
  ) -> dict[str, torch.Tensor]:
    if isinstance(obs, TensorDict):
      obs = {k: obs[k] for k in obs.keys()}
    else:
      obs = dict(obs)
    return {
      key: self._sanitize_tensor(value, clip=self.OBS_CLIP)
      if isinstance(value, torch.Tensor)
      else value
      for key, value in obs.items()
    }

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
        outputs = self.actor_critic(model_batch)
      return outputs["actions_mean"]

    return _policy

  def _split_reward_groups(self) -> tuple[torch.Tensor, torch.Tensor]:
    term_names, step_rewards = self._weighted_step_rewards()

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
    current_amp_obs = torch.nan_to_num(current_amp_obs, nan=0.0, posinf=0.0, neginf=0.0)
    next_amp_obs = torch.nan_to_num(next_amp_obs, nan=0.0, posinf=0.0, neginf=0.0)
    joint_pos_t = current_amp_obs[:, :23]
    joint_vel_t = current_amp_obs[:, 23:46]
    joint_pos_t1 = next_amp_obs[:, :23]
    joint_vel_t1 = next_amp_obs[:, 23:46]
    transition = torch.cat((joint_pos_t, joint_vel_t, joint_pos_t1, joint_vel_t1), dim=-1)
    return torch.nan_to_num(transition, nan=0.0, posinf=0.0, neginf=0.0)

  def _compute_amp_reward(self, policy_transition: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      normalized_transition = self.amp_normalizer.normalize(policy_transition)
      normalized_transition = self._sanitize_tensor(
        normalized_transition,
        clip=self.AMP_TRANSITION_CLIP,
      )
      scores = self.amp_discriminator(normalized_transition)
      return self._sanitize_tensor(AmpDiscriminator.adversarial_reward(scores))

  def _compute_amp_discriminator_loss(
    self,
    expert: torch.Tensor,
    policy: torch.Tensor,
    *,
    gradient_penalty_coef: float,
  ) -> dict[str, torch.Tensor]:
    expert_scores = self._sanitize_tensor(self.amp_discriminator(expert))
    policy_scores = self._sanitize_tensor(self.amp_discriminator(policy))
    wasserstein_loss = AmpDiscriminator.wasserstein_loss_from_scores(
      expert_scores,
      policy_scores,
    )

    batch = min(expert.shape[0], policy.shape[0])
    if batch == 0:
      gradient_penalty = torch.tensor(0.0, device=self.device)
    else:
      alpha = torch.rand((batch, 1), device=self.device)
      mixed = alpha * expert[:batch] + (1.0 - alpha) * policy[:batch]
      mixed.requires_grad_(True)
      mixed_scores = self.amp_discriminator(mixed)
      grad = torch.autograd.grad(
        outputs=mixed_scores.sum(),
        inputs=mixed,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
      )[0]
      grad = self._sanitize_tensor(grad)
      gradient_penalty = gradient_penalty_coef * (
        grad.norm(2, dim=1) - 1.0
      ).pow(2).mean()

    total_loss = self._sanitize_tensor(
      wasserstein_loss + gradient_penalty,
      clip=gradient_penalty_coef,
    )
    return {
      "expert_scores": expert_scores,
      "policy_scores": policy_scores,
      "wasserstein_loss": wasserstein_loss,
      "gradient_penalty": gradient_penalty,
      "total_loss": total_loss,
    }

  def _update_amp_discriminator(
    self,
    policy_transitions: list[torch.Tensor],
  ) -> dict[str, float]:
    if not policy_transitions:
      return self._empty_amp_update()

    policy_batch = self._sanitize_tensor(torch.cat(policy_transitions, dim=0))
    expert_batch = self._sanitize_tensor(
      self.amp_loader.sample_transition_batch(batch_size=policy_batch.shape[0])
    )
    normalizer_batch = torch.cat((expert_batch, policy_batch), dim=0)
    if self.distributed:
      batch_sum = normalizer_batch.sum(dim=0)
      batch_sq_sum = normalizer_batch.square().sum(dim=0)
      batch_count = torch.tensor(
        float(normalizer_batch.shape[0]),
        device=self.device,
        dtype=torch.float32,
      )
      dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)
      dist.all_reduce(batch_sq_sum, op=dist.ReduceOp.SUM)
      dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
      global_mean = batch_sum / batch_count.clamp(min=1.0)
      global_var = batch_sq_sum / batch_count.clamp(min=1.0) - global_mean.square()
      self.amp_normalizer.update_from_moments(global_mean, global_var, batch_count)
    else:
      self.amp_normalizer.update(normalizer_batch)
    expert_norm = self._sanitize_tensor(
      self.amp_normalizer.normalize(expert_batch),
      clip=self.AMP_TRANSITION_CLIP,
    )
    policy_norm = self._sanitize_tensor(
      self.amp_normalizer.normalize(policy_batch),
      clip=self.AMP_TRANSITION_CLIP,
    )
    loss_dict = self._compute_amp_discriminator_loss(
      expert_norm,
      policy_norm,
      gradient_penalty_coef=self.amp_gradient_penalty_coef,
    )
    skip_loss_update = self._distributed_any(
      not bool(torch.isfinite(loss_dict["total_loss"]).item())
    )
    if skip_loss_update:
      return self._empty_amp_update()
    self.amp_optimizer.zero_grad(set_to_none=True)
    loss_dict["total_loss"].backward()
    has_nonfinite_grad = False
    for param in self.amp_discriminator.parameters():
      if param.grad is None:
        continue
      if not torch.all(torch.isfinite(param.grad)):
        has_nonfinite_grad = True
        break
    skip_grad_update = self._distributed_any(has_nonfinite_grad)
    if skip_grad_update:
      self.amp_optimizer.zero_grad(set_to_none=True)
      return self._empty_amp_update()
    self.amp_optimizer.step()
    amp_reward = AmpDiscriminator.adversarial_reward(loss_dict["policy_scores"]).mean()
    expert_reward = AmpDiscriminator.adversarial_reward(loss_dict["expert_scores"]).mean()
    expert_score_mean = float(loss_dict["expert_scores"].detach().mean().item())
    policy_score_mean = float(loss_dict["policy_scores"].detach().mean().item())
    return {
      "amp_discriminator": self._distributed_reduce_scalar(
        float(loss_dict["total_loss"].detach().item()),
        reduce_op="mean",
      ),
      "amp_reward": self._distributed_reduce_scalar(
        float(amp_reward.detach().item()),
        reduce_op="mean",
      ),
      "amp/expert_score_mean": self._distributed_reduce_scalar(
        expert_score_mean,
        reduce_op="mean",
      ),
      "amp/policy_score_mean": self._distributed_reduce_scalar(
        policy_score_mean,
        reduce_op="mean",
      ),
      "amp/score_gap": self._distributed_reduce_scalar(
        expert_score_mean - policy_score_mean,
        reduce_op="mean",
      ),
      "amp/wasserstein_loss": self._distributed_reduce_scalar(
        float(loss_dict["wasserstein_loss"].detach().item()),
        reduce_op="mean",
      ),
      "amp/gradient_penalty": self._distributed_reduce_scalar(
        float(loss_dict["gradient_penalty"].detach().item()),
        reduce_op="mean",
      ),
      "amp/expert_reward_mean": self._distributed_reduce_scalar(
        float(expert_reward.detach().item()),
        reduce_op="mean",
      ),
      "amp/policy_reward_mean": self._distributed_reduce_scalar(
        float(amp_reward.detach().item()),
        reduce_op="mean",
      ),
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
      rollout_health = self._init_rollout_health()
      reward_diagnostics = self._init_reward_diagnostics()
      amp_schedule_scale = self._amp_reward_schedule_scale(next_iteration)

      for _rollout_idx in range(rollout_steps):
        rollout_health["obs_nonfinite_count"] += self._count_nonfinite_in_batch(obs)
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
        rollout_health["next_obs_nonfinite_count"] += self._count_nonfinite_in_batch(next_obs)
        rollout_health["reward_nonfinite_count"] += self._count_nonfinite_tensor(rewards)
        action_abs = action_dict["actions"].detach().abs()
        rollout_health["action_abs_mean"] += float(action_abs.mean().item())
        rollout_health["action_abs_max"] = max(
          rollout_health["action_abs_max"],
          float(action_abs.max().item()),
        )
        safe_rewards = self._sanitize_tensor(rewards)
        reward_abs = safe_rewards.detach().abs()
        rollout_health["reward_abs_mean"] += float(reward_abs.mean().item())
        rollout_health["reward_abs_max"] = max(
          rollout_health["reward_abs_max"],
          float(reward_abs.max().item()),
        )
        self._last_step_extras = extras
        self._accumulate_episode_log(episode_log_totals, episode_log_counts, extras)
        next_batch = self._tensor_dict_to_batch(next_obs)

        policy_transition = self._build_policy_amp_transition(
          batch["amp"],
          next_batch["amp"],
        )
        rollout_health["amp_transition_abs_max"] = max(
          rollout_health["amp_transition_abs_max"],
          float(policy_transition.detach().abs().max().item()),
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
        amp_scale = (
          float(self.env.unwrapped.step_dt)
          if bool(getattr(self.env.unwrapped.cfg, "scale_rewards_by_dt", False))
          else 1.0
        )
        amp_contribution = (
          amp_scale * self.amp_reward_weight * amp_schedule_scale * amp_reward
        )
        aux_rewards = aux_rewards + amp_contribution
        self._accumulate_reward_diagnostics(
          reward_diagnostics,
          amp_contribution=amp_contribution,
        )
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
      update_dict.update(amp_update)
      finalized_rollout_health = self._finalize_rollout_health(
        rollout_health,
        rollout_steps=rollout_steps,
      )
      finalized_reward_diagnostics = self._finalize_reward_diagnostics(
        reward_diagnostics,
        rollout_steps=rollout_steps,
      )
      self._latest_logs = self._compose_log_dict(
        update_dict,
        amp_reward=amp_update["amp_reward"]
        if policy_transitions
        else amp_reward_running / max(rollout_steps, 1),
        amp_reward_rollout=amp_reward_running / max(rollout_steps, 1),
        amp_schedule_scale=amp_schedule_scale,
        rollout_health=finalized_rollout_health,
        reward_diagnostics=finalized_reward_diagnostics,
      )
      self._latest_logs = self._distributed_reduce_log_dict(self._latest_logs)
      self.current_learning_iteration += 1
      save_interval = int(self.cfg.get("save_interval", 0) or 0)
      periodic_eval_due = self._eval_due(self.current_learning_iteration)
      if self.log_dir is not None:
        self._write_tensorboard_scalars(
          self.current_learning_iteration, self._latest_logs
        )
        saved_checkpoint_path: Path | None = None
        if (
          self.rank == 0
          and save_interval > 0
          and self.current_learning_iteration % save_interval == 0
        ):
          saved_checkpoint_path = Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"
          self.save(str(saved_checkpoint_path))
          self._prune_old_checkpoints()
        if periodic_eval_due:
          if self.rank == 0:
            self._run_periodic_eval(
              self.current_learning_iteration,
              checkpoint_path=saved_checkpoint_path,
            )
          else:
            self._wait_for_periodic_eval(self.current_learning_iteration)
      total_timesteps = (
        self.current_learning_iteration
        * self.env.num_envs
        * rollout_steps
        * max(self.world_size, 1)
      )
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
