"""Minimal reactive soccer runner scaffold."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from mjlab.tasks.soccer_reactive.models.history_encoder_actor_critic import (
  ReactiveSoccerActorCritic,
)
from mjlab.tasks.soccer_reactive.rl.amp_discriminator import AmpDiscriminator
from mjlab.tasks.soccer_reactive.rl.amp_motion_loader import ReactiveSoccerAmpMotionLoader
from mjlab.tasks.soccer_reactive.rl.amp_normalizer import ReactiveSoccerAmpNormalizer
from mjlab.tasks.soccer_reactive.rl.reactive_soccer_ppo import ReactiveSoccerPPO


class ReactiveSoccerRunner:
  """Lightweight runner that wires the reactive soccer components together."""

  GOAL_REWARD_TERMS = frozenset(
    {
      'goal_scored',
      'ball_approach',
      'goal_progress',
      'ball_goal_alignment',
      'touch_ball',
    }
  )

  def __init__(
    self,
    env,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = 'cpu',
    **_: Any,
  ) -> None:
    self.env = env
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir
    self.current_learning_iteration = 0
    self.rank = int(os.environ.get('RANK', '0'))
    self.logger = SimpleNamespace(save_model=lambda *args, **kwargs: None)
    self.tb_log_dir = Path(log_dir) / 'tensorboard' if log_dir is not None else None
    use_tensorboard = (
      self.tb_log_dir is not None
      and train_cfg.get('logger', 'tensorboard') == 'tensorboard'
      and self.rank == 0
    )
    self.tb_writer = SummaryWriter(log_dir=str(self.tb_log_dir)) if use_tensorboard else None

    actor_current_dim = self._flat_dim(env.unwrapped.observation_manager.group_obs_dim['actor_current'])
    actor_history_dim = env.unwrapped.observation_manager.group_obs_dim['actor_history'][-1]
    history_steps = env.unwrapped.observation_manager.group_obs_dim['actor_history'][0]
    privileged_dim = self._flat_dim(env.unwrapped.observation_manager.group_obs_dim['critic_privileged'])
    action_dim = env.unwrapped.action_manager.total_action_dim

    self.actor_critic = ReactiveSoccerActorCritic(
      actor_obs_dim=actor_current_dim,
      history_obs_dim=actor_history_dim,
      history_steps=history_steps,
      action_dim=action_dim,
      privileged_dim=privileged_dim,
      reconstruct_dim=privileged_dim,
    ).to(device)
    self.alg = ReactiveSoccerPPO.construct_algorithm(
      device=device,
      actor_critic=self.actor_critic,
      action_dim=action_dim,
      learning_rate=float(train_cfg.get('algorithm', {}).get('learning_rate', 3.0e-4)),
      gamma=float(train_cfg.get('algorithm', {}).get('gamma', 0.99)),
      lam=float(train_cfg.get('algorithm', {}).get('lam', 0.95)),
      clip_param=float(train_cfg.get('algorithm', {}).get('clip_param', 0.2)),
    )
    self.alg.initialize_storage(
      num_envs=env.num_envs,
      num_steps=train_cfg.get('num_steps_per_env', 4),
      actor_obs_shape=(actor_current_dim,),
      action_shape=(action_dim,),
      reconstruction_shape=(privileged_dim,),
      actor_history_shape=tuple(env.unwrapped.observation_manager.group_obs_dim['actor_history']),
      privileged_shape=(privileged_dim,),
    )

    motion_root = Path('data/soccer_amp/motions_unified')
    self.amp_loader = ReactiveSoccerAmpMotionLoader(motion_root=motion_root, device=device)
    self.amp_normalizer = ReactiveSoccerAmpNormalizer(
      feature_dim=self.amp_loader.transition_dim,
      device=device,
    )
    self.amp_discriminator = AmpDiscriminator(
      input_dim=self.amp_loader.transition_dim,
      hidden_dims=(128, 64),
    ).to(device)

  def _apply_training_curriculum(self, iteration: int) -> None:
    curriculum = getattr(self.env.unwrapped.cfg, 'training_curriculum', None)
    if curriculum is None:
      return
    if iteration < curriculum.phase1_end_iter:
      pose_range = curriculum.phase1_ball_pose_range
    elif iteration < curriculum.phase2_end_iter:
      pose_range = curriculum.phase2_ball_pose_range
    else:
      pose_range = curriculum.phase3_ball_pose_range
    self.env.unwrapped.event_manager.get_term_cfg('reset_ball').params['pose_range'] = pose_range
    self.env.unwrapped.event_manager.get_term_cfg('reset_ball_only').params['pose_range'] = pose_range

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

  def _compose_log_dict(self, update_dict: dict[str, float], amp_reward: float) -> dict[str, float]:
    return {
      'surrogate': float(update_dict.get('surrogate', 0.0)),
      'goal_value': float(update_dict.get('goal_value', 0.0)),
      'aux_value': float(update_dict.get('aux_value', 0.0)),
      'reconstruction': float(update_dict.get('reconstruction', 0.0)),
      'symmetry': float(update_dict.get('symmetry', 0.0)),
      'amp_discriminator': float(update_dict.get('amp_discriminator', 0.0)),
      'amp_reward': float(amp_reward),
    }

  def add_git_repo_to_log(self, *_args: Any, **_kwargs: Any) -> None:
    return None

  def _write_tensorboard_scalars(self, iteration: int, log_dict: dict[str, float]) -> None:
    if self.tb_writer is None:
      return
    for key in (
      'surrogate',
      'goal_value',
      'aux_value',
      'reconstruction',
      'symmetry',
      'amp_discriminator',
      'amp_reward',
    ):
      if key in log_dict:
        self.tb_writer.add_scalar(f'train/{key}', float(log_dict[key]), iteration)

  def save(self, path: str, infos=None) -> None:
    state = {
      'iter': self.current_learning_iteration,
      'algo': self.alg.save(),
      'infos': infos or {},
    }
    torch.save(state, path)

  def load(self, path: str, load_cfg=None, strict: bool = True, map_location: str | None = None):
    del load_cfg, strict
    state = torch.load(path, map_location=map_location or self.device, weights_only=False)
    self.current_learning_iteration = state.get('iter', 0)
    self.alg.load(state.get('algo', {}))
    return state.get('infos', {})

  def _tensor_dict_to_batch(self, obs: TensorDict | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(obs, TensorDict):
      return {k: obs[k] for k in obs.keys()}
    return dict(obs)

  def get_inference_policy(self, device: str | None = None) -> Callable[[TensorDict | dict[str, torch.Tensor]], torch.Tensor]:
    policy_device = device or self.device

    def _policy(obs: TensorDict | dict[str, torch.Tensor]) -> torch.Tensor:
      batch = self._tensor_dict_to_batch(obs)
      model_batch = {
        'actor_current': batch['actor_current'].to(policy_device),
        'actor_history': batch['actor_history'].to(policy_device),
        'critic_privileged': batch['critic_privileged'].to(policy_device),
      }
      with torch.no_grad():
        outputs = self.actor_critic.forward_train(model_batch)
      return outputs['actions_mean']

    return _policy

  def _split_reward_groups(self) -> tuple[torch.Tensor, torch.Tensor]:
    reward_manager = self.env.unwrapped.reward_manager
    term_names = list(reward_manager.active_terms)
    step_rewards = reward_manager._step_reward
    scale = self.env.unwrapped.step_dt if self.env.unwrapped.cfg.scale_rewards_by_dt else 1.0

    goal_indices = [idx for idx, name in enumerate(term_names) if name in self.GOAL_REWARD_TERMS]
    aux_indices = [idx for idx, name in enumerate(term_names) if name not in self.GOAL_REWARD_TERMS]

    goal_reward = step_rewards[:, goal_indices].sum(dim=1) * scale if goal_indices else torch.zeros(self.env.num_envs, device=self.device)
    aux_reward = step_rewards[:, aux_indices].sum(dim=1) * scale if aux_indices else torch.zeros(self.env.num_envs, device=self.device)
    return goal_reward, aux_reward

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True) -> None:
    del init_at_random_ep_len
    obs, _ = self.env.reset()
    for _ in range(num_learning_iterations):
      self._apply_training_curriculum(self.current_learning_iteration)
      assert self.alg.storage is not None
      self.alg.storage.reset()
      rollout_steps = int(self.cfg.get('num_steps_per_env', 4))
      for _rollout_idx in range(rollout_steps):
        batch = self._tensor_dict_to_batch(obs)
        action_dict = self.alg.act(
          {
            'actor_current': batch['actor_current'],
            'actor_history': batch['actor_history'],
            'critic_privileged': batch['critic_privileged'],
          }
        )
        next_obs, rewards, dones, extras = self.env.step(action_dict['actions'])
        self._last_step_extras = extras
        goal_rewards, aux_rewards = self._split_reward_groups()
        self.alg.process_env_step(
          {
            'actor_current': batch['actor_current'],
            'actor_history': batch['actor_history'],
            'critic_privileged': batch['critic_privileged'],
          },
          {'goal': goal_rewards, 'aux': aux_rewards, 'total': rewards},
          dones,
          extras,
        )
        obs = next_obs

      self.alg.compute_returns(self._tensor_dict_to_batch(obs))
      amp_batch = self.amp_loader.sample_transition_batch(batch_size=min(4, self.env.num_envs))
      self.amp_normalizer.update(amp_batch)
      norm_batch = self.amp_normalizer.normalize(amp_batch)
      amp_scores = self.amp_discriminator(norm_batch)
      update_dict = self.alg.update()
      update_dict['amp_discriminator'] = float(amp_scores.mean().item())
      self._latest_logs = self._compose_log_dict(
        update_dict,
        amp_reward=float(AmpDiscriminator.bounded_reward(amp_scores).mean().item()),
      )
      self.current_learning_iteration += 1
      save_interval = int(self.cfg.get('save_interval', 0) or 0)
      if self.log_dir is not None:
        self._write_tensorboard_scalars(
          self.current_learning_iteration, self._latest_logs
        )
        if save_interval > 0 and self.current_learning_iteration % save_interval == 0:
          self.save(str(Path(self.log_dir) / f'model_{self.current_learning_iteration}.pt'))

  def close(self) -> None:
    if self.tb_writer is not None:
      self.tb_writer.flush()
      self.tb_writer.close()
