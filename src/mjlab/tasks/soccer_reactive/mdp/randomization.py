"""Training randomization helpers for reactive soccer."""

from __future__ import annotations

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.soccer_reactive.mdp.events import reset_ball_only
from mjlab.utils.lab_api.math import sample_uniform

_BALL_CFG = SceneEntityCfg("ball")
_ROBOT_BASE_CFG = SceneEntityCfg("robot", body_names=("pelvis",))


def _sample_trigger_env_ids(
  env,
  env_ids: torch.Tensor | None,
  probability: float,
) -> torch.Tensor:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if probability >= 1.0:
    return env_ids
  if len(env_ids) == 0:
    return env_ids
  trigger_mask = torch.rand(len(env_ids), device=env.device) < probability
  return env_ids[trigger_mask]


@requires_model_fields("body_mass", "geom_friction", recompute=RecomputeLevel.set_const)
def ball_physics_randomization(
  env,
  env_ids: torch.Tensor | None,
  mass_range: tuple[float, float] = (0.40, 0.46),
  friction_range: tuple[float, float] = (0.35, 0.85),
  asset_cfg: SceneEntityCfg = _BALL_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  ball = env.scene[asset_cfg.name]
  body_id = int(ball.indexing.body_ids[0])
  geom_id = int(ball.indexing.geom_ids[0])
  env.sim.model.body_mass[env_ids, body_id] = sample_uniform(
    mass_range[0],
    mass_range[1],
    (len(env_ids),),
    device=env.device,
  )
  env.sim.model.geom_friction[env_ids, geom_id, 0] = sample_uniform(
    friction_range[0],
    friction_range[1],
    (len(env_ids),),
    device=env.device,
  )


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def robot_base_mass_randomization(
  env,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float] = (0.9, 1.1),
  asset_cfg: SceneEntityCfg = _ROBOT_BASE_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  body_id = int(asset_cfg.body_ids[0])
  default_mass = env.sim.get_default_field("body_mass")[body_id]
  scale = sample_uniform(
    scale_range[0],
    scale_range[1],
    (len(env_ids),),
    device=env.device,
  )
  env.sim.model.body_mass[env_ids, body_id] = default_mass * scale


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def robot_base_com_randomization(
  env,
  env_ids: torch.Tensor | None,
  offset_range: tuple[float, float] = (-0.02, 0.02),
  asset_cfg: SceneEntityCfg = _ROBOT_BASE_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  body_id = int(asset_cfg.body_ids[0])
  default_ipos = env.sim.get_default_field("body_ipos")[body_id, :3]
  offset = sample_uniform(
    offset_range[0],
    offset_range[1],
    (len(env_ids), 3),
    device=env.device,
  )
  env.sim.model.body_ipos[env_ids, body_id, :3] = default_ipos.unsqueeze(0) + offset


def ball_teleport_disturbance(
  env,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  probability: float = 1.0,
) -> None:
  trigger_env_ids = _sample_trigger_env_ids(env, env_ids, probability)
  if len(trigger_env_ids) == 0:
    return
  reset_ball_only(env, trigger_env_ids, pose_range=pose_range)


def probabilistic_velocity_disturbance(
  env,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  probability: float = 1.0,
  asset_cfg: SceneEntityCfg = _BALL_CFG,
) -> None:
  trigger_env_ids = _sample_trigger_env_ids(env, env_ids, probability)
  if len(trigger_env_ids) == 0:
    return
  envs_mdp.push_by_setting_velocity(
    env,
    trigger_env_ids,
    velocity_range=velocity_range,
    asset_cfg=asset_cfg,
  )
