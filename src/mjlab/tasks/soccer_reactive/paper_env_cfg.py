"""Minimal paper-style reactive soccer environment config."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mjlab.asset_zoo.robots import G1_COMP_BODY_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.soccer.field_specs import M_FIELD
from mjlab.tasks.soccer.soccer_env_cfg import make_soccer_env_cfg
from mjlab.tasks.soccer_reactive import mdp

_ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=G1_COMP_BODY_JOINT_NAMES)


def _make_actor_terms() -> dict[str, ObservationTermCfg]:
  return {
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": _ROBOT_JOINT_CFG},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": _ROBOT_JOINT_CFG},
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "ball_pos_b": ObservationTermCfg(func=mdp.perceived_ball_pos_b),
    "ball_mask": ObservationTermCfg(func=mdp.perceived_ball_mask),
    "goal_pos_b": ObservationTermCfg(func=mdp.perceived_goal_pos_b),
    "goal_dir_b": ObservationTermCfg(func=mdp.perceived_goal_dir_b),
  }


def _make_privileged_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_lin_vel": ObservationTermCfg(func=mdp.privileged_base_lin_vel),
    "base_height": ObservationTermCfg(func=mdp.privileged_base_height),
  }


def _make_amp_terms() -> dict[str, ObservationTermCfg]:
  return {
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": _ROBOT_JOINT_CFG},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": _ROBOT_JOINT_CFG},
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }


def make_reactive_soccer_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Return the minimal Task 4 env cfg for reactive soccer."""
  cfg = make_soccer_env_cfg(play=play)
  actor_terms = _make_actor_terms()
  privileged_terms = _make_privileged_terms()

  cfg.observations = {
    # Compatibility aliases for existing generic task tests.
    "actor": ObservationGroupCfg(
      terms=_make_actor_terms(),
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms={**actor_terms, **privileged_terms},
      concatenate_terms=True,
      enable_corruption=False,
    ),
    # Task-2-specific groups.
    "actor_current": ObservationGroupCfg(
      terms=_make_actor_terms(),
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "actor_history": ObservationGroupCfg(
      terms=_make_actor_terms(),
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=50,
      flatten_history_dim=False,
    ),
    "critic_current": ObservationGroupCfg(
      terms=_make_actor_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic_privileged": ObservationGroupCfg(
      terms=_make_privileged_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "amp": ObservationGroupCfg(
      terms=_make_amp_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.rewards = {
    "goal_scored": RewardTermCfg(func=mdp.goal_scored_reward, weight=10.0),
    "ball_approach": RewardTermCfg(func=mdp.ball_approach_reward, weight=0.6),
    "goal_progress": RewardTermCfg(func=mdp.goal_progress_reward, weight=1.0),
    "ball_goal_alignment": RewardTermCfg(
      func=mdp.ball_goal_alignment_reward, weight=0.25
    ),
    "touch_ball": RewardTermCfg(func=mdp.touch_ball_impulse_reward, weight=0.75),
    "survival": RewardTermCfg(func=mdp.survival_reward, weight=0.05),
    "stagnation": RewardTermCfg(func=mdp.stagnation_penalty, weight=-0.05),
    "robot_ball_alignment": RewardTermCfg(
      func=mdp.robot_ball_alignment_reward, weight=0.2
    ),
    "head_pitch_alignment": RewardTermCfg(
      func=mdp.head_pitch_alignment_penalty, weight=-0.01
    ),
    "head_yaw_alignment": RewardTermCfg(
      func=mdp.head_yaw_alignment_penalty, weight=-0.01
    ),
    "sideways_kick": RewardTermCfg(func=mdp.sideways_kick_reward, weight=0.0),
    "forward_kick_penalty": RewardTermCfg(
      func=mdp.forward_kick_penalty, weight=0.0
    ),
    "foot_proximity_penalty": RewardTermCfg(
      func=mdp.foot_proximity_penalty, weight=-0.01
    ),
    "near_ball_control": RewardTermCfg(
      func=mdp.near_ball_control_reward, weight=0.15
    ),
    "leg_action_rate_penalty": RewardTermCfg(
      func=mdp.leg_action_rate_penalty, weight=-0.01
    ),
    "joint_limit_penalty": RewardTermCfg(func=mdp.joint_limit_penalty, weight=-0.01),
    "base_acc_penalty": RewardTermCfg(func=mdp.base_acc_penalty, weight=0.0),
    "non_foot_collision_penalty": RewardTermCfg(
      func=mdp.non_foot_collision_penalty, weight=0.0
    ),
    "amp_style": RewardTermCfg(func=mdp.amp_style_reward, weight=0.0),
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "robot_fallen": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.42, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "ball_out_of_bounds": TerminationTermCfg(
      func=mdp.ball_out_of_bounds,
      params={
        "field_length": M_FIELD.field_length,
        "field_width": M_FIELD.field_width,
      },
    ),
    "goal_scored": TerminationTermCfg(
      func=mdp.goal_scored,
      params={
        "goal_x": M_FIELD.field_length * 0.5,
        "goal_width": M_FIELD.goal_width,
      },
    ),
  }

  reset_from_motion_prob = 0.5
  cfg.reset_from_motion_prob = reset_from_motion_prob
  training_curriculum = SimpleNamespace(
    phase1_end_iter=500,
    phase2_end_iter=1500,
    phase1_ball_pose_range={
      "x": (0.5, 2.0),
      "y": (-1.0, 1.0),
      "z": (0.0, 0.0),
    },
    phase2_ball_pose_range={
      "x": (0.0, 4.0),
      "y": (-2.0, 2.0),
      "z": (0.0, 0.0),
    },
    phase3_ball_pose_range={
      "x": (-4.0, 4.0),
      "y": (-3.5, 3.5),
      "z": (0.0, 0.0),
    },
  )
  cfg.training_curriculum = training_curriculum

  cfg.events["motion_clip_reset"] = EventTermCfg(
    func=mdp.MotionClipResetEvent,
    mode="reset",
    params={
      "motion_root": str(Path("data/soccer_amp/motions_unified")),
      "probability": reset_from_motion_prob,
      "asset_cfg": SceneEntityCfg("robot", joint_names=G1_COMP_BODY_JOINT_NAMES),
    },
  )

  cfg.events["reset_ball_only"] = EventTermCfg(
    func=mdp.reset_ball_only,
    mode="reset",
    params={"pose_range": training_curriculum.phase1_ball_pose_range},
  )
  cfg.control_hz = 50
  cfg.perception = SimpleNamespace(camera_hz=30, odom_hz=20)
  cfg.episode_length_s = 60.0
  if play:
    cfg.episode_length_s = int(1e9)
  return cfg
