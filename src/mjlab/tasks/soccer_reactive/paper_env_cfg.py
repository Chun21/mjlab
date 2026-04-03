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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.soccer.field_specs import M_FIELD
from mjlab.tasks.soccer.soccer_env_cfg import make_soccer_env_cfg
from mjlab.tasks.soccer_reactive import mdp
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import perlin_noise

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
    "ball_pos_b": ObservationTermCfg(func=mdp.PerceivedBallPosObs),
    "ball_mask": ObservationTermCfg(func=mdp.PerceivedBallMaskObs),
    "goal_pos_b": ObservationTermCfg(func=mdp.PerceivedGoalPosObs),
    "goal_dir_b": ObservationTermCfg(func=mdp.PerceivedGoalDirObs),
    "ball_obs_age": ObservationTermCfg(func=mdp.BallObservationAgeObs),
    "goal_obs_age": ObservationTermCfg(func=mdp.GoalObservationAgeObs),
  }


def _make_privileged_terms() -> dict[str, ObservationTermCfg]:
  return {
    "ball_true_pos_b": ObservationTermCfg(func=mdp.privileged_ball_true_pos_b),
    "base_lin_vel": ObservationTermCfg(func=mdp.privileged_base_lin_vel),
    "base_height": ObservationTermCfg(func=mdp.privileged_base_height),
    "base_mass_randomization": ObservationTermCfg(
      func=mdp.privileged_base_mass_randomization
    ),
    "base_com_randomization": ObservationTermCfg(
      func=mdp.privileged_base_com_randomization
    ),
    "ball_vel_w": ObservationTermCfg(func=mdp.privileged_ball_vel_w),
    "ball_friction_w": ObservationTermCfg(func=mdp.privileged_ball_friction_w),
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
  if cfg.scene.terrain is not None:
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
      size=(M_FIELD.field_length, M_FIELD.field_width),
      border_width=1.0,
      num_rows=1,
      num_cols=1,
      curriculum=False,
      sub_terrains={
        "mild_uneven": perlin_noise(
          proportion=1.0,
          height_range=(0.0, 0.02),
          scale=15.0,
          horizontal_scale=0.25,
          resolution=0.25,
          octaves=2,
          persistence=0.15,
          border_width=0.5,
        )
      },
      add_lights=False,
    )
  non_foot_collision_cfg = ContactSensorCfg(
    name="non_foot_ground_collision",
    primary=ContactMatch(
      mode="body",
      pattern=".*",
      entity="robot",
      exclude=(r".*foot.*",),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  existing_sensors = tuple(cfg.scene.sensors or ())
  cfg.scene.sensors = (*existing_sensors, non_foot_collision_cfg)
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

  # Tuned with reward-share diagnostics so AMP is no longer numerically drowned out
  # by collision penalties during early reactive-soccer training.
  cfg.rewards = {
    "survival": RewardTermCfg(func=mdp.survival_reward, weight=3.0),
    "upright": RewardTermCfg(func=mdp.upright_reward, weight=2.0),
    "termination": RewardTermCfg(func=mdp.fall_termination_penalty, weight=-1000.0),
    "stagnation": RewardTermCfg(func=mdp.StagnationPenalty, weight=-100.0),
    "goal_scored": RewardTermCfg(func=mdp.goal_scored_reward, weight=15.0),
    "ball_approach": RewardTermCfg(func=mdp.ball_approach_reward, weight=50.0),
    "goal_progress": RewardTermCfg(func=mdp.goal_progress_reward, weight=500.0),
    "ball_goal_alignment": RewardTermCfg(
      func=mdp.ball_goal_alignment_reward, weight=0.0
    ),
    "touch_ball": RewardTermCfg(func=mdp.touch_ball_impulse_reward, weight=15.0),
    "robot_ball_alignment": RewardTermCfg(
      func=mdp.robot_ball_alignment_reward, weight=10.0
    ),
    "head_pitch_alignment": RewardTermCfg(
      func=mdp.head_pitch_alignment_penalty, weight=-0.5
    ),
    "head_yaw_alignment": RewardTermCfg(
      func=mdp.head_yaw_alignment_penalty, weight=-0.5
    ),
    "sideways_kick": RewardTermCfg(func=mdp.sideways_kick_reward, weight=20.0),
    "forward_kick_penalty": RewardTermCfg(
      func=mdp.forward_kick_penalty, weight=-20.0
    ),
    "foot_proximity_penalty": RewardTermCfg(
      func=mdp.foot_proximity_penalty, weight=-5.0
    ),
    "near_ball_control": RewardTermCfg(
      func=mdp.near_ball_control_reward, weight=5.0
    ),
    "head_action_rate_penalty": RewardTermCfg(
      func=mdp.head_action_rate_penalty,
      weight=-15.0,
      params={"head_action_indices": ()},
    ),
    "leg_action_rate_penalty": RewardTermCfg(
      func=mdp.leg_action_rate_penalty, weight=-1.0
    ),
    "joint_limit_penalty": RewardTermCfg(func=mdp.joint_limit_penalty, weight=-100.0),
    "base_acc_penalty": RewardTermCfg(func=mdp.BaseAccelerationPenalty, weight=-0.001),
    "non_foot_collision_penalty": RewardTermCfg(
      func=mdp.non_foot_collision_penalty,
      weight=-100.0,
      params={"sensor_name": non_foot_collision_cfg.name, "force_threshold": 10.0},
    ),
    "amp_style": RewardTermCfg(func=mdp.amp_style_reward, weight=40.0),
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

  final_reset_from_motion_prob = 0.5
  training_curriculum = SimpleNamespace(
    phase1_end_iter=1000,
    phase2_end_iter=3000,
    phase1_ball_pose_range={
      "x": (-1.15, -0.45),
      "y": (-0.6, 0.6),
      "z": (0.0, 0.0),
    },
    phase2_ball_pose_range={
      "x": (-1.25, 1.0),
      "y": (-1.5, 1.5),
      "z": (0.0, 0.0),
    },
    phase3_ball_pose_range={
      "x": (-4.0, 4.0),
      "y": (-3.5, 3.5),
      "z": (0.0, 0.0),
    },
    phase1_reset_from_motion_prob=0.5,
    phase2_reset_from_motion_prob=0.2,
    phase3_reset_from_motion_prob=final_reset_from_motion_prob,
    phase1_ball_teleport_prob=0.0,
    phase2_ball_teleport_prob=0.15,
    phase3_ball_teleport_prob=0.35,
    phase1_ball_velocity_prob=0.0,
    phase2_ball_velocity_prob=0.25,
    phase3_ball_velocity_prob=0.5,
    phase1_robot_velocity_prob=0.0,
    phase2_robot_velocity_prob=0.1,
    phase3_robot_velocity_prob=0.4,
    phase1_robot_push_prob=0.0,
    phase2_robot_push_prob=0.1,
    phase3_robot_push_prob=0.3,
  )
  cfg.reset_from_motion_prob = training_curriculum.phase1_reset_from_motion_prob
  cfg.training_curriculum = training_curriculum

  cfg.events["motion_clip_reset"] = EventTermCfg(
    func=mdp.MotionClipResetEvent,
    mode="reset",
    params={
      "motion_root": str(Path("data/soccer_amp/motions_unified")),
      "probability": cfg.reset_from_motion_prob,
      "asset_cfg": SceneEntityCfg("robot", joint_names=G1_COMP_BODY_JOINT_NAMES),
    },
  )

  cfg.events["reset_ball_only"] = EventTermCfg(
    func=mdp.reset_ball_only,
    mode="reset",
    params={"pose_range": training_curriculum.phase1_ball_pose_range},
  )
  cfg.events["ball_physics_randomization"] = EventTermCfg(
    func=mdp.ball_physics_randomization,
    mode="reset",
    params={},
  )
  cfg.events["robot_base_mass_randomization"] = EventTermCfg(
    func=mdp.robot_base_mass_randomization,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
  )
  cfg.events["robot_base_com_randomization"] = EventTermCfg(
    func=mdp.robot_base_com_randomization,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
  )
  cfg.events["ball_teleport_disturbance"] = EventTermCfg(
    func=mdp.ball_teleport_disturbance,
    mode="interval",
    interval_range_s=(4.0, 8.0),
    params={
      "pose_range": training_curriculum.phase1_ball_pose_range,
      "probability": training_curriculum.phase1_ball_teleport_prob,
    },
  )
  cfg.events["ball_velocity_disturbance"] = EventTermCfg(
    func=mdp.probabilistic_velocity_disturbance,
    mode="interval",
    interval_range_s=(2.0, 5.0),
    params={
      "velocity_range": {
        "x": (-1.5, 1.5),
        "y": (-1.5, 1.5),
      },
      "probability": training_curriculum.phase1_ball_velocity_prob,
      "asset_cfg": SceneEntityCfg("ball"),
    },
  )
  cfg.events["robot_velocity_disturbance"] = EventTermCfg(
    func=mdp.probabilistic_velocity_disturbance,
    mode="interval",
    interval_range_s=(2.0, 5.0),
    params={
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "roll": (-0.3, 0.3),
        "pitch": (-0.3, 0.3),
      },
      "probability": training_curriculum.phase1_robot_velocity_prob,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.events["robot_push_disturbance"] = EventTermCfg(
    func=envs_mdp.apply_body_impulse,
    mode="step",
    params={
      "force_range": (-40.0, 40.0),
      "torque_range": (-5.0, 5.0),
      "duration_s": (0.05, 0.15),
      "cooldown_s": (1.5, 3.0),
      "probability": training_curriculum.phase1_robot_push_prob,
      "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
      "body_point_offset": (0.0, 0.0, 0.1),
    },
  )
  cfg.control_hz = 50
  cfg.perception = SimpleNamespace(
    camera_hz=30,
    odometry_history_steps=50,
    odometry_update_hz=20,
    odom_hz=20,
    localization_mode="particle_filter",
    odometry_model_path="",
    detection_mean_hz=25.36,
    detection_std_hz=1.06,
    latency_mean_s=0.116,
    latency_std_s=0.018,
    landmark_mean_hz=25.36,
    landmark_std_hz=1.06,
    landmark_latency_mean_s=0.116,
    landmark_latency_std_s=0.018,
    landmark_fov_half_angle_deg=90.0,
    landmark_max_range_m=12.0,
    landmark_near_probability=0.9,
    landmark_guaranteed_range_m=7.0,
    landmark_decay_distance_m=3.0,
    landmark_dropout_prob=0.05,
    landmark_noise_std_base_m=0.03,
    landmark_noise_std_scale_per_m=0.01,
    pf_num_particles=128,
    pf_resample_threshold=0.5,
    pf_motion_noise_xy=0.03,
    pf_motion_noise_yaw=0.05,
    pf_measurement_sigma=0.15,
    pf_initial_xy_std=0.02,
    pf_initial_yaw_std=0.05,
  )
  cfg.scale_rewards_by_dt = True
  cfg.partial_reset_term_names = ("goal_scored", "ball_out_of_bounds")
  cfg.partial_reset_event_name = "reset_ball_only"
  cfg.partial_reset_as_done = False
  cfg.preserve_observation_history_on_partial_reset = True
  cfg.preserve_action_history_on_partial_reset = True
  cfg.episode_length_s = 60.0
  if play:
    cfg.episode_length_s = int(1e9)
    if cfg.scene.terrain is not None:
      cfg.scene.terrain.terrain_type = "plane"
      cfg.scene.terrain.terrain_generator = None
    for event_name in (
      "motion_clip_reset",
      "ball_physics_randomization",
      "robot_base_mass_randomization",
      "robot_base_com_randomization",
      "ball_teleport_disturbance",
      "ball_velocity_disturbance",
      "robot_velocity_disturbance",
      "robot_push_disturbance",
    ):
      cfg.events.pop(event_name, None)
  return cfg
