"""Base configuration for the single-robot soccer task."""

from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots import (
  G1_COMP_ACTION_SCALE,
  G1_COMP_BODY_JOINT_NAMES,
  get_g1_comp_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.soccer import mdp
from mjlab.tasks.soccer.field_specs import (
  M_FIELD,
  build_field_line_specs,
  build_goal_frame_specs,
  compute_single_g1_spawn_pose,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

_LEFT_GOAL_POS_W = (-M_FIELD.field_length * 0.5, 0.0, M_FIELD.ball_radius)
_RIGHT_GOAL_POS_W = (M_FIELD.field_length * 0.5, 0.0, M_FIELD.ball_radius)
_SOCCER_GRASS_TEXTURE = spec_cfg.TextureCfg(
  name="soccer_grass",
  type="2d",
  builtin="checker",
  mark="none",
  rgb1=(0.16, 0.48, 0.18),
  rgb2=(0.12, 0.40, 0.14),
  width=512,
  height=512,
)
_SOCCER_GRASS_MATERIAL = spec_cfg.MaterialCfg(
  name="soccer_grass",
  texuniform=True,
  texrepeat=(8.0, 8.0),
  reflectance=0.1,
  texture="soccer_grass",
  geom_names_expr=("terrain$",),
)


def get_soccer_ball_spec(
  radius: float = M_FIELD.ball_radius,
  mass: float = 0.43,
) -> mujoco.MjSpec:
  """Create the free-moving soccer ball spec."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint(name="ball_joint")
  body.add_geom(
    name="ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius, 0.0, 0.0),
    mass=mass,
    rgba=(0.95, 0.95, 0.95, 1.0),
  )
  return spec


def get_goal_spec(side: str) -> mujoco.MjSpec:
  """Create a fixed goal frame spec for the requested side."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=f"{side}_goal")
  prefix = f"{side}_"
  for frame_spec in build_goal_frame_specs(M_FIELD):
    if not frame_spec.name.startswith(prefix):
      continue
    body.add_geom(
      name=frame_spec.name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=tuple(dim * 0.5 for dim in frame_spec.size),
      pos=frame_spec.position,
      rgba=(1.0, 1.0, 1.0, 1.0),
    )
  return spec


def add_field_lines_to_spec(spec: mujoco.MjSpec) -> None:
  """Attach visual-only field line boxes to the scene spec."""
  body = spec.worldbody.add_body(name="soccer_field_lines")
  for line_spec in build_field_line_specs(M_FIELD):
    geom = body.add_geom(
      name=line_spec.name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=tuple(dim * 0.5 for dim in line_spec.size),
      pos=line_spec.position,
      quat=line_spec.orientation,
      rgba=(1.0, 1.0, 1.0, 1.0),
    )
    geom.contype = 0
    geom.conaffinity = 0


def make_soccer_env_cfg(
  robot_cfg: EntityCfg | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the base single-robot soccer task configuration."""
  robot_cfg = robot_cfg or get_g1_comp_robot_cfg()
  spawn_x, spawn_y, spawn_z, _spawn_yaw = compute_single_g1_spawn_pose(M_FIELD)
  robot_cfg.init_state.pos = (spawn_x, spawn_y, spawn_z)

  ball_cfg = EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(spawn_x + 1.0, spawn_y, M_FIELD.ball_radius),
    ),
    spec_fn=get_soccer_ball_spec,
  )

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=G1_COMP_BODY_JOINT_NAMES)
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=G1_COMP_BODY_JOINT_NAMES)
      },
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "ball_pos_b": ObservationTermCfg(
      func=mdp.ball_pos_b,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "ball_vel_b": ObservationTermCfg(
      func=mdp.ball_vel_b,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "goal_pos_b": ObservationTermCfg(
      func=mdp.goal_pos_b,
      params={"goal_pos_w": _RIGHT_GOAL_POS_W},
    ),
  }

  critic_terms = {
    **actor_terms,
    "ball_pos_w": ObservationTermCfg(func=mdp.ball_pos_w),
    "ball_vel_w": ObservationTermCfg(func=mdp.ball_vel_w),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=G1_COMP_BODY_JOINT_NAMES,
      scale=G1_COMP_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  events = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default,
      mode="reset",
    ),
    "reset_robot": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.15, 0.15),
          "y": (-0.25, 0.25),
          "z": (0.0, 0.0),
          "yaw": (-0.2, 0.2),
        },
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reset_ball": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.30, 0.30),
          "y": (-0.40, 0.40),
          "z": (0.0, 0.0),
        },
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("ball"),
      },
    ),
  }

  rewards = {
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.05),
    "upright": RewardTermCfg(func=mdp.upright_reward, weight=0.25),
    "approach_ball": RewardTermCfg(func=mdp.approach_ball, weight=0.6),
    "face_ball": RewardTermCfg(func=mdp.face_ball, weight=0.15),
    "near_ball_bonus": RewardTermCfg(func=mdp.near_ball_bonus, weight=0.2),
    "ball_to_goal_progress": RewardTermCfg(
      func=mdp.ball_to_goal_progress,
      weight=1.0,
      params={"goal_pos_w": _RIGHT_GOAL_POS_W},
    ),
    "action_smoothness_penalty": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.01,
    ),
    "joint_velocity_penalty": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-1.0e-4,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=G1_COMP_BODY_JOINT_NAMES)
      },
    ),
  }

  terminations = {
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
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        textures=(_SOCCER_GRASS_TEXTURE,),
        materials=(_SOCCER_GRASS_MATERIAL,),
      ),
      num_envs=1,
      entities={
        "robot": robot_cfg,
        "ball": ball_cfg,
        "left_goal": EntityCfg(spec_fn=lambda: get_goal_spec("left")),
        "right_goal": EntityCfg(spec_fn=lambda: get_goal_spec("right")),
      },
      spec_fn=add_field_lines_to_spec,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=5.5,
      fovy=55.0,
      elevation=-18.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=128,
      njmax=512,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
