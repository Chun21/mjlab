"""G1 Comp robot constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  ACTUATOR_5020,
  ACTUATOR_7520_14,
  ACTUATOR_7520_22,
  DAMPING_5020,
  DAMPING_7520_14,
  DAMPING_7520_22,
  STIFFNESS_5020,
  STIFFNESS_7520_14,
  STIFFNESS_7520_22,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

G1_COMP_URDF: Path = Path(__file__).parent / "urdf" / "g1_comp.urdf"
assert G1_COMP_URDF.exists()

G1_COMP_BODY_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
)

G1_COMP_SENSOR_JOINT_NAMES: tuple[str, ...] = (
  "xl330_joint",
  "d455_joint",
)

G1_COMP_BODY_ACTUATORS: tuple[BuiltinPositionActuatorCfg, ...] = (
  BuiltinPositionActuatorCfg(
    target_names_expr=(
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
    ),
    stiffness=STIFFNESS_5020,
    damping=DAMPING_5020,
    effort_limit=ACTUATOR_5020.effort_limit,
    armature=ACTUATOR_5020.reflected_inertia,
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(
      "left_hip_pitch_joint",
      "left_hip_yaw_joint",
      "right_hip_pitch_joint",
      "right_hip_yaw_joint",
      "waist_yaw_joint",
    ),
    stiffness=STIFFNESS_7520_14,
    damping=DAMPING_7520_14,
    effort_limit=ACTUATOR_7520_14.effort_limit,
    armature=ACTUATOR_7520_14.reflected_inertia,
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(
      "left_hip_roll_joint",
      "left_knee_joint",
      "right_hip_roll_joint",
      "right_knee_joint",
    ),
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    effort_limit=ACTUATOR_7520_22.effort_limit,
    armature=ACTUATOR_7520_22.reflected_inertia,
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
    ),
    stiffness=STIFFNESS_5020 * 2.0,
    damping=DAMPING_5020 * 2.0,
    effort_limit=ACTUATOR_5020.effort_limit * 2.0,
    armature=ACTUATOR_5020.reflected_inertia * 2.0,
  ),
)

G1_COMP_SENSOR_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=G1_COMP_SENSOR_JOINT_NAMES,
  stiffness=5.0,
  damping=0.5,
  effort_limit=5.0,
  armature=0.001,
)

G1_COMP_ARTICULATION = EntityArticulationInfoCfg(
  actuators=G1_COMP_BODY_ACTUATORS + (G1_COMP_SENSOR_ACTUATOR,),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_comp_spec() -> mujoco.MjSpec:
  """Load the G1 Comp URDF and convert it to a floating-base MuJoCo spec."""
  spec = mujoco.MjSpec.from_file(str(G1_COMP_URDF))
  spec.body("pelvis").add_freejoint(name="floating_base_joint")
  return spec


def get_g1_comp_robot_cfg() -> EntityCfg:
  """Get a fresh G1 Comp robot configuration instance."""
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.78),
      joint_pos={
        "left_hip_pitch_joint": -0.20,
        "left_knee_joint": 0.42,
        "left_ankle_pitch_joint": -0.23,
        "right_hip_pitch_joint": -0.20,
        "right_knee_joint": 0.42,
        "right_ankle_pitch_joint": -0.23,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.16,
        "right_shoulder_pitch_joint": 0.35,
        "left_elbow_joint": 0.87,
        "right_elbow_joint": 0.87,
        "xl330_joint": 0.0,
        "d455_joint": 0.0,
      },
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_g1_comp_spec,
    articulation=G1_COMP_ARTICULATION,
  )


G1_COMP_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in G1_COMP_BODY_ACTUATORS:
  assert actuator_cfg.effort_limit is not None
  for joint_name in actuator_cfg.target_names_expr:
    G1_COMP_ACTION_SCALE[joint_name] = (
      0.25 * actuator_cfg.effort_limit / actuator_cfg.stiffness
    )
