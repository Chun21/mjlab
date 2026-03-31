"""Soccer field specifications and helper builders."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SoccerFieldConfig:
  """Soccer field dimensions and markings."""

  field_length: float
  field_width: float
  goal_depth: float
  goal_width: float
  goal_height: float
  goal_area_length: float
  goal_area_width: float
  penalty_area_length: float
  penalty_area_width: float
  penalty_mark_dist: float
  center_circle_dia: float
  border_strip_width: float
  line_width: float
  mark_size: float
  spawn_height: float = 0.78
  ball_radius: float = 0.11


M_FIELD = SoccerFieldConfig(
  field_length=14.0,
  field_width=9.0,
  goal_depth=1.0,
  goal_width=2.6,
  goal_height=1.8,
  goal_area_length=1.0,
  goal_area_width=4.0,
  penalty_area_length=3.0,
  penalty_area_width=6.0,
  penalty_mark_dist=2.0,
  center_circle_dia=3.0,
  border_strip_width=1.0,
  line_width=0.07,
  mark_size=0.10,
)


@dataclass(frozen=True)
class FieldLineSpec:
  name: str
  size: tuple[float, float, float]
  position: tuple[float, float, float]
  orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GoalFrameSpec:
  name: str
  size: tuple[float, float, float]
  position: tuple[float, float, float]


def compute_single_g1_spawn_pose(
  field: SoccerFieldConfig = M_FIELD,
) -> tuple[float, float, float, float]:
  """Return the Soccer_Lab single-agent spawn mapped from the a4 reference pose."""
  del field  # The Soccer_Lab a4 pose is fixed for the M preset we mirror here.
  return (-1.75, 0.0, 0.78, 0.0)


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
  half_yaw = 0.5 * yaw
  return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def build_field_line_specs(
  field: SoccerFieldConfig = M_FIELD,
  line_height: float = 0.01,
  z_offset: float = 0.005,
) -> list[FieldLineSpec]:
  """Build cuboid field lines following the Soccer_Lab M-field layout."""
  lw = field.line_width
  length = field.field_length
  width = field.field_width
  ga_length = field.goal_area_length
  ga_width = field.goal_area_width
  pa_length = field.penalty_area_length
  pa_width = field.penalty_area_width
  half_length = length * 0.5

  line_specs = [
    FieldLineSpec(
      "sideline_top",
      (length + 2 * lw, lw, line_height),
      (0.0, width * 0.5 + lw * 0.5, z_offset),
    ),
    FieldLineSpec(
      "sideline_bottom",
      (length + 2 * lw, lw, line_height),
      (0.0, -(width * 0.5 + lw * 0.5), z_offset),
    ),
    FieldLineSpec(
      "goal_line_left",
      (lw, width, line_height),
      (-(length * 0.5 + lw * 0.5), 0.0, z_offset),
    ),
    FieldLineSpec(
      "goal_line_right",
      (lw, width, line_height),
      (length * 0.5 + lw * 0.5, 0.0, z_offset),
    ),
    FieldLineSpec("center_line", (lw, width, line_height), (0.0, 0.0, z_offset)),
    FieldLineSpec(
      "goal_area_left_top",
      (ga_length, lw, line_height),
      (-(length * 0.5 - ga_length * 0.5), ga_width * 0.5 + lw * 0.5, z_offset),
    ),
    FieldLineSpec(
      "goal_area_left_bottom",
      (ga_length, lw, line_height),
      (-(length * 0.5 - ga_length * 0.5), -(ga_width * 0.5 + lw * 0.5), z_offset),
    ),
    FieldLineSpec(
      "goal_area_left_front",
      (lw, ga_width + 2 * lw, line_height),
      (-(length * 0.5 - ga_length), 0.0, z_offset),
    ),
    FieldLineSpec(
      "goal_area_right_top",
      (ga_length, lw, line_height),
      ((length * 0.5 - ga_length * 0.5), ga_width * 0.5 + lw * 0.5, z_offset),
    ),
    FieldLineSpec(
      "goal_area_right_bottom",
      (ga_length, lw, line_height),
      ((length * 0.5 - ga_length * 0.5), -(ga_width * 0.5 + lw * 0.5), z_offset),
    ),
    FieldLineSpec(
      "goal_area_right_front",
      (lw, ga_width + 2 * lw, line_height),
      ((length * 0.5 - ga_length), 0.0, z_offset),
    ),
    FieldLineSpec(
      "penalty_area_left_top",
      (pa_length, lw, line_height),
      (-(length * 0.5 - pa_length * 0.5), pa_width * 0.5 + lw * 0.5, z_offset),
    ),
    FieldLineSpec(
      "penalty_area_left_bottom",
      (pa_length, lw, line_height),
      (-(length * 0.5 - pa_length * 0.5), -(pa_width * 0.5 + lw * 0.5), z_offset),
    ),
    FieldLineSpec(
      "penalty_area_left_front",
      (lw, pa_width + 2 * lw, line_height),
      (-(length * 0.5 - pa_length), 0.0, z_offset),
    ),
    FieldLineSpec(
      "penalty_area_right_top",
      (pa_length, lw, line_height),
      ((length * 0.5 - pa_length * 0.5), pa_width * 0.5 + lw * 0.5, z_offset),
    ),
    FieldLineSpec(
      "penalty_area_right_bottom",
      (pa_length, lw, line_height),
      ((length * 0.5 - pa_length * 0.5), -(pa_width * 0.5 + lw * 0.5), z_offset),
    ),
    FieldLineSpec(
      "penalty_area_right_front",
      (lw, pa_width + 2 * lw, line_height),
      ((length * 0.5 - pa_length), 0.0, z_offset),
    ),
    FieldLineSpec(
      "penalty_mark_left",
      (field.mark_size, field.mark_size, line_height),
      (-(half_length - field.penalty_mark_dist), 0.0, z_offset),
    ),
    FieldLineSpec(
      "penalty_mark_right",
      (field.mark_size, field.mark_size, line_height),
      ((half_length - field.penalty_mark_dist), 0.0, z_offset),
    ),
  ]

  center_circle_radius = field.center_circle_dia * 0.5
  num_segments = 40
  if center_circle_radius > 0.0:
    segment_angle = 2.0 * math.pi / num_segments
    segment_length = max(
      2.0 * center_circle_radius * math.sin(0.5 * segment_angle),
      lw,
    )
    for segment_idx in range(num_segments):
      theta = segment_idx * segment_angle
      line_specs.append(
        FieldLineSpec(
          name=f"center_circle_{segment_idx:02d}",
          size=(segment_length, lw, line_height),
          position=(
            center_circle_radius * math.cos(theta),
            center_circle_radius * math.sin(theta),
            z_offset,
          ),
          orientation=yaw_to_quat_wxyz(theta + math.pi * 0.5),
        )
      )

  return line_specs


def build_goal_frame_specs(field: SoccerFieldConfig = M_FIELD) -> list[GoalFrameSpec]:
  """Build simple cuboid posts and crossbars for both goals."""
  post_diameter = 0.1
  post_size = (post_diameter, post_diameter, field.goal_height)
  crossbar_size = (post_diameter, field.goal_width, post_diameter)
  rear_bar_size = (field.goal_depth, post_diameter, post_diameter)
  half_length = field.field_length * 0.5
  half_goal_width = field.goal_width * 0.5
  post_z = field.goal_height * 0.5

  return [
    GoalFrameSpec("left_post_top", post_size, (-half_length, half_goal_width, post_z)),
    GoalFrameSpec(
      "left_post_bottom", post_size, (-half_length, -half_goal_width, post_z)
    ),
    GoalFrameSpec(
      "left_crossbar", crossbar_size, (-half_length, 0.0, field.goal_height)
    ),
    GoalFrameSpec(
      "left_rear_top",
      rear_bar_size,
      (-(half_length + field.goal_depth * 0.5), half_goal_width, field.goal_height),
    ),
    GoalFrameSpec(
      "left_rear_bottom",
      rear_bar_size,
      (-(half_length + field.goal_depth * 0.5), -half_goal_width, field.goal_height),
    ),
    GoalFrameSpec("right_post_top", post_size, (half_length, half_goal_width, post_z)),
    GoalFrameSpec(
      "right_post_bottom", post_size, (half_length, -half_goal_width, post_z)
    ),
    GoalFrameSpec(
      "right_crossbar", crossbar_size, (half_length, 0.0, field.goal_height)
    ),
    GoalFrameSpec(
      "right_rear_top",
      rear_bar_size,
      (half_length + field.goal_depth * 0.5, half_goal_width, field.goal_height),
    ),
    GoalFrameSpec(
      "right_rear_bottom",
      rear_bar_size,
      (half_length + field.goal_depth * 0.5, -half_goal_width, field.goal_height),
    ),
  ]
