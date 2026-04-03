"""Dedicated training entrypoint for the reactive soccer task."""

from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import mjlab
import tyro

from mjlab.scripts.train import TrainConfig, launch_training as _launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

TASK_ID = "Mjlab-Soccer-G1-Comp-ReactivePaper"


@dataclass(frozen=True)
class ReactiveSoccerTrainConfig(TrainConfig):
  training_profile: Literal["default", "day1_8x4090"] = "default"

  @staticmethod
  def from_task(task_id: str) -> "ReactiveSoccerTrainConfig":
    return build_reactive_soccer_train_cfg(task_id=task_id)


def _validate_task_id(task_id: str) -> str:
  if task_id != TASK_ID:
    raise ValueError(
      f"Reactive soccer 专用 train 入口只支持任务 {TASK_ID!r}，收到 {task_id!r}。"
    )
  return task_id


def _normalize_cli_args(argv: list[str]) -> list[str]:
  normalized: list[str] = []
  idx = 0
  while idx < len(argv):
    arg = argv[idx]
    if arg == '--gpu-ids' and idx + 1 < len(argv) and argv[idx + 1] == '':
      normalized.extend([arg, 'None'])
      idx += 2
      continue
    if arg == '--gpu-ids' and idx + 1 < len(argv):
      raw_value = argv[idx + 1].strip()
      if raw_value.startswith('[') and raw_value.endswith(']'):
        inner = raw_value[1:-1].strip()
        if inner == '':
          normalized.extend([arg, 'None'])
        else:
          normalized.append(arg)
          normalized.extend(
            token.strip() for token in inner.split(',') if token.strip() != ''
          )
        idx += 2
        continue
    normalized.append(arg)
    idx += 1
  return normalized


def _apply_training_profile(cfg: ReactiveSoccerTrainConfig) -> ReactiveSoccerTrainConfig:
  if cfg.training_profile != 'day1_8x4090':
    return cfg
  default_agent_cfg = load_rl_cfg(TASK_ID)
  if cfg.agent.num_steps_per_env == default_agent_cfg.num_steps_per_env:
    cfg.agent.num_steps_per_env = 32
  if cfg.agent.save_interval == default_agent_cfg.save_interval:
    cfg.agent.save_interval = 50
  if cfg.gpu_ids == [0]:
    object.__setattr__(cfg, "gpu_ids", "all")
  existing_tags = tuple(cfg.agent.wandb_tags)
  if "day1_8x4090" not in existing_tags:
    cfg.agent.wandb_tags = (*existing_tags, "day1_8x4090")
  return cfg


def build_reactive_soccer_train_cfg(
  *,
  task_id: str = TASK_ID,
  training_profile: Literal["default", "day1_8x4090"] = "default",
) -> ReactiveSoccerTrainConfig:
  task_id = _validate_task_id(task_id)
  cfg = ReactiveSoccerTrainConfig(
    env=load_env_cfg(task_id),
    agent=load_rl_cfg(task_id),
    training_profile=training_profile,
  )
  cfg = _restore_reactive_env_extensions(cfg, task_id=task_id)
  return _apply_training_profile(cfg)


def _restore_reactive_env_extensions(
  cfg: ReactiveSoccerTrainConfig,
  *,
  task_id: str = TASK_ID,
) -> ReactiveSoccerTrainConfig:
  task_id = _validate_task_id(task_id)
  template_env = load_env_cfg(task_id)
  for attr_name in (
    "perception",
    "control_hz",
    "training_curriculum",
    "reset_from_motion_prob",
  ):
    if not hasattr(cfg.env, attr_name) and hasattr(template_env, attr_name):
      setattr(cfg.env, attr_name, deepcopy(getattr(template_env, attr_name)))
  return cfg


def parse_reactive_soccer_train_args(argv: list[str]) -> ReactiveSoccerTrainConfig:
  args = tyro.cli(
    ReactiveSoccerTrainConfig,
    args=_normalize_cli_args(argv),
    default=build_reactive_soccer_train_cfg(),
    prog=sys.argv[0],
    config=mjlab.TYRO_FLAGS,
  )
  args = _restore_reactive_env_extensions(args)
  return _apply_training_profile(args)


def launch_training(args: ReactiveSoccerTrainConfig | None = None) -> None:
  cfg = args or build_reactive_soccer_train_cfg()
  cfg = _restore_reactive_env_extensions(cfg, task_id=TASK_ID)
  cfg = _apply_training_profile(cfg)
  _launch_training(TASK_ID, args=cfg)


def main() -> None:
  args = parse_reactive_soccer_train_args(sys.argv[1:])
  launch_training(args=args)


if __name__ == "__main__":
  main()
