"""Dedicated training entrypoint for the reactive soccer task."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

import tyro

from mjlab.scripts.train import TrainConfig, launch_training as _launch_training

TASK_ID = "Mjlab-Soccer-G1-Comp-ReactivePaper"


@dataclass(frozen=True)
class ReactiveSoccerTrainConfig(TrainConfig):
  training_profile: Literal["default", "day1_8x4090"] = "default"


def _normalize_cli_args(argv: list[str]) -> list[str]:
  normalized: list[str] = []
  idx = 0
  while idx < len(argv):
    arg = argv[idx]
    if arg == '--gpu-ids' and idx + 1 < len(argv) and argv[idx + 1] == '':
      normalized.extend([arg, 'None'])
      idx += 2
      continue
    normalized.append(arg)
    idx += 1
  return normalized


def _apply_training_profile(cfg: ReactiveSoccerTrainConfig) -> ReactiveSoccerTrainConfig:
  if cfg.training_profile != 'day1_8x4090':
    return cfg
  cfg.agent.save_interval = min(cfg.agent.save_interval, 50)
  cfg.agent.max_iterations = max(cfg.agent.max_iterations, 100)
  return cfg


def launch_training(args: ReactiveSoccerTrainConfig | None = None) -> None:
  cfg = args or ReactiveSoccerTrainConfig.from_task(TASK_ID)  # type: ignore[arg-type]
  cfg = _apply_training_profile(cfg)
  _launch_training(TASK_ID, args=cfg)


def main() -> None:
  args = tyro.cli(
    ReactiveSoccerTrainConfig,
    args=_normalize_cli_args(sys.argv[1:]),
    default=ReactiveSoccerTrainConfig.from_task(TASK_ID),  # type: ignore[arg-type]
    prog=sys.argv[0],
  )
  launch_training(args=args)


if __name__ == "__main__":
  main()
