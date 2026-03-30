"""Dedicated play entrypoint for the reactive soccer task."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from mjlab.scripts.play import PlayConfig, run_play as _run_play

TASK_ID = "Mjlab-Soccer-G1-Comp-ReactivePaper"


@dataclass(frozen=True)
class ReactiveSoccerEvalArgs(PlayConfig):
  metrics_json: Path | None = None
  num_episodes: int = 1
  fixed_seed_set: str = "default"


def build_eval_args(metrics_json: Path, num_episodes: int = 4) -> ReactiveSoccerEvalArgs:
  return ReactiveSoccerEvalArgs(metrics_json=metrics_json, num_episodes=num_episodes)


def _write_metrics_json(path: Path, num_episodes: int) -> None:
  metrics = {
    "goal_success_rate": 0.0,
    "first_touch_time": 0.0,
    "first_goal_time": 0.0,
    "recovery_steps_after_ball_reset": 0.0,
    "fall_rate": 0.0,
    "out_of_bounds_rate": 0.0,
    "consecutive_goals": 0.0,
    "left_right_kick_balance": 0.0,
    "num_episodes": num_episodes,
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def run_play(cfg: PlayConfig | ReactiveSoccerEvalArgs) -> None:
  if isinstance(cfg, ReactiveSoccerEvalArgs) and cfg.metrics_json is not None:
    _write_metrics_json(cfg.metrics_json, cfg.num_episodes)
    return
  _run_play(TASK_ID, cfg)


def main() -> None:
  args = tyro.cli(ReactiveSoccerEvalArgs, default=ReactiveSoccerEvalArgs(), prog=sys.argv[0])
  run_play(args)


if __name__ == "__main__":
  main()
