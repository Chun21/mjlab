"""Ablation evaluation entrypoint for reactive soccer."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from mjlab.tasks.soccer_reactive.scripts.play import (
  ReactiveSoccerEvalArgs,
  _collect_eval_metrics,
)


@dataclass(frozen=True)
class ReactiveSoccerAblationArgs:
  output_json: Path = Path("logs/rsl_rl/g1_comp_reactive_soccer/ablation.json")
  num_episodes: int = 16
  checkpoint_file: str | None = None
  device: str | None = None
  num_envs: int | None = None
  agent: Literal["zero", "random", "trained"] = "zero"
  no_decoder: bool = False
  single_critic: bool = False
  no_amp: bool = False
  no_symmetry: bool = False
  oracle_ball: bool = False


def _collect_ablation_metrics(args: ReactiveSoccerAblationArgs) -> dict[str, object]:
  eval_args = ReactiveSoccerEvalArgs(
    agent=args.agent,
    checkpoint_file=args.checkpoint_file,
    device=args.device,
    num_envs=args.num_envs,
    num_episodes=args.num_episodes,
  )
  payload = _collect_eval_metrics(eval_args)
  payload["ablation_flags"] = {
    "no_decoder": args.no_decoder,
    "single_critic": args.single_critic,
    "no_amp": args.no_amp,
    "no_symmetry": args.no_symmetry,
    "oracle_ball": args.oracle_ball,
  }
  return payload


def run_eval_ablation(args: ReactiveSoccerAblationArgs) -> dict[str, object]:
  payload = _collect_ablation_metrics(args)
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
  return payload


def main() -> None:
  args = tyro.cli(
    ReactiveSoccerAblationArgs,
    prog=sys.argv[0],
    default=ReactiveSoccerAblationArgs(),
  )
  run_eval_ablation(args)


if __name__ == "__main__":
  main()
