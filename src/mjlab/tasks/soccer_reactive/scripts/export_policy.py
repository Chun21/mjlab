"""Dedicated export entrypoint for the reactive soccer task."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.soccer_reactive.config.g1_comp.env_cfgs import (
  g1_comp_reactive_soccer_env_cfg,
)
from mjlab.tasks.soccer_reactive.config.g1_comp.rl_cfg import (
  g1_comp_reactive_soccer_runner_cfg,
)
from mjlab.tasks.soccer_reactive.rl.reactive_soccer_runner import ReactiveSoccerRunner


@dataclass(frozen=True)
class ReactiveSoccerExportArgs:
  checkpoint: str | None = None
  output_dir: str = "logs/rsl_rl/g1_comp_reactive_soccer/exported"


def export_policy(checkpoint: str | None, output_dir: str) -> Path:
  env = ManagerBasedRlEnv(g1_comp_reactive_soccer_env_cfg(play=True), device="cpu")
  try:
    wrapped = RslRlVecEnvWrapper(env)
    runner = ReactiveSoccerRunner(
      wrapped,
      g1_comp_reactive_soccer_runner_cfg().__dict__,
      device="cpu",
    )
    if checkpoint is not None:
      runner.load(checkpoint, map_location="cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "policy.pt"
    torch.save({"state": runner.alg.save()}, out_path)
    return out_path
  finally:
    env.close()


def main() -> None:
  args = tyro.cli(
    ReactiveSoccerExportArgs,
    prog=sys.argv[0],
    default=ReactiveSoccerExportArgs(),
  )
  export_policy(args.checkpoint, args.output_dir)


if __name__ == "__main__":
  main()
