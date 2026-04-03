"""Dedicated export entrypoint for the reactive soccer task."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mjlab
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.train import _cfg_to_dict
from mjlab.tasks.soccer_reactive.config.g1_comp.env_cfgs import (
  g1_comp_reactive_soccer_env_cfg,
)
from mjlab.tasks.soccer_reactive.config.g1_comp.rl_cfg import (
  g1_comp_reactive_soccer_runner_cfg,
)
from mjlab.tasks.soccer_reactive.odometry_artifacts import (
  ODOMETRY_INFO_KEY,
  apply_odometry_artifact_binding,
  apply_training_odometry_artifact_binding,
  extract_odometry_info_from_checkpoint,
)
from mjlab.tasks.soccer_reactive.rl.reactive_soccer_runner import ReactiveSoccerRunner


@dataclass(frozen=True)
class ReactiveSoccerExportArgs:
  checkpoint: str | None = None
  output_dir: str = "logs/rsl_rl/g1_comp_reactive_soccer/exported"
  odometry_model_path: str | None = None
  localization_mode: str | None = None


def export_policy(
  checkpoint: str | None,
  output_dir: str,
  *,
  odometry_model_path: str | None = None,
  localization_mode: str | None = None,
) -> Path:
  env_cfg = g1_comp_reactive_soccer_env_cfg(play=True)
  if localization_mode is not None or odometry_model_path is not None:
    odometry_payload = apply_odometry_artifact_binding(
      env_cfg,
      checkpoint_path=checkpoint,
      odometry_model_path=odometry_model_path,
      localization_mode=localization_mode,
    )
  else:
    odometry_payload = apply_training_odometry_artifact_binding(
      env_cfg,
      checkpoint_path=checkpoint,
      bootstrap_localization_mode="ground_truth",
    )
    if odometry_payload.get("binding_source") == "bootstrap_localization":
      print(
        "[WARN] Export is bootstrapping with ground-truth localization "
        "because no learned odometry checkpoint could be resolved.",
        flush=True,
      )
  env = ManagerBasedRlEnv(env_cfg, device="cpu")
  try:
    wrapped = RslRlVecEnvWrapper(env)
    runner = ReactiveSoccerRunner(
      wrapped,
      _cfg_to_dict(g1_comp_reactive_soccer_runner_cfg()),
      device="cpu",
      enable_distributed=False,
    )
    checkpoint_infos: dict[str, object] = {}
    if checkpoint is not None:
      checkpoint_infos = runner.load(checkpoint, map_location="cpu") or {}
      recorded_payload = checkpoint_infos.get(ODOMETRY_INFO_KEY, {})
      if isinstance(recorded_payload, dict):
        odometry_payload = {**recorded_payload, **odometry_payload}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "policy.pt"
    torch.save(
      {
        "state": runner.alg.save(),
        "infos": {
          ODOMETRY_INFO_KEY: odometry_payload,
        },
        "metadata": {
          ODOMETRY_INFO_KEY: odometry_payload,
        },
      },
      out_path,
    )
    metadata_path = out_dir / "odometry_metadata.json"
    metadata_path.write_text(json.dumps(odometry_payload, indent=2, ensure_ascii=False))
    return out_path
  finally:
    env.close()


def main() -> None:
  args = tyro.cli(
    ReactiveSoccerExportArgs,
    prog=sys.argv[0],
    default=ReactiveSoccerExportArgs(),
    config=mjlab.TYRO_FLAGS,
  )
  export_policy(
    args.checkpoint,
    args.output_dir,
    odometry_model_path=args.odometry_model_path,
    localization_mode=args.localization_mode,
  )


if __name__ == "__main__":
  main()
