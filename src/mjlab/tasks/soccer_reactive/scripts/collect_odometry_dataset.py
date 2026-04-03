"""Collect offline odometry supervision data from reactive soccer policy rollouts."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mjlab
import torch
import tyro

from mjlab.asset_zoo.robots import G1_COMP_BODY_JOINT_NAMES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.soccer_reactive.config.g1_comp.env_cfgs import (
  g1_comp_reactive_soccer_env_cfg,
)
from mjlab.tasks.soccer_reactive.config.g1_comp.rl_cfg import (
  g1_comp_reactive_soccer_runner_cfg,
)
from mjlab.tasks.soccer_reactive.mdp.odometry import (
  build_odometry_proprio_vector,
  get_ground_truth_field_pose,
)
from mjlab.tasks.soccer_reactive.mdp.localization_filter import compute_body_frame_delta
from mjlab.tasks.soccer_reactive.odometry_artifacts import apply_odometry_artifact_binding
from mjlab.tasks.soccer_reactive.rl.reactive_soccer_runner import ReactiveSoccerRunner
from mjlab.utils.buffers.circular_buffer import CircularBuffer


@dataclass(frozen=True)
class CollectOdometryDatasetArgs:
  output_path: str = "data/soccer_reactive/odometry/odometry_dataset.pt"
  experiment_dir: str = "logs/rsl_rl/g1_comp_reactive_soccer"
  checkpoint_files: tuple[str, ...] = ()
  default_recent_checkpoints: int = 5
  samples_per_checkpoint: int = 8192
  num_envs: int = 32
  play_env: bool = False
  device: str | None = None
  seed: int = 20260401


def _device_or_default(device: str | None) -> str:
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def _extract_iteration_from_path(path: Path) -> int:
  stem = path.stem
  if "_" not in stem:
    return -1
  try:
    return int(stem.split("_")[-1])
  except ValueError:
    return -1


def _load_json(path: Path) -> dict[str, Any]:
  try:
    payload = json.loads(path.read_text())
  except Exception:
    return {}
  return payload if isinstance(payload, dict) else {}


def _select_recent_eval_checkpoints(experiment_dir: str, limit: int) -> list[Path]:
  root = Path(experiment_dir)
  if not root.exists():
    raise FileNotFoundError(f"Experiment directory not found: {root}")

  candidates: list[tuple[float, int, Path]] = []
  for checkpoint in root.glob("**/model_*.pt"):
    iteration = _extract_iteration_from_path(checkpoint)
    if iteration < 0:
      continue
    metrics_path = checkpoint.parent / "eval" / f"metrics_{iteration}.json"
    if not metrics_path.exists():
      continue
    metrics = _load_json(metrics_path)
    goal_success_rate = metrics.get("goal_success_rate")
    if goal_success_rate is None:
      continue
    try:
      success_rate = float(goal_success_rate)
    except (TypeError, ValueError):
      continue
    if not math.isfinite(success_rate):
      continue
    candidates.append((checkpoint.stat().st_mtime, iteration, checkpoint))

  candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
  selected = [path for _mtime, _iteration, path in candidates[:limit]]
  if not selected:
    raise FileNotFoundError(
      "No stable reactive checkpoints with eval metrics were found under "
      f"{root}."
    )
  return selected


def _resolve_checkpoint_list(args: CollectOdometryDatasetArgs) -> list[Path]:
  if args.checkpoint_files:
    checkpoints = [Path(path) for path in args.checkpoint_files]
    missing = [str(path) for path in checkpoints if not path.exists()]
    if missing:
      raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")
    return checkpoints
  return _select_recent_eval_checkpoints(
    args.experiment_dir,
    limit=int(args.default_recent_checkpoints),
  )


def _build_policy_and_env(
  checkpoint_path: Path,
  *,
  num_envs: int,
  play_env: bool,
  device: str,
  seed: int,
) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper, ReactiveSoccerRunner]:
  env_cfg = g1_comp_reactive_soccer_env_cfg(play=play_env)
  env_cfg.scene.num_envs = int(num_envs)
  env_cfg.seed = int(seed)
  apply_odometry_artifact_binding(
    env_cfg,
    checkpoint_path=checkpoint_path,
    localization_mode="ground_truth",
    odometry_model_path="",
    require_if_needed=False,
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env)
  runner = ReactiveSoccerRunner(
    wrapped,
    asdict(g1_comp_reactive_soccer_runner_cfg()),
    device=device,
  )
  runner.load(str(checkpoint_path), map_location=device)
  return env, wrapped, runner


def _collect_from_checkpoint(
  checkpoint_path: Path,
  *,
  samples_target: int,
  num_envs: int,
  play_env: bool,
  device: str,
  seed: int,
) -> dict[str, torch.Tensor | list[str] | list[int] | dict[str, Any]]:
  env, wrapped, runner = _build_policy_and_env(
    checkpoint_path,
    num_envs=num_envs,
    play_env=play_env,
    device=device,
    seed=seed,
  )
  joint_cfg = SceneEntityCfg("robot", joint_names=G1_COMP_BODY_JOINT_NAMES)
  joint_cfg.resolve(env.scene)
  history_steps = int(getattr(env.cfg.perception, "odometry_history_steps", 50))
  odom_period = 1.0 / float(getattr(env.cfg.perception, "odometry_update_hz", 20.0))

  proprio_history = CircularBuffer(
    max_len=history_steps,
    batch_size=env.num_envs,
    device=env.device,
  )
  obs, _ = wrapped.reset()
  del obs
  policy = runner.get_inference_policy(device=device)

  def current_proprio() -> torch.Tensor:
    return build_odometry_proprio_vector(
      projected_gravity=envs_mdp.projected_gravity(env),
      base_ang_vel=envs_mdp.base_ang_vel(env),
      joint_pos=envs_mdp.joint_pos_rel(env, asset_cfg=joint_cfg),
      joint_vel=envs_mdp.joint_vel_rel(env, asset_cfg=joint_cfg),
      previous_action=envs_mdp.last_action(env),
    )

  proprio_history.append(current_proprio())
  last_boundary_pose = get_ground_truth_field_pose(env)
  pending_history = proprio_history.buffer.clone()
  pending_prev_delta = torch.zeros((env.num_envs, 3), device=env.device)
  time_since_boundary = torch.zeros(env.num_envs, device=env.device)

  collected_history: list[torch.Tensor] = []
  collected_prev_delta: list[torch.Tensor] = []
  collected_target_delta: list[torch.Tensor] = []
  source_iteration: list[int] = []
  checkpoint_iteration = _extract_iteration_from_path(checkpoint_path)

  try:
    while sum(tensor.shape[0] for tensor in collected_target_delta) < samples_target:
      actions = policy(wrapped.get_observations())
      _obs, _rew, dones, _extras = wrapped.step(actions)
      done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
      if len(done_ids) > 0:
        proprio_history.reset(batch_ids=done_ids)
        time_since_boundary[done_ids] = 0.0
        pending_prev_delta[done_ids] = 0.0

      proprio_history.append(current_proprio())
      current_pose = get_ground_truth_field_pose(env)
      if len(done_ids) > 0:
        last_boundary_pose[done_ids] = current_pose[done_ids]
        pending_history[done_ids] = proprio_history.buffer[done_ids]

      time_since_boundary += env.step_dt
      boundary_ids = (time_since_boundary >= (odom_period - 1.0e-6)).nonzero(as_tuple=False).squeeze(-1)
      if len(boundary_ids) == 0:
        continue

      target_delta = compute_body_frame_delta(
        last_boundary_pose[boundary_ids],
        current_pose[boundary_ids],
      )
      collected_history.append(pending_history[boundary_ids].detach().cpu())
      collected_prev_delta.append(pending_prev_delta[boundary_ids].detach().cpu())
      collected_target_delta.append(target_delta.detach().cpu())
      source_iteration.extend([checkpoint_iteration] * int(boundary_ids.numel()))

      last_boundary_pose[boundary_ids] = current_pose[boundary_ids]
      pending_prev_delta[boundary_ids] = target_delta
      pending_history[boundary_ids] = proprio_history.buffer[boundary_ids]
      time_since_boundary[boundary_ids] = (
        time_since_boundary[boundary_ids] - odom_period
      ).clamp(min=0.0)
  finally:
    wrapped.close()

  if not collected_target_delta:
    raise RuntimeError(f"No odometry samples were collected from {checkpoint_path}.")

  return {
    "proprio_history": torch.cat(collected_history, dim=0)[:samples_target],
    "prev_delta_pose": torch.cat(collected_prev_delta, dim=0)[:samples_target],
    "target_delta_pose": torch.cat(collected_target_delta, dim=0)[:samples_target],
    "checkpoint_iterations": source_iteration[:samples_target],
  }


def collect_odometry_dataset(args: CollectOdometryDatasetArgs) -> Path:
  torch.manual_seed(args.seed)
  device = _device_or_default(args.device)
  checkpoints = _resolve_checkpoint_list(args)

  all_history: list[torch.Tensor] = []
  all_prev_delta: list[torch.Tensor] = []
  all_target_delta: list[torch.Tensor] = []
  checkpoint_sources: list[str] = []
  checkpoint_iterations: list[int] = []

  for idx, checkpoint in enumerate(checkpoints):
    shard = _collect_from_checkpoint(
      checkpoint,
      samples_target=int(args.samples_per_checkpoint),
      num_envs=int(args.num_envs),
      play_env=bool(args.play_env),
      device=device,
      seed=int(args.seed + idx),
    )
    all_history.append(shard["proprio_history"])
    all_prev_delta.append(shard["prev_delta_pose"])
    all_target_delta.append(shard["target_delta_pose"])
    checkpoint_sources.append(str(checkpoint))
    checkpoint_iterations.extend(shard["checkpoint_iterations"])

  payload = {
    "proprio_history": torch.cat(all_history, dim=0),
    "prev_delta_pose": torch.cat(all_prev_delta, dim=0),
    "target_delta_pose": torch.cat(all_target_delta, dim=0),
    "metadata": {
      "checkpoint_sources": checkpoint_sources,
      "checkpoint_iterations": checkpoint_iterations,
      "history_steps": int(all_history[0].shape[1]),
      "proprio_dim": int(all_history[0].shape[2]),
      "samples_per_checkpoint": int(args.samples_per_checkpoint),
      "num_envs": int(args.num_envs),
      "play_env": bool(args.play_env),
      "seed": int(args.seed),
    },
  }

  output_path = Path(args.output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(payload, output_path)
  return output_path


def main() -> None:
  args = tyro.cli(
    CollectOdometryDatasetArgs,
    prog=sys.argv[0],
    default=CollectOdometryDatasetArgs(),
    config=mjlab.TYRO_FLAGS,
  )
  output_path = collect_odometry_dataset(args)
  print(f"[INFO] Saved odometry dataset to {output_path}")


if __name__ == "__main__":
  main()
