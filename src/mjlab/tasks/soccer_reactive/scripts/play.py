"""Dedicated play / evaluation entrypoint for the reactive soccer task."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.play import PlayConfig
from mjlab.scripts.train import _cfg_to_dict
from mjlab.tasks.soccer_reactive.config.g1_comp.env_cfgs import (
  g1_comp_reactive_soccer_env_cfg,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.soccer_reactive.odometry_artifacts import (
  apply_odometry_artifact_binding,
  apply_training_odometry_artifact_binding,
)

TASK_ID = "Mjlab-Soccer-G1-Comp-ReactivePaper"
_FIXED_EVAL_SEED_SETS = {
  "default": 20260331,
  "paper": 424242,
}


@dataclass(frozen=True)
class ReactiveSoccerEvalArgs(PlayConfig):
  metrics_json: Path | None = None
  num_episodes: int = 1
  fixed_seed_set: str = "default"
  odometry_model_path: str | None = None
  localization_mode: str | None = None


def build_eval_args(
  metrics_json: Path,
  num_episodes: int = 4,
) -> ReactiveSoccerEvalArgs:
  return ReactiveSoccerEvalArgs(
    agent="zero",
    metrics_json=metrics_json,
    num_episodes=num_episodes,
  )


def build_eval_metrics_payload(
  *,
  num_episodes: int,
  goal_success_rate: float = 0.0,
  first_touch_time: float = 0.0,
  first_goal_time: float = 0.0,
  recovery_steps_after_ball_reset: float = 0.0,
  fall_rate: float = 0.0,
  out_of_bounds_rate: float = 0.0,
  consecutive_goals: float = 0.0,
  left_right_kick_balance: float = 0.0,
  region_success_rate: dict[str, float] | None = None,
  ablation_flags: dict[str, bool] | None = None,
) -> dict[str, object]:
  payload: dict[str, object] = {
    "goal_success_rate": goal_success_rate,
    "first_touch_time": first_touch_time,
    "first_goal_time": first_goal_time,
    "recovery_steps_after_ball_reset": recovery_steps_after_ball_reset,
    "fall_rate": fall_rate,
    "out_of_bounds_rate": out_of_bounds_rate,
    "consecutive_goals": consecutive_goals,
    "left_right_kick_balance": left_right_kick_balance,
    "num_episodes": num_episodes,
    "region_success_rate": region_success_rate
    or {
      "front_near": 0.0,
      "front_mid": 0.0,
      "wing": 0.0,
      "backfield": 0.0,
    },
  }
  if ablation_flags is not None:
    payload["ablation_flags"] = ablation_flags
  return payload


def _make_policy(cfg: ReactiveSoccerEvalArgs, env: RslRlVecEnvWrapper) -> callable:
  if cfg.agent == "zero":

    def _zero_policy(_obs) -> torch.Tensor:
      return torch.zeros(
        (env.num_envs, env.num_actions),
        device=env.device,
        dtype=torch.float32,
      )

    return _zero_policy

  if cfg.agent == "random":

    def _random_policy(_obs) -> torch.Tensor:
      return 2.0 * torch.rand(
        (env.num_envs, env.num_actions),
        device=env.device,
        dtype=torch.float32,
      ) - 1.0

    return _random_policy

  if cfg.checkpoint_file is None:
    raise ValueError("真实评测模式需要提供 checkpoint_file。")

  agent_cfg = load_rl_cfg(TASK_ID)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(
    env,
    _cfg_to_dict(agent_cfg),
    device=str(env.device),
    enable_distributed=False,
  )
  runner.load(str(cfg.checkpoint_file), map_location=str(env.device))
  return runner.get_inference_policy(device=str(env.device))


def _bucket_region(ball_xy: torch.Tensor) -> str:
  x = float(ball_xy[0].item())
  y = float(ball_xy[1].item())
  if x >= 3.0 and abs(y) <= 1.5:
    return "front_near"
  if x >= 0.0 and abs(y) <= 2.5:
    return "front_mid"
  if abs(y) > 2.5:
    return "wing"
  return "backfield"


def _canonical_ball_xy_for_region(
  ball_xy_w: torch.Tensor,
  env_origin_xy: torch.Tensor,
) -> torch.Tensor:
  return ball_xy_w[:2] - env_origin_xy[:2]


def _build_eval_env_cfg(num_envs: int, fixed_seed_set: str = "default"):
  from mjlab.tasks.soccer_reactive.config.g1_comp.env_cfgs import (
    g1_comp_reactive_soccer_env_cfg,
  )

  env_cfg = g1_comp_reactive_soccer_env_cfg(play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.episode_length_s = 60.0
  env_cfg.seed = _FIXED_EVAL_SEED_SETS.get(
    fixed_seed_set,
    _FIXED_EVAL_SEED_SETS["default"],
  )
  return env_cfg


def _read_soccer_reactive_step_events(
  env: ManagerBasedRlEnv,
  terminated: torch.Tensor,
  time_outs: torch.Tensor,
  extras: dict,
) -> dict[str, torch.Tensor]:
  """Read raw task events even when partial resets are suppressed from policy done."""
  bool_terminated = terminated.to(dtype=torch.bool)
  bool_time_outs = time_outs.to(dtype=torch.bool)
  raw_terms = getattr(env, "raw_termination_terms", {})
  goal_scored = raw_terms.get(
    "goal_scored",
    torch.zeros_like(bool_terminated, dtype=torch.bool),
  )
  ball_out_of_bounds = raw_terms.get(
    "ball_out_of_bounds",
    torch.zeros_like(bool_terminated, dtype=torch.bool),
  )
  robot_fallen = raw_terms.get(
    "robot_fallen",
    torch.zeros_like(bool_terminated, dtype=torch.bool),
  )
  raw_time_out = raw_terms.get("time_out", bool_time_outs)
  trial_done = goal_scored | ball_out_of_bounds | robot_fallen | raw_time_out
  return {
    "goal_scored": goal_scored,
    "ball_out_of_bounds": ball_out_of_bounds,
    "robot_fallen": robot_fallen,
    "time_out": raw_time_out,
    "episode_done": bool_terminated | bool_time_outs,
    "trial_done": trial_done,
    "partial_reset_env_ids": extras.get("soccer_reactive_events", {}).get(
      "partial_reset_env_ids",
      torch.empty(0, device=env.device, dtype=torch.long),
    ),
  }


def _collect_eval_metrics(cfg: ReactiveSoccerEvalArgs) -> dict[str, object]:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  from mjlab.tasks.soccer_reactive.rl.reactive_soccer_runner import (
    ReactiveSoccerRunner,
  )

  num_envs = cfg.num_envs or min(max(cfg.num_episodes, 1), 32)
  env_cfg = _build_eval_env_cfg(
    num_envs=num_envs,
    fixed_seed_set=cfg.fixed_seed_set,
  )
  if cfg.agent in {"zero", "random"}:
    apply_odometry_artifact_binding(
      env_cfg,
      checkpoint_path=None,
      odometry_model_path=cfg.odometry_model_path,
      localization_mode=(
        cfg.localization_mode
        if cfg.localization_mode is not None
        else "ground_truth"
      ),
    )
  elif cfg.localization_mode is not None or cfg.odometry_model_path is not None:
    apply_odometry_artifact_binding(
      env_cfg,
      checkpoint_path=cfg.checkpoint_file,
      odometry_model_path=cfg.odometry_model_path,
      localization_mode=cfg.localization_mode,
    )
  else:
    odometry_payload = apply_training_odometry_artifact_binding(
      env_cfg,
      checkpoint_path=cfg.checkpoint_file,
      bootstrap_localization_mode="ground_truth",
    )
    if odometry_payload.get("binding_source") == "bootstrap_localization":
      print(
        "[WARN] Periodic reactive-soccer eval is bootstrapping with "
        "ground-truth localization because no learned odometry checkpoint "
        "could be resolved from the training checkpoint.",
        flush=True,
      )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK_ID).clip_actions)
  try:
    policy = _make_policy(cfg, wrapped)
    obs, _ = wrapped.reset()
    start_ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2].clone()
    episode_step = torch.zeros(num_envs, device=env.device, dtype=torch.long)
    first_touch_step = torch.full((num_envs,), -1, device=env.device, dtype=torch.long)
    pending_recovery = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    completed = 0
    goal_count = 0
    fall_count = 0
    out_of_bounds_count = 0
    first_touch_samples: list[float] = []
    first_goal_samples: list[float] = []
    recovery_samples: list[float] = []
    region_attempts = {k: 0 for k in ("front_near", "front_mid", "wing", "backfield")}
    region_success = {k: 0 for k in ("front_near", "front_mid", "wing", "backfield")}
    current_goal_streak = 0
    max_goal_streak = 0
    left_kick_count = 0
    right_kick_count = 0

    while completed < cfg.num_episodes:
      actions = policy(obs)
      obs, _rew, dones, _extras = wrapped.step(actions)
      episode_step += 1

      ball_xy = env.scene["ball"].data.root_link_pos_w[:, :2]
      ball_speed = torch.norm(env.scene["ball"].data.root_link_lin_vel_w[:, :2], dim=1)
      robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
      ball_dist = torch.norm(ball_xy - robot_xy, dim=1)
      touch_mask = (first_touch_step < 0) & (ball_dist < 0.45) & (ball_speed > 0.1)
      if torch.any(touch_mask):
        first_touch_step[touch_mask] = episode_step[touch_mask]
      recovery_mask = pending_recovery & touch_mask
      if torch.any(recovery_mask):
        recovery_samples.extend(episode_step[recovery_mask].float().cpu().tolist())
        pending_recovery[recovery_mask] = False

      step_events = _read_soccer_reactive_step_events(
        env,
        dones,
        _extras.get(
          "time_outs",
          torch.zeros(num_envs, device=env.device, dtype=torch.bool),
        ),
        _extras,
      )
      done_ids = step_events["trial_done"].nonzero(as_tuple=False).squeeze(-1)
      if len(done_ids) == 0:
        continue

      goal_done = step_events["goal_scored"][done_ids]
      oob_done = step_events["ball_out_of_bounds"][done_ids]
      fall_done = step_events["robot_fallen"][done_ids]

      for local_idx, env_id in enumerate(done_ids.tolist()):
        if completed >= cfg.num_episodes:
          break
        region = _bucket_region(
          _canonical_ball_xy_for_region(
            start_ball_xy[env_id],
            env.scene.env_origins[env_id, :2],
          )
        )
        region_attempts[region] += 1
        if first_touch_step[env_id] >= 0:
          first_touch_samples.append(
            float(first_touch_step[env_id].item()) * env.step_dt
          )
        if bool(goal_done[local_idx].item()):
          goal_count += 1
          region_success[region] += 1
          first_goal_samples.append(float(episode_step[env_id].item()) * env.step_dt)
          current_goal_streak += 1
          max_goal_streak = max(max_goal_streak, current_goal_streak)
          left_leg = torch.sum(torch.abs(actions[env_id, :6])).item()
          right_leg = torch.sum(torch.abs(actions[env_id, 6:12])).item()
          if left_leg >= right_leg:
            left_kick_count += 1
          else:
            right_kick_count += 1
        else:
          current_goal_streak = 0
        if bool(oob_done[local_idx].item()):
          out_of_bounds_count += 1
        if bool(fall_done[local_idx].item()):
          fall_count += 1
        if bool(goal_done[local_idx].item() or oob_done[local_idx].item()):
          pending_recovery[env_id] = True

        completed += 1
        episode_step[env_id] = 0
        first_touch_step[env_id] = -1
        start_ball_xy[env_id] = env.scene["ball"].data.root_link_pos_w[env_id, :2]

    total_kicks = left_kick_count + right_kick_count
    balance = 0.0
    if total_kicks > 0:
      balance = 1.0 - abs(left_kick_count - right_kick_count) / total_kicks
    region_success_rate = {
      region: (
        float(region_success[region]) / float(region_attempts[region])
        if region_attempts[region] > 0
        else 0.0
      )
      for region in region_attempts
    }
    return build_eval_metrics_payload(
      num_episodes=cfg.num_episodes,
      goal_success_rate=goal_count / max(cfg.num_episodes, 1),
      first_touch_time=sum(first_touch_samples) / max(len(first_touch_samples), 1),
      first_goal_time=sum(first_goal_samples) / max(len(first_goal_samples), 1),
      recovery_steps_after_ball_reset=sum(recovery_samples)
      / max(len(recovery_samples), 1),
      fall_rate=fall_count / max(cfg.num_episodes, 1),
      out_of_bounds_rate=out_of_bounds_count / max(cfg.num_episodes, 1),
      consecutive_goals=float(max_goal_streak),
      left_right_kick_balance=balance,
      region_success_rate=region_success_rate,
    )
  finally:
    wrapped.close()


def _write_metrics_json(path: Path, metrics: dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))



def _run_standard_play(cfg: PlayConfig | ReactiveSoccerEvalArgs) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = g1_comp_reactive_soccer_env_cfg(play=True)
  agent_cfg = load_rl_cfg(TASK_ID)

  dummy_mode = cfg.agent in {"zero", "random"}
  trained_mode = not dummy_mode

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if trained_mode:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  if dummy_mode:
    apply_odometry_artifact_binding(
      env_cfg,
      checkpoint_path=None,
      odometry_model_path=getattr(cfg, "odometry_model_path", None),
      localization_mode=(
        getattr(cfg, "localization_mode", None)
        if getattr(cfg, "localization_mode", None) is not None
        else "ground_truth"
      ),
    )
  elif (
    getattr(cfg, "localization_mode", None) is not None
    or getattr(cfg, "odometry_model_path", None) is not None
  ):
    apply_odometry_artifact_binding(
      env_cfg,
      checkpoint_path=resume_path,
      odometry_model_path=getattr(cfg, "odometry_model_path", None),
      localization_mode=getattr(cfg, "localization_mode", None),
    )
  else:
    odometry_payload = apply_training_odometry_artifact_binding(
      env_cfg,
      checkpoint_path=resume_path,
      bootstrap_localization_mode="ground_truth",
    )
    if odometry_payload.get("binding_source") == "bootstrap_localization":
      print(
        "[WARN] Play mode is bootstrapping with ground-truth localization "
        "because no learned odometry checkpoint could be resolved.",
        flush=True,
      )

  render_mode = "rgb_array" if (trained_mode and cfg.video) else None
  if cfg.video and dummy_mode:
    print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if trained_mode and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if dummy_mode:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(
      env,
      _cfg_to_dict(agent_cfg),
      device=device,
      enable_distributed=False,
    )
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()

def run_play(cfg: PlayConfig | ReactiveSoccerEvalArgs) -> None:
  if isinstance(cfg, ReactiveSoccerEvalArgs) and cfg.metrics_json is not None:
    metrics = _collect_eval_metrics(cfg)
    _write_metrics_json(cfg.metrics_json, metrics)
    return
  _run_standard_play(cfg)


def main() -> None:
  args = tyro.cli(
    ReactiveSoccerEvalArgs,
    default=ReactiveSoccerEvalArgs(),
    prog=sys.argv[0],
    config=mjlab.TYRO_FLAGS,
  )
  run_play(args)


if __name__ == "__main__":
  main()
