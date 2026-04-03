"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import torch
import torch.distributed as dist
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder

_REACTIVE_SOCCER_TASK_ID = "Mjlab-Soccer-G1-Comp-ReactivePaper"


def _maybe_init_reactive_soccer_distributed(
  task_id: str,
  device: str,
  *,
  rank: int,
) -> bool:
  if task_id != _REACTIVE_SOCCER_TASK_ID:
    return False
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  if world_size <= 1 or not dist.is_available() or dist.is_initialized():
    return False
  backend = "nccl" if str(device).startswith("cuda") else "gloo"
  dist.init_process_group(backend=backend, init_method="env://")
  if rank == 0:
    print(
      f"[INFO] Initialized distributed reactive soccer training "
      f"(backend={backend}, world_size={world_size})"
    )
  return True


def _maybe_shutdown_distributed(initialized_by_this_process: bool) -> None:
  if (
    not initialized_by_this_process
    or not dist.is_available()
    or not dist.is_initialized()
  ):
    return
  dist.destroy_process_group()


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _cfg_to_dict(value):
  """Convert configs to serializable dicts while preserving dynamic attributes."""
  if is_dataclass(value):
    data = {
      field_info.name: _cfg_to_dict(getattr(value, field_info.name))
      for field_info in fields(value)
    }
    declared_field_names = {field_info.name for field_info in fields(value)}
    for key, nested_value in vars(value).items():
      if key.startswith("_") or key in declared_field_names:
        continue
      data[key] = _cfg_to_dict(nested_value)
    return data
  if isinstance(value, SimpleNamespace):
    return {key: _cfg_to_dict(nested_value) for key, nested_value in vars(value).items()}
  if isinstance(value, dict):
    return {key: _cfg_to_dict(nested_value) for key, nested_value in value.items()}
  if isinstance(value, (list, tuple)):
    return [_cfg_to_dict(item) for item in value]
  return deepcopy(value)


def _restore_missing_dynamic_attrs(target, template) -> None:
  if isinstance(target, dict) and isinstance(template, dict):
    for attr_name, attr_value in template.items():
      if attr_name not in target:
        target[attr_name] = deepcopy(attr_value)
      elif isinstance(target[attr_name], (dict, SimpleNamespace)) and isinstance(
        attr_value, (dict, SimpleNamespace)
      ):
        _restore_missing_dynamic_attrs(target[attr_name], attr_value)
    return

  if not hasattr(target, "__dict__") or not hasattr(template, "__dict__"):
    return

  target_vars = vars(target)
  for attr_name, attr_value in vars(template).items():
    if attr_name not in target_vars:
      setattr(target, attr_name, deepcopy(attr_value))
      continue
    if isinstance(target_vars[attr_name], (dict, SimpleNamespace)) and isinstance(
      attr_value, (dict, SimpleNamespace)
    ):
      _restore_missing_dynamic_attrs(target_vars[attr_name], attr_value)


def _restore_task_dynamic_train_cfg_extensions(
  task_id: str,
  cfg: TrainConfig,
) -> TrainConfig:
  template_cfg = TrainConfig.from_task(task_id)
  _restore_missing_dynamic_attrs(cfg.env, template_cfg.env)
  _restore_missing_dynamic_attrs(cfg.agent, template_cfg.agent)
  return cfg


def _apply_task_specific_train_env_bindings(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  *,
  rank: int,
  resume_path: Path | None,
) -> None:
  if task_id != _REACTIVE_SOCCER_TASK_ID:
    return

  from mjlab.tasks.soccer_reactive.odometry_artifacts import (
    DEFAULT_REACTIVE_SOCCER_ODOMETRY_MODEL_PATH,
    apply_training_odometry_artifact_binding,
  )

  default_env_cfg = load_env_cfg(task_id)
  default_perception = getattr(default_env_cfg, "perception", SimpleNamespace())
  current_perception = getattr(env_cfg, "perception", SimpleNamespace())
  allow_bootstrap = (
    str(getattr(current_perception, "localization_mode", ""))
    == str(getattr(default_perception, "localization_mode", ""))
    and str(getattr(current_perception, "odometry_model_path", "") or "")
    == str(getattr(default_perception, "odometry_model_path", "") or "")
  )
  payload = apply_training_odometry_artifact_binding(
    env_cfg,
    checkpoint_path=resume_path,
    bootstrap_localization_mode=("ground_truth" if allow_bootstrap else None),
  )
  if rank != 0:
    return
  binding_source = str(payload.get("binding_source", ""))
  if binding_source == "checkpoint":
    print(
      "[INFO] Reactive soccer odometry restored from resume checkpoint: "
      f"{payload.get('resolved_path', '')}"
    )
  elif binding_source == "default_checkpoint":
    print(
      "[INFO] Reactive soccer using default odometry checkpoint: "
      f"{payload.get('resolved_path', DEFAULT_REACTIVE_SOCCER_ODOMETRY_MODEL_PATH)}"
    )
  elif binding_source == "bootstrap_localization":
    print(
      "[WARN] Reactive soccer odometry checkpoint is unavailable; "
      "bootstrapping training with ground-truth localization. "
      f"Requested mode: {payload.get('requested_localization_mode', 'unknown')}. "
      f"Expected offline odometry checkpoint: {DEFAULT_REACTIVE_SOCCER_ODOMETRY_MODEL_PATH}"
    )


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cfg = _restore_task_dynamic_train_cfg_extensions(task_id, cfg)
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  local_rank = 0
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")
  initialized_distributed = _maybe_init_reactive_soccer_distributed(
    task_id,
    device,
    rank=rank,
  )

  registry_name: str | None = None
  env = None
  runner = None

  try:
    # Check if this is a tracking task by checking for motion command.
    is_tracking_task = "motion" in cfg.env.commands and isinstance(
      cfg.env.commands["motion"], MotionCommandCfg
    )

    if is_tracking_task:
      motion_cmd = cfg.env.commands["motion"]
      assert isinstance(motion_cmd, MotionCommandCfg)

      # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
      if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
        print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
      elif cfg.registry_name:
        # Download from WandB registry.
        registry_name = cast(str, cfg.registry_name)
        if ":" not in registry_name:
          registry_name = registry_name + ":latest"
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
      else:
        raise ValueError(
          "For tracking tasks, provide either:\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)\n"
          "  --env.commands.motion.motion-file /path/to/motion.npz (local file)"
        )

    # Enable NaN guard if requested.
    if cfg.enable_nan_guard:
      cfg.env.sim.nan_guard.enabled = True
      print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

    if rank == 0:
      print(f"[INFO] Logging experiment in directory: {log_dir}")

    log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

    resume_path: Path | None = None
    if cfg.agent.resume:
      if cfg.wandb_run_path is not None:
        # Load checkpoint from W&B.
        resume_path, was_cached = get_wandb_checkpoint_path(
          log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
        )
        if rank == 0:
          run_id = resume_path.parent.name
          checkpoint_name = resume_path.name
          cached_str = "cached" if was_cached else "downloaded"
          print(
            f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
            f"(run: {run_id}, {cached_str})"
          )
      else:
        # Load checkpoint from local filesystem.
        resume_path = get_checkpoint_path(
          log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
        )

    _apply_task_specific_train_env_bindings(
      task_id,
      cfg.env,
      rank=rank,
      resume_path=resume_path,
    )

    env = ManagerBasedRlEnv(
      cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
    )

    # Only record videos on rank 0 to avoid multiple workers writing to the same files.
    if cfg.video and rank == 0:
      env = VideoRecorder(
        env,
        video_folder=Path(log_dir) / "videos" / "train",
        step_trigger=lambda step: step % cfg.video_interval == 0,
        video_length=cfg.video_length,
        disable_logger=True,
      )
      print("[INFO] Recording videos during training.")

    env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

    agent_cfg = _cfg_to_dict(cfg.agent)
    env_cfg = _cfg_to_dict(cfg.env)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
      runner_cls = MjlabOnPolicyRunner

    runner_kwargs = {}
    if is_tracking_task:
      runner_kwargs["registry_name"] = registry_name

    # Write config files before runner creation, since the runner mutates agent_cfg
    # in-place (e.g., injecting non-serializable objects).
    if rank == 0:
      dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
      dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

    runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)
    if resume_path is not None:
      print(f"[INFO]: Loading model checkpoint from: {resume_path}")
      runner.load(str(resume_path))

    runner.learn(
      num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
    )
  finally:
    try:
      if runner is not None and hasattr(runner, "close"):
        runner.close()
    finally:
      try:
        if env is not None:
          env.close()
      finally:
        _maybe_shutdown_distributed(initialized_distributed)


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)
  args = _restore_task_dynamic_train_cfg_extensions(task_id, args)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
