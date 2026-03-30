"""Motion-clip reset helpers for reactive soccer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.asset_zoo.robots import G1_COMP_BODY_JOINT_NAMES
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg


class MotionClipInitializer:
  """Load AMP motion clips and sample robot state frames."""

  def __init__(self, motion_root: Path, device: str):
    self.motion_root = Path(motion_root)
    self.device = device
    self._joint_names = tuple(G1_COMP_BODY_JOINT_NAMES)
    self._clips = self._load_clips()

  def _load_clips(self) -> list[dict[str, torch.Tensor]]:
    motion_files = sorted(self.motion_root.rglob("*.npz"))
    if not motion_files:
      raise ValueError(f"No motion clips found under {self.motion_root}")

    clips: list[dict[str, torch.Tensor]] = []
    for motion_file in motion_files:
      data = np.load(motion_file, allow_pickle=False)
      joint_names = [str(name) for name in data["joint_names"]]
      joint_indices = [joint_names.index(name) for name in self._joint_names]

      clips.append(
        {
          "root_pos_w": torch.tensor(
            data["body_pos_w"][:, 0, :], device=self.device, dtype=torch.float32
          ),
          "root_quat_w": torch.tensor(
            data["body_quat_w"][:, 0, :], device=self.device, dtype=torch.float32
          ),
          "root_lin_vel_w": torch.tensor(
            data["body_lin_vel_w"][:, 0, :], device=self.device, dtype=torch.float32
          ),
          "root_ang_vel_w": torch.tensor(
            data["body_ang_vel_w"][:, 0, :], device=self.device, dtype=torch.float32
          ),
          "joint_pos": torch.tensor(
            data["joint_pos"][:, joint_indices],
            device=self.device,
            dtype=torch.float32,
          ),
          "joint_vel": torch.tensor(
            data["joint_vel"][:, joint_indices],
            device=self.device,
            dtype=torch.float32,
          ),
        }
      )
    return clips

  def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
    """Sample random frames from the loaded motion clips."""
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")

    file_ids = torch.randint(len(self._clips), (batch_size,), device=self.device)
    samples: dict[str, list[torch.Tensor]] = {
      "root_pos_w": [],
      "root_quat_w": [],
      "root_lin_vel_w": [],
      "root_ang_vel_w": [],
      "joint_pos": [],
      "joint_vel": [],
    }

    for file_id in file_ids.tolist():
      clip = self._clips[file_id]
      frame_id = int(
        torch.randint(clip["joint_pos"].shape[0], (1,), device=self.device).item()
      )
      for key in samples:
        samples[key].append(clip[key][frame_id])

    return {key: torch.stack(value, dim=0) for key, value in samples.items()}


class MotionClipResetEvent:
  """Reset selected robot environments from sampled motion clips."""

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._probability = float(cfg.params["probability"])
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._initializer = MotionClipInitializer(
      motion_root=Path(cfg.params["motion_root"]), device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids  # Stateless hook.

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    probability: float,
    motion_root: str,
    asset_cfg: SceneEntityCfg,
  ) -> None:
    del probability, motion_root, asset_cfg

    if env_ids is None or isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0 or self._probability <= 0.0:
      return

    sampled_mask = torch.rand(len(env_ids), device=env.device) < self._probability
    selected_env_ids = env_ids[sampled_mask]
    if len(selected_env_ids) == 0:
      return

    robot = env.scene[self._asset_cfg.name]
    motion_state = self._initializer.sample(batch_size=len(selected_env_ids))

    root_pose = torch.cat(
      [motion_state["root_pos_w"], motion_state["root_quat_w"]], dim=-1
    )
    root_pose[:, :2] += env.scene.env_origins[selected_env_ids, :2]
    root_velocity = torch.cat(
      [motion_state["root_lin_vel_w"], motion_state["root_ang_vel_w"]], dim=-1
    )

    robot.write_root_link_pose_to_sim(root_pose, env_ids=selected_env_ids)
    robot.write_root_link_velocity_to_sim(root_velocity, env_ids=selected_env_ids)
    robot.write_joint_state_to_sim(
      motion_state["joint_pos"],
      motion_state["joint_vel"],
      joint_ids=self._asset_cfg.joint_ids,
      env_ids=selected_env_ids,
    )
