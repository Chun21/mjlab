"""Learned odometry model for reactive soccer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


class ReactiveSoccerOdometryMLP(nn.Module):
  """Paper-style odometry MLP operating on 1-second proprio history.

  The model consumes a batched history window with shape ``(B, T, D)`` together with
  the previous odometry output ``(B, 3)`` and predicts a body-frame pose increment
  ``[dx_body, dy_body, d_yaw]``.
  """

  INPUT_CLIP = 100.0
  OUTPUT_CLIP = 5.0

  def __init__(
    self,
    proprio_dim: int,
    history_steps: int = 50,
    hidden_dims: tuple[int, ...] = (1024, 128),
    activation: type[nn.Module] = nn.ELU,
  ) -> None:
    super().__init__()
    self.proprio_dim = int(proprio_dim)
    self.history_steps = int(history_steps)
    self.hidden_dims = tuple(int(dim) for dim in hidden_dims)

    input_dim = self.proprio_dim * self.history_steps + 3
    dims = (input_dim, *self.hidden_dims, 3)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=False):
      layers.append(nn.Linear(in_dim, out_dim))
      layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    self.network = nn.Sequential(*layers)

  @staticmethod
  def _sanitize_tensor(
    tensor: torch.Tensor,
    *,
    clip: float | None = None,
  ) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
      tensor = torch.clamp(tensor, min=-clip, max=clip)
    return tensor

  def forward(
    self,
    proprio_history: torch.Tensor,
    prev_delta_pose: torch.Tensor,
  ) -> torch.Tensor:
    proprio_history = self._sanitize_tensor(proprio_history, clip=self.INPUT_CLIP)
    prev_delta_pose = self._sanitize_tensor(prev_delta_pose, clip=self.INPUT_CLIP)
    if proprio_history.ndim != 3:
      raise ValueError(
        f"Expected proprio_history with shape (B, T, D), got {tuple(proprio_history.shape)}"
      )
    if proprio_history.shape[1] != self.history_steps or proprio_history.shape[2] != self.proprio_dim:
      raise ValueError(
        "Unexpected proprio_history shape: "
        f"expected (*, {self.history_steps}, {self.proprio_dim}), "
        f"got {tuple(proprio_history.shape)}"
      )
    if prev_delta_pose.shape[-1] != 3:
      raise ValueError(
        f"Expected prev_delta_pose last dim 3, got {tuple(prev_delta_pose.shape)}"
      )
    flat_history = proprio_history.reshape(proprio_history.shape[0], -1)
    output = self.network(torch.cat((flat_history, prev_delta_pose), dim=-1))
    return self._sanitize_tensor(output, clip=self.OUTPUT_CLIP)


@dataclass(frozen=True)
class ReactiveSoccerOdometryModelMetadata:
  version: str = "reactive_soccer_odometry_v1"
  proprio_dim: int = 75
  history_steps: int = 50
  hidden_dims: tuple[int, ...] = (1024, 128)

  def to_dict(self) -> dict[str, Any]:
    return {
      "version": self.version,
      "proprio_dim": int(self.proprio_dim),
      "history_steps": int(self.history_steps),
      "hidden_dims": list(self.hidden_dims),
    }

  @classmethod
  def from_dict(
    cls,
    payload: dict[str, Any] | None,
  ) -> "ReactiveSoccerOdometryModelMetadata":
    if payload is None:
      return cls()
    return cls(
      version=str(payload.get("version", "reactive_soccer_odometry_v1")),
      proprio_dim=int(payload.get("proprio_dim", 75)),
      history_steps=int(payload.get("history_steps", 50)),
      hidden_dims=tuple(int(dim) for dim in payload.get("hidden_dims", (1024, 128))),
    )


def build_reactive_soccer_odometry_model(
  metadata: ReactiveSoccerOdometryModelMetadata,
  device: str = "cpu",
) -> ReactiveSoccerOdometryMLP:
  model = ReactiveSoccerOdometryMLP(
    proprio_dim=metadata.proprio_dim,
    history_steps=metadata.history_steps,
    hidden_dims=metadata.hidden_dims,
  )
  return model.to(device)


def save_reactive_soccer_odometry_checkpoint(
  path: str | Path,
  model: ReactiveSoccerOdometryMLP,
  *,
  metadata: ReactiveSoccerOdometryModelMetadata,
  extras: dict[str, Any] | None = None,
) -> Path:
  checkpoint = {
    "state_dict": model.state_dict(),
    "metadata": metadata.to_dict(),
    "extras": extras or {},
  }
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(checkpoint, path)
  return path


def peek_reactive_soccer_odometry_checkpoint(
  path: str | Path,
  device: str = "cpu",
) -> tuple[ReactiveSoccerOdometryModelMetadata, dict[str, Any]]:
  checkpoint = torch.load(path, map_location=device, weights_only=False)
  if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    metadata = ReactiveSoccerOdometryModelMetadata.from_dict(checkpoint.get("metadata"))
    extras = checkpoint.get("extras", {})
    return metadata, extras if isinstance(extras, dict) else {}
  if isinstance(checkpoint, dict):
    return ReactiveSoccerOdometryModelMetadata(), {}
  raise ValueError(f"Unsupported odometry checkpoint format: {type(checkpoint)!r}")


def load_reactive_soccer_odometry_checkpoint(
  path: str | Path,
  device: str = "cpu",
) -> tuple[ReactiveSoccerOdometryMLP, ReactiveSoccerOdometryModelMetadata, dict[str, Any]]:
  checkpoint = torch.load(path, map_location=device, weights_only=False)
  if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    metadata = ReactiveSoccerOdometryModelMetadata.from_dict(
      checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    )
    model = build_reactive_soccer_odometry_model(metadata, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    extras = checkpoint.get("extras", {}) if isinstance(checkpoint, dict) else {}
    return model, metadata, extras

  if isinstance(checkpoint, dict):
    metadata = ReactiveSoccerOdometryModelMetadata()
    model = build_reactive_soccer_odometry_model(metadata, device=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, metadata, {}

  raise ValueError(f"Unsupported odometry checkpoint format: {type(checkpoint)!r}")
