"""Train the offline learned odometry MLP for reactive soccer."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mjlab
import torch
import tyro
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from mjlab.tasks.soccer_reactive.models.odometry_mlp import (
  ReactiveSoccerOdometryModelMetadata,
  build_reactive_soccer_odometry_model,
  save_reactive_soccer_odometry_checkpoint,
)


@dataclass(frozen=True)
class TrainOdometryArgs:
  dataset_path: str = "data/soccer_reactive/odometry/odometry_dataset.pt"
  output_path: str = "logs/rsl_rl/g1_comp_reactive_soccer/odometry/odometry_model.pt"
  batch_size: int = 1024
  epochs: int = 20
  learning_rate: float = 1.0e-3
  weight_decay: float = 1.0e-5
  train_split: float = 0.9
  grad_clip_norm: float = 1.0
  hidden_dims: tuple[int, ...] = (1024, 128)
  device: str | None = None
  seed: int = 20260401


def _device_or_default(device: str | None) -> str:
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_dataset(path: str | Path) -> tuple[TensorDataset, dict]:
  payload = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(payload, dict):
    raise ValueError(f"Unsupported odometry dataset format: {type(payload)!r}")
  proprio_history = payload["proprio_history"].float()
  prev_delta_pose = payload["prev_delta_pose"].float()
  target_delta_pose = payload["target_delta_pose"].float()
  dataset = TensorDataset(proprio_history, prev_delta_pose, target_delta_pose)
  metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
  return dataset, metadata


def train_odometry(args: TrainOdometryArgs) -> Path:
  torch.manual_seed(args.seed)
  device = _device_or_default(args.device)
  dataset, dataset_metadata = _load_dataset(args.dataset_path)
  if len(dataset) == 0:
    raise ValueError("Odometry dataset is empty.")

  proprio_history, _prev_delta_pose, _target_delta_pose = dataset[0]
  metadata = ReactiveSoccerOdometryModelMetadata(
    proprio_dim=int(proprio_history.shape[-1]),
    history_steps=int(proprio_history.shape[0]),
    hidden_dims=tuple(int(dim) for dim in args.hidden_dims),
  )
  model = build_reactive_soccer_odometry_model(metadata, device=device)
  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=float(args.learning_rate),
    weight_decay=float(args.weight_decay),
  )
  criterion = nn.MSELoss()

  if len(dataset) == 1:
    train_dataset = dataset
    val_dataset = dataset
  else:
    train_size = min(len(dataset) - 1, max(1, int(len(dataset) * float(args.train_split))))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
      dataset,
      [train_size, val_size],
      generator=torch.Generator().manual_seed(args.seed),
    )
  train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False)

  history: list[dict[str, float]] = []
  best_val_loss = float("inf")
  best_state_dict = None
  for epoch in range(int(args.epochs)):
    model.train()
    train_loss_total = 0.0
    train_count = 0
    for proprio_history_batch, prev_delta_batch, target_batch in train_loader:
      proprio_history_batch = proprio_history_batch.to(device)
      prev_delta_batch = prev_delta_batch.to(device)
      target_batch = target_batch.to(device)
      optimizer.zero_grad(set_to_none=True)
      prediction = model(proprio_history_batch, prev_delta_batch)
      loss = criterion(prediction, target_batch)
      if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite odometry loss detected at epoch {epoch}: {loss}")
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
      optimizer.step()
      train_loss_total += float(loss.item()) * int(proprio_history_batch.shape[0])
      train_count += int(proprio_history_batch.shape[0])

    model.eval()
    val_loss_total = 0.0
    val_count = 0
    with torch.no_grad():
      for proprio_history_batch, prev_delta_batch, target_batch in val_loader:
        proprio_history_batch = proprio_history_batch.to(device)
        prev_delta_batch = prev_delta_batch.to(device)
        target_batch = target_batch.to(device)
        prediction = model(proprio_history_batch, prev_delta_batch)
        loss = criterion(prediction, target_batch)
        val_loss_total += float(loss.item()) * int(proprio_history_batch.shape[0])
        val_count += int(proprio_history_batch.shape[0])

    train_loss = train_loss_total / max(train_count, 1)
    val_loss = val_loss_total / max(val_count, 1)
    history.append({"epoch": float(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

  if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

  extras = {
    "dataset_path": str(args.dataset_path),
    "dataset_metadata": dataset_metadata,
    "train_history": history,
    "best_val_loss": best_val_loss,
  }
  output_path = save_reactive_soccer_odometry_checkpoint(
    args.output_path,
    model,
    metadata=metadata,
    extras=extras,
  )
  metrics_path = Path(output_path).with_suffix(".json")
  metrics_path.write_text(json.dumps(extras, indent=2, ensure_ascii=False))
  return output_path


def main() -> None:
  args = tyro.cli(
    TrainOdometryArgs,
    prog=sys.argv[0],
    default=TrainOdometryArgs(),
    config=mjlab.TYRO_FLAGS,
  )
  output_path = train_odometry(args)
  print(f"[INFO] Saved odometry checkpoint to {output_path}")


if __name__ == "__main__":
  main()
