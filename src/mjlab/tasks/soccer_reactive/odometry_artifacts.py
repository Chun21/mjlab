"""Helpers for explicit odometry artifact binding in reactive soccer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from mjlab.tasks.soccer_reactive.models.odometry_mlp import (
  peek_reactive_soccer_odometry_checkpoint,
)

ODOMETRY_INFO_KEY = "reactive_soccer_odometry"
_LEARNED_LOCALIZATION_MODES = frozenset({"learned_only", "particle_filter"})
DEFAULT_REACTIVE_SOCCER_ODOMETRY_MODEL_PATH = Path(
  "logs/rsl_rl/g1_comp_reactive_soccer/odometry/odometry_model.pt"
)


def localization_mode_requires_learned_odom(localization_mode: str | None) -> bool:
  return str(localization_mode or "") in _LEARNED_LOCALIZATION_MODES


def load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  return checkpoint if isinstance(checkpoint, dict) else {}


def load_checkpoint_infos(checkpoint_path: str | Path) -> dict[str, Any]:
  checkpoint = load_checkpoint_payload(checkpoint_path)
  infos = checkpoint.get("infos", {})
  return infos if isinstance(infos, dict) else {}


def extract_odometry_info_from_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
  checkpoint = load_checkpoint_payload(checkpoint_path)
  infos = checkpoint.get("infos", {})
  if isinstance(infos, dict):
    payload = infos.get(ODOMETRY_INFO_KEY, {})
    if isinstance(payload, dict) and payload:
      return payload
  metadata = checkpoint.get("metadata", {})
  if isinstance(metadata, dict):
    payload = metadata.get(ODOMETRY_INFO_KEY, {})
    if isinstance(payload, dict):
      return payload
  return {}


def build_odometry_info_from_env(env) -> dict[str, Any]:
  perception_cfg = getattr(env.cfg, "perception", SimpleNamespace())
  payload: dict[str, Any] = {
    "resolved_path": str(getattr(perception_cfg, "odometry_model_path", "") or ""),
    "localization_mode": str(getattr(perception_cfg, "localization_mode", "ground_truth")),
  }
  runtime = getattr(env, "_soccer_reactive_perception_runtime", None)
  if runtime is not None and hasattr(runtime, "odometry"):
    payload.update(runtime.odometry.odometry_artifact_info())
    return payload

  resolved_path = payload["resolved_path"]
  if resolved_path and Path(resolved_path).exists():
    metadata, extras = peek_reactive_soccer_odometry_checkpoint(resolved_path)
    payload["metadata"] = metadata.to_dict()
    payload["extras"] = extras
  return payload


def apply_odometry_artifact_binding(
  env_cfg,
  *,
  checkpoint_path: str | Path | None = None,
  odometry_model_path: str | None = None,
  localization_mode: str | None = None,
  require_if_needed: bool = True,
) -> dict[str, Any]:
  perception_cfg = getattr(env_cfg, "perception", None)
  if perception_cfg is None:
    perception_cfg = SimpleNamespace()
    setattr(env_cfg, "perception", perception_cfg)

  checkpoint_payload: dict[str, Any] = {}
  if checkpoint_path is not None:
    checkpoint_payload = extract_odometry_info_from_checkpoint(checkpoint_path)

  if localization_mode is not None:
    perception_cfg.localization_mode = str(localization_mode)
  elif getattr(perception_cfg, "localization_mode", None) in (None, "") and checkpoint_payload:
    perception_cfg.localization_mode = str(checkpoint_payload.get("localization_mode", "ground_truth"))

  if odometry_model_path is not None:
    perception_cfg.odometry_model_path = str(odometry_model_path)
  elif not str(getattr(perception_cfg, "odometry_model_path", "") or "") and checkpoint_payload:
    recorded_path = str(
      checkpoint_payload.get("resolved_path") or checkpoint_payload.get("path") or ""
    )
    perception_cfg.odometry_model_path = recorded_path

  resolved_path = str(getattr(perception_cfg, "odometry_model_path", "") or "")
  resolved_mode = str(getattr(perception_cfg, "localization_mode", "ground_truth"))
  payload: dict[str, Any] = {
    "resolved_path": resolved_path,
    "localization_mode": resolved_mode,
  }

  if localization_mode_requires_learned_odom(resolved_mode):
    if not resolved_path:
      raise FileNotFoundError(
        "Reactive soccer requires perception.odometry_model_path when "
        f"localization_mode={resolved_mode!r}."
      )
    if require_if_needed and not Path(resolved_path).exists():
      raise FileNotFoundError(f"Odometry checkpoint not found: {resolved_path}")
    if Path(resolved_path).exists():
      metadata, extras = peek_reactive_soccer_odometry_checkpoint(resolved_path)
      payload["metadata"] = metadata.to_dict()
      payload["extras"] = extras
  elif checkpoint_payload:
    if "metadata" in checkpoint_payload:
      payload["metadata"] = checkpoint_payload.get("metadata")
    if "extras" in checkpoint_payload:
      payload["extras"] = checkpoint_payload.get("extras")

  return payload


def apply_training_odometry_artifact_binding(
  env_cfg,
  *,
  checkpoint_path: str | Path | None = None,
  default_odometry_model_path: str | Path | None = DEFAULT_REACTIVE_SOCCER_ODOMETRY_MODEL_PATH,
  bootstrap_localization_mode: str | None = "ground_truth",
) -> dict[str, Any]:
  """Resolve reactive-soccer odometry before training env creation.

  Training is allowed to bootstrap from ground-truth localization when the
  offline learned odometry artifact is not available yet. The resolution order
  is:

  1. honor an explicit config path if provided;
  2. recover the bound odometry artifact from a resume checkpoint;
  3. use the canonical offline odometry checkpoint path if it exists;
  4. fall back to ``bootstrap_localization_mode`` when it is not ``None``.
  """
  perception_cfg = getattr(env_cfg, "perception", None)
  if perception_cfg is None:
    perception_cfg = SimpleNamespace()
    setattr(env_cfg, "perception", perception_cfg)

  requested_mode = str(getattr(perception_cfg, "localization_mode", "ground_truth"))
  requested_path = str(getattr(perception_cfg, "odometry_model_path", "") or "")
  if not localization_mode_requires_learned_odom(requested_mode):
    return {
      "resolved_path": requested_path,
      "localization_mode": requested_mode,
      "binding_source": "passthrough",
    }

  if requested_path:
    payload = apply_odometry_artifact_binding(
      env_cfg,
      checkpoint_path=checkpoint_path,
    )
    payload["binding_source"] = "explicit_config"
    return payload

  last_error: FileNotFoundError | None = None
  if checkpoint_path is not None:
    try:
      payload = apply_odometry_artifact_binding(
        env_cfg,
        checkpoint_path=checkpoint_path,
      )
    except FileNotFoundError as exc:
      last_error = exc
    else:
      payload["binding_source"] = "checkpoint"
      return payload

  default_path = (
    Path(default_odometry_model_path)
    if default_odometry_model_path is not None
    else None
  )
  if default_path is not None and default_path.exists():
    payload = apply_odometry_artifact_binding(
      env_cfg,
      odometry_model_path=str(default_path),
    )
    payload["binding_source"] = "default_checkpoint"
    return payload

  if bootstrap_localization_mode is None:
    if last_error is not None:
      raise last_error
    raise FileNotFoundError(
      "Reactive soccer training could not resolve a learned odometry checkpoint "
      f"for localization_mode={requested_mode!r}. "
      f"Expected default path: {default_path}"
    )

  perception_cfg.localization_mode = str(bootstrap_localization_mode)
  perception_cfg.odometry_model_path = ""
  payload: dict[str, Any] = {
    "resolved_path": "",
    "localization_mode": str(bootstrap_localization_mode),
    "binding_source": "bootstrap_localization",
    "requested_localization_mode": requested_mode,
  }
  if default_path is not None:
    payload["default_odometry_model_path"] = str(default_path)
  if last_error is not None:
    payload["reason"] = str(last_error)
  return payload
