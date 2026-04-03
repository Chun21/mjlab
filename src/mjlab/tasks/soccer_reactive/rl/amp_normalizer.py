"""Minimal running-stat normalizer for AMP transitions."""

from __future__ import annotations

import torch


class ReactiveSoccerAmpNormalizer:
  """Track running mean/var and normalize AMP transitions."""

  def __init__(self, feature_dim: int, device: str = 'cpu', eps: float = 1e-6) -> None:
    self.feature_dim = feature_dim
    self.device = device
    self.eps = eps
    self.count = torch.tensor(0.0, device=device)
    self.mean = torch.zeros(feature_dim, device=device)
    self.var = torch.ones(feature_dim, device=device)

  def update_from_moments(
    self,
    batch_mean: torch.Tensor,
    batch_var: torch.Tensor,
    batch_count: torch.Tensor,
  ) -> None:
    batch_mean = torch.nan_to_num(batch_mean, nan=0.0, posinf=0.0, neginf=0.0)
    batch_var = torch.clamp(
      torch.nan_to_num(batch_var, nan=0.0, posinf=0.0, neginf=0.0),
      min=self.eps,
    )
    batch_count = torch.nan_to_num(batch_count, nan=0.0, posinf=0.0, neginf=0.0)
    if float(batch_count.item()) <= 0.0:
      return
    total = self.count + batch_count
    delta = batch_mean - self.mean
    new_mean = self.mean + delta * (batch_count / total)
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    correction = delta.pow(2) * (self.count * batch_count / total.clamp(min=1.0))
    new_var = (m_a + m_b + correction) / total.clamp(min=1.0)
    self.mean = torch.nan_to_num(new_mean, nan=0.0, posinf=0.0, neginf=0.0)
    self.var = torch.clamp(new_var, min=self.eps)
    self.count = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

  def update(self, batch: torch.Tensor) -> None:
    batch = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
    if batch.numel() == 0:
      return
    batch_mean = batch.mean(dim=0)
    batch_var = batch.var(dim=0, unbiased=False)
    batch_count = torch.tensor(float(batch.shape[0]), device=self.device)
    self.update_from_moments(batch_mean, batch_var, batch_count)

  def normalize(self, batch: torch.Tensor) -> torch.Tensor:
    batch = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = (batch - self.mean) / torch.sqrt(self.var + self.eps)
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
