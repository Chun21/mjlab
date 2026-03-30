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

  def update(self, batch: torch.Tensor) -> None:
    if batch.numel() == 0:
      return
    batch_mean = batch.mean(dim=0)
    batch_var = batch.var(dim=0, unbiased=False)
    batch_count = torch.tensor(float(batch.shape[0]), device=self.device)
    total = self.count + batch_count
    delta = batch_mean - self.mean
    new_mean = self.mean + delta * (batch_count / total)
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    correction = delta.pow(2) * (self.count * batch_count / total.clamp(min=1.0))
    new_var = (m_a + m_b + correction) / total.clamp(min=1.0)
    self.mean = new_mean
    self.var = torch.clamp(new_var, min=self.eps)
    self.count = total

  def normalize(self, batch: torch.Tensor) -> torch.Tensor:
    return (batch - self.mean) / torch.sqrt(self.var + self.eps)
