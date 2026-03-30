"""Minimal AMP discriminator for reactive soccer."""

from __future__ import annotations

import torch
from torch import nn


class AmpDiscriminator(nn.Module):
  """Small MLP discriminator that scores expert/policy transitions."""

  def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (128, 64)) -> None:
    super().__init__()
    dims = (input_dim, *hidden_dims, 1)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=False):
      layers.append(nn.Linear(in_dim, out_dim))
      layers.append(nn.ELU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    self.network = nn.Sequential(*layers)

  def forward(self, transitions: torch.Tensor) -> torch.Tensor:
    return self.network(transitions)

  @staticmethod
  def bounded_reward(scores: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(scores)

  def gradient_penalty(
    self,
    expert: torch.Tensor,
    policy: torch.Tensor,
    lambda_: float = 10.0,
  ) -> torch.Tensor:
    batch = min(expert.shape[0], policy.shape[0])
    if batch == 0:
      return torch.tensor(0.0, device=expert.device)
    alpha = torch.rand((batch, 1), device=expert.device)
    mixed = alpha * expert[:batch] + (1.0 - alpha) * policy[:batch]
    mixed.requires_grad_(True)
    scores = self(mixed)
    grad = torch.autograd.grad(
      outputs=scores.sum(),
      inputs=mixed,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    penalty = (grad.norm(2, dim=1) - 1.0).pow(2).mean()
    return lambda_ * penalty
