"""AMP discriminator and losses for reactive soccer."""

from __future__ import annotations

import torch
from torch import nn


class AmpDiscriminator(nn.Module):
  """MLP discriminator for expert vs policy motion transitions."""

  def __init__(
    self,
    input_dim: int,
    hidden_dims: tuple[int, ...] = (256, 256, 128),
  ) -> None:
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
  def adversarial_reward(scores: torch.Tensor) -> torch.Tensor:
    return -torch.tanh(0.4 * scores)

  @staticmethod
  def bounded_reward(scores: torch.Tensor) -> torch.Tensor:
    return AmpDiscriminator.adversarial_reward(scores)

  @staticmethod
  def wasserstein_loss_from_scores(
    expert_scores: torch.Tensor,
    policy_scores: torch.Tensor,
  ) -> torch.Tensor:
    return -torch.tanh(0.4 * expert_scores).mean() + torch.tanh(0.4 * policy_scores).mean()

  def gradient_penalty(
    self,
    expert: torch.Tensor,
    policy: torch.Tensor,
    lambda_: float = 50.0,
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

  def discriminator_loss(
    self,
    expert: torch.Tensor,
    policy: torch.Tensor,
    gradient_penalty_coef: float = 50.0,
  ) -> dict[str, torch.Tensor]:
    expert_scores = self(expert)
    policy_scores = self(policy)
    wasserstein_loss = self.wasserstein_loss_from_scores(expert_scores, policy_scores)
    gradient_penalty = self.gradient_penalty(
      expert,
      policy,
      lambda_=gradient_penalty_coef,
    )
    total_loss = wasserstein_loss + gradient_penalty
    return {
      "expert_scores": expert_scores,
      "policy_scores": policy_scores,
      "wasserstein_loss": wasserstein_loss,
      "gradient_penalty": gradient_penalty,
      "total_loss": total_loss,
    }
