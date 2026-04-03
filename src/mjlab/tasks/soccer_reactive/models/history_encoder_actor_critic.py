"""Minimal history encoder-decoder actor-critic for reactive soccer."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
  """Simple MLP helper used by the reactive soccer model."""

  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    activation: type[nn.Module] = nn.ELU,
  ) -> None:
    super().__init__()
    dims = (input_dim, *hidden_dims, output_dim)
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=False):
      layers.append(nn.Linear(in_dim, out_dim))
      layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    self.network = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.network(x)


class ReactiveSoccerActorCritic(nn.Module):
  """Unified model with paper-style history encoder, decoder, actor and dual critics."""

  OBS_CLIP = 100.0
  LATENT_CLIP = 50.0

  def __init__(
    self,
    actor_obs_dim: int,
    history_obs_dim: int,
    history_steps: int,
    action_dim: int,
    latent_dim: int = 64,
    reconstruct_dim: int = 14,
    privileged_dim: int | None = None,
    actor_hidden_dims: tuple[int, ...] = (256, 256, 128),
    critic_hidden_dims: tuple[int, ...] = (256, 256, 128),
    activation: type[nn.Module] = nn.ELU,
  ) -> None:
    super().__init__()
    self.actor_obs_dim = actor_obs_dim
    self.history_obs_dim = history_obs_dim
    self.history_steps = history_steps
    self.action_dim = action_dim
    self.latent_dim = latent_dim
    self.reconstruct_dim = reconstruct_dim
    self.privileged_dim = reconstruct_dim if privileged_dim is None else privileged_dim

    self.history_encoder = MLP(
      history_obs_dim * history_steps,
      latent_dim,
      hidden_dims=(1024, 128),
      activation=activation,
    )
    self.actor_head = MLP(
      actor_obs_dim + latent_dim,
      action_dim,
      hidden_dims=actor_hidden_dims,
      activation=activation,
    )
    self.decoder_head = MLP(
      latent_dim,
      reconstruct_dim,
      hidden_dims=(128, 128),
      activation=activation,
    )
    critic_input_dim = actor_obs_dim + latent_dim + self.privileged_dim
    self.goal_critic_head = MLP(
      critic_input_dim,
      1,
      hidden_dims=critic_hidden_dims,
      activation=activation,
    )
    self.aux_critic_head = MLP(
      critic_input_dim,
      1,
      hidden_dims=critic_hidden_dims,
      activation=activation,
    )

  @classmethod
  def _sanitize_tensor(cls, x: torch.Tensor, *, clip: float | None = None) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None:
      x = torch.clamp(x, min=-clip, max=clip)
    return x

  def encode_history(self, actor_history: torch.Tensor) -> torch.Tensor:
    actor_history = self._sanitize_tensor(actor_history, clip=self.OBS_CLIP)
    batch_size = actor_history.shape[0]
    flat_history = actor_history.reshape(batch_size, self.history_obs_dim * self.history_steps)
    latent = self.history_encoder(flat_history)
    return self._sanitize_tensor(latent, clip=self.LATENT_CLIP)

  def forward_train(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    actor_current = self._sanitize_tensor(batch["actor_current"], clip=self.OBS_CLIP)
    actor_history = self._sanitize_tensor(batch["actor_history"], clip=self.OBS_CLIP)
    critic_current = self._sanitize_tensor(
      batch.get("critic_current", actor_current),
      clip=self.OBS_CLIP,
    )
    critic_privileged = self._sanitize_tensor(
      batch["critic_privileged"],
      clip=self.OBS_CLIP,
    )

    latent = self.encode_history(actor_history)
    actor_input = torch.cat((actor_current, latent), dim=-1)
    critic_input = torch.cat((critic_current, latent, critic_privileged), dim=-1)
    actions_mean = torch.tanh(self.actor_head(actor_input))
    reconstruction = self._sanitize_tensor(self.decoder_head(latent), clip=self.OBS_CLIP)
    goal_value = self._sanitize_tensor(self.goal_critic_head(critic_input))
    aux_value = self._sanitize_tensor(self.aux_critic_head(critic_input))

    return {
      "actions_mean": actions_mean,
      "latent": latent,
      "reconstruction": reconstruction,
      "goal_value": goal_value,
      "aux_value": aux_value,
    }

  def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return self.forward_train(batch)
