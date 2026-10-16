"""Shared decentralized actor and centralized team-value MAPPO utilities."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def mlp(inputs, outputs, hidden):
    layers = [nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
              nn.Linear(hidden, outputs)]
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, 0.01 if outputs == 5 else 1.0)
    return nn.Sequential(*layers)


class RecurrentActor(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh())
        self.memory = nn.GRUCell(hidden, hidden)
        self.head = nn.Linear(hidden, 5)
        nn.init.orthogonal_(self.head.weight, 0.01)
        nn.init.zeros_(self.head.bias)
        self.hidden = hidden

    def forward(self, obs, memory=None):
        shape = obs.shape[:-1]
        encoded = self.encode(obs).reshape(-1, self.hidden)
        if memory is None:
            memory = torch.zeros_like(encoded)
        memory = self.memory(encoded, memory.reshape(-1, self.hidden))
        return self.head(memory).reshape(*shape, 5), memory.reshape(*shape, self.hidden)


class MAPPO(nn.Module):
    def __init__(self, obs_dim=28, state_dim=36, hidden=128, recurrent=False):
        super().__init__()
        self.actor = RecurrentActor(obs_dim, hidden) if recurrent else mlp(obs_dim, 5, hidden)
        self.critic = mlp(state_dim, 1, hidden)
        self.register_buffer("value_mean", torch.tensor(0.0))
        self.register_buffer("value_std", torch.tensor(1.0))
        self.register_buffer("value_updates", torch.tensor(0))
        self.spec = dict(obs_dim=obs_dim, state_dim=state_dim, hidden=hidden, recurrent=recurrent)
        self._memory = None

    def reset_memory(self):
        self._memory = None

    def policy(self, obs, memory=None):
        if self.spec["recurrent"]:
            logits, memory = self.actor(obs, memory)
        else:
            logits = self.actor(obs)
        return Categorical(logits=logits), memory

    def distribution(self, obs):
        return self.policy(obs)[0]

    def value(self, state):
        return self.critic(state).squeeze(-1) * self.value_std + self.value_mean

    @torch.no_grad()
    def normalize_value_targets(self, targets):
        """PopArt: update target scale while preserving all unnormalized predictions."""
        old_mean, old_std = self.value_mean.clone(), self.value_std.clone()
        weight = 1.0 if self.value_updates.item() == 0 else 0.01
        new_mean = (1 - weight) * old_mean + weight * targets.mean()
        second_moment = ((1 - weight) * (old_std.square() + old_mean.square())
                         + weight * targets.square().mean())
        new_std = (second_moment - new_mean.square()).clamp_min(0.01).sqrt()
        last = self.critic[-1]
        last.weight.mul_(old_std / new_std)
        last.bias.copy_((old_std * last.bias + old_mean - new_mean) / new_std)
        self.value_mean.copy_(new_mean)
        self.value_std.copy_(new_std)
        self.value_updates.add_(1)
        return (targets - new_mean) / new_std

    @torch.no_grad()
    def moves(self, obs, deterministic=True):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        dist, self._memory = self.policy(obs, self._memory)
        return (dist.logits.argmax(dim=-1) if deterministic else dist.sample()).numpy()


def gae(rewards, values, next_values, boundaries, terminals=None, gamma=0.99, lam=0.95):
    """Bootstrap at truncations, but stop the advantage trace at every reset."""
    if terminals is None:
        terminals = np.zeros_like(boundaries)
    advantages = np.empty_like(rewards)
    carry = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - terminals[t]) - values[t]
        carry = delta + gamma * lam * (1 - boundaries[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def load_policy(path):
    # TorchVersion appeared in early locally created pilot metadata.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MAPPO(**checkpoint["model_spec"])
    # Early pilot checkpoints predate value normalization; their raw values use mean=0/std=1.
    state = checkpoint["model"]
    for key in ("value_mean", "value_std", "value_updates"):
        if key not in state:
            state[key] = model.state_dict()[key]
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


