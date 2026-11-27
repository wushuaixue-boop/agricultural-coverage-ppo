"""Lightweight multi-branch CNN + MLP feature extractor for Stable-Baselines3 PPO.

Architecture:
  - local_map (5, H_loc, W_loc)  -> 3-layer strided CNN -> Linear -> 64-D (when enabled)
  - global_map (6, H_glob, W_glob) -> 3-layer strided CNN -> Linear -> 64-D (when enabled)
  - proprio + capabilities (23-D) -> 2-layer MLP -> 64-D
  - progress + obstacles (62-D)   -> 2-layer MLP -> 64-D
  Concatenated features (192-D or 256-D) feed SB3's Actor and Critic MLPs ([128, 128]).
"""
from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn


def _make_map_cnn(in_channels: int, h: int, w: int, out_dim: int) -> nn.Sequential:
    conv = nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, h, w)
        flat_dim = int(conv(dummy).shape[1])
    return nn.Sequential(
        conv,
        nn.Linear(flat_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
    )


class AgCoverageFeaturesExtractor(BaseFeaturesExtractor):
    """Multi-modal CNN+MLP encoder for Dict observations from PPOCoverageEnv."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        local_dim: int = 64,
        global_dim: int = 64,
        state_dim: int = 64,
        sensor_dim: int = 64,
    ) -> None:
        spaces = observation_space.spaces
        self.has_local = "local_map" in spaces
        self.has_global = "global_map" in spaces
        self.has_vehicle_axis = "vehicle_pose" in spaces
        num_vehicles = spaces["vehicle_pose"].shape[0] if self.has_vehicle_axis else 1

        total_dim = num_vehicles * (state_dim + sensor_dim + (local_dim if self.has_local else 0))
        if self.has_global:
            total_dim += global_dim

        super().__init__(observation_space, features_dim=total_dim)

        if self.has_local:
            c, h, w = spaces["local_map"].shape[-3:]
            self.local_cnn = _make_map_cnn(c, h, w, local_dim)
        else:
            self.local_cnn = None

        if self.has_global:
            c, h, w = spaces["global_map"].shape
            self.global_cnn = _make_map_cnn(c, h, w, global_dim)
        else:
            self.global_cnn = None

        prop_dim = spaces["proprio"].shape[-1]
        cap_dim = spaces["capabilities"].shape[-1]
        pose_dim = spaces["vehicle_pose"].shape[-1] if self.has_vehicle_axis else 0
        self.state_mlp = nn.Sequential(
            nn.Linear(prop_dim + cap_dim + pose_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
            nn.ReLU(),
        )

        prog_dim = int(spaces["progress"].shape[0])
        obs_dim = int(spaces["obstacles"].shape[-1])
        self.sensor_mlp = nn.Sequential(
            nn.Linear(prog_dim + obs_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, sensor_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # Fleet observations have a vehicle axis after the batch axis. Apply
        # the same encoders to every vehicle, retaining vehicle order for PPO.
        state = [observations["proprio"], observations["capabilities"]]
        progress = observations["progress"]
        local_map = observations.get("local_map")
        batch_size = progress.shape[0]
        if self.has_vehicle_axis:
            poses = observations["vehicle_pose"]
            state.append(poses)
            progress = progress.unsqueeze(1).expand(-1, poses.shape[1], -1)
            if local_map is not None:
                local_map = local_map.flatten(0, 1)

        parts: list[torch.Tensor] = []
        if self.local_cnn is not None:
            parts.append(self.local_cnn(local_map).reshape(batch_size, -1))
        if self.global_cnn is not None:
            parts.append(self.global_cnn(observations["global_map"]))

        state_in = torch.cat(state, dim=-1)
        parts.append(self.state_mlp(state_in).reshape(batch_size, -1))

        sensor_in = torch.cat([progress, observations["obstacles"]], dim=-1)
        parts.append(self.sensor_mlp(sensor_in).reshape(batch_size, -1))
        return torch.cat(parts, dim=1)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
