"""Central PPO training configuration and future large-scale Kaggle training presets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any

from .dynamics import ControlInterface
from .env import ObservationConfig, PPOCoverageEnv, PPOEnvConfig
from .field_world import CollisionMode
from .rewards import RewardConfig, get_reward_preset


def compute_physical_time_gamma(dt_action_s: float, tau_s: float) -> float:
    """Compute effective PPO discount factor gamma = exp(-dt_action / tau) in simulated physical time."""
    if tau_s <= 0.0:
        raise ValueError(f"tau_s must be positive, got {tau_s}")
    return float(math.exp(-dt_action_s / tau_s))


@dataclass
class PPOTrainConfig:
    """Centralized reproducible PPO + Environment + Curriculum configuration."""
    experiment_name: str = "ppo_stage1_pilot"
    robot_id: str = "robot_1_reference"
    curriculum_stage: int = 1
    max_curriculum_stage: int = 1
    enable_curriculum_progression: bool = False
    allow_sampled_robots: bool = False
    dataset_manifest: str | None = None

    # Environment & Observation & Reward
    dt_action_s: float = 0.50
    dt_phys_s: float = 0.02
    coverage_resolution_m: float = 0.50
    control_interface: str = ControlInterface.TARGET_SPEED_CURVATURE.value
    collision_mode: str = CollisionMode.TERMINATE.value
    coverage_target_pct: float | None = None   # None => use stage default
    randomize_start_pose: bool = True
    include_local_map: bool = True
    include_global_map: bool = True
    local_map_size: int = 24
    local_map_extent_m: float = 36.0
    global_map_size: int = 32
    global_map_frame: str = "ego_aligned"
    n_lidar_rays: int = 24
    lidar_max_range_m: float = 36.0
    reward_preset: str = "simple_v1"

    # PPO Hyperparameters (feedforward MultiInputPolicy baseline)
    learning_rate: float = 3e-4
    n_steps: int = 1024                 # rollout steps per env
    batch_size: int = 256
    n_epochs: int = 4
    discount_timescale_tau_s: float = 60.0
    gamma_override: float | None = None
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.01
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50
    net_arch_pi: tuple[int, ...] = (128, 128)
    net_arch_vf: tuple[int, ...] = (128, 128)
    normalize_reward: bool = True
    n_envs: int = 2
    use_subproc: bool = False
    total_timesteps: int = 150_000
    eval_freq_steps: int = 25_000
    eval_episodes: int = 8
    checkpoint_freq_steps: int = 50_000
    seed: int = 42
    device: str = "auto"
    num_vehicles: int = 1
    inter_vehicle_clearance_m: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.num_vehicles, bool) or not isinstance(self.num_vehicles, Integral) or self.num_vehicles < 1:
            raise ValueError("num_vehicles must be a positive integer")
        if not math.isfinite(self.inter_vehicle_clearance_m) or self.inter_vehicle_clearance_m < 0:
            raise ValueError("inter_vehicle_clearance_m must be finite and nonnegative")

    def make_env(self, *, env_config: PPOEnvConfig | None = None, **kwargs) -> PPOCoverageEnv:
        env_config = env_config if env_config is not None else self.build_env_config()
        if self.num_vehicles == 1:
            return PPOCoverageEnv(env_config=env_config, **kwargs)

        from .multi_vehicle_env import MultiVehicleCoverageEnv

        return MultiVehicleCoverageEnv(
            env_config=env_config,
            num_vehicles=self.num_vehicles,
            inter_vehicle_clearance_m=self.inter_vehicle_clearance_m,
            **kwargs,
        )

    def effective_gamma(self) -> float:
        if self.gamma_override is not None:
            return float(self.gamma_override)
        return compute_physical_time_gamma(self.dt_action_s, self.discount_timescale_tau_s)

    def build_env_config(self) -> PPOEnvConfig:
        rc: RewardConfig = get_reward_preset(self.reward_preset)
        oc = ObservationConfig(
            include_local_map=self.include_local_map,
            include_global_map=self.include_global_map,
            local_map_size=self.local_map_size,
            local_map_extent_m=self.local_map_extent_m,
            global_map_size=self.global_map_size,
            global_map_frame=self.global_map_frame,
            n_lidar_rays=self.n_lidar_rays,
            lidar_max_range_m=self.lidar_max_range_m,
        )
        return PPOEnvConfig(
            dt_action_s=self.dt_action_s,
            dt_phys_s=self.dt_phys_s,
            coverage_resolution_m=self.coverage_resolution_m,
            control_interface=ControlInterface(self.control_interface),
            collision_mode=CollisionMode(self.collision_mode),
            coverage_target_pct=self.coverage_target_pct if self.coverage_target_pct is not None else 99.0,
            randomize_start_pose=self.randomize_start_pose,
            obs=oc,
            reward=rc,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["effective_gamma"] = round(self.effective_gamma(), 6)
        return d

    def save_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PPOTrainConfig:
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "net_arch_pi" in valid:
            valid["net_arch_pi"] = tuple(valid["net_arch_pi"])
        if "net_arch_vf" in valid:
            valid["net_arch_vf"] = tuple(valid["net_arch_vf"])
        return cls(**valid)


def get_future_scale_presets() -> dict[str, PPOTrainConfig]:
    """Documented multi-session Kaggle scaling configurations (1M, 5M, 10M, 50M, 100M steps)."""
    return {
        "kaggle_1m_stage1_2": PPOTrainConfig(
            experiment_name="kaggle_1m_stage1_2",
            curriculum_stage=1,
            max_curriculum_stage=2,
            enable_curriculum_progression=True,
            dt_action_s=0.50,
            discount_timescale_tau_s=80.0,
            n_steps=2048,
            batch_size=256,
            n_envs=4,
            use_subproc=True,
            total_timesteps=1_000_000,
            eval_freq_steps=50_000,
            checkpoint_freq_steps=100_000,
            device="auto",
        ),
        "kaggle_5m_stage1_4": PPOTrainConfig(
            experiment_name="kaggle_5m_stage1_4",
            curriculum_stage=1,
            max_curriculum_stage=4,
            enable_curriculum_progression=True,
            dt_action_s=0.50,
            discount_timescale_tau_s=100.0,
            n_steps=2048,
            batch_size=256,
            n_envs=4,
            use_subproc=True,
            total_timesteps=5_000_000,
            eval_freq_steps=100_000,
            checkpoint_freq_steps=250_000,
            device="auto",
        ),
        "kaggle_10m_stage1_6": PPOTrainConfig(
            experiment_name="kaggle_10m_stage1_6",
            curriculum_stage=1,
            max_curriculum_stage=6,
            enable_curriculum_progression=True,
            dt_action_s=0.50,
            discount_timescale_tau_s=120.0,
            n_steps=2048,
            batch_size=256,
            n_envs=4,
            use_subproc=True,
            total_timesteps=10_000_000,
            eval_freq_steps=200_000,
            checkpoint_freq_steps=500_000,
            device="auto",
        ),
        "kaggle_50m_full_curriculum": PPOTrainConfig(
            experiment_name="kaggle_50m_full_curriculum",
            curriculum_stage=1,
            max_curriculum_stage=7,
            enable_curriculum_progression=True,
            allow_sampled_robots=False,
            dt_action_s=0.50,
            discount_timescale_tau_s=150.0,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=512,
            n_envs=4,
            use_subproc=True,
            total_timesteps=50_000_000,
            eval_freq_steps=500_000,
            checkpoint_freq_steps=1_000_000,
            device="auto",
        ),
        "kaggle_100m_heterogeneous": PPOTrainConfig(
            experiment_name="kaggle_100m_heterogeneous",
            curriculum_stage=1,
            max_curriculum_stage=7,
            enable_curriculum_progression=True,
            allow_sampled_robots=True,
            dt_action_s=0.50,
            discount_timescale_tau_s=150.0,
            learning_rate=2.0e-4,
            n_steps=2048,
            batch_size=512,
            n_envs=4,
            use_subproc=True,
            total_timesteps=100_000_000,
            eval_freq_steps=1_000_000,
            checkpoint_freq_steps=2_000_000,
            device="auto",
        ),
    }
