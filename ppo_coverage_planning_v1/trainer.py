"""Stable-Baselines3 PPO training framework, TensorBoard monitoring, and resumable checkpointing.

Features:
  - Vectorized DummyVecEnv / SubprocVecEnv with optional VecNormalize (norm_obs=False, norm_reward=True).
  - Comprehensive TensorBoard & JSONL logging of learning, agricultural, safety, and generalization metrics.
  - Validation-based best checkpoint selection (strictly prioritizing agricultural coverage & safety,
    NOT raw training reward).
  - Full checkpoint saving and multi-session resumption (weights, optimizer state, VecNormalize
    running stats, config, curriculum state, seed metadata, and total timesteps).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch

from . import ENV_VERSION
from .config import PPOTrainConfig
from .curriculum import CurriculumManager, build_validation_suite_for_stage
from .datasets import AgriculturalFieldDataset
from .networks import AgCoverageFeaturesExtractor, count_trainable_parameters
from .robot_specs import get_robot_by_id


REPRODUCIBILITY_NOTES = (
    "Exact restoration: PPO policy/value weights, Adam optimizer moments, learning-rate schedule state, "
    "VecNormalize reward-return running variance/mean, curriculum stage & pass counters, training config, "
    "and completed timestep counter. Non-bitwise across restarts: OS process scheduling order in multi-env "
    "workers, fresh RNG stream advance inside environment resets after resume, and non-deterministic GPU "
    "floating-point reductions when CUDA is active."
)


def _make_env_fn(
    cfg: PPOTrainConfig,
    rank: int,
    stage_id: int,
    dataset_path: str | None,
) -> Any:
    def _init() -> Monitor:
        ds = AgriculturalFieldDataset(dataset_path) if dataset_path else None
        fixed_robot = None if stage_id == 7 else get_robot_by_id(cfg.robot_id)
        env = cfg.make_env(
            dataset=ds,
            curriculum_stage=stage_id,
            split="train",
            fixed_robot=fixed_robot,
            allow_sampled_robots=cfg.allow_sampled_robots,
        )
        env.reset(seed=cfg.seed + 1000 * rank)
        return Monitor(env)
    return _init


def save_checkpoint_bundle(
    target_dir: Path,
    model: PPO,
    vec_env: Any,
    cfg: PPOTrainConfig,
    curriculum: CurriculumManager,
    *,
    timesteps: int,
    val_metrics: dict[str, Any] | None = None,
    dataset_meta: dict[str, Any] | None = None,
) -> None:
    """Save complete self-contained resumable checkpoint directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(target_dir / "model"))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(target_dir / "vecnormalize.pkl"))
    meta = {
        "env_version": ENV_VERSION,
        "saved_at_unix": time.time(),
        "total_timesteps": int(timesteps),
        "train_config": cfg.to_dict(),
        "robot_spec": get_robot_by_id(cfg.robot_id).to_dict(),
        "curriculum_state": curriculum.state_dict(),
        "dataset_manifest_meta": dataset_meta or {},
        "validation_metrics": val_metrics or {},
        "reproducibility_notes": REPRODUCIBILITY_NOTES,
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def evaluate_policy_on_suite(
    model: PPO | None,
    suite: list[tuple[Any, Any, int]],
    cfg: PPOTrainConfig,
    *,
    stage_id: int | None = None,
    deterministic: bool = True,
    record_trajectories: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate a PPO policy (or random policy when model is None) on a fixed validation suite."""
    env_cfg = cfg.build_env_config()
    # Strict evaluation always uses TERMINATE collision mode!
    from dataclasses import replace
    from .curriculum import CURRICULUM_STAGES
    from .field_world import CollisionMode
    strict_cfg = replace(env_cfg, collision_mode=CollisionMode.TERMINATE)
    if cfg.coverage_target_pct is not None:
        eval_target_pct = float(cfg.coverage_target_pct)
    elif stage_id is not None and stage_id in CURRICULUM_STAGES:
        eval_target_pct = float(CURRICULUM_STAGES[stage_id].coverage_target_pct)
    else:
        eval_target_pct = 99.0
    eval_time_mult = (
        float(CURRICULUM_STAGES[stage_id].time_budget_multiplier)
        if (stage_id is not None and stage_id in CURRICULUM_STAGES)
        else strict_cfg.time_budget_multiplier
    )
    eval_min_time = (
        float(CURRICULUM_STAGES[stage_id].min_episode_time_s)
        if (stage_id is not None and stage_id in CURRICULUM_STAGES)
        else strict_cfg.min_episode_time_s
    )

    summaries: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    rng = np.random.default_rng(cfg.seed + 777)

    for field_rec, robot_spec, ep_seed in suite:
        world = field_rec.to_world(sdf_resolution_m=0.5)
        env = cfg.make_env(env_config=strict_cfg, world=world, robot_spec=robot_spec)
        obs, _ = env.reset(
            seed=ep_seed,
            options={
                "coverage_target_pct": eval_target_pct,
                "time_budget_multiplier": eval_time_mult,
                "min_episode_time_s": eval_min_time,
                "randomize_start_pose": cfg.randomize_start_pose,
            },
        )
        t0 = time.perf_counter()
        inf_time = 0.0
        n_actions = 0
        done = False
        last_info: dict[str, Any] = {}
        while not done:
            ti0 = time.perf_counter()
            if model is None:
                action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            inf_time += time.perf_counter() - ti0
            n_actions += 1
            obs, _rew, terminated, truncated, last_info = env.step(action, record_trajectory=record_trajectories)
            done = terminated or truncated
        rollout_wall_s = time.perf_counter() - t0
        ep_sum = dict(last_info.get("episode_summary", {}))
        ep_sum["inference_ms_per_action"] = 1000.0 * inf_time / max(1, n_actions)
        ep_sum["rollout_wall_s"] = rollout_wall_s
        ep_sum["ep_seed"] = ep_seed
        summaries.append(ep_sum)
        if record_trajectories:
            poses = env.get_trajectory_poses()
            trajectories.append({
                "field_id": field_rec.field_id,
                "family": field_rec.family,
                "robot_id": robot_spec.robot_id,
                "wkb_hex": field_rec.wkb_hex,
                "ep_seed": ep_seed,
                "summary": ep_sum,
                "trajectory_poses": poses.tolist() if isinstance(poses, np.ndarray) else poses,
            })
        env.close()

    covs = [s["coverage_pct"] for s in summaries]
    metrics = {
        "n_episodes": float(len(summaries)),
        "mean_coverage_pct": float(np.mean(covs)) if covs else 0.0,
        "median_coverage_pct": float(np.median(covs)) if covs else 0.0,
        "max_coverage_pct": float(np.max(covs)) if covs else 0.0,
        "min_coverage_pct": float(np.min(covs)) if covs else 0.0,
        "stage_completion_rate": float(np.mean([1.0 if s["stage_completed"] else 0.0 for s in summaries])),
        "final_99_completion_rate": float(np.mean([1.0 if s["final_99_completed"] else 0.0 for s in summaries])),
        "body_collision_rate": float(np.mean([1.0 if s["body_collision"] else 0.0 for s in summaries])),
        "exterior_collision_rate": float(np.mean([1.0 if s["exterior_collision"] else 0.0 for s in summaries])),
        "obstacle_collision_rate": float(np.mean([1.0 if s["obstacle_collision"] else 0.0 for s in summaries])),
        "clearance_violation_rate": float(np.mean([1.0 if s["clearance_violation"] else 0.0 for s in summaries])),
        "mean_trajectory_length_m": float(np.mean([s["distance_m"] for s in summaries])),
        "mean_repeated_treatment_m2": float(np.mean([s["repeated_treatment_m2"] for s in summaries])),
        "mean_repeated_treatment_pct": float(np.mean([s["repeated_treatment_pct"] for s in summaries])),
        "mean_largest_omission_m2": float(np.mean([s["largest_omission_m2"] for s in summaries])),
        "mean_operating_time_s": float(np.mean([s["operating_time_s"] for s in summaries])),
        "mean_episode_reward": float(np.mean([s["episode_reward"] for s in summaries])),
        "mean_saturation_rate": float(np.mean([s["saturation_rate"] for s in summaries])),
        "mean_constraint_violations": float(np.mean([s["constraint_violations"] for s in summaries])),
        "mean_inference_ms_per_action": float(np.mean([s["inference_ms_per_action"] for s in summaries])),
        "mean_rollout_wall_s": float(np.mean([s["rollout_wall_s"] for s in summaries])),
    }
    # Composite validation selection score (agricultural coverage + safety, NOT raw training reward!)
    metrics["val_selection_score"] = (
        metrics["median_coverage_pct"]
        + 25.0 * metrics["final_99_completion_rate"]
        + 10.0 * metrics["stage_completion_rate"]
        - 30.0 * metrics["body_collision_rate"]
        - 18.0 * metrics["clearance_violation_rate"]
        - 0.15 * metrics["mean_repeated_treatment_pct"]
    )
    if cfg.num_vehicles > 1:
        metrics["vehicle_collision_rate"] = float(np.mean([s["vehicle_collision"] for s in summaries]))
        metrics["vehicle_clearance_violation_rate"] = float(
            np.mean([s["vehicle_clearance_breach"] for s in summaries])
        )
    return metrics, trajectories


class AgriculturalMonitorCallback(BaseCallback):
    """Logs agricultural & safety metrics, runs periodic validation, checkpoints, and curriculum progression."""

    def __init__(
        self,
        *,
        cfg: PPOTrainConfig,
        output_dir: Path,
        curriculum: CurriculumManager,
        dataset: AgriculturalFieldDataset | None,
        eval_suites: dict[int, list[tuple[Any, Any, int]]],
        save_checkpoint_steps: set[int] | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.output_dir = output_dir
        self.curriculum = curriculum
        self.dataset = dataset
        self.eval_suites = eval_suites
        self.save_checkpoint_steps = set(save_checkpoint_steps or ())

        self._t_start = time.perf_counter()
        self._steps_at_start = 0
        self._last_eval_step = 0
        self._last_ckpt_step = 0
        self._best_score = -1e9
        self._last_eval_metrics: dict[str, float] | None = None
        self._last_eval_trajs: list[dict[str, Any]] = []
        self._last_eval_timestep: int = -1
        self._recent_ep_summaries: list[dict[str, Any]] = []
        self._eval_csv_path = output_dir / "eval_history.csv"
        self._ep_jsonl_path = output_dir / "train_episodes.jsonl"

    def _on_training_start(self) -> None:
        self._t_start = time.perf_counter()
        self._steps_at_start = int(self.num_timesteps)
        self._last_eval_step = int(self.num_timesteps)
        self._last_ckpt_step = int(self.num_timesteps)
        # Run initial baseline evaluation at step 0 if starting fresh
        if self.num_timesteps == 0:
            self._run_eval_and_checkpoint(force_checkpoint_tag="step_0")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_sum = info.get("episode_summary")
            if ep_sum is not None:
                self._recent_ep_summaries.append(ep_sum)
                with self._ep_jsonl_path.open("a") as f:
                    f.write(json.dumps({"timesteps": int(self.num_timesteps), **ep_sum}) + "\n")

        # Flush training episode statistics to TensorBoard periodically
        if len(self._recent_ep_summaries) >= 4:
            self._log_train_episodes()

        # Exact milestone checkpoints (e.g. 10k, 50k, 100k, 250k, 500k)
        ran_milestone_eval = False
        for target_step in sorted(list(self.save_checkpoint_steps)):
            if self.num_timesteps >= target_step:
                self.save_checkpoint_steps.remove(target_step)
                self._run_eval_and_checkpoint(force_checkpoint_tag=f"step_{target_step}")
                self._last_eval_step = int(self.num_timesteps)
                ran_milestone_eval = True

        if (not ran_milestone_eval) and (self.num_timesteps - self._last_eval_step >= self.cfg.eval_freq_steps):
            self._last_eval_step = int(self.num_timesteps)
            self._run_eval_and_checkpoint(force_checkpoint_tag=None)

        if self.num_timesteps - self._last_ckpt_step >= self.cfg.checkpoint_freq_steps:
            self._last_ckpt_step = int(self.num_timesteps)
            save_checkpoint_bundle(
                self.output_dir / "checkpoints" / f"step_{self.num_timesteps}",
                self.model,
                self.training_env,
                self.cfg,
                self.curriculum,
                timesteps=self.num_timesteps,
                dataset_meta=self.dataset.manifest_meta if self.dataset else {},
            )
            save_checkpoint_bundle(
                self.output_dir / "latest",
                self.model,
                self.training_env,
                self.cfg,
                self.curriculum,
                timesteps=self.num_timesteps,
                dataset_meta=self.dataset.manifest_meta if self.dataset else {},
            )
        return True

    def _log_train_episodes(self) -> None:
        eps = self._recent_ep_summaries
        self.logger.record("ag_train/mean_coverage_pct", float(np.mean([e["coverage_pct"] for e in eps])))
        self.logger.record("ag_train/median_coverage_pct", float(np.median([e["coverage_pct"] for e in eps])))
        self.logger.record("ag_train/repeated_pct", float(np.mean([e["repeated_treatment_pct"] for e in eps])))
        self.logger.record("ag_train/largest_omission_m2", float(np.mean([e["largest_omission_m2"] for e in eps])))
        self.logger.record("ag_train/distance_m", float(np.mean([e["distance_m"] for e in eps])))
        self.logger.record("ag_train/operating_time_s", float(np.mean([e["operating_time_s"] for e in eps])))
        self.logger.record("ag_train/unproductive_dist_m", float(np.mean([e["unproductive_dist_m"] for e in eps])))
        self.logger.record("ag_train/time_without_new_cov_s", float(np.mean([e["time_without_new_cov_s"] for e in eps])))
        self.logger.record("ag_train/obstacle_treatment_m2", float(np.mean([e["obstacle_treatment_m2"] for e in eps])))
        self.logger.record("safety_train/body_collision_rate", float(np.mean([1.0 if e["body_collision"] else 0.0 for e in eps])))
        self.logger.record("safety_train/clearance_violation_rate", float(np.mean([1.0 if e["clearance_violation"] else 0.0 for e in eps])))
        self.logger.record("safety_train/saturation_rate", float(np.mean([e["saturation_rate"] for e in eps])))
        self.logger.record("safety_train/constraint_violations", float(np.mean([e["constraint_violations"] for e in eps])))
        self.logger.record("curriculum/stage", float(self.curriculum.current_stage))
        self._recent_ep_summaries = []

    def _run_eval_and_checkpoint(self, *, force_checkpoint_tag: str | None) -> None:
        stage = self.curriculum.current_stage
        suite = self.eval_suites[stage]
        metrics, trajs = evaluate_policy_on_suite(
            self.model, suite, self.cfg, stage_id=stage, deterministic=True, record_trajectories=(force_checkpoint_tag is not None)
        )
        elapsed = max(1e-6, time.perf_counter() - self._t_start)
        sps = (self.num_timesteps - self._steps_at_start) / elapsed
        metrics["timesteps"] = float(self.num_timesteps)
        metrics["stage"] = float(stage)
        metrics["train_steps_per_sec"] = float(sps)
        self._last_eval_metrics = dict(metrics)
        self._last_eval_trajs = trajs
        self._last_eval_timestep = int(self.num_timesteps)

        for k, v in metrics.items():
            self.logger.record(f"ag_val/{k}", v)

        # Append to CSV history
        write_header = not self._eval_csv_path.exists()
        with self._eval_csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

        if force_checkpoint_tag is not None:
            cdir = self.output_dir / "checkpoints" / force_checkpoint_tag
            save_checkpoint_bundle(
                cdir, self.model, self.training_env, self.cfg, self.curriculum,
                timesteps=self.num_timesteps, val_metrics=metrics,
                dataset_meta=self.dataset.manifest_meta if self.dataset else {},
            )
            if trajs:
                (cdir / "val_trajectories.json").write_text(json.dumps(trajs))

        if metrics["val_selection_score"] > self._best_score:
            self._best_score = metrics["val_selection_score"]
            save_checkpoint_bundle(
                self.output_dir / "best", self.model, self.training_env, self.cfg, self.curriculum,
                timesteps=self.num_timesteps, val_metrics=metrics,
                dataset_meta=self.dataset.manifest_meta if self.dataset else {},
            )
            if trajs:
                (self.output_dir / "best" / "val_trajectories.json").write_text(json.dumps(trajs))

        if self.cfg.enable_curriculum_progression:
            advanced, reason = self.curriculum.evaluate_progression(metrics, int(self.num_timesteps))
            if advanced:
                new_stage = self.curriculum.current_stage
                self.training_env.env_method("set_curriculum_stage", new_stage)
                if new_stage not in self.eval_suites:
                    fixed_robot = None if new_stage == 7 else get_robot_by_id(self.cfg.robot_id)
                    self.eval_suites[new_stage] = build_validation_suite_for_stage(
                        new_stage, dataset=self.dataset, n_episodes=self.cfg.eval_episodes,
                        seed=self.cfg.seed + 5000, fixed_robot=fixed_robot,
                    )
                if self.verbose:
                    print(f"[Curriculum] step={self.num_timesteps} -> {reason}")

        if self.verbose:
            print(
                f"[Val step={self.num_timesteps:7d} | stg={stage}] "
                f"cov mean={metrics['mean_coverage_pct']:5.1f}% med={metrics['median_coverage_pct']:5.1f}% max={metrics['max_coverage_pct']:5.1f}% | "
                f"comp={100*metrics['stage_completion_rate']:4.0f}% (99%={100*metrics['final_99_completion_rate']:4.0f}%) | "
                f"body_coll={100*metrics['body_collision_rate']:4.0f}% clr_viol={100*metrics['clearance_violation_rate']:4.0f}% | "
                f"rep={metrics['mean_repeated_treatment_pct']:4.1f}% rew={metrics['mean_episode_reward']:+6.1f} | {sps:5.1f} sps"
            )


def train_ppo(
    cfg: PPOTrainConfig,
    output_dir: Path | str,
    *,
    resume_from: Path | str | None = None,
    milestone_steps: tuple[int, ...] = (),
    verbose: int = 1,
) -> dict[str, Any]:
    """Run (or resume) end-to-end SB3 PPO training and return structured training summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out / "train_config.json")

    dataset = AgriculturalFieldDataset(cfg.dataset_manifest) if cfg.dataset_manifest else None
    curriculum = CurriculumManager(initial_stage=cfg.curriculum_stage, max_stage=cfg.max_curriculum_stage)

    resume_dir = Path(resume_from) if resume_from is not None else None
    if resume_dir is not None:
        meta_path = resume_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if "curriculum_state" in meta:
                curriculum.load_state_dict(meta["curriculum_state"])
                # Allow explicit manual stage promotion on resume (e.g., Stage 1 -> Stage 2 continuation)
                if cfg.curriculum_stage > curriculum.current_stage:
                    curriculum.current_stage = cfg.curriculum_stage
                if cfg.max_curriculum_stage > curriculum.max_stage:
                    curriculum.max_stage = cfg.max_curriculum_stage

    stage = curriculum.current_stage
    env_fns = [_make_env_fn(cfg, rank=i, stage_id=stage, dataset_path=cfg.dataset_manifest) for i in range(cfg.n_envs)]
    if cfg.use_subproc and cfg.n_envs > 1:
        base_vec = SubprocVecEnv(env_fns)
    else:
        base_vec = DummyVecEnv(env_fns)

    if cfg.normalize_reward:
        vecnorm_path = (resume_dir / "vecnormalize.pkl") if resume_dir is not None else None
        if vecnorm_path is not None and vecnorm_path.exists():
            vec_env: Any = VecNormalize.load(str(vecnorm_path), base_vec)
            vec_env.training = True
            vec_env.norm_obs = False
            vec_env.norm_reward = True
        else:
            vec_env = VecNormalize(base_vec, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=cfg.effective_gamma())
    else:
        vec_env = base_vec

    # Fixed validation suites per stage (deterministic, separate from training seeds!)
    eval_suites: dict[int, list[tuple[Any, Any, int]]] = {}
    for s_id in range(min(cfg.curriculum_stage, stage), cfg.max_curriculum_stage + 1):
        fixed_robot = None if s_id == 7 else get_robot_by_id(cfg.robot_id)
        eval_suites[s_id] = build_validation_suite_for_stage(
            s_id, dataset=dataset, n_episodes=cfg.eval_episodes, seed=cfg.seed + 5000, fixed_robot=fixed_robot
        )

    tb_dir = out / "tb_logs"
    if resume_dir is not None:
        model = PPO.load(
            str(resume_dir / "model"),
            env=vec_env,
            device=cfg.device,
            tensorboard_log=str(tb_dir),
        )
        reset_num_timesteps = False
        remaining_steps = max(0, cfg.total_timesteps - int(model.num_timesteps))
    else:
        policy_kwargs = {
            "features_extractor_class": AgCoverageFeaturesExtractor,
            "features_extractor_kwargs": {"local_dim": 64, "global_dim": 64, "state_dim": 64, "sensor_dim": 64},
            "net_arch": {"pi": list(cfg.net_arch_pi), "vf": list(cfg.net_arch_vf)},
            "normalize_images": False,
        }
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.effective_gamma(),
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tb_dir),
            seed=cfg.seed,
            device=cfg.device,
            verbose=0,
        )
        reset_num_timesteps = True
        remaining_steps = int(cfg.total_timesteps)

    n_params = count_trainable_parameters(model.policy)
    if verbose:
        print(
            f"[PPO] {cfg.experiment_name} | device={model.device} | params={n_params:,} | "
            f"dt_action={cfg.dt_action_s}s | gamma={cfg.effective_gamma():.4f} | "
            f"n_envs={cfg.n_envs} | target_steps={cfg.total_timesteps} (remaining={remaining_steps})"
        )

    cb = AgriculturalMonitorCallback(
        cfg=cfg,
        output_dir=out,
        curriculum=curriculum,
        dataset=dataset,
        eval_suites=eval_suites,
        save_checkpoint_steps=set(milestone_steps),
        verbose=verbose,
    )

    t0 = time.perf_counter()
    if remaining_steps > 0:
        model.learn(
            total_timesteps=remaining_steps,
            callback=cb,
            tb_log_name=cfg.experiment_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
    wall_s = time.perf_counter() - t0

    # Final validation evaluation (reuse if callback already evaluated at exact final timestep!)
    final_stage = curriculum.current_stage
    if cb._last_eval_timestep == int(model.num_timesteps) and cb._last_eval_metrics is not None and cb._last_eval_trajs:
        final_metrics = dict(cb._last_eval_metrics)
        final_trajs = cb._last_eval_trajs
    else:
        final_metrics, final_trajs = evaluate_policy_on_suite(
            model, eval_suites[final_stage], cfg, stage_id=final_stage, deterministic=True, record_trajectories=True
        )
    final_metrics["timesteps"] = float(model.num_timesteps)
    final_metrics["train_steps_per_sec"] = float(remaining_steps / max(1e-6, wall_s))
    save_checkpoint_bundle(
        out / "latest", model, vec_env, cfg, curriculum,
        timesteps=model.num_timesteps, val_metrics=final_metrics,
        dataset_meta=dataset.manifest_meta if dataset else {},
    )
    (out / "latest" / "val_trajectories.json").write_text(json.dumps(final_trajs))
    if final_metrics["val_selection_score"] > cb._best_score:
        save_checkpoint_bundle(
            out / "best", model, vec_env, cfg, curriculum,
            timesteps=model.num_timesteps, val_metrics=final_metrics,
            dataset_meta=dataset.manifest_meta if dataset else {},
        )
        (out / "best" / "val_trajectories.json").write_text(json.dumps(final_trajs))

    summary = {
        "experiment_name": cfg.experiment_name,
        "output_dir": str(out),
        "device": str(model.device),
        "trainable_parameters": n_params,
        "total_timesteps": int(model.num_timesteps),
        "trained_steps_this_session": int(remaining_steps),
        "wall_time_s": round(wall_s, 3),
        "train_steps_per_sec": round(remaining_steps / max(1e-6, wall_s), 2),
        "final_stage": final_stage,
        "final_validation_metrics": final_metrics,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    vec_env.close()
    return summary
