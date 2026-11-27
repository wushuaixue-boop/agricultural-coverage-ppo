"""Standalone CLI script for evaluating a saved PPO checkpoint (or random policy) without training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PPOTrainConfig
from .curriculum import build_validation_suite_for_stage
from .datasets import AgriculturalFieldDataset
from .evaluator import evaluate_episodes, load_checkpoint
from .robot_specs import get_robot_by_id
from .trainer import evaluate_policy_on_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PPO policy or random baseline. Coverage is measured with Shapely for "
        "single-vehicle episodes and with the shared raster for fleets."
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint dir containing model.zip & metadata.json. Omit for random baseline.")
    parser.add_argument("--dataset-manifest", type=Path, default=None, help="Optional real_fields_splits.json path.")
    parser.add_argument("--stage", type=int, default=1, help="Evaluation curriculum stage (0..7).")
    parser.add_argument("--robot-id", type=str, default=None, help="Override RobotSpec ID for evaluation.")
    parser.add_argument("--n-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--stochastic", action="store_true", help="Evaluate stochastic policy instead of deterministic.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.checkpoint is not None:
        model, cfg, meta = load_checkpoint(args.checkpoint, device=args.device)
        print(f"Loaded checkpoint {args.checkpoint} (timesteps={meta.get('total_timesteps')})")
    else:
        model = None
        cfg = PPOTrainConfig(curriculum_stage=args.stage)
        print("Evaluating untrained random baseline policy")

    ds = AgriculturalFieldDataset(args.dataset_manifest) if args.dataset_manifest else None
    robot_id = args.robot_id or cfg.robot_id
    fixed_robot = None if args.stage == 7 and args.robot_id is None else get_robot_by_id(robot_id)
    suite = build_validation_suite_for_stage(
        args.stage, dataset=ds, n_episodes=args.n_episodes, seed=args.seed, fixed_robot=fixed_robot
    )
    if cfg.num_vehicles > 1:
        aggregate, trajectories = evaluate_policy_on_suite(
            model, suite, cfg, stage_id=args.stage, deterministic=not args.stochastic, record_trajectories=True
        )
        res = {
            "coverage_method": "shared_raster",
            "aggregate": aggregate,
            "episodes": [trajectory["summary"] for trajectory in trajectories],
            "trajectories": trajectories,
        }
    else:
        res = evaluate_episodes(
            model,
            cfg,
            suite,
            deterministic=not args.stochastic,
            use_exact_reference_coverage=True,
        )
    print(json.dumps(res["aggregate"], indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
