"""Standalone entry point for the two separately pinned training environments."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["1.0", "1.2"], default="1.0")
    parser.add_argument("--backend", choices=["fast", "reference"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024, help="Additional aggregate steps, rounded to a complete rollout")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--num-vehicles", type=int, help="Vehicles sharing one field per environment (V1.0, default: 1)")
    parser.add_argument("--inter-vehicle-clearance-m", type=float, help="Extra gap between vehicle bodies (V1.0, default: 0.5 m)")
    parser.add_argument("--stage", type=int, choices=range(8))
    parser.add_argument("--robot")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "fields.json")
    parser.add_argument("--interval", type=int, default=25000, help="Validate and checkpoint at this interval and at the end")
    parser.add_argument("--resume", type=Path, help="Checkpoint directory from this training package")
    args = parser.parse_args()
    if args.version == "1.2":
        sys.path.insert(0, str(ROOT / "versions" / "v1_2"))
    backend = args.backend or ("fast" if args.version == "1.0" else "reference")
    if args.version == "1.2" and backend == "fast":
        parser.error("The compiled backend is qualified for the packaged V1.0 environment.")
    fleet_args = (args.num_vehicles, args.inter_vehicle_clearance_m)
    if args.resume and any(x is not None for x in (args.workers, args.stage, args.robot, args.seed, *fleet_args)):
        parser.error("Resume restores workers, stage, robot, seed and vehicle settings from the checkpoint.")
    if args.version == "1.2" and any(x is not None for x in fleet_args):
        parser.error("Vehicle settings are supported by --version 1.0.")
    if args.num_vehicles is not None and args.num_vehicles < 1:
        parser.error("--num-vehicles must be a positive integer.")
    gap = args.inter_vehicle_clearance_m
    if gap is not None and (not math.isfinite(gap) or gap < 0):
        parser.error("--inter-vehicle-clearance-m must be finite and nonnegative.")
    if args.workers is not None and args.workers not in (1,2,4):
        parser.error("Choose 1, 2 or 4 workers to retain 512 aggregate steps per rollout.")
    if not args.data.is_file():
        parser.error(f"Dataset manifest missing: {args.data}")
    os.environ["AG_PPO_BACKEND"] = backend
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import torch
    import session
    from ppo_coverage_planning_v1 import PPOTrainConfig
    torch.set_num_threads(2)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Requested CUDA device is unavailable.")
    session.activate_backend()
    cfg = None
    if not args.resume:
        n = args.workers or 1
        stage = args.stage if args.stage is not None else 1
        fleet_config = {
            name: value
            for name, value in zip(("num_vehicles", "inter_vehicle_clearance_m"), fleet_args, strict=True)
            if value is not None
        }
        cfg = PPOTrainConfig(n_envs=n, n_steps=512//n, use_subproc=n>1, batch_size=64,
                             seed=args.seed if args.seed is not None else 42,
                             robot_id=args.robot or "robot_1_reference",
                             curriculum_stage=stage, max_curriculum_stage=stage,
                             enable_curriculum_progression=False, eval_episodes=2, **fleet_config)
        if args.version == "1.2":
            cfg.reward_preset = "v1_2_anti_circle"
            cfg.control_interface = "curvature_only"
            cfg.include_global_untreated_cue = True
    result = session.train(args.out, additional=args.steps, cfg=cfg, resume=args.resume,
                           data=args.data.resolve(), device=args.device,
                           checkpoint_interval=args.interval, evaluation_interval=args.interval)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
