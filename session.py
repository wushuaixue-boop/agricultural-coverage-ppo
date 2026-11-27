"""PPO session runner. Checkpoints are taken after complete on-policy updates."""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from ppo_coverage_planning_v1 import (AgriculturalFieldDataset, CurriculumManager, ENV_VERSION,
    PPOCoverageEnv, PPOTrainConfig, build_validation_suite_for_stage, get_robot_by_id)
from ppo_coverage_planning_v1.networks import AgCoverageFeaturesExtractor
from ppo_coverage_planning_v1.trainer import evaluate_policy_on_suite, save_checkpoint_bundle


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")


def check_data(manifest):
    if manifest is None:
        return "synthetic"
    manifest = Path(manifest)
    root = manifest.parent
    checks = json.loads((root / "checksums.json").read_bytes())
    if manifest.name not in checks or not all(sha(root / name) == value for name,value in checks.items()):
        raise ValueError("Field package checksum mismatch")
    return sha(manifest)


def manifest_path():
    from ppo_coverage_planning_v1 import __file__ as package_file
    return Path(package_file).resolve().parent.parent / "source_manifest.json"


def activate_backend():
    if os.environ.get("AG_PPO_BACKEND") == "fast":
        from ppo_fast import enable
        enable()


def source_signature():
    manifest = json.loads(manifest_path().read_bytes())
    for name, value in manifest["files"].items():
        if sha(Path(__file__).parent / name) != value:
            raise ValueError(f"Source integrity mismatch for {name}")
    return manifest["source_commit"]


def val_suite(cfg, dataset, stage, robot_mode):
    robot = None if robot_mode == "curriculum" and stage == 7 else get_robot_by_id(cfg.robot_id)
    return build_validation_suite_for_stage(stage,dataset=dataset,n_episodes=cfg.eval_episodes,
                                           seed=cfg.seed+5000,fixed_robot=robot)


def make_env(cfg, path, stage, robot_mode):
    def create():
        activate_backend()
        dataset = AgriculturalFieldDataset(path) if path else None
        robot = None if robot_mode == "curriculum" and stage == 7 else get_robot_by_id(cfg.robot_id)
        # V1.2 still uses its original single-vehicle environment.
        create_env = getattr(cfg, "make_env", PPOCoverageEnv)
        return Monitor(create_env(env_config=cfg.build_env_config(), dataset=dataset,
                                  curriculum_stage=stage, split="train", fixed_robot=robot,
                                  allow_sampled_robots=cfg.allow_sampled_robots))
    return create


class MeasuredPPO(PPO):
    def train(self):
        start = time.perf_counter()
        super().train()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.optimization_seconds = getattr(self,"optimization_seconds",0.) + time.perf_counter()-start


def train(out, *, additional=1024, cfg=None, resume=None, data=None, device="cpu",
          checkpoint_interval=1024, evaluation_interval=1024, robot_mode=None,
          legacy=False, warm_start=None):
    out = Path(out)
    if resume and warm_start:
        raise ValueError("Choose one initialization mode")
    source = source_signature()
    package_hash = sha(manifest_path())
    field_hash = check_data(data)
    seed_offset = 0
    best = None
    ck = Path(resume) if resume else None
    if ck:
        if cfg is not None:
            raise ValueError("Continuation takes the archived configuration")
        meta = json.loads((ck/"metadata.json").read_bytes())
        if meta["env_version"] != ENV_VERSION:
            raise ValueError("Environment version changed")
        state_file = ck/"session.json"
        if state_file.exists():
            state = json.loads(state_file.read_bytes())
            archived_package = state["package_sha256"]
            compatible_packages = json.loads(manifest_path().read_bytes()).get("compatible_single_vehicle_packages", [])
            package_matches = archived_package == package_hash or (
                meta["train_config"].get("num_vehicles", 1) == 1 and archived_package in compatible_packages
            )
            if (state["source_commit"] != source or state["fields"] != field_hash or
                not package_matches or
                any(sha(ck/f)!=h for f,h in state["files"].items())):
                raise ValueError("Checkpoint, simulator or field bindings changed")
            best = state["best_score"]
            if robot_mode is not None and robot_mode != state["robot_mode"]:
                raise ValueError("Robot sampling mode changed")
            robot_mode = state["robot_mode"]
        elif not legacy or meta["curriculum_state"]["max_stage"] > 2:
            raise ValueError("Use an audited synthetic legacy checkpoint or a sealed session checkpoint")
        else:
            registry=json.loads(Path(__file__).with_name("legacy_models.json").read_bytes())
            approved=registry.get(sha(ck/"model.zip"))
            archived_config=dict(meta["train_config"])
            archived_config.pop("dataset_manifest",None)
            contract=[meta["env_version"],archived_config,meta["curriculum_state"],meta["total_timesteps"]]
            identity=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            if not approved or approved["contract_sha256"]!=identity or approved["vecnormalize_sha256"]!=sha(ck/"vecnormalize.pkl"):
                raise ValueError("Legacy policy, training semantics or normalization differ from both audited bundles")
        cfg = PPOTrainConfig.from_dict(meta["train_config"])
        seed_offset = int(meta["total_timesteps"])
    cfg = cfg or PPOTrainConfig()
    robot_mode = robot_mode or "curriculum"
    if robot_mode not in ("curriculum","fixed"):
        raise ValueError("Unknown robot sampling mode")
    if robot_mode=="curriculum" and cfg.enable_curriculum_progression and cfg.curriculum_stage<7<=cfg.max_curriculum_stage:
        raise ValueError("Start a separate stage 7 run for robot randomization; fixed-robot workers persist across stage changes")
    if additional<=0 or min(checkpoint_interval,evaluation_interval,cfg.eval_episodes)<1:
        raise ValueError("Budgets and evaluation episode count must be positive")
    cfg = replace(cfg,device=device,dataset_manifest=str(data) if data else None)
    dataset = AgriculturalFieldDataset(data) if data else None
    if cfg.max_curriculum_stage >= 3 and (dataset is None or not dataset.records):
        raise ValueError("Real-field curriculum requires the checked field attachment")
    curriculum = CurriculumManager(initial_stage=cfg.curriculum_stage,max_stage=cfg.max_curriculum_stage)
    if ck:
        curriculum.load_state_dict(meta["curriculum_state"])
    out.mkdir(parents=True,exist_ok=False)
    stage = curriculum.current_stage
    fns = [make_env(cfg,data,stage,robot_mode) for _ in range(cfg.n_envs)]
    base = SubprocVecEnv(fns,start_method="forkserver") if cfg.use_subproc and len(fns)>1 else DummyVecEnv(fns)
    env = base
    try:
        if cfg.normalize_reward:
            if ck:
                env = VecNormalize.load(str(ck/"vecnormalize.pkl"),base)
                if env.norm_obs or not env.norm_reward or not math.isclose(env.gamma,cfg.effective_gamma()):
                    raise ValueError("Normalization contract mismatch")
                env.training = True
            else:
                env = VecNormalize(base,norm_obs=False,norm_reward=True,clip_reward=10.,gamma=cfg.effective_gamma())
        log = str(out/"tensorboard")
        if ck:
            model = MeasuredPPO.load(ck/"model.zip",env=env,device=device,tensorboard_log=log)
            if model.num_timesteps != seed_offset or model.n_envs != cfg.n_envs or not isinstance(model.learning_rate,(int,float)):
                raise ValueError("Inconsistent legacy counter, worker count or learning rate schedule")
            # One historical stage-2 receipt declared a different LR than the serialized optimizer.
            cfg.learning_rate = float(model.learning_rate)
            if not all(math.isclose(group["lr"],cfg.learning_rate) for group in model.policy.optimizer.param_groups):
                raise ValueError("Optimizer learning rate disagrees with constant schedule")
            if not all(math.isclose(model.clip_range(x),cfg.clip_range) for x in (0.,0.5,1.)):
                raise ValueError("Only the registered constant clip schedule supports chunked learning")
        else:
            model = MeasuredPPO("MultiInputPolicy",env,learning_rate=cfg.learning_rate,n_steps=cfg.n_steps,
                batch_size=cfg.batch_size,n_epochs=cfg.n_epochs,gamma=cfg.effective_gamma(),
                gae_lambda=cfg.gae_lambda,clip_range=cfg.clip_range,ent_coef=cfg.ent_coef,
                vf_coef=cfg.vf_coef,max_grad_norm=cfg.max_grad_norm,tensorboard_log=log,
                seed=cfg.seed,device=device,policy_kwargs=dict(features_extractor_class=AgCoverageFeaturesExtractor,
                    features_extractor_kwargs=dict(local_dim=64,global_dim=64,state_dim=64,sensor_dim=64),
                    net_arch=dict(pi=list(cfg.net_arch_pi),vf=list(cfg.net_arch_vf)),normalize_images=False))
            if warm_start:
                old = PPO.load(Path(warm_start)/"model.zip",env=env,device=device)
                model.policy.load_state_dict(old.policy.state_dict(),strict=True)
                del old
        if ck:
            # Environment episodes and rollout buffers start fresh. Avoid repeating the first seed sequence.
            model.set_random_seed(cfg.seed+seed_offset)
        quantum = model.n_envs*model.n_steps
        step0 = model.num_timesteps
        end = step0+math.ceil(additional/quantum)*quantum
        cfg.total_timesteps = end
        eval_every = math.ceil(evaluation_interval/quantum)*quantum
        save_every = math.ceil(checkpoint_interval/quantum)*quantum
        next_eval, next_save = step0+eval_every, step0+save_every
        dump(out/"config.json",cfg.to_dict())
        model.optimization_seconds = 0.
        learning_s = cpu_learning_s = 0.
        start = time.perf_counter()
        while model.num_timesteps < end:
            stop = min(next_eval,next_save,end)
            wall = time.perf_counter()
            cpu = time.process_time()
            model.learn(total_timesteps=stop-model.num_timesteps,reset_num_timesteps=False,tb_log_name="training")
            learning_s += time.perf_counter()-wall
            cpu_learning_s += time.process_time()-cpu
            t = int(model.num_timesteps)
            grade = {}
            is_best = False
            if t == next_eval or t == end:
                stage = curriculum.current_stage
                grade, traces = evaluate_policy_on_suite(model,val_suite(cfg,dataset,stage,robot_mode),cfg,
                    stage_id=stage,record_trajectories=True)
                grade.update(timesteps=t,stage=stage)
                with (out/"evaluation.jsonl").open("a") as f:
                    f.write(json.dumps(grade,allow_nan=False)+"\n")
                dump(out/"validation_trajectory.json",traces[0])
                score = grade["val_selection_score"]
                is_best = best is None or score > best
                if is_best:
                    best = score
                if cfg.enable_curriculum_progression:
                    advanced,_ = curriculum.evaluate_progression(grade,t)
                    if advanced:
                        env.env_method("set_curriculum_stage",curriculum.current_stage)
                next_eval += eval_every
            if t == next_save or t == end or is_best:
                tag = "step_"+str(t)
                temp = out/(tag+".partial")
                save_checkpoint_bundle(temp,model,env,cfg,curriculum,timesteps=t,val_metrics=grade)
                receipt = {"source_commit":source,"fields":field_hash,"package_sha256":package_hash,"best_score":best,"robot_mode":robot_mode,
                    "steps":t,"completed_updates":model._n_updates,"reward_rms_count":float(env.ret_rms.count)
                    if isinstance(env,VecNormalize) else None,"files":{p.name:sha(p) for p in temp.iterdir()}}
                dump(temp/"session.json",receipt)
                temp.rename(out/tag)
                dump(out/"latest.json",{"checkpoint":tag})
                if is_best:
                    dump(out/"best.json",{"checkpoint":tag,"validation_score":best})
                if t == next_save:
                    next_save += save_every
        wall_s = time.perf_counter()-start
        result = {"initial_steps":step0,"steps":end,"requested_additional":additional,"rollout_quantum":quantum,
                  "actual_additional":end-step0,"optimizer_updates":model._n_updates,"device":str(model.device),
                  "learning_wall_s":learning_s,"learning_parent_cpu_s":cpu_learning_s,
                  "optimizer_wall_s":model.optimization_seconds,"train_eval_save_wall_s":wall_s,
                  "learning_steps_s":(end-step0)/learning_s,"session_steps_s":(end-step0)/wall_s,
                  "peak_parent_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated() if model.device.type=="cuda" else None,
                  "final_validation":grade,"best_score_across_sessions":best}
        dump(out/"summary.json",result)
        return result
    finally:
        env.close()
