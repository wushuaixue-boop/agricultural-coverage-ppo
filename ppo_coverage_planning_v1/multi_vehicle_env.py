"""Centralized control of multiple vehicles covering one agricultural field."""
from dataclasses import replace
from itertools import combinations
import math
from numbers import Integral

import gymnasium as gym
import numpy as np

from .env import PPOCoverageEnv
from .field_world import CollisionMode
from .rewards import compute_step_reward


def _closest_vehicle_distance(paths: np.ndarray) -> float:
    """Minimum separation along synchronized relative-motion segments."""
    closest = math.inf
    for left, right in combinations(paths[:, :, :2], 2):
        relative = left - right
        if len(relative) == 1:
            closest = min(closest, float(np.linalg.norm(relative[0])))
            continue
        starts = relative[:-1]
        directions = np.diff(relative, axis=0)
        squared_lengths = np.sum(directions**2, axis=1)
        fractions = np.clip(
            -np.sum(starts * directions, axis=1) / np.maximum(squared_lengths, 1e-30), 0.0, 1.0
        )
        closest = min(closest, float(np.min(np.linalg.norm(starts + fractions[:, None] * directions, axis=1))))
    return closest


class MultiVehicleCoverageEnv(PPOCoverageEnv):
    """Share one field, one coverage map, and one reward across a fixed fleet.

    Accepts PPOCoverageEnv's keyword arguments. For multiple vehicles, actions
    are flattened in vehicle order, and observations retain the vehicle axis.
    Each episode uses a common robot specification. Pass an (N, 3) array of
    (x, y, heading) as reset's initial_poses to control placement.
    """

    def __init__(self, *, num_vehicles: int = 1, inter_vehicle_clearance_m: float = 0.5, **kwargs):
        if isinstance(num_vehicles, bool) or not isinstance(num_vehicles, Integral) or num_vehicles < 1:
            raise ValueError("num_vehicles must be a positive integer")
        if not math.isfinite(inter_vehicle_clearance_m) or inter_vehicle_clearance_m < 0:
            raise ValueError("inter_vehicle_clearance_m must be finite and nonnegative")
        super().__init__(**kwargs)
        self.num_vehicles = int(num_vehicles)
        self.inter_vehicle_clearance_m = inter_vehicle_clearance_m
        self._vehicles = []
        if self.num_vehicles == 1:
            return
        if self.cfg.collision_mode != CollisionMode.TERMINATE:
            raise ValueError("Multi-vehicle episodes require CollisionMode.TERMINATE")
        if self.render_mode not in (None, "ansi"):
            raise ValueError("Multi-vehicle episodes support render_mode='ansi'")

        self.metadata = {**self.metadata, "render_modes": ["ansi"]}
        self.robot_spec, self.spec = self.spec, None
        self._fleet_trajectory = []
        observer_config = replace(self.cfg, obs=replace(self.cfg.obs, include_global_map=False))
        self._vehicles = [PPOCoverageEnv(env_config=observer_config) for _ in range(self.num_vehicles)]
        self._action_dim = self.cfg.control_interface.action_dim
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.num_vehicles * self._action_dim,), dtype=np.float32)
        spaces = dict(self.observation_space.spaces)
        for key, space in self._vehicles[0].observation_space.spaces.items():
            if key != "progress":
                shape = (self.num_vehicles, *space.shape)
                spaces[key] = gym.spaces.Box(
                    np.broadcast_to(space.low, shape), np.broadcast_to(space.high, shape), dtype=space.dtype
                )
        spaces["vehicle_pose"] = gym.spaces.Box(-1.0, 1.0, (self.num_vehicles, 4), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    @property
    def _required_separation(self) -> float:
        return 2 * self.robot_spec.body_radius_m + self.inter_vehicle_clearance_m

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        poses = opts.pop("initial_poses", None)
        if poses is not None:
            poses = np.asarray(poses, dtype=np.float64)
            if poses.shape != (self.num_vehicles, 3) or not np.isfinite(poses).all():
                raise ValueError(f"initial_poses must have shape ({self.num_vehicles}, 3) and be finite")
            if opts.get("initial_pose") is not None:
                raise ValueError("Choose initial_pose or initial_poses")
            opts["initial_pose"] = poses[0]
        if self.num_vehicles == 1:
            return super().reset(seed=seed, options=opts)

        try:
            _, info = super().reset(seed=seed, options=opts)
        finally:
            # The single-vehicle implementation keeps a RobotSpec in the
            # attribute Gymnasium reserves for its own EnvSpec.
            self.robot_spec, self.spec = self.spec, None
            self._done = True
        if poses is None:
            state = self.sim.state
            sampled = [(state.x, state.y, state.theta)]
            randomize = self.cfg.randomize_start_pose and opts.get("randomize_start_pose", True)
            placement_rng = self._rng if randomize else np.random.default_rng(0)
            for _ in range(1, self.num_vehicles):
                for _ in range(200):
                    pose = self.world.sample_valid_start_pose(
                        self.robot_spec, placement_rng, margin_m=self.cfg.start_clearance_margin_m
                    )
                    if all(math.dist(pose[:2], other[:2]) >= self._required_separation for other in sampled):
                        sampled.append(pose)
                        break
                else:
                    raise ValueError(f"Could not place {self.num_vehicles} vehicles safely in {self.world.field_id}")
            poses = np.asarray(sampled)
        if any(not self.world.is_point_valid_start(x, y, self.robot_spec, margin_m=0) for x, y, _ in poses):
            raise ValueError("Every vehicle must start within the navigable field")
        if _closest_vehicle_distance(poses[:, None]) < self._required_separation:
            raise ValueError("Initial vehicle poses violate inter-vehicle clearance")

        for vehicle, pose in zip(self._vehicles, poses, strict=True):
            vehicle.world, vehicle.spec, vehicle.coverage = self.world, self.robot_spec, self.coverage
            vehicle.sim = type(self.sim)(self.robot_spec, dt_phys=self.cfg.dt_phys_s, interface=self.cfg.control_interface)
            vehicle.sim.reset(*pose, v=self.sim.state.v, kappa=self.sim.state.kappa)
            vehicle._prev_action.fill(0)
            vehicle._last_diag = None
            vehicle._max_sim_time_s = self._max_sim_time_s
            vehicle._l_scale_m = self._l_scale_m
            vehicle._recent_new_areas = self._recent_new_areas
        self.sim = self._vehicles[0].sim
        self._fleet_trajectory = [self._trajectory_frame()]
        self._done = False
        return self._fleet_observation(), {**info, "num_vehicles": self.num_vehicles}

    def _trajectory_frame(self):
        return np.asarray([
            (state.t, state.x, state.y, state.theta, state.v, state.kappa)
            for state in (vehicle.sim.state for vehicle in self._vehicles)
        ])

    def _poses(self):
        return self._trajectory_frame()[:, 1:4]

    def _fleet_observation(self):
        for vehicle in self._vehicles:
            vehicle._time_since_new_cov_s = self._time_since_new_cov_s
        observations = [vehicle._build_observation() for vehicle in self._vehicles]
        obs = {key: np.stack([item[key] for item in observations]) for key in observations[0] if key != "progress"}
        obs["progress"] = observations[0]["progress"].copy()
        distance = sum(vehicle.sim.state.s for vehicle in self._vehicles)
        obs["progress"][3] = min(1.0, distance / (5 * self._l_scale_m))
        poses = self._poses()
        offset = poses[:, :2] - [self.coverage.field_cx, self.coverage.field_cy]
        position = offset / max(1e-6, self.coverage.field_diag_m)
        obs["vehicle_pose"] = np.column_stack((position, np.sin(poses[:, 2]), np.cos(poses[:, 2]))).astype(np.float32)
        if self.cfg.obs.include_global_map:
            maps = [
                self.coverage.extract_global_map(*pose, global_size=self.cfg.obs.global_map_size, frame="world")
                for pose in poses
            ]
            obs["global_map"] = maps[0]
            obs["global_map"][4:] = np.maximum.reduce([image[4:] for image in maps])
        return obs

    def step(self, action, *, physical_command=False, record_trajectory=True):
        if self.num_vehicles == 1:
            return super().step(action, physical_command=physical_command, record_trajectory=record_trajectory)
        if self._done:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode")
        actions = np.asarray(action, dtype=np.float32)
        if actions.shape not in (self.action_space.shape, (self.num_vehicles, self._action_dim)) or not np.isfinite(actions).all():
            raise ValueError("Expected one finite action per vehicle")
        actions = actions.reshape(self.num_vehicles, self._action_dim)
        simulators = [vehicle.sim for vehicle in self._vehicles]
        previous = [sim.state.copy() for sim in simulators]
        dt = self.cfg.dt_action_s
        diagnostics = [
            sim.step(command, dt, normalized_action=not physical_command)[1]
            for sim, command in zip(simulators, actions, strict=True)
        ]
        paths = np.asarray([diag.substep_states for diag in diagnostics])
        spec = self.robot_spec
        wall_reports = [self.world.check_motion_substeps(path, spec) for path in paths]
        separation = _closest_vehicle_distance(paths)
        vehicle_collision = separation < 2 * spec.body_radius_m
        vehicle_clearance_breach = not vehicle_collision and separation < self._required_separation
        body_collision = vehicle_collision or any(report.body_collision for report in wall_reports)
        clearance_breach = vehicle_clearance_breach or any(report.clearance_breach for report in wall_reports)
        unsafe = body_collision or clearance_breach
        if unsafe:
            # Keep the same elapsed time for every vehicle. Check chronological
            # physics prefixes and replay only the shared safe prefix.
            lo, hi = 0, len(paths[0]) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                prefix = paths[:, :mid + 1]
                if (
                    _closest_vehicle_distance(prefix) >= self._required_separation
                    and all(self.world.check_motion_substeps(path, spec).safe for path in prefix)
                ):
                    lo = mid
                else:
                    hi = mid
            dt = lo * diagnostics[0].dt_phys
            for sim, state, command in zip(simulators, previous, actions, strict=True):
                sim.state = state.copy()
                if lo:
                    sim.step(command, dt, normalized_action=not physical_command)
            paths = paths[:, :lo + 1]

        deltas = [
            self.coverage.update_from_substeps(path, spec.working_width_m, tool_active=spec.tool_always_active)
            for path in paths
        ]
        new_area = sum(delta.newly_treated_m2 for delta in deltas)
        step_distance = sum(sim.state.s - start.s for sim, start in zip(simulators, previous, strict=True))
        self._time_since_new_cov_s = 0.0 if new_area >= self.cfg.meaningful_new_area_m2 else self._time_since_new_cov_s + dt
        self._recent_new_areas.append((new_area, dt))
        self._step_count += 1
        snapshot = self.coverage.snapshot()
        success = not unsafe and snapshot.coverage_pct >= self._stage_target_pct
        terminated = unsafe or success
        truncated = not terminated and (self._step_count >= self._max_steps or self.sim.state.t >= self._max_sim_time_s - 1e-9)
        self._done = terminated or truncated
        margin = min(
            separation - self._required_separation,
            *(r.min_exterior_dist_m - spec.exterior_center_clearance_m for r in wall_reports),
            *(r.min_obstacle_dist_m - spec.obstacle_center_clearance_m for r in wall_reports),
        )
        reward, terms = compute_step_reward(
            self.cfg.reward, field_area_m2=self.world.source_area_m2, l_scale_m=self._l_scale_m, t_scale_s=self._t_scale_s,
            newly_treated_m2=new_area, repeated_treated_m2=sum(delta.repeated_treated_m2 for delta in deltas),
            distance_traveled_m=step_distance, dt_elapsed_s=dt, body_collision=body_collision,
            clearance_breach=clearance_breach, safety_filter_intervened=False,
            new_obstacle_intrusion_m2=sum(delta.new_obstacle_intrusion_m2 for delta in deltas),
            clearance_margin_above_req_m=margin, time_since_new_coverage_s=self._time_since_new_cov_s,
            coverage_completed_safely=success,
        )
        for vehicle, command, diag in zip(self._vehicles, actions, diagnostics, strict=True):
            vehicle._last_diag = diag
            vehicle._prev_action[:] = (1.0, np.clip(command[0], -1, 1)) if self._action_dim == 1 else np.clip(command, -1, 1)
        if record_trajectory:
            self._fleet_trajectory.append(self._trajectory_frame())

        reason = next((report.collision_type for report in wall_reports if not report.safe), "running")
        if vehicle_collision:
            reason = "vehicle_collision"
        elif vehicle_clearance_breach:
            reason = "vehicle_clearance_breach"
        elif success:
            reason = "coverage_target_reached"
        elif truncated:
            reason = "time_budget_exhausted"
        info = {
            "num_vehicles": self.num_vehicles, "field_id": self.world.field_id, "coverage_pct": snapshot.coverage_pct,
            "newly_treated_m2": new_area, "repeated_treatment_m2": snapshot.repeated_treatment_m2,
            "distance_m": sum(sim.state.s for sim in simulators), "operating_time_s": self.sim.state.t,
            "body_collision": body_collision, "clearance_breach": clearance_breach,
            "vehicle_collision": vehicle_collision, "vehicle_clearance_breach": vehicle_clearance_breach,
            "termination_reason": reason, "reward_terms": terms,
        }
        return self._fleet_observation(), reward, terminated, truncated, info

    def get_trajectory_xy(self):
        """Return (steps + 1, vehicles, 2) for a fleet."""
        if self.num_vehicles == 1:
            return super().get_trajectory_xy()
        return self.get_trajectory_poses()[:, :, 1:3]

    def get_trajectory_poses(self):
        """Return (steps + 1, vehicles, 6) with t, x, y, heading, speed, curvature."""
        if self.num_vehicles == 1:
            return super().get_trajectory_poses()
        return np.asarray(self._fleet_trajectory).reshape(-1, self.num_vehicles, 6)

    def render(self):
        if self.num_vehicles == 1:
            return super().render()
        if self.render_mode == "ansi" and self.sim is not None:
            return f"t={self.sim.state.t:.1f}s vehicles={self._poses().tolist()} cov={self.coverage.snapshot().coverage_pct:.1f}%"
        return None
