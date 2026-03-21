from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from transformers.feature_extraction_utils import BatchFeature
import torch
import yaml

from .cost_constraint_objective import CostConstraintObjective


INFERENCE_CONFIG_PATH = Path(__file__).with_name("cost_guided_inference.yaml")


class InferenceConfigLoader:
    """Hot-reloading YAML config loader for inference-only settings."""

    def __init__(self, path: Path = INFERENCE_CONFIG_PATH):
        self.path = path
        self._cached_mtime_ns: Optional[int] = None
        self._cached_config: Optional[dict[str, Any]] = None

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Missing cost-guided inference config: {self.path}")

        stat = self.path.stat()
        if self._cached_config is None or self._cached_mtime_ns != stat.st_mtime_ns:
            with self.path.open("r", encoding="utf-8") as file_obj:
                config = yaml.safe_load(file_obj) or {}
            if not isinstance(config, dict):
                raise ValueError(f"Expected YAML mapping in {self.path}, found {type(config)}")
            config["embodiments"] = {
                int(embodiment_id): spec
                for embodiment_id, spec in config.get("embodiments", {}).items()
            }
            self._cached_config = config
            self._cached_mtime_ns = stat.st_mtime_ns
        return self._cached_config


class Gr00tN1d6InferenceEngine:
    """Config-driven inference engine for vanilla, linear-combo, and FKC sampling."""

    def __init__(self, action_head, config_loader: Optional[InferenceConfigLoader] = None):
        self.action_head = action_head
        self.config_loader = config_loader or InferenceConfigLoader()

    def sample(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        raw_state: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        config = self.config_loader.load()
        mode = config.get("mode", "vanilla")

        if mode == "vanilla":
            actions = self._vanilla_sample(
                backbone_features=backbone_features,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
            )
        elif mode == "linear_combo":
            actions = self._linear_combo_sample(
                config=config,
                backbone_features=backbone_features,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
                raw_state=raw_state,
            )
        elif mode == "fkc":
            actions = self._fkc_sample(
                config=config,
                backbone_features=backbone_features,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
                raw_state=raw_state,
            )
        else:
            raise ValueError(f"Unsupported inference mode: {mode}")

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": backbone_features,
                "state_features": state_features,
            }
        )

    def _build_objective(self, config: dict[str, Any]) -> CostConstraintObjective:
        return CostConstraintObjective(
            objective_config=config.get("objective", {}),
            embodiments=config.get("embodiments", {}),
        )

    def _num_steps(self, config: dict[str, Any], mode: str) -> int:
        if mode == "vanilla":
            return int(self.action_head.num_inference_timesteps)
        return int(config.get("time_grid", {}).get("num_steps", self.action_head.num_inference_timesteps))

    def _broadcast_scalar(self, value: float, ref: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float(value), dtype=ref.dtype, device=ref.device)

    def _flow_source_tensor(
        self,
        value: float | list[float] | tuple[float, ...],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
        while tensor.dim() < ref.dim():
            tensor = tensor.unsqueeze(0)
        return tensor

    def _score_from_velocity(
        self,
        actions: torch.Tensor,
        velocity: torch.Tensor,
        t_cont: float,
        config: dict[str, Any],
    ) -> torch.Tensor:
        score_cfg = config.get("score", {})
        eps_score = float(score_cfg.get("eps", 1.0e-6))
        source_cfg = config.get("flow_source", {})

        source_mean = self._flow_source_tensor(source_cfg.get("mean", 0.0), actions)
        source_variance = self._flow_source_tensor(source_cfg.get("variance", 1.0), actions)
        denom = max(1.0 - t_cont, eps_score)
        return (t_cont * velocity + source_mean - actions) / (denom * source_variance)

    def _sigma_schedule(
        self,
        t_cont: float,
        config: dict[str, Any],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        sigma_cfg = config.get("sigma_schedule", {})
        kind = sigma_cfg.get("kind", "zero_ends")
        alpha = float(sigma_cfg.get("alpha", 0.25))

        if kind == "zero_ends":
            value = alpha * math.sqrt(max(t_cont * (1.0 - t_cont), 0.0))
        elif kind == "non_singular":
            # GR00T uses t=0 for source noise and t=1 for clean actions, so
            # the paper's endpoint-aware schedule must be mirrored here.
            value = alpha * math.sqrt(max(1.0 - t_cont, 0.0))
        elif kind == "constant":
            value = alpha
        else:
            raise ValueError(f"Unsupported sigma schedule kind: {kind}")

        return self._broadcast_scalar(value, ref)

    def _beta_schedule(self, t_cont: float, config: dict[str, Any]) -> float:
        beta_cfg = config.get("beta_schedule", {})
        kind = beta_cfg.get("kind", "constant")
        guidance_strength = float(beta_cfg.get("guidance_strength", 1.0))

        if kind == "constant":
            gamma_t = guidance_strength
        elif kind == "linear_ramp":
            gamma_t = guidance_strength * t_cont
        else:
            raise ValueError(f"Unsupported beta schedule kind: {kind}")

        return -gamma_t

    def _repeat_batch(self, value: Any, num_particles: int):
        if value is None:
            return None
        if torch.is_tensor(value):
            expanded = value.unsqueeze(1).expand(value.shape[0], num_particles, *value.shape[1:])
            return expanded.reshape(value.shape[0] * num_particles, *value.shape[1:])
        if isinstance(value, BatchFeature):
            return BatchFeature(
                data={key: self._repeat_batch(inner_value, num_particles) for key, inner_value in value.items()}
            )
        if isinstance(value, dict):
            return {key: self._repeat_batch(inner_value, num_particles) for key, inner_value in value.items()}
        return value

    def _gather_particle_actions(
        self,
        actions: torch.Tensor,
        particle_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_particles = particle_indices.shape
        traj = actions.reshape(batch_size, num_particles, *actions.shape[1:])
        gather_index = particle_indices
        while gather_index.dim() < traj.dim():
            gather_index = gather_index.unsqueeze(-1)
        gather_index = gather_index.expand(batch_size, num_particles, *traj.shape[2:])
        gathered = torch.gather(traj, 1, gather_index)
        return gathered.reshape(batch_size * num_particles, *actions.shape[1:])

    def _systematic_resample(self, log_weights: torch.Tensor) -> torch.Tensor:
        batch_size, num_particles = log_weights.shape
        stable_logw = log_weights - log_weights.max(dim=1, keepdim=True).values
        weights = torch.softmax(stable_logw, dim=1)

        invalid = (~torch.isfinite(weights)).any(dim=1) | (weights.sum(dim=1) <= 0)
        if invalid.any():
            weights = weights.clone()
            weights[invalid] = 1.0 / num_particles

        cdf = torch.cumsum(weights, dim=1)
        cdf[:, -1] = 1.0
        base = torch.rand(batch_size, 1, device=log_weights.device, dtype=log_weights.dtype) / num_particles
        steps = torch.arange(num_particles, device=log_weights.device, dtype=log_weights.dtype)
        thresholds = base + steps.unsqueeze(0) / num_particles
        thresholds = thresholds.clamp(min=0.0, max=1.0 - 1.0e-7)
        return torch.searchsorted(cdf, thresholds).clamp(max=num_particles - 1).long()

    def _sample_particle_index(self, log_weights: torch.Tensor) -> torch.Tensor:
        stable_logw = log_weights - log_weights.max(dim=1, keepdim=True).values
        weights = torch.softmax(stable_logw, dim=1)
        invalid = (~torch.isfinite(weights)).any(dim=1) | (weights.sum(dim=1) <= 0)
        if invalid.any():
            weights = weights.clone()
            weights[invalid] = 1.0 / log_weights.shape[1]
        return torch.multinomial(weights, num_samples=1).squeeze(1)

    def _vanilla_sample(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        batch_size = backbone_features.shape[0]
        actions = torch.randn(
            size=(batch_size, self.action_head.config.action_horizon, self.action_head.action_dim),
            dtype=backbone_features.dtype,
            device=backbone_features.device,
        )
        dt = 1.0 / self.action_head.num_inference_timesteps

        for step_idx in range(self.action_head.num_inference_timesteps):
            t_cont = step_idx / float(self.action_head.num_inference_timesteps)
            velocity = self.action_head._predict_velocity(
                actions=actions,
                t_cont=t_cont,
                backbone_features=backbone_features,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
            )
            actions = actions + dt * velocity

        return actions

    def _linear_combo_sample(
        self,
        config: dict[str, Any],
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        raw_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        objective = self._build_objective(config)
        objective.validate_embodiment_ids(embodiment_id)

        num_steps = self._num_steps(config, mode="linear_combo")
        dt = 1.0 / max(1, num_steps)
        actions = torch.randn(
            size=(backbone_features.shape[0], self.action_head.config.action_horizon, self.action_head.action_dim),
            dtype=backbone_features.dtype,
            device=backbone_features.device,
        )

        for step_idx in range(num_steps):
            t_cont = step_idx / float(num_steps)
            velocity = self.action_head._predict_velocity(
                actions=actions,
                t_cont=t_cont,
                backbone_features=backbone_features,
                state_features=state_features,
                embodiment_id=embodiment_id,
                backbone_output=backbone_output,
            )
            sigma_t = self._sigma_schedule(t_cont, config, actions)
            score = self._score_from_velocity(actions, velocity, t_cont, config)
            beta_t = self._beta_schedule(t_cont, config)

            _, grad_actions = objective.J_and_grad(actions, raw_state, embodiment_id)
            drift = velocity + 0.5 * sigma_t.square() * (score + beta_t * grad_actions)
            noise = sigma_t * math.sqrt(dt) * torch.randn_like(actions)
            actions = actions + dt * drift + noise

        return actions

    def _fkc_sample(
        self,
        config: dict[str, Any],
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        raw_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        objective = self._build_objective(config)
        objective.validate_embodiment_ids(embodiment_id)

        fkc_cfg = config.get("fkc", {})
        num_particles = max(1, int(fkc_cfg.get("num_particles", 8)))
        resample_interval = max(1, int(fkc_cfg.get("resample_interval", 1)))
        resample_t_min = float(fkc_cfg.get("resample_t_min", 0.05))
        resample_t_max = float(fkc_cfg.get("resample_t_max", 0.95))
        center_increment = bool(fkc_cfg.get("center_log_weight_increment", True))

        num_steps = self._num_steps(config, mode="fkc")
        dt = 1.0 / max(1, num_steps)

        batch_size = backbone_features.shape[0]
        actions = torch.randn(
            size=(batch_size * num_particles, self.action_head.config.action_horizon, self.action_head.action_dim),
            dtype=backbone_features.dtype,
            device=backbone_features.device,
        )
        log_weights = torch.zeros(batch_size, num_particles, dtype=torch.float32, device=actions.device)

        backbone_features_exp = self._repeat_batch(backbone_features, num_particles)
        state_features_exp = self._repeat_batch(state_features, num_particles)
        embodiment_id_exp = self._repeat_batch(embodiment_id, num_particles)
        backbone_output_exp = self._repeat_batch(backbone_output, num_particles)
        raw_state_exp = self._repeat_batch(raw_state, num_particles)

        for step_idx in range(num_steps):
            t_cont = step_idx / float(num_steps)
            next_t = min((step_idx + 1) / float(num_steps), 1.0)

            velocity = self.action_head._predict_velocity(
                actions=actions,
                t_cont=t_cont,
                backbone_features=backbone_features_exp,
                state_features=state_features_exp,
                embodiment_id=embodiment_id_exp,
                backbone_output=backbone_output_exp,
            )
            sigma_t = self._sigma_schedule(t_cont, config, actions)
            score = self._score_from_velocity(actions, velocity, t_cont, config)
            beta_t = self._beta_schedule(t_cont, config)
            beta_next = self._beta_schedule(next_t, config)

            objective_value, grad_actions = objective.J_and_grad(actions, raw_state_exp, embodiment_id_exp)
            drift = velocity + 0.5 * sigma_t.square() * (score + beta_t * grad_actions)
            noise = sigma_t * math.sqrt(dt) * torch.randn_like(actions)
            actions = actions + dt * drift + noise

            if resample_t_min <= t_cont <= resample_t_max:
                velocity_inner = (grad_actions * velocity).reshape(batch_size * num_particles, -1).sum(dim=-1)
                dlogw = beta_t * velocity_inner * dt + (beta_next - beta_t) * objective_value.to(
                    velocity_inner.dtype
                )
                dlogw = dlogw.reshape(batch_size, num_particles)
                if center_increment:
                    dlogw = dlogw - dlogw.mean(dim=1, keepdim=True)
                log_weights = log_weights + dlogw.to(log_weights.dtype)

                should_resample = (step_idx + 1) % resample_interval == 0 and (step_idx + 1) < num_steps
                if should_resample:
                    particle_indices = self._systematic_resample(log_weights)
                    actions = self._gather_particle_actions(actions, particle_indices)
                    log_weights.zero_()

        picked_indices = self._sample_particle_index(log_weights)
        actions = actions.reshape(batch_size, num_particles, *actions.shape[1:])
        return actions[torch.arange(batch_size, device=actions.device), picked_indices]
