from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F


class CostConstraintObjective:
    """Config-driven augmented objective for cost-guided GR00T inference."""

    def __init__(
        self,
        objective_config: Optional[dict[str, Any]] = None,
        embodiments: Optional[dict[Any, dict[str, Any]]] = None,
    ):
        self.objective_config = objective_config or {}
        self.embodiments = {
            int(embodiment_id): spec for embodiment_id, spec in (embodiments or {}).items()
        }
        self.weights = self.objective_config.get("weights", {})
        self.placeholder_terms = self.objective_config.get("placeholder_terms", {})

        self.cost_weight = float(self.weights.get("cost", 1.0))
        self.equality_weight = float(self.weights.get("equality", 1.0))
        self.inequality_weight = float(self.weights.get("inequality", 1.0))
        self.softplus_beta = float(self.placeholder_terms.get("softplus_beta", 10.0))

    def validate_embodiment_ids(self, embodiment_id: torch.Tensor | int | list[int] | tuple[int, ...]) -> None:
        embodiment_ids = self._normalize_embodiment_ids(embodiment_id, batch_size=None)
        for current_id in embodiment_ids:
            self._get_embodiment_spec(current_id)

    def _normalize_embodiment_ids(
        self,
        embodiment_id: torch.Tensor | int | list[int] | tuple[int, ...],
        batch_size: Optional[int],
    ) -> list[int]:
        if isinstance(embodiment_id, torch.Tensor):
            ids = [int(v) for v in embodiment_id.detach().view(-1).cpu().tolist()]
        elif isinstance(embodiment_id, (list, tuple)):
            ids = [int(v) for v in embodiment_id]
        else:
            ids = [int(embodiment_id)]

        if batch_size is None:
            return ids

        if len(ids) == 1 and batch_size > 1:
            ids = ids * batch_size
        if len(ids) != batch_size:
            raise ValueError(
                f"Expected {batch_size} embodiment ids, but received {len(ids)}."
            )
        return ids

    def _get_embodiment_spec(self, embodiment_id: int) -> dict[str, Any]:
        spec = self.embodiments.get(int(embodiment_id))
        if spec is None:
            raise NotImplementedError(
                f"Cost-guided inference is not configured for embodiment_id={embodiment_id}."
            )
        if spec.get("control_kind", "non_eef") != "eef":
            raise NotImplementedError(
                "Cost-guided inference is only implemented for end-effector control. "
                f"embodiment_id={embodiment_id} is configured as {spec.get('control_kind', 'non_eef')}."
            )
        return spec

    def _extract_indices(
        self,
        values: Optional[torch.Tensor],
        indices: Optional[list[int]],
    ) -> Optional[torch.Tensor]:
        if values is None or not indices:
            return None
        return values[..., indices]

    def _as_like(
        self,
        value: float | list[float] | tuple[float, ...],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)

    def _smooth_max(self, values: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self.softplus_beta * values, dim=-1) / self.softplus_beta

    def _batch_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3:
            return actions
        if actions.ndim == 2:
            return actions.unsqueeze(0)
        raise ValueError(
            f"Expected actions to have shape [B, T, D] or [T, D], received {tuple(actions.shape)}."
        )

    def _batch_states(
        self,
        states: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if states is None:
            return None

        if states.ndim == 3:
            batch_states = states
        elif states.ndim == 2:
            if states.shape[0] == batch_size:
                batch_states = states
            else:
                batch_states = states.unsqueeze(0)
        elif states.ndim == 1:
            batch_states = states.unsqueeze(0)
        else:
            raise ValueError(
                f"Expected states to have shape [B, T_state, D], [B, D], or [D], received {tuple(states.shape)}."
            )

        if batch_states.shape[0] == 1 and batch_size > 1:
            expand_shape = (batch_size,) + tuple(batch_states.shape[1:])
            batch_states = batch_states.expand(expand_shape)

        if batch_states.shape[0] != batch_size:
            raise ValueError(
                f"State batch size {batch_states.shape[0]} does not match action batch size {batch_size}."
            )
        return batch_states

    def _decode_value(
        self,
        value: Optional[torch.Tensor],
        decoder_cfg: Optional[dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        cfg = decoder_cfg or {}
        kind = cfg.get("kind", "identity")

        if kind == "identity":
            return value
        if kind == "affine":
            scale = self._as_like(cfg.get("scale", 1.0), value)
            shift = self._as_like(cfg.get("shift", 0.0), value)
            return value * scale + shift
        if kind == "minmax":
            min_val = self._as_like(cfg["min"], value)
            max_val = self._as_like(cfg["max"], value)
            return 0.5 * (value + 1.0) * (max_val - min_val) + min_val
        if kind == "meanstd":
            mean = self._as_like(cfg["mean"], value)
            std = self._as_like(cfg["std"], value)
            return value * std + mean

        raise ValueError(f"Unsupported decoder kind: {kind}")

    def _extract_view(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        embodiment_id: int,
    ) -> dict[str, Optional[torch.Tensor]]:
        spec = self._get_embodiment_spec(embodiment_id)
        action_layout = spec.get("action_layout", {})
        state_layout = spec.get("state_layout", {})

        action_position = self._extract_indices(actions, action_layout.get("position_idx"))
        action_rotation = self._extract_indices(actions, action_layout.get("rotation_idx"))
        action_gripper = self._extract_indices(actions, action_layout.get("gripper_idx"))

        state_position = self._extract_indices(states, state_layout.get("position_idx"))
        state_rotation = self._extract_indices(states, state_layout.get("rotation_idx"))
        state_gripper = self._extract_indices(states, state_layout.get("gripper_idx"))

        if spec.get("objective_space", "normalized") == "decoded_cartesian":
            decoder = spec.get("decoder", {})
            action_decoder = decoder.get("action", {})
            state_decoder = decoder.get("state", {})

            action_position = self._decode_value(action_position, action_decoder.get("position"))
            action_rotation = self._decode_value(action_rotation, action_decoder.get("rotation"))
            action_gripper = self._decode_value(action_gripper, action_decoder.get("gripper"))

            state_position = self._decode_value(state_position, state_decoder.get("position"))
            state_rotation = self._decode_value(state_rotation, state_decoder.get("rotation"))
            state_gripper = self._decode_value(state_gripper, state_decoder.get("gripper"))

        return {
            "action_position": action_position,
            "action_rotation": action_rotation,
            "action_gripper": action_gripper,
            "state_position": state_position,
            "state_rotation": state_rotation,
            "state_gripper": state_gripper,
        }

    def _cost_single(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        embodiment_id: int,
    ) -> torch.Tensor:
        view = self._extract_view(actions, states, embodiment_id)
        total = actions.new_zeros(())

        position = view["action_position"]
        rotation = view["action_rotation"]
        gripper = view["action_gripper"]

        if position is not None and position.shape[0] > 1:
            position_diff = position[1:] - position[:-1]
            total = total + float(
                self.placeholder_terms.get("position_smoothness_weight", 1.0)
            ) * position_diff.square().sum()
        if rotation is not None and rotation.shape[0] > 1:
            rotation_diff = rotation[1:] - rotation[:-1]
            total = total + float(
                self.placeholder_terms.get("rotation_smoothness_weight", 0.1)
            ) * rotation_diff.square().sum()
        if gripper is not None and gripper.shape[0] > 1:
            gripper_diff = gripper[1:] - gripper[:-1]
            total = total + float(
                self.placeholder_terms.get("gripper_smoothness_weight", 0.05)
            ) * gripper_diff.square().sum()

        return total

    def _equality_penalty_single(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        embodiment_id: int,
    ) -> torch.Tensor:
        view = self._extract_view(actions, states, embodiment_id)
        position = view["action_position"]
        if position is None:
            return actions.new_zeros(())

        target_position = self.placeholder_terms.get("target_position")
        if target_position is None:
            return actions.new_zeros(())

        target = self._as_like(target_position, position)
        return float(self.placeholder_terms.get("terminal_position_weight", 1.0)) * (
            position[-1] - target
        ).square().sum()

    def _cylinder_avoidance_penalty(self, position: torch.Tensor) -> torch.Tensor:
        # Task-specific obstacle geometry from the LIBERO cylinder scene.
        cylinder_center_xy = self._as_like([-0.01145, 0.02955], position)
        cylinder_center_z = self._as_like(0.20, position)

        cylinder_radius = 0.02
        cylinder_half_height = 0.20
        epsilon = 0.05

        delta_xy = position[..., :2] - cylinder_center_xy
        radial_distance = torch.sqrt(delta_xy.square().sum(dim=-1) + 1.0e-12)

        radial_clearance = radial_distance - (cylinder_radius + epsilon)
        vertical_clearance = torch.abs(position[..., 2] - cylinder_center_z) - (
            cylinder_half_height + epsilon
        )

        clearance = self._smooth_max(
            torch.stack([radial_clearance, vertical_clearance], dim=-1)
        )
        violation = F.softplus(-self.softplus_beta * clearance) / self.softplus_beta
        return violation.square().sum()

    def _nonnegative_y_penalty(self, position: torch.Tensor) -> torch.Tensor:
        y_position = position[..., 1]
        violation = F.softplus(self.softplus_beta * y_position) / self.softplus_beta
        return violation.square().sum()

    def _nonnegative_x_penalty(self, position: torch.Tensor) -> torch.Tensor:
        x_position = position[..., 0]
        violation = F.softplus(self.softplus_beta * x_position) / self.softplus_beta
        return violation.square().sum()

    def _inequality_penalty_single(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        embodiment_id: int,
    ) -> torch.Tensor:
        view = self._extract_view(actions, states, embodiment_id)
        position = view["action_position"]
        if position is None or position.shape[-1] < 2:
            return actions.new_zeros(())

        return self._cylinder_avoidance_penalty(position)
        #return self._nonnegative_x_penalty(position) + self._nonnegative_y_penalty(position)

    def cost(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        embodiment_id: torch.Tensor | int | list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        batch_actions = self._batch_actions(actions)
        batch_states = self._batch_states(states, batch_actions.shape[0])

        embodiment_ids = self._normalize_embodiment_ids(embodiment_id, batch_actions.shape[0])
        outputs = []
        for batch_index, current_id in enumerate(embodiment_ids):
            state_item = None if batch_states is None else batch_states[batch_index]
            outputs.append(self._cost_single(batch_actions[batch_index], state_item, current_id))
        return torch.stack(outputs, dim=0)

    def equality_penalty(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        embodiment_id: torch.Tensor | int | list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        batch_actions = self._batch_actions(actions)
        batch_states = self._batch_states(states, batch_actions.shape[0])

        embodiment_ids = self._normalize_embodiment_ids(embodiment_id, batch_actions.shape[0])
        outputs = []
        for batch_index, current_id in enumerate(embodiment_ids):
            state_item = None if batch_states is None else batch_states[batch_index]
            outputs.append(
                self._equality_penalty_single(batch_actions[batch_index], state_item, current_id)
            )
        return torch.stack(outputs, dim=0)

    def inequality_penalty(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        embodiment_id: torch.Tensor | int | list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        batch_actions = self._batch_actions(actions)
        batch_states = self._batch_states(states, batch_actions.shape[0])

        embodiment_ids = self._normalize_embodiment_ids(embodiment_id, batch_actions.shape[0])
        outputs = []
        for batch_index, current_id in enumerate(embodiment_ids):
            state_item = None if batch_states is None else batch_states[batch_index]
            outputs.append(
                self._inequality_penalty_single(batch_actions[batch_index], state_item, current_id)
            )
        return torch.stack(outputs, dim=0)

    def J(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        embodiment_id: torch.Tensor | int | list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        return (
            self.cost_weight * self.cost(actions, states, embodiment_id)
            + self.equality_weight * self.equality_penalty(actions, states, embodiment_id)
            + self.inequality_weight * self.inequality_penalty(actions, states, embodiment_id)
        )

    def J_and_grad(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        embodiment_id: torch.Tensor | int | list[int] | tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode(False), torch.enable_grad():
            actions_req = actions.detach().to(torch.float32).clone().requires_grad_(True)
            states_req = None
            if states is not None:
                states_req = states.detach().to(torch.float32).clone()
            objective_value = self.J(actions_req, states_req, embodiment_id)
            if objective_value.requires_grad:
                grad_actions = torch.autograd.grad(
                    objective_value.sum(), actions_req, create_graph=False
                )[0]
            else:
                grad_actions = torch.zeros_like(actions_req)

        return objective_value.detach().to(actions.dtype), grad_actions.detach().to(actions.dtype)
