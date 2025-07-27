# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from typing import Dict

import torch
from jaxtyping import Bool

_log = logging.getLogger(__name__)


class ConstraintChecker:
    """Checks the constraints in a cost dict against the desired tolerances."""

    def __init__(self, constraint_config: Dict[str, Dict[str, float]], default_tol: float = 0.0):
        self.constraint_config = constraint_config
        self.default_tol = default_tol
        # Map from (constraint_type, name) to tolerance for quicker indexing
        self.constraint_to_tol = {}
        for constraint_type, constraint_info in constraint_config.items():
            for name, tol in constraint_info.items():
                self.constraint_to_tol[(constraint_type, name)] = tol
        self._warned_default_tol = set()

    def _get_tol(self, constraint_type: str, name: str) -> float:
        key = (constraint_type, name)
        if key in self.constraint_to_tol:
            return self.constraint_to_tol[key]

        default_key = (constraint_type, "default")
        if default_key in self.constraint_to_tol:
            return self.constraint_to_tol[default_key]

        if key not in self._warned_default_tol:
            _log.warning(f"No tolerance found for {constraint_type} {name}, using default {self.default_tol}")
            self._warned_default_tol.add(key)
        return self.default_tol

    def get_full_mask(self, cost_dict: Dict[str, dict]) -> Dict[str, Dict[str, Bool[torch.Tensor, "num_particles"]]]:
        """Get the "full" mask that maps the constraints to the satisfying masks."""
        mask_dict = {}
        for cost_type, cost_info in cost_dict.items():
            # Ignore non-constraints
            if cost_info["type"] != "constraint":
                continue

            mask_dict[cost_type] = {
                name: values <= self._get_tol(cost_type, name) for name, values in cost_info["values"].items()
            }
        return mask_dict

    def get_mask(self, cost_dict: Dict[str, dict], verbose: bool = True) -> Bool[torch.Tensor, "num_particles"]:
        """Get the satisfying mask for the constraints given the costs."""
        overall_mask = None
        for cost_type, cost_info in cost_dict.items():
            # Ignore non-constraints
            if cost_info["type"] != "constraint":
                continue

            for name, values in cost_info["values"].items():
                tol = self._get_tol(cost_type, name)
                mask = values <= tol
                if mask.ndim == 2:
                    mask = mask.all(dim=1)  # satisfy over time dimension
                overall_mask = mask if overall_mask is None else overall_mask & mask

                if verbose:
                    _log.debug(
                        f"[{cost_type}] {name} <= {tol} has {mask.sum()}/{mask.shape[0]} satisfying, "
                        f"{overall_mask.sum()} remaining"
                    )

        assert overall_mask is not None, "No constraints found in cost dict"
        return overall_mask
