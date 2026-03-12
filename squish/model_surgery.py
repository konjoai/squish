"""ModelSurgery — In-place transformer architecture patching.

Records a plan for which layers to remove and which attention heads to prune,
then applies that plan to a weight dictionary (``{layer_name: np.ndarray}``).
Layer removal drops every weight tensor whose key contains ``"layer_{i}"`` for
each ``i`` in ``layers_to_remove``.  Head pruning tracks the number of heads
removed per layer and is reflected in the ``SurgeryResult``.

Practical use-cases include depth pruning (removing entire transformer blocks)
and width pruning (removing individual attention heads) to produce smaller
models without retraining.

Usage::

    import numpy as np
    from squish.model_surgery import SurgeryPlan, SurgeryResult, ModelSurgeon

    surgeon = ModelSurgeon()

    plan = surgeon.plan(n_layers=32, n_heads=32, head_dim=128)
    plan.layers_to_remove = [30, 31]          # drop last two layers
    plan.heads_to_prune   = {0: [0, 1], 1: [0]}  # prune heads in layer 0 and 1
    plan.validate()

    weights = {
        "layer_0.mlp.weight": np.ones((128, 512), dtype=np.float32),
        "layer_30.mlp.weight": np.ones((128, 512), dtype=np.float32),
        "layer_31.attn.weight": np.ones((128, 512), dtype=np.float32),
    }

    result = surgeon.apply(plan, weights)
    print(result.layers_removed, result.heads_pruned)
    print(surgeon.stats)
"""

from __future__ import annotations

__all__ = ["SurgeryPlan", "SurgeryResult", "ModelSurgeon", "SurgeryStats"]

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class SurgeryPlan:
    """Specification of which layers and heads to remove from a model.

    Attributes:
        layers_to_remove: Indices of transformer layers to drop entirely.
        heads_to_prune: Mapping ``{layer_idx: [head_indices]}`` of attention
            heads to prune per layer.
        n_layers: Total number of layers in the original model.
        n_heads:  Total number of attention heads per layer.
        head_dim: Dimension of each attention head (d_head).
    """

    layers_to_remove: List[int] = field(default_factory=list)
    heads_to_prune:   Dict[int, List[int]] = field(default_factory=dict)
    n_layers: int = 32
    n_heads:  int = 32
    head_dim: int = 128

    def validate(self) -> None:
        """Validate that all layer and head indices are within bounds.

        Raises:
            ValueError: If any layer index >= ``n_layers`` or any head index
                >= ``n_heads``.
        """
        for idx in self.layers_to_remove:
            if idx >= self.n_layers:
                raise ValueError(
                    f"Layer index {idx} is out of range for a model with "
                    f"{self.n_layers} layers."
                )
        for layer_idx, head_indices in self.heads_to_prune.items():
            if layer_idx >= self.n_layers:
                raise ValueError(
                    f"Head-prune layer index {layer_idx} is out of range for "
                    f"a model with {self.n_layers} layers."
                )
            for h in head_indices:
                if h >= self.n_heads:
                    raise ValueError(
                        f"Head index {h} in layer {layer_idx} is out of range "
                        f"for a model with {self.n_heads} heads."
                    )


@dataclass
class SurgeryResult:
    """Summary of a completed :meth:`ModelSurgeon.apply` operation.

    Attributes:
        layers_removed: Number of weight keys actually deleted.
        heads_pruned: Total number of head slots pruned across all layers.
        estimated_param_reduction: Estimated fraction of parameters removed
            (0.0 – 1.0).
    """

    layers_removed: int = 0
    heads_pruned:   int = 0
    estimated_param_reduction: float = 0.0


@dataclass
class SurgeryStats:
    """Running statistics for a :class:`ModelSurgeon` instance.

    Attributes:
        total_plan_calls:  Number of :meth:`ModelSurgeon.plan` invocations.
        total_apply_calls: Number of :meth:`ModelSurgeon.apply` invocations.
    """

    total_plan_calls:  int = 0
    total_apply_calls: int = 0


class ModelSurgeon:
    """Creates surgery plans and applies them to transformer weight dicts.

    All mutating operations work on a *copy* of the supplied weight dictionary;
    the original mapping is never modified.

    Example::

        surgeon = ModelSurgeon()
        plan    = surgeon.plan(32, 32, 128)
        plan.layers_to_remove = [0]
        result  = surgeon.apply(plan, weights)
    """

    def __init__(self) -> None:
        self._stats = SurgeryStats()

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    def plan(
        self,
        n_layers: int,
        n_heads:  int,
        head_dim: int,
    ) -> SurgeryPlan:
        """Create an empty :class:`SurgeryPlan` with the given model dimensions.

        Args:
            n_layers: Total number of transformer layers.
            n_heads:  Number of attention heads per layer.
            head_dim: Dimension per attention head.

        Returns:
            A :class:`SurgeryPlan` with empty ``layers_to_remove`` and
            ``heads_to_prune``.
        """
        self._stats.total_plan_calls += 1
        return SurgeryPlan(
            layers_to_remove=[],
            heads_to_prune={},
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
        )

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_reduction(plan: SurgeryPlan) -> float:
        """Estimate the fraction of parameters removed by *plan*.

        The estimate uses equal weighting between layer removal and head
        pruning::

            layer_fraction = len(layers_to_remove) / n_layers
            head_fraction  = total_heads_pruned / (n_layers * n_heads)
            reduction      = 0.5 * layer_fraction + 0.5 * head_fraction

        Args:
            plan: A validated or unvalidated :class:`SurgeryPlan`.

        Returns:
            Estimated parameter reduction in [0, 1].
        """
        layer_fraction = len(plan.layers_to_remove) / plan.n_layers

        total_heads_pruned = sum(
            len(v) for v in plan.heads_to_prune.values()
        )
        head_fraction = total_heads_pruned / (plan.n_layers * plan.n_heads)

        return 0.5 * layer_fraction + 0.5 * head_fraction

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(
        self,
        plan: SurgeryPlan,
        weight_dict: Dict[str, np.ndarray],
    ) -> SurgeryResult:
        """Apply *plan* to *weight_dict* and return a surgery result summary.

        Layer removal: every key containing ``"layer_{i}"`` for each
        ``i`` in ``plan.layers_to_remove`` is deleted from the copy.

        Head pruning: keys containing ``"layer_{i}.attn"`` are noted, and the
        pruned head count is accumulated.  Full weight-slice manipulation is
        tracked by count (the exact tensor surgery depends on the target
        architecture layout and is represented here by the row-selection
        logic described below).

        For head pruning on attention weight matrices of shape
        ``(n_heads * head_dim, *)``, this method removes the row slices
        belonging to pruned heads and updates the array in the working copy.

        Args:
            plan: Surgery plan (need not be pre-validated; validate() is called
                internally).
            weight_dict: Mapping of weight names to numpy arrays.

        Returns:
            :class:`SurgeryResult` with counts and estimated reduction.
        """
        plan.validate()

        # Work on a shallow copy — values (arrays) are not copied
        working: Dict[str, np.ndarray] = dict(weight_dict)

        # ── Layer removal ─────────────────────────────────────────────
        layers_removed = 0
        keys_to_delete = []
        for key in working:
            for layer_idx in plan.layers_to_remove:
                marker = f"layer_{layer_idx}"
                # Match "layer_N" but not "layer_N1" etc. by checking the
                # character immediately after the marker (dot, underscore,
                # end-of-string, or bracket).
                pos = key.find(marker)
                if pos != -1:
                    after = pos + len(marker)
                    if after >= len(key) or key[after] in (".", "_", "[", "]", "/"):
                        keys_to_delete.append(key)
                        break

        for key in keys_to_delete:
            del working[key]
            layers_removed += 1

        # ── Head pruning ──────────────────────────────────────────────
        total_heads_pruned = 0
        for layer_idx, head_indices in plan.heads_to_prune.items():
            if not head_indices:
                continue
            total_heads_pruned += len(head_indices)

            # Apply row-level pruning to attention weight matrices that are
            # still present after layer removal.
            attn_marker = f"layer_{layer_idx}.attn"
            rows_to_keep = [
                r
                for r in range(plan.n_heads)
                if r not in set(head_indices)
            ]
            keep_rows: list[int] = []
            for r in rows_to_keep:
                start = r * plan.head_dim
                keep_rows.extend(range(start, start + plan.head_dim))

            for key in list(working):
                if attn_marker in key:
                    arr = working[key]
                    if arr.ndim >= 1 and arr.shape[0] == plan.n_heads * plan.head_dim:
                        working[key] = arr[keep_rows]

        estimated_reduction = self.estimate_reduction(plan)

        self._stats.total_apply_calls += 1

        return SurgeryResult(
            layers_removed=layers_removed,
            heads_pruned=total_heads_pruned,
            estimated_param_reduction=estimated_reduction,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> SurgeryStats:
        """Running statistics for this surgeon."""
        return self._stats
