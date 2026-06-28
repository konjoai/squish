"""The matrix definition: reuse levels x context lengths, and order counterbalancing.

Reuse levels and context lengths are the two axes the sprint mandates. Systems
are the depth dimension. ``counterbalanced_order`` rotates which system runs
first per cell so no system is systematically measured first (cool) or last
(hot) — complementary to the per-system cold-baseline isolation in ``cell``.
Pure module: unit-tested without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

REUSE_LEVELS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
CONTEXT_LENGTHS: tuple[int, ...] = (4000, 8000, 16000, 32000)

# The kill-test cell: run this ONE cell, show it, and WAIT for approval.
KILL_TEST_REUSE = 0.50
KILL_TEST_CTX = 8000

# Cross-version Ollama cross-check is scoped to these cells only (cost control).
CROSS_CHECK_CELLS: tuple[tuple[float, int], ...] = ((0.0, 8000), (0.50, 8000))


@dataclass(frozen=True)
class Cell:
    reuse: float
    ctx_tokens: int

    @property
    def cell_id(self) -> str:
        return f"r{int(self.reuse * 100):03d}_c{self.ctx_tokens}"


def all_cells() -> list[Cell]:
    """Full matrix in a stable, reproducible order (reuse outer, context inner)."""
    return [Cell(r, c) for r in REUSE_LEVELS for c in CONTEXT_LENGTHS]


def kill_test_cell() -> Cell:
    return Cell(KILL_TEST_REUSE, KILL_TEST_CTX)


def counterbalanced_order(system_names: list[str], cell_index: int) -> list[str]:
    """Rotate system order by cell index so first-position is shared evenly.

    Head-to-head systems are always kept adjacent and rotated together so the
    paired comparison is never separated by an unrelated capability run.
    """
    if not system_names:
        return []
    k = cell_index % len(system_names)
    return system_names[k:] + system_names[:k]


def second_model_rows() -> list[Cell]:
    """The 0% and 50% rows recommended for the smaller cross-model (anti-artifact)."""
    return [Cell(0.0, c) for c in CONTEXT_LENGTHS] + [Cell(0.50, c) for c in CONTEXT_LENGTHS]
