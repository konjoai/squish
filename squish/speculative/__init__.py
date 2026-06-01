"""squish.speculative — speculative-decoding helpers.

Re-exports the public names that ``squish.server`` and ``squish.cli`` import
from this package. Without these re-exports the v4 ``--draft-model`` flag
import-errors at server startup; see ``results/benchmarks_v4/PRECHECK.md``.
"""

from squish.speculative.speculative import (
    EagleDraftHead,
    FSMGammaController,
    NgramTable,
    SpeculativeGenerator,
    load_draft_model,
)

__all__ = [
    "EagleDraftHead",
    "FSMGammaController",
    "NgramTable",
    "SpeculativeGenerator",
    "load_draft_model",
]
