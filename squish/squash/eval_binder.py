"""squish/squash/eval_binder.py — DEPRECATED backward-compat shim.

Wave 19: ``EvalBinder`` now lives in :mod:`squish.squash.sbom_builder`.
This module is retained so all existing callers continue to work without
modification.

Do not add new code here.  Import from ``squish.squash.sbom_builder`` directly.
"""

from squish.squash.sbom_builder import EvalBinder  # noqa: F401

__all__ = ["EvalBinder"]
