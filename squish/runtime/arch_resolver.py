"""squish/runtime/arch_resolver.py — dispatch a checkpoint to mlx_lm or mlx_vlm.

``mlx_lm`` only implements text-only decoder architectures. A growing set of
``mlx-community`` checkpoints declare a ``model_type`` mlx_lm doesn't know
(fused/multimodal families like Gemma 4, plus the VLM/omni long tail). This
module decides, from ``config.json`` alone and without loading any weights,
which runtime backend a given model directory should load through.

``mlx_lm`` stays the default/fast path — every model type it already
supports resolves to ``"mlx_lm"`` with a single cheap ``importlib`` probe.
Only unsupported types fall through to ``mlx_vlm``, and only when the
``multimodal`` extra is installed (``pip install squish-ai[multimodal]``).
Neither ``mlx_lm`` nor ``mlx_vlm`` is imported at module scope, matching the
platform-gating rule for MLX imports elsewhere in the codebase.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Literal

_LOG = logging.getLogger("squish.runtime.arch_resolver")

RuntimeName = Literal["mlx_lm", "mlx_vlm"]

_SIDECAR_NAME = ".squish_runtime.json"

# Mirrors mlx_lm.utils.MODEL_REMAPPING. Kept as an independent copy so this
# probe doesn't reach into mlx_lm's private module internals — it only needs
# to know which module name to try importing, not mlx_lm's implementation.
_MODEL_REMAPPING = {
    "mistral": "llama",
    "llava": "mistral3",
    "phi-msft": "phixtral",
    "falcon_mamba": "mamba",
    "joyai_llm_flash": "deepseek_v3",
    "kimi_k2": "deepseek_v3",
    "qwen2_5_vl": "qwen2_vl",
    "minimax_m2": "minimax",
    "iquestcoder": "llama",
}


class UnsupportedArchitectureError(RuntimeError):
    """Neither mlx_lm nor an installed mlx_vlm can load this model_type."""


def _read_model_type(model_dir: str | Path) -> str:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_dir}")
    with open(config_path) as f:
        config = json.load(f)
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError(f"{config_path} has no 'model_type' field")
    return model_type


def _mlx_lm_supports(model_type: str) -> bool:
    """Cheap probe: does mlx_lm have a matching models.<type> module?

    No weights are loaded — this mirrors mlx_lm.utils._get_classes's own
    dispatch (remap, then ``importlib.import_module``) without calling it
    directly, so the probe works even when mlx_lm isn't installed at all.
    """
    mapped = _MODEL_REMAPPING.get(model_type, model_type)
    try:
        importlib.import_module(f"mlx_lm.models.{mapped}")
        return True
    except ImportError:
        return False


def _mlx_vlm_available() -> bool:
    try:
        importlib.import_module("mlx_vlm")
        return True
    except ImportError:
        return False


def _sidecar_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / _SIDECAR_NAME


def _read_sidecar(model_dir: str | Path) -> RuntimeName | None:
    path = _sidecar_path(model_dir)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _LOG.warning("Ignoring unreadable runtime sidecar %s: %s", path, exc)
        return None
    runtime = data.get("runtime")
    return runtime if runtime in ("mlx_lm", "mlx_vlm") else None


def _write_sidecar(model_dir: str | Path, runtime: RuntimeName) -> None:
    path = _sidecar_path(model_dir)
    try:
        with open(path, "w") as f:
            json.dump({"runtime": runtime}, f)
    except OSError as exc:
        _LOG.warning("Could not write runtime sidecar %s: %s", path, exc)


def resolve_runtime(model_dir: str | Path, *, use_cache: bool = True) -> RuntimeName:
    """Decide whether *model_dir* should load via ``mlx_lm`` or ``mlx_vlm``.

    Reads ``config.json``'s ``model_type`` and probes mlx_lm's model
    registry via a cheap ``importlib`` import (no weights loaded). Falls
    back to ``mlx_vlm`` when the extra is installed; raises
    :class:`UnsupportedArchitectureError` with an actionable install
    instruction when neither backend can load it.

    Caches the resolution in a small JSON sidecar next to *model_dir* so
    repeated ``squish run`` calls don't re-probe. Pass ``use_cache=False``
    to force a fresh probe.
    """
    if use_cache:
        cached = _read_sidecar(model_dir)
        if cached is not None:
            return cached

    model_type = _read_model_type(model_dir)

    if _mlx_lm_supports(model_type):
        runtime: RuntimeName = "mlx_lm"
    elif _mlx_vlm_available():
        _LOG.info("model_type=%r not supported by mlx_lm; routing to mlx_vlm", model_type)
        runtime = "mlx_vlm"
    else:
        raise UnsupportedArchitectureError(
            f"model_type={model_type!r} is not supported by mlx_lm, and the "
            "mlx_vlm backend isn't installed. Install it with: "
            "pip install 'squish-ai[multimodal]'"
        )

    if use_cache:
        _write_sidecar(model_dir, runtime)
    return runtime
