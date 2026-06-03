"""squish.squash.qtip_loader — load-only bridge for pre-quantized QTIP/YAQA checkpoints.

This module intentionally **does not** implement QTIP/YAQA quantization kernels.
It only downloads and loads already-quantized checkpoints from HuggingFace Hub
through standard runtime libraries.

License boundary note:
- QTIP/YAQA ecosystems may involve GPL runtime components.
- Squish does not bundle or distribute GPL kernels.
- This loader uses user-provided runtime environments and logs an explicit warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_GPL_BOUNDARY_WARNING = (
    "WARNING: QTIP model loaded via GPL-compatible runtime path — "
    "squish does not distribute GPL code"
)


class QtipLoaderError(RuntimeError):
    """Raised when a QTIP/YAQA checkpoint cannot be loaded."""


@dataclass
class PreQuantizedLoadHandle:
    """Loaded model handle metadata for pre-quantized checkpoints."""

    family: str
    repo_id: str
    local_path: Path
    model: Any
    tokenizer: Any


def _import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for load-qtip. Install with: pip install huggingface-hub"
        ) from e
    return snapshot_download


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers is required for load-qtip. Install with: pip install transformers"
        ) from e
    return AutoModelForCausalLM, AutoTokenizer


def _load_prequantized_checkpoint(
    *,
    family: str,
    hf_repo_id: str,
    local_dir: Path | None = None,
    hf_token: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> PreQuantizedLoadHandle:
    if not hf_repo_id.strip():
        raise ValueError("hf_repo_id is required")

    snapshot_download = _import_snapshot_download()
    AutoModelForCausalLM, AutoTokenizer = _import_transformers()

    if local_dir is not None:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=str(local_dir) if local_dir is not None else None,
            token=hf_token,
            revision=revision,
        )
    )

    # Required governance log line for GPL-adjacent runtime paths.
    log.warning(_GPL_BOUNDARY_WARNING)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=trust_remote_code,
    )

    return PreQuantizedLoadHandle(
        family=family,
        repo_id=hf_repo_id,
        local_path=model_dir,
        model=model,
        tokenizer=tokenizer,
    )


def load_qtip_checkpoint(
    hf_repo_id: str,
    *,
    local_dir: Path | None = None,
    hf_token: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> PreQuantizedLoadHandle:
    """Download and load a pre-quantized QTIP checkpoint from HF Hub."""
    return _load_prequantized_checkpoint(
        family="qtip",
        hf_repo_id=hf_repo_id,
        local_dir=local_dir,
        hf_token=hf_token,
        revision=revision,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )


def load_yaqa_checkpoint(
    hf_repo_id: str,
    *,
    local_dir: Path | None = None,
    hf_token: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> PreQuantizedLoadHandle:
    """Download and load a pre-quantized YAQA checkpoint from HF Hub."""
    return _load_prequantized_checkpoint(
        family="yaqa",
        hf_repo_id=hf_repo_id,
        local_dir=local_dir,
        hf_token=hf_token,
        revision=revision,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
