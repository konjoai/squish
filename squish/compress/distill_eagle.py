# [Experimental] This module is part of Squish v42+ (Wave 68).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""squish/compress/distill_eagle.py — EAGLE Draft Head Distillation.

Wave 68: trains a lightweight 3-layer transformer draft head from the target
model's hidden states, following Li et al. EAGLE-2 (ICML 2024) and the
SQUIZD draft-head appendix specification.

Overview
────────
During calibration, 2,000 prompt samples are fed through the target model and
the hidden states at two configurable source layers (default: 50th and 75th
percentile layers) are recorded.  A lightweight draft head — a 3-layer
transformer with ``d_model = d_target // d_hidden_ratio`` — is trained to
predict the next-token hidden state from those captured states.  The trained
draft head is serialised to a ``.squizd-eagle`` file and can later be merged
into the parent ``.squizd`` via ``squish compose --draft``.

EAGLE draft head appendix layout in .squizd
────────────────────────────────────────────
```
+─────────────────────────────┐
│ b"EAGL"   4 bytes           │  tag constant (SQUIZD_EAGLE_TAG)
│ payload_len  8 bytes        │  uint64 LE — length of JSON manifest
│ JSON manifest (UTF-8)       │  keys: version, d_model, d_hidden, n_layers,
│                             │        source_layer_indices, calibration_hash,
│                             │        weight_format, weight_offset, weight_len
│ weight bytes                │  raw NumPy .npy stream, INT4 or FP16
└─────────────────────────────┘
```

Usage::

    from squish.compress.distill_eagle import EAGLEConfig, EAGLEDistiller

    cfg      = EAGLEConfig(n_model_layers=32, d_model=4096)
    distiller = EAGLEDistiller(cfg)

    # hidden_states_fn: callable(prompt: str) -> list[np.ndarray]
    # Returns one array per layer, shape (seq_len, d_model)
    result = distiller.distill(hidden_states_fn, calibration_prompts)
    distiller.save(result, "qwen3-8b-draft.squizd-eagle")

    # Or use the high-level helper:
    from squish.compress.distill_eagle import download_pretrained_head
    download_pretrained_head("qwen3:8b", dest="~/.squish/models/")
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EAGLEConfig",
    "EAGLELayerWeights",
    "EAGLEHeadWeights",
    "EAGLEDistiller",
    "save_eagle_head",
    "load_eagle_head",
    "download_pretrained_head",
    "SQUIZD_EAGLE_TAG",
    "EAGLE_FORMAT_VERSION",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQUIZD_EAGLE_TAG: bytes = b"EAGL"
EAGLE_FORMAT_VERSION: int = 1
_DEFAULT_N_SAMPLES: int = 2000
_DEFAULT_N_EPOCHS: int = 3
_DEFAULT_LR: float = 3e-4
_DEFAULT_D_HIDDEN_RATIO: int = 4   # d_head = d_model // 4
_FALLBACK_ACCEPTANCE_THRESHOLD: float = 0.55


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EAGLEConfig:
    """Configuration for EAGLE head distillation.

    Attributes:
        n_model_layers: Total number of transformer layers in the target model.
        d_model: Hidden dimension of the target model.
        n_draft_layers: Number of layers in the draft head (default 3).
        d_hidden_ratio: ``d_head = d_model // d_hidden_ratio`` (default 4).
        n_samples: Number of calibration prompts (default 2000).
        n_epochs: Training epochs per calibration run (default 3).
        lr: AdamW learning rate (default 3e-4).
        n_draft_tokens: Draft tokens per generation step (default 5).
        acceptance_fallback_threshold: Acceptance rate below which the runtime
            falls back to n-gram drafting (default 0.55).
    """

    n_model_layers: int
    d_model: int
    n_draft_layers: int = 3
    d_hidden_ratio: int = _DEFAULT_D_HIDDEN_RATIO
    n_samples: int = _DEFAULT_N_SAMPLES
    n_epochs: int = _DEFAULT_N_EPOCHS
    lr: float = _DEFAULT_LR
    n_draft_tokens: int = 5
    acceptance_fallback_threshold: float = _FALLBACK_ACCEPTANCE_THRESHOLD

    def __post_init__(self) -> None:
        if self.n_model_layers < 1:
            raise ValueError("n_model_layers must be >= 1")
        if self.d_model < 1:
            raise ValueError("d_model must be >= 1")
        if self.d_hidden_ratio < 1:
            raise ValueError("d_hidden_ratio must be >= 1")
        if not (0.0 < self.acceptance_fallback_threshold < 1.0):
            raise ValueError("acceptance_fallback_threshold must be in (0, 1)")

    @property
    def d_head(self) -> int:
        """Hidden dimension of the draft head."""
        return self.d_model // self.d_hidden_ratio

    @property
    def source_layer_indices(self) -> tuple[int, int]:
        """Indices of the two target model layers used as hidden-state input.

        Selects the 50th and 75th percentile layers following Li et al.
        """
        p50 = max(0, round(self.n_model_layers * 0.50) - 1)
        p75 = max(0, round(self.n_model_layers * 0.75) - 1)
        return (p50, p75)


# ---------------------------------------------------------------------------
# Weight containers
# ---------------------------------------------------------------------------

@dataclass
class EAGLELayerWeights:
    """Weights for one transformer layer of the EAGLE draft head.

    Attributes:
        W_q: Query projection, shape ``(d_head, d_head)``.
        W_k: Key projection, shape ``(d_head, d_head)``.
        W_v: Value projection, shape ``(d_head, d_head)``.
        W_o: Output projection, shape ``(d_head, d_head)``.
        W_ff1: First FFN projection, shape ``(d_head * 4, d_head)``.
        W_ff2: Second FFN projection, shape ``(d_head, d_head * 4)``.
        ln_attn_gamma: LayerNorm weight before attention, shape ``(d_head,)``.
        ln_ff_gamma: LayerNorm weight before FFN, shape ``(d_head,)``.
    """

    W_q: np.ndarray
    W_k: np.ndarray
    W_v: np.ndarray
    W_o: np.ndarray
    W_ff1: np.ndarray
    W_ff2: np.ndarray
    ln_attn_gamma: np.ndarray
    ln_ff_gamma: np.ndarray


@dataclass
class EAGLEHeadWeights:
    """Complete trained EAGLE draft head.

    Attributes:
        config: The :class:`EAGLEConfig` used during distillation.
        layers: List of per-layer weights (``config.n_draft_layers`` items).
        input_proj: Input projection from ``d_model * 2 → d_head``, shape
            ``(d_head, d_model * 2)``.  Concatenates the two hidden states from
            ``source_layer_indices`` before passing to the transformer stack.
        output_proj: Output projection ``d_head → vocab_size``, shape
            ``(vocab_size, d_head)``.
        vocab_size: Vocabulary size of the target model.
        calibration_hash: SHA-256 hex digest of the calibration dataset
            (first 16 chars).  Stored in the appendix header.
    """

    config: EAGLEConfig
    layers: List[EAGLELayerWeights]
    input_proj: np.ndarray
    output_proj: np.ndarray
    vocab_size: int
    calibration_hash: str = ""


# ---------------------------------------------------------------------------
# Reference transformer block
# ---------------------------------------------------------------------------

def _layer_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMS LayerNorm (no bias) — matches Qwen3 architecture."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-8)


def _attention(
    h: np.ndarray,
    W_q: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_o: np.ndarray,
) -> np.ndarray:
    """Scaled dot-product attention (single head, single token)."""
    seq_len, d = h.shape
    scale = 1.0 / np.sqrt(d)
    Q = h @ W_q.T
    K = h @ W_k.T
    V = h @ W_v.T
    scores = _softmax((Q @ K.T) * scale)  # (seq, seq)
    attn_out = scores @ V                  # (seq, d)
    return attn_out @ W_o.T               # (seq, d)


def _ffn(h: np.ndarray, W_ff1: np.ndarray, W_ff2: np.ndarray) -> np.ndarray:
    """Two-layer FFN with SiLU activation."""
    gate = h @ W_ff1.T
    gate = gate * (1.0 / (1.0 + np.exp(-gate)))  # SiLU
    return gate @ W_ff2.T


def _eagle_forward(h: np.ndarray, weights: EAGLEHeadWeights) -> np.ndarray:
    """Single forward pass through the EAGLE draft head.

    Args:
        h: Concatenated hidden states from the two source layers,
           shape ``(1, d_model * 2)``.
        weights: Trained :class:`EAGLEHeadWeights`.

    Returns:
        Log-probability distribution over the vocabulary, shape ``(vocab_size,)``.
    """
    # Project from 2*d_model to d_head
    x = (h @ weights.input_proj.T).reshape(1, -1)  # (1, d_head)

    for layer in weights.layers:
        # Pre-norm attention
        residual = x
        x_ln = _layer_norm(x, layer.ln_attn_gamma)
        x = residual + _attention(x_ln, layer.W_q, layer.W_k, layer.W_v, layer.W_o)

        # Pre-norm FFN
        residual = x
        x_ln = _layer_norm(x, layer.ln_ff_gamma)
        x = residual + _ffn(x_ln, layer.W_ff1, layer.W_ff2)

    # Output logits
    logits = (x[0] @ weights.output_proj.T)  # (vocab_size,)
    # Log-softmax
    logits = logits - np.max(logits)
    log_probs = logits - np.log(np.sum(np.exp(logits)) + 1e-8)
    return log_probs


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------

class EAGLEDistiller:
    """Trains an EAGLE draft head via hidden-state distillation.

    The distiller uses a NumPy SGD-with-AdamW reference implementation.
    In production, MLX or PyTorch are used for GPU/ANE acceleration; this
    reference serves as a correctness baseline and for unit testing.

    Args:
        config: :class:`EAGLEConfig` controlling architecture and training.
        rng_seed: Random seed for reproducible weight initialisation.
    """

    def __init__(self, config: EAGLEConfig, *, rng_seed: int = 42) -> None:
        self.config = config
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_layer(self, d: int) -> EAGLELayerWeights:
        """Xavier-initialise one transformer layer."""
        scale = np.sqrt(2.0 / d)
        randn = self._rng.standard_normal

        def w(rows: int, cols: int) -> np.ndarray:
            return (randn((rows, cols)) * scale).astype(np.float32)

        return EAGLELayerWeights(
            W_q=w(d, d),
            W_k=w(d, d),
            W_v=w(d, d),
            W_o=w(d, d),
            W_ff1=w(d * 4, d),
            W_ff2=w(d, d * 4),
            ln_attn_gamma=np.ones(d, dtype=np.float32),
            ln_ff_gamma=np.ones(d, dtype=np.float32),
        )

    def _init_weights(self, vocab_size: int) -> EAGLEHeadWeights:
        """Initialise full draft head weights."""
        cfg = self.config
        d = cfg.d_head
        scale_in = np.sqrt(2.0 / (cfg.d_model * 2))
        scale_out = np.sqrt(2.0 / d)
        return EAGLEHeadWeights(
            config=cfg,
            layers=[self._init_layer(d) for _ in range(cfg.n_draft_layers)],
            input_proj=(self._rng.standard_normal((d, cfg.d_model * 2)) * scale_in
                        ).astype(np.float32),
            output_proj=(self._rng.standard_normal((vocab_size, d)) * scale_out
                         ).astype(np.float32),
            vocab_size=vocab_size,
            calibration_hash="",
        )

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    def distill(
        self,
        hidden_states_fn: Callable[[str], List[np.ndarray]],
        calibration_prompts: Sequence[str],
        vocab_size: int = 32000,
        *,
        progress: bool = False,
    ) -> EAGLEHeadWeights:
        """Distil a draft head from hidden states collected by *hidden_states_fn*.

        Args:
            hidden_states_fn: Callable that accepts a prompt string and returns
                a list of ``n_layers`` arrays of shape ``(seq_len, d_model)``.
            calibration_prompts: Sequence of prompt strings (recommend ≥ 2000).
            vocab_size: Vocabulary size of the target model.
            progress: If ``True``, print epoch/sample progress.

        Returns:
            Trained :class:`EAGLEHeadWeights`.
        """
        weights = self._init_weights(vocab_size)
        cfg = self.config
        p50, p75 = cfg.source_layer_indices

        # Compute calibration hash from prompts
        h = hashlib.sha256("".join(calibration_prompts).encode()).hexdigest()[:16]
        weights.calibration_hash = h

        # AdamW moment buffers (flat dict keyed by array id)
        m1: dict[int, np.ndarray] = {}
        m2: dict[int, np.ndarray] = {}
        step = 0

        for epoch in range(cfg.n_epochs):
            for idx, prompt in enumerate(calibration_prompts[: cfg.n_samples]):
                try:
                    all_states = hidden_states_fn(prompt)
                except Exception:
                    continue

                if len(all_states) <= max(p50, p75):
                    continue

                h_a = all_states[p50]     # (seq, d_model)
                h_b = all_states[p75]     # (seq, d_model)
                n_tok = min(h_a.shape[0], h_b.shape[0]) - 1
                if n_tok < 1:
                    continue

                for t in range(n_tok):
                    step += 1
                    h_in = np.concatenate([h_a[t], h_b[t]])[np.newaxis, :].astype(
                        np.float32
                    )  # (1, 2*d_model)
                    log_probs = _eagle_forward(h_in, weights)

                    # Target: next hidden state → soften with uniform label
                    # Project next hidden state from d_model → d_head before
                    # computing similarity (output_proj is (vocab_size, d_head)).
                    next_h_full = h_a[t + 1].astype(np.float32)  # (d_model,)
                    # Use the first d_model columns of input_proj as an
                    # approximate embedding into the d_head space.
                    next_h = weights.input_proj[:, : cfg.d_model] @ next_h_full  # (d_head,)
                    sim = weights.output_proj @ next_h
                    sim = sim - sim.max()
                    target = np.exp(sim) / (np.sum(np.exp(sim)) + 1e-8)

                    # Cross-entropy loss gradient (output projection only for
                    # speed — full backprop via external trainer in production)
                    grad_logits = np.exp(log_probs) - target
                    grad_W_out = np.outer(
                        grad_logits,
                        _layer_norm(
                            ((h_in @ weights.input_proj.T).reshape(1, -1))[0],
                            np.ones(weights.input_proj.shape[0], dtype=np.float32),
                        ),
                    )

                    # AdamW update for output_proj
                    wid = id(weights.output_proj)
                    if wid not in m1:
                        m1[wid] = np.zeros_like(weights.output_proj)
                        m2[wid] = np.zeros_like(weights.output_proj)
                    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
                    m1[wid] = beta1 * m1[wid] + (1 - beta1) * grad_W_out
                    m2[wid] = beta2 * m2[wid] + (1 - beta2) * grad_W_out ** 2
                    m1_hat = m1[wid] / (1 - beta1 ** step)
                    m2_hat = m2[wid] / (1 - beta2 ** step)
                    weights.output_proj -= cfg.lr * (
                        m1_hat / (np.sqrt(m2_hat) + eps_adam)
                        + 1e-2 * weights.output_proj  # weight decay
                    )

                if progress and idx % 100 == 0:
                    print(
                        f"  [eagle distill] epoch={epoch+1}/{cfg.n_epochs}  "
                        f"sample={idx}/{min(cfg.n_samples, len(calibration_prompts))}"
                    )

        return weights


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_eagle_head(weights: EAGLEHeadWeights, path: str | Path) -> Path:
    """Serialise *weights* to a ``.squizd-eagle`` file.

    The file layout follows the SQUIZD draft-head appendix spec:

    * 4-byte tag ``b"EAGL"``
    * 8-byte uint64 LE: JSON manifest length
    * JSON manifest (UTF-8)
    * Raw NumPy weight bytes (concatenated ``.npy`` streams per array)

    Args:
        weights: Trained :class:`EAGLEHeadWeights`.
        path: Destination file path.

    Returns:
        The written :class:`~pathlib.Path`.
    """
    path = Path(path)
    cfg = weights.config

    # Collect all arrays in deterministic order
    arrays: list[tuple[str, np.ndarray]] = [
        ("input_proj", weights.input_proj),
        ("output_proj", weights.output_proj),
    ]
    for li, layer in enumerate(weights.layers):
        for name in ("W_q", "W_k", "W_v", "W_o", "W_ff1", "W_ff2",
                     "ln_attn_gamma", "ln_ff_gamma"):
            arrays.append((f"layer_{li}_{name}", getattr(layer, name)))

    # Serialise each array to an in-memory .npy stream
    weight_buf = io.BytesIO()
    weight_layout: list[dict] = []
    for array_name, arr in arrays:
        offset = weight_buf.tell()
        np.save(weight_buf, arr)
        weight_layout.append({
            "name": array_name,
            "offset": offset,
            "length": weight_buf.tell() - offset,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        })

    weight_bytes = weight_buf.getvalue()

    manifest = {
        "version": EAGLE_FORMAT_VERSION,
        "d_model": cfg.d_model,
        "d_head": cfg.d_head,
        "n_draft_layers": cfg.n_draft_layers,
        "d_hidden_ratio": cfg.d_hidden_ratio,
        "n_draft_tokens": cfg.n_draft_tokens,
        "acceptance_fallback_threshold": cfg.acceptance_fallback_threshold,
        "source_layer_indices": list(cfg.source_layer_indices),
        "vocab_size": weights.vocab_size,
        "calibration_hash": weights.calibration_hash,
        "weight_format": "npy",
        "weight_layout": weight_layout,
    }

    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()

    with path.open("wb") as f:
        f.write(SQUIZD_EAGLE_TAG)
        f.write(struct.pack("<Q", len(manifest_bytes)))
        f.write(manifest_bytes)
        f.write(weight_bytes)

    return path


def load_eagle_head(path: str | Path) -> EAGLEHeadWeights:
    """Load an EAGLE draft head from a ``.squizd-eagle`` file.

    Args:
        path: Path to the serialised draft head file.

    Returns:
        :class:`EAGLEHeadWeights` ready for inference.

    Raises:
        ValueError: If the file magic tag is incorrect.
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    with path.open("rb") as f:
        tag = f.read(4)
        if tag != SQUIZD_EAGLE_TAG:
            raise ValueError(
                f"Invalid EAGLE head file: expected tag {SQUIZD_EAGLE_TAG!r}, "
                f"got {tag!r}"
            )
        manifest_len = struct.unpack("<Q", f.read(8))[0]
        manifest = json.loads(f.read(manifest_len).decode())
        weight_bytes = f.read()

    cfg = EAGLEConfig(
        n_model_layers=1,   # not stored; set from target model at runtime
        d_model=manifest["d_model"],
        n_draft_layers=manifest["n_draft_layers"],
        d_hidden_ratio=manifest["d_hidden_ratio"],
        n_draft_tokens=manifest["n_draft_tokens"],
        acceptance_fallback_threshold=manifest["acceptance_fallback_threshold"],
    )

    # Reconstruct arrays from the weight layout
    array_map: dict[str, np.ndarray] = {}
    for entry in manifest["weight_layout"]:
        buf = io.BytesIO(weight_bytes[entry["offset"]: entry["offset"] + entry["length"]])
        array_map[entry["name"]] = np.load(buf)

    n_layers = manifest["n_draft_layers"]
    layers: list[EAGLELayerWeights] = []
    for li in range(n_layers):
        layers.append(EAGLELayerWeights(
            W_q=array_map[f"layer_{li}_W_q"],
            W_k=array_map[f"layer_{li}_W_k"],
            W_v=array_map[f"layer_{li}_W_v"],
            W_o=array_map[f"layer_{li}_W_o"],
            W_ff1=array_map[f"layer_{li}_W_ff1"],
            W_ff2=array_map[f"layer_{li}_W_ff2"],
            ln_attn_gamma=array_map[f"layer_{li}_ln_attn_gamma"],
            ln_ff_gamma=array_map[f"layer_{li}_ln_ff_gamma"],
        ))

    return EAGLEHeadWeights(
        config=cfg,
        layers=layers,
        input_proj=array_map["input_proj"],
        output_proj=array_map["output_proj"],
        vocab_size=manifest["vocab_size"],
        calibration_hash=manifest["calibration_hash"],
    )


# ---------------------------------------------------------------------------
# HuggingFace download helper
# ---------------------------------------------------------------------------

HF_EAGLE_REPO: str = "squish-community/eagle-heads"


def download_pretrained_head(
    model_id: str,
    dest: str | Path = "~/.squish/models",
    *,
    token: Optional[str] = None,
    revision: str = "main",
) -> Path:
    """Download a pre-distilled EAGLE head from the squish-community HuggingFace repo.

    The head is downloaded only if the destination file does not already exist.

    Args:
        model_id: Model identifier in the squish catalog format (e.g. ``"qwen3:8b"``).
        dest: Directory to save the head file.  Defaults to ``~/.squish/models``.
        token: HuggingFace API token.  Reads ``HF_TOKEN`` from environment if not set.
        revision: HuggingFace repo revision / branch (default ``"main"``).

    Returns:
        Path to the downloaded ``.squizd-eagle`` file.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        RuntimeError: If the download fails.
    """
    import os
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download pre-trained EAGLE heads. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    dest_dir = Path(dest).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)

    safe_id = model_id.replace(":", "-").replace("/", "-")
    filename = f"{safe_id}.squizd-eagle"
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path

    api_token = token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )

    try:
        downloaded = hf_hub_download(
            repo_id=HF_EAGLE_REPO,
            filename=filename,
            revision=revision,
            token=api_token,
            local_dir=str(dest_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download EAGLE head for {model_id!r} from "
            f"{HF_EAGLE_REPO!r}: {exc}"
        ) from exc

    return Path(downloaded)
