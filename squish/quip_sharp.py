"""
squish/quip_sharp.py

QuIP# — Trellis-Coded E8 Quantization
======================================

Extends the SpinQuant incoherence preprocessing from ``spin_quant.py``
(which provides Step 1: random orthogonal / Cayley-SGD rotation) with

  Step 2: E8 lattice quantization and per-block scale compression.

For each 8-D weight chunk the nearest entry in a 256-vector E8-derived
codebook is stored as a uint8 index.  A float16 *residual scale* (the L2
norm of the original chunk) allows the codeword to be rescaled on load.

Reconstruction (quip_dequantize):
    W_chunk ≈ residual_scale * E8Lattice.codebook[e8_index]
    W       = W_rot @ R          (where W_rot = W @ R.T at compress time)

Background
----------
QuIP# (Tseng et al. 2024, arXiv:2402.04396) applies incoherence processing
(Hadamard or learned rotation) to each weight matrix and then quantizes each
8-D block to the nearest point in the E8 lattice.  The E8 lattice is optimal
for packing spheres in 8 dimensions and achieves the highest kissing number
(240 minimal vectors), making it an excellent quantization codebook for
isotropically distributed rotated weights.

This module implements a self-contained NumPy-only offline quantizer
(no MLX required) plus MLX-compatible dequantization via mx.take().

Usage
-----
    from squish.quip_sharp import (
        E8Lattice, QuIPSharpConfig, QuIPSharpQuantizer,
        quip_dequantize, quantize_model_quip,
    )

    cfg = QuIPSharpConfig(use_hadamard=True, scalar_bits=2)
    q   = QuIPSharpQuantizer(cfg, seed=42)

    layer = q.quantize(W_float32)   # W: (out, in)
    W_hat = quip_dequantize(layer)  # float16 (out, in)

    # Model-level compression
    compressed = quantize_model_quip(weight_dict, cfg)
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# E8 codebook construction (once at import, deterministic, ~4 KB)
# ---------------------------------------------------------------------------

def _build_e8_codebook() -> np.ndarray:
    """
    Build the 256-entry E8-derived codebook (float16), projected to unit sphere.

    Three families of E8 lattice vectors (all with norm sqrt(2) or 2 before
    normalisation):

    1. **Half-integer family** — (±1/2)^8 with an *even* number of −1/2
       entries: 128 vectors, norm sqrt(2).

    2. **Integer-pair family** — ±e_i ± e_j for 0 ≤ i < j ≤ 7, all four
       sign combinations: 28 × 4 = 112 vectors, norm sqrt(2).

    3. **Cartesian family** — ±2·e_i for i = 0..7: 16 vectors, norm 2.

    Total: 128 + 112 + 16 = 256 distinct vectors.

    After L2 normalisation every vector lies on the unit 8-sphere.  Vectors
    from different families have different sparsity patterns and are always
    distinct after normalisation.
    """
    vectors: list[np.ndarray] = []

    # ── Family 1: half-integer (±1/2)^8, even number of minus signs ─────────
    for mask in range(256):
        if bin(mask).count("1") % 2 == 0:
            v = np.array(
                [(-0.5 if (mask >> i) & 1 else 0.5) for i in range(8)],
                dtype=np.float64,
            )
            vectors.append(v)

    # ── Family 2: integer-pair ±e_i ± e_j ───────────────────────────────────
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8, dtype=np.float64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)

    # ── Family 3: Cartesian ±2·e_i ──────────────────────────────────────────
    for i in range(8):
        for s in (1.0, -1.0):
            v = np.zeros(8, dtype=np.float64)
            v[i] = 2.0 * s
            vectors.append(v)

    vecs = np.array(vectors, dtype=np.float64)           # (256, 8)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)  # (256, 1)
    return (vecs / norms).astype(np.float16)              # unit sphere, float16


# ---------------------------------------------------------------------------
# E8Lattice — static codebook container
# ---------------------------------------------------------------------------

class E8Lattice:
    """
    Static container for the precomputed 256-entry E8-derived codebook.

    ``E8Lattice.codebook`` is a **(256, 8) float16** array of unit vectors
    covering the 8-dimensional unit sphere, built deterministically at import
    time (no random seed, no learned parameters).

    For MLX dequantization::

        import mlx.core as mx
        cb  = mx.array(E8Lattice.codebook)   # (256, 8)
        W_q = mx.take(cb, e8_indices, axis=0) * residual_scales[:, None]
    """

    codebook: np.ndarray = _build_e8_codebook()   # class attribute, (256, 8) float16


# ---------------------------------------------------------------------------
# QuIPSharpConfig
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class QuIPSharpConfig:
    """
    Configuration for QuIP# trellis-coded E8 quantization.

    Parameters
    ----------
    use_hadamard : bool
        True  → generate a random orthogonal (Hadamard-style) incoherence
                 rotation per weight matrix at quantise time.
        False → reuse a pre-computed SpinQuant rotation supplied via the
                 ``rotation_matrix`` argument of :class:`QuIPSharpQuantizer`.
                 Falls back to a random orthogonal if none is provided.
    scalar_bits : int
        Bits used for the per-block scalar representation (2 or 3).
        The residual scale is always stored as full float16 for lossless
        reconstruction; this field documents the intended compression level.
    group_size : int
        Dimensionality of each E8 lattice group.  Must be **8** (the
        dimension of the E8 lattice).
    """

    use_hadamard: bool = True
    scalar_bits: int = 2
    group_size: int = 8

    def __post_init__(self) -> None:
        if self.group_size != 8:
            raise ValueError(
                f"QuIPSharpConfig.group_size must be 8 (E8 dimensionality); "
                f"got {self.group_size}"
            )
        if self.scalar_bits not in (2, 3):
            raise ValueError(
                f"QuIPSharpConfig.scalar_bits must be 2 or 3; got {self.scalar_bits}"
            )


# ---------------------------------------------------------------------------
# QuIPSharpLayer
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class QuIPSharpLayer:
    """
    Compressed linear-layer representation produced by :class:`QuIPSharpQuantizer`.

    Attributes
    ----------
    e8_indices : np.ndarray
        uint8 array of shape **(N,)** where N = out_features × n_groups.
        Each entry is an index (0–255) into ``E8Lattice.codebook``.
    residual_scales : np.ndarray
        float16 array of shape **(N,)**.  The L2 norm of the original 8-D
        weight chunk, used to scale the unit-sphere codeword at dequantize time.
    rotation_matrix : np.ndarray or None
        float16 array of shape **(in_features, in_features)** — the orthogonal
        rotation applied before E8 projection.  None if no rotation was stored.
    original_shape : tuple[int, ...]
        The ``(out_features, in_features)`` shape of the uncompressed weight.
    config : QuIPSharpConfig
        The configuration used during quantization.
    """

    e8_indices: np.ndarray        # uint8 (N,)
    residual_scales: np.ndarray   # float16 (N,)
    rotation_matrix: np.ndarray | None  # float16 (d_in, d_in) or None
    original_shape: tuple[int, ...]
    config: QuIPSharpConfig


# ---------------------------------------------------------------------------
# QuIPSharpQuantizer
# ---------------------------------------------------------------------------

class QuIPSharpQuantizer:
    """
    Offline quantizer: compresses a 2-D weight matrix into a
    :class:`QuIPSharpLayer` using E8 lattice quantization.

    **Step 1 — incoherence preprocessing**
        A random orthogonal matrix R is applied: ``W_rot = W @ R.T``.
        This makes the weight distribution roughly isotropic, which minimises
        E8 quantization error.  When ``config.use_hadamard=False`` a
        pre-computed SpinQuant rotation is accepted via *rotation_matrix*.

    **Step 2 — E8 lattice projection**
        Each 8-D chunk of W_rot is normalized to the unit sphere and matched
        against the 256-entry E8 codebook by maximum inner product (equivalent
        to minimum L2 distance for unit vectors).  The uint8 codebook index
        and the chunk's L2 norm (float16) are stored for later reconstruction.

    Parameters
    ----------
    config : QuIPSharpConfig or None
        Quantization settings.  Defaults to ``QuIPSharpConfig()``.
    seed : int
        Random seed for the orthogonal rotation generator.  The internal RNG
        advances its state over successive :meth:`quantize` calls so each
        weight matrix in a model receives a unique rotation.
    rotation_matrix : np.ndarray or None
        Pre-computed orthogonal rotation (e.g. from ``spin_quant.run_rotation``)
        used when ``config.use_hadamard=False``.  If None a random rotation is
        generated as a fallback regardless of the ``use_hadamard`` flag.
    """

    def __init__(
        self,
        config: QuIPSharpConfig | None = None,
        *,
        seed: int = 42,
        rotation_matrix: np.ndarray | None = None,
    ) -> None:
        self.config = config if config is not None else QuIPSharpConfig()
        self._rng = np.random.default_rng(seed)
        self._ext_rotation = rotation_matrix

    def _make_rotation(self, dim: int) -> np.ndarray:
        """
        Return a ``(dim, dim)`` float32 orthogonal rotation matrix.

        Prefers the external SpinQuant rotation when it matches *dim*;
        otherwise generates a fresh Haar-uniform random orthogonal via QR
        (same construction as ``spin_quant._random_orthogonal``).
        """
        if self._ext_rotation is not None and self._ext_rotation.shape == (dim, dim):
            return self._ext_rotation.astype(np.float32)

        A = self._rng.standard_normal((dim, dim)).astype(np.float32)
        Q, R = np.linalg.qr(A)
        diag = np.sign(np.diag(R))
        diag[diag == 0] = 1.0
        return (Q * diag).astype(np.float32)

    def quantize(self, W: np.ndarray) -> QuIPSharpLayer:
        """
        Quantize a 2-D weight matrix into a :class:`QuIPSharpLayer`.

        Parameters
        ----------
        W : np.ndarray
            Float weight matrix of shape ``(out_features, in_features)``.
            Any float dtype is accepted; conversion to float32 is applied
            internally.

        Returns
        -------
        QuIPSharpLayer
            Compressed representation with E8 indices, residual scales, and
            the rotation matrix required for dequantization.

        Raises
        ------
        ValueError
            If *W* is not 2-D.
        """
        if W.ndim != 2:
            raise ValueError(
                f"QuIPSharpQuantizer.quantize expects a 2-D array; got ndim={W.ndim}"
            )

        original_shape = (int(W.shape[0]), int(W.shape[1]))
        out_features, in_features = original_shape
        W_f32 = W.astype(np.float32)

        # ── Step 1: incoherence rotation ─────────────────────────────────────
        R = self._make_rotation(in_features)        # (d_in, d_in) float32
        W_rot = W_f32 @ R.T                         # (out, d_in)

        # ── Step 2: E8 lattice quantization ──────────────────────────────────
        gs = self.config.group_size                 # 8

        # Pad in_features dimension to a multiple of group_size
        pad = (-in_features) % gs
        if pad:
            W_rot = np.pad(W_rot, ((0, 0), (0, pad)))

        n_groups = W_rot.shape[1] // gs
        N = out_features * n_groups
        chunks = W_rot.reshape(N, gs)               # (N, 8)

        # Per-chunk L2 norm (used as the residual scale at dequantize time)
        scales = np.linalg.norm(chunks, axis=1)     # (N,) float32
        safe_scales = np.where(scales == 0.0, 1.0, scales)
        chunks_norm = chunks / safe_scales[:, np.newaxis]  # (N, 8) unit vectors

        # Nearest E8 codeword: for unit vectors, argmax of dot-product = argmin L2
        cb = E8Lattice.codebook.astype(np.float32)  # (256, 8)
        dots = chunks_norm @ cb.T                   # (N, 256)
        e8_indices = np.argmax(dots, axis=1).astype(np.uint8)  # (N,)

        return QuIPSharpLayer(
            e8_indices=e8_indices,
            residual_scales=scales.astype(np.float16),
            rotation_matrix=R.astype(np.float16),
            original_shape=original_shape,
            config=self.config,
        )


# ---------------------------------------------------------------------------
# quip_dequantize
# ---------------------------------------------------------------------------

def quip_dequantize(layer: QuIPSharpLayer) -> np.ndarray:
    """
    Reconstruct a weight matrix from a :class:`QuIPSharpLayer`.

    Reconstruction steps:

    1. Look up the E8 codeword for each block:
       ``codeword = E8Lattice.codebook[e8_indices[i]]``
    2. Scale by the stored residual:
       ``chunk ≈ residual_scales[i] * codeword``
    3. Reshape to ``(out_features, in_features_padded)``.
    4. Strip zero-padding columns.
    5. Apply inverse rotation:
       ``W ≈ W_rot @ R``  (since at compress time ``W_rot = W @ R.T``).

    For MLX inference this is equivalent to::

        cb      = mx.array(E8Lattice.codebook)
        W_rot   = mx.take(cb, e8_indices, axis=0) * residual_scales[:, None]
        W       = W_rot.reshape(out, in) @ R

    Parameters
    ----------
    layer : QuIPSharpLayer
        Compressed layer as returned by :meth:`QuIPSharpQuantizer.quantize`.

    Returns
    -------
    np.ndarray
        float16 array of shape ``layer.original_shape`` — the reconstructed
        weight matrix.
    """
    out_features, in_features = layer.original_shape
    gs = layer.config.group_size                        # 8

    pad = (-in_features) % gs
    in_features_padded = in_features + pad
    n_groups = in_features_padded // gs
    N = out_features * n_groups

    # ── Gather E8 codewords ───────────────────────────────────────────────────
    cb = E8Lattice.codebook.astype(np.float32)          # (256, 8)
    idx = layer.e8_indices.astype(np.intp)              # (N,)
    codewords = cb[idx]                                 # (N, 8)

    # ── Scale by residual norms ───────────────────────────────────────────────
    scales = layer.residual_scales.astype(np.float32)   # (N,)
    W_chunks = codewords * scales[:, np.newaxis]        # (N, 8)

    # ── Reshape and strip padding ──────────────────────────────────────────────
    W_rot = W_chunks.reshape(out_features, in_features_padded)
    if pad:
        W_rot = W_rot[:, :in_features]

    # ── Inverse rotation: W_rot = W @ R.T  →  W ≈ W_rot @ R ─────────────────
    if layer.rotation_matrix is not None:
        R = layer.rotation_matrix.astype(np.float32)   # (d_in, d_in)
        W = W_rot @ R
    else:
        W = W_rot

    return W.astype(np.float16)


# ---------------------------------------------------------------------------
# quantize_model_quip
# ---------------------------------------------------------------------------

def quantize_model_quip(
    model: dict[str, Any],
    config: QuIPSharpConfig | None = None,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Quantize eligible linear weights in a model weight dict using QuIP# E8.

    Walks *model* (a flat ``name → np.ndarray`` mapping) and replaces every
    2-D float weight array with a :class:`QuIPSharpLayer`.  1-D tensors
    (biases, layer norms, etc.) are returned unchanged.

    Parameters
    ----------
    model : dict[str, Any]
        Flat weight mapping: ``"layer.weight" → np.ndarray``.
    config : QuIPSharpConfig or None
        Quantization settings.  Defaults to ``QuIPSharpConfig()``.
    seed : int
        Base seed for the rotation RNG.  The RNG is shared across all weights
        so each layer receives a unique (but reproducible) rotation.

    Returns
    -------
    dict[str, Any]
        Weight dict with 2-D arrays replaced by :class:`QuIPSharpLayer`
        objects and all other entries passed through unmodified.
    """
    effective_config = config if config is not None else QuIPSharpConfig()
    quantizer = QuIPSharpQuantizer(effective_config, seed=seed)
    result: dict[str, Any] = {}
    for name, weight in model.items():
        if isinstance(weight, np.ndarray) and weight.ndim == 2:
            result[name] = quantizer.quantize(weight)
        else:
            result[name] = weight
    return result
