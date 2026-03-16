#!/usr/bin/env python3
"""
Smoke test: save_int4_npy_dir + INT4 dequantize path in compressed_loader.

Creates a minimal synthetic npy-dir, runs save_int4_npy_dir(), then calls
_dequantize_npy_dir() and verifies INT4 round-trip cosine quality.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Vectro optional dependency — set VECTRO_DIR env var or place at ~/vectro
_vectro = Path(os.environ.get("VECTRO_DIR", Path.home() / "vectro"))
if _vectro.exists():
    sys.path.insert(0, str(_vectro))

from squish.quant.compressed_loader import (  # noqa: E402
    _INT4_READY,
    _dequantize_npy_dir,
    save_int4_npy_dir,
)


# ── Build a tiny synthetic npy-dir ──────────────────────────────────────────
def make_synthetic_npy_dir(root: Path, n: int = 64, d: int = 512) -> list[str]:
    """Save two INT8 tensors and one passthrough tensor."""
    tensor_dir = root / "tensors"
    tensor_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    keys = []

    for name in ("tensor_a", "tensor_b"):
        # 2D weight tensor (n × d)
        arr = rng.standard_normal((n, d)).astype(np.float32)
        # INT8 per-row quantize (simple)
        row_max = np.abs(arr).max(axis=1, keepdims=True)
        scales = (row_max / 127.0).astype(np.float32).squeeze()
        q = np.clip(np.round(arr / scales[:, None]), -127, 127).astype(np.int8)
        np.save(str(tensor_dir / f"{name}__q.npy"), q)
        np.save(str(tensor_dir / f"{name}__s.npy"), scales)
        np.save(str(tensor_dir / f"{name}__shape.npy"), np.array(arr.shape))
        keys.append(name)

    # Passthrough float16 tensor
    bias = rng.standard_normal((d,)).astype(np.float16)
    np.save(str(tensor_dir / "bias__pt.npy"), bias)
    np.save(str(tensor_dir / "bias__shape.npy"), np.array(bias.shape))
    keys.append("bias")

    # Minimal manifest
    manifest = {f"model.{k}": k for k in keys}
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return keys


def test_int4_conversion_and_round_trip():
    """Smoke test: save_int4_npy_dir + INT4 dequantize round-trip quality."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        keys = make_synthetic_npy_dir(root)

        # Step 1: convert INT8 → INT4
        result = save_int4_npy_dir(str(root), verbose=False)
        assert (root / _INT4_READY).exists(), "Sentinel not written"
        assert result["n_converted"] == 2, f"Expected 2 converted, got {result['n_converted']}"
        assert result["n_skipped"]   == 1, f"Expected 1 skipped (bias), got {result['n_skipped']}"
        savings = result["savings_pct"]
        # group_size=32 uses 2× as many scale values as group_size=64, reducing
        # raw byte savings from ~50% to ~38%.  The accuracy improvement is worth it.
        assert 32 < savings < 60, f"Expected ~38-50% savings, got {savings:.1f}%"

        # Step 2: verify INT4 round-trip quality
        tensor_dir = root / "tensors"
        rng = np.random.default_rng(42)
        for name in ("tensor_a", "tensor_b"):
            n, d = 64, 512
            arr_orig = rng.standard_normal((n, d)).astype(np.float32)
            arr_rec = _dequantize_npy_dir(tensor_dir, name)
            assert arr_rec.shape == (n, d), f"Shape mismatch: {arr_rec.shape}"
            cos = np.mean([
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                for a, b in zip(arr_orig[:8], arr_rec[:8], strict=False)
            ])
            assert cos > 0.98, f"Cosine too low for INT4: {cos:.5f}"
