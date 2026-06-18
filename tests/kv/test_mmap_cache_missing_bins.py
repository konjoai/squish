"""Regression: stale meta.json + missing .bin files must reset n_tokens to 0.

MMapKVLayer restored n_tokens from meta.json regardless of whether the k.bin /
v.bin data files still existed. If the bins were lost while meta survived, the
layer opened fresh zeroed memmaps yet reported n_tokens > 0, so get() returned
all-zero K/V with no error.
"""
from __future__ import annotations

import numpy as np

from squish.kv.mmap_cache import MMapKVLayer


def test_missing_bins_resets_n_tokens(tmp_path):
    root = tmp_path / "layer0"
    n_heads, head_dim = 2, 4

    layer = MMapKVLayer(root, capacity=8, n_heads=n_heads, head_dim=head_dim)
    layer.append(np.ones((n_heads, head_dim), dtype=np.float16),
                 np.ones((n_heads, head_dim), dtype=np.float16))
    layer.append(np.ones((n_heads, head_dim), dtype=np.float16) * 2,
                 np.ones((n_heads, head_dim), dtype=np.float16) * 2)
    assert layer.n_tokens == 2
    layer.close()

    # Simulate partial loss: data files gone, meta.json survives.
    (root / "k.bin").unlink()
    (root / "v.bin").unlink()
    assert (root / "meta.json").exists()

    reopened = MMapKVLayer(root, capacity=8, n_heads=n_heads, head_dim=head_dim)
    # Must not claim to hold tokens it can no longer back with real data.
    assert reopened.n_tokens == 0
    reopened.close()


def test_intact_reopen_preserves_tokens(tmp_path):
    # Guard: a normal reopen (bins present) still restores n_tokens.
    root = tmp_path / "layer1"
    n_heads, head_dim = 2, 4
    layer = MMapKVLayer(root, capacity=8, n_heads=n_heads, head_dim=head_dim)
    layer.append(np.ones((n_heads, head_dim), dtype=np.float16),
                 np.ones((n_heads, head_dim), dtype=np.float16))
    layer.close()

    reopened = MMapKVLayer(root, capacity=8, n_heads=n_heads, head_dim=head_dim)
    assert reopened.n_tokens == 1
    reopened.close()
