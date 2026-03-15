"""pack_bitnet_bf16.py

Convert a BitNet b1.58 BF16 checkpoint (quantization_mode="online") into the
packed uint8 + weight_scale format that mlx_lm's BitLinear kernel expects.

The BF16 checkpoint stores latent float weights; mlx_lm's BitLinear layer
expects weights already packed as 4 ternary values per uint8 byte using the
custom Metal kernel layout (see bitlinear_layers.py).

Usage::

    python dev/scripts/pack_bitnet_bf16.py \\
        --src models/bitnet-b1.58-2B-4T-bf16 \\
        --dst models/bitnet-b1.58-2B-4T-mlx

Packed format
-------------
For each linear weight of shape (out_features, in_features):

1. scale   = mean(|W|)
2. threshold = 0.5 × scale
3. ternary   = sign(W) where |W| >= threshold, else 0  ∈ {-1, 0, +1}
4. codes     = ternary + 1                              ∈ {0, 1, 2}

Bit-packing layout (matches make_bitlinear_kernel() in bitlinear_layers.py):
  packed[row_idx, j] contains four ternary codes packed as 2-bit groups:
    bits 0-1  → code for output feature  row_idx
    bits 2-3  → code for output feature  row_idx +   out_features//4
    bits 4-5  → code for output feature  row_idx + 2*out_features//4
    bits 6-7  → code for output feature  row_idx + 3*out_features//4

  packed shape = (out_features // 4, in_features), dtype uint8

weight_scale is saved as a float32 array of shape (1,).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Ternary packing helpers
# ---------------------------------------------------------------------------

def ternarize(w_f32: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ternary codes ∈ {-1, 0, +1} and global scale for a 2-D weight."""
    scale = float(np.mean(np.abs(w_f32)))
    threshold = 0.5 * scale
    ternary = np.zeros(w_f32.shape, dtype=np.int8)
    ternary[w_f32 >  threshold] =  1
    ternary[w_f32 < -threshold] = -1
    return ternary, scale


def pack_ternary(ternary: np.ndarray) -> np.ndarray:
    """Pack a 2-D int8 ternary array into the BitLinear uint8 kernel layout.

    Input shape:  (out_features, in_features)  — values in {-1, 0, +1}
    Output shape: (out_features // 4, in_features) — packed uint8

    Requires out_features % 4 == 0.
    """
    out_features, in_features = ternary.shape
    assert out_features % 4 == 0, (
        f"out_features ({out_features}) must be divisible by 4"
    )

    # Encode {-1, 0, +1} → {0, 1, 2} so each code fits in 2 bits
    codes = (ternary + 1).astype(np.uint8)

    # Reshape to (4 groups, out_features//4, in_features) using C-order so
    # that group g contains output features g*(N/4) .. (g+1)*(N/4)-1
    codes_r = codes.reshape(4, out_features // 4, in_features)

    # Pack four 2-bit groups into one uint8
    packed = (
        (codes_r[0])
        | (codes_r[1] << 2)
        | (codes_r[2] << 4)
        | (codes_r[3] << 6)
    )
    return packed  # shape (out_features//4, in_features)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

EMBED_PASSTHROUGH = {"model.embed_tokens.weight"}


def convert(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(src_dir.glob("model*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No model*.safetensors files in {src_dir}")

    print(f"Source : {src_dir}")
    print(f"Dest   : {dst_dir}")
    print(f"Shards : {len(shard_paths)}")

    for shard_path in shard_paths:
        print(f"\n  Loading {shard_path.name} …")
        weights_mx = mx.load(str(shard_path))

        out_dict: dict[str, mx.array] = {}
        n_packed = 0
        n_passthrough = 0

        for name, arr in weights_mx.items():
            arr_f32 = np.array(arr.astype(mx.float32))

            if arr_f32.ndim == 2 and name not in EMBED_PASSTHROUGH:
                # Linear weight → ternarize + pack
                out_features, in_features = arr_f32.shape

                if out_features % 4 != 0:
                    # Edge case: pad to nearest multiple of 4
                    pad_rows = 4 - (out_features % 4)
                    arr_f32 = np.pad(arr_f32, ((0, pad_rows), (0, 0)))
                    print(f"    [padded] {name} {(out_features, in_features)} → {arr_f32.shape}")

                ternary, scale = ternarize(arr_f32)
                packed = pack_ternary(ternary)

                out_dict[name] = mx.array(packed)                          # uint8
                # weight_scale must match activation dtype (BF16) so the
                # BitLinear Metal kernel's dtype check passes.
                out_dict[f"{name[:-7]}.weight_scale"] = mx.array(         # bfloat16
                    np.array([scale], dtype=np.float32)
                ).astype(mx.bfloat16)
                n_packed += 1

                if n_packed <= 3 or name == "model.layers.0.self_attn.q_proj.weight":
                    sparsity = float(np.mean(ternary == 0))
                    print(
                        f"    [packed] {name}  "
                        f"{(out_features, in_features)} → {packed.shape} uint8  "
                        f"scale={scale:.4f}  sparsity={sparsity:.2%}"
                    )
            else:
                # Embedding / 1-D tensor → pass through at BF16
                out_dict[name] = arr
                n_passthrough += 1

        dst_path = dst_dir / shard_path.name
        mx.save_safetensors(str(dst_path), out_dict)
        print(f"  Saved  {dst_path.name}  "
              f"({n_packed} packed + {n_passthrough} passthrough, "
              f"{dst_path.stat().st_size / 1e9:.2f} GB)")

    # ------------------------------------------------------------------
    # Copy ancillary files (config, tokenizer, …)
    # ------------------------------------------------------------------
    copy_patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    print("\n  Copying ancillary files …")
    for pattern in copy_patterns:
        for src_file in src_dir.glob(pattern):
            shutil.copy2(src_file, dst_dir / src_file.name)
            print(f"    {src_file.name}")

    print("\nDone.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        default="models/bitnet-b1.58-2B-4T-bf16",
        help="Path to source BF16 model directory",
    )
    p.add_argument(
        "--dst",
        default="models/bitnet-b1.58-2B-4T-mlx",
        help="Path to destination packed model directory",
    )
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.is_absolute():
        src = Path(__file__).parent.parent.parent / args.src

    dst = Path(args.dst).expanduser().resolve()
    if not dst.is_absolute():
        dst = Path(__file__).parent.parent.parent / args.dst

    convert(src, dst)


if __name__ == "__main__":
    main()
