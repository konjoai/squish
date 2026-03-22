"""
rope.mojo — Mojo implementation of Rotary Position Embedding (RoPE).

Reference implementation stub.
"""

from math import cos, sin, pi
from algorithm import vectorize
from sys.info import simdwidthof

alias F32 = DType.float32


fn rope_f32(
    x: DTypePointer[F32],
    out: DTypePointer[F32],
    positions: DTypePointer[DType.int32],
    n_heads: Int,
    head_dim: Int,
    seq_len: Int,
    base: Float32,
) -> None:
    """Apply RoPE rotations in-place.

    For each (position, head) pair, rotates pairs of features in the
    (cos θ, −sin θ; sin θ, cos θ) rotation matrix where θ_i = pos / base^(2i/d).
    """
    let half_dim = head_dim // 2

    for s in range(seq_len):
        let pos = positions[s]
        for h in range(n_heads):
            let base_offset = (s * n_heads + h) * head_dim
            for i in range(half_dim):
                let theta = Float32(pos) / (base ** (Float32(2 * i) / Float32(head_dim)))
                let c = cos(theta)
                let sin_val = sin(theta)
                let j0 = base_offset + i
                let j1 = base_offset + half_dim + i
                let x0 = x[j0]
                let x1 = x[j1]
                out[j0] = x0 * c - x1 * sin_val
                out[j1] = x0 * sin_val + x1 * c
