# rotary_embed.mojo — Rotary Position Embedding (RoPE) Mojo kernel.
#
# Wave 59b — MojoRotaryEmbed
#
# Fuses 6 NumPy dispatches (np.cos, np.sin, split, multiply, negate, concat)
# into one vectorize pass with @parameter on head_dim (64, 128),
# inline 2×2 rotation via SIMD FMA, parallelize(n_heads * T) tasks.
#
# Rotation per pair (r, i):
#     out_r = r * cos - i * sin
#     out_i = r * sin + i * cos
#
# Specialization:
#     @parameter on head_dim ∈ {64, 128}
#     parallelize over n_heads * T tokens
#
# Reference:
#     Su et al. (arXiv 2104.09864, 2023) — RoFormer.

from algorithm import parallelize, vectorize
from sys.info import simdwidthof

alias float32 = DType.float32
alias simd_width = simdwidthof[float32]()


fn apply_rope[head_dim: Int](
    x: DTypePointer[float32],       # (n_heads, T, head_dim)
    cos_vals: DTypePointer[float32], # (T, head_dim // 2)
    sin_vals: DTypePointer[float32], # (T, head_dim // 2)
    output: DTypePointer[float32],  # (n_heads, T, head_dim)
    n_heads: Int,
    T: Int,
):
    """
    Apply rotary embeddings to all (head, token) pairs.

    Each token's head_dim values are split into two halves (real, imag).
    Rotated as: out_r = r * cos - i * sin, out_i = r * sin + i * cos.
    Uses SIMD FMA for the 2×2 rotation and parallelize over (head, token).
    """
    alias half = head_dim // 2

    @parameter
    fn rotate_token(task_id: Int):
        var h = task_id // T
        var t = task_id % T

        var x_ptr = x + (h * T + t) * head_dim
        var out_ptr = output + (h * T + t) * head_dim
        var cos_ptr = cos_vals + t * half
        var sin_ptr = sin_vals + t * half

        # Process real half (indices 0..half)
        @parameter
        fn rot_real[width: Int](i: Int):
            var r = SIMD[float32, width].load(x_ptr + i)
            var im = SIMD[float32, width].load(x_ptr + half + i)
            var c = SIMD[float32, width].load(cos_ptr + i)
            var s = SIMD[float32, width].load(sin_ptr + i)
            SIMD[float32, width].store(out_ptr + i, r * c - im * s)
            SIMD[float32, width].store(out_ptr + half + i, r * s + im * c)
        vectorize[rot_real, simd_width](half)

    parallelize[rotate_token](n_heads * T)
