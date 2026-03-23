# squish/kernels/mojo/kernels/cake_entropy.mojo
#
# CAKE per-head normalised entropy — Mojo stub.
#
# For each attention head h the kernel:
#   1. Iterates over obs_window observation queries.
#   2. Computes dot-product attention scores against all T key tokens.
#   3. Applies softmax then computes Shannon entropy normalised by ln(T).
#   4. Averages the per-head entropy over obs_window.
#
# Parallelism:
#   - Outer `parallelize[compute_head](n_heads)` — one task per head.
#   - Inner dot product uses `vectorize[dot_elem, SIMD_W](head_dim)`.
#
# Mojo version target: MAX 25.x (UnsafePointer API).

from algorithm import parallelize, vectorize
from math import exp, log, sqrt, max
from memory import memset_zero

alias SIMD_W: Int = 8


fn cake_entropy_f32(
    q_obs_ptr: UnsafePointer[Float32],  # [obs_window, n_heads, head_dim]
    k_ptr: UnsafePointer[Float32],      # [T, n_heads, head_dim]
    entropy_ptr: UnsafePointer[Float32], # [n_heads] — output
    n_heads: Int,
    head_dim: Int,
    obs_window: Int,
    T: Int,
    temperature: Float32,
) -> None:
    """Compute normalised per-head Shannon entropy for CAKE eviction."""
    let inv_scale = 1.0 / (sqrt(Float32(head_dim)) * temperature)
    let log_T: Float32 = log(Float32(max(T, 2)))  # guard log(1)==0

    @parameter
    fn compute_head(h: Int):
        var head_ent: Float32 = 0.0

        for q_pos in range(obs_window):
            # --- dot products: scores[t] = q_obs[q_pos, h, :] @ k[t, h, :] ---
            let q_off = (q_pos * n_heads + h) * head_dim
            var scores = UnsafePointer[Float32].alloc(T)
            var max_score: Float32 = -3.4e38

            for t in range(T):
                let k_off = (t * n_heads + h) * head_dim
                var acc: Float32 = 0.0

                @parameter
                fn dot_elem[w: Int](i: Int):
                    acc += q_obs_ptr[q_off + i] * k_ptr[k_off + i]

                vectorize[dot_elem, SIMD_W](head_dim)
                let score = acc * inv_scale
                scores[t] = score
                if score > max_score:
                    max_score = score

            # --- softmax ---
            var sum_exp: Float32 = 0.0
            for t in range(T):
                scores[t] = exp(scores[t] - max_score)
                sum_exp += scores[t]
            for t in range(T):
                scores[t] = scores[t] / sum_exp

            # --- Shannon entropy, normalised by ln(T) ---
            var ent: Float32 = 0.0
            for t in range(T):
                let p = scores[t]
                if p > 1e-12:
                    ent -= p * log(p)
            ent = ent / log_T

            head_ent += ent
            scores.free()

        entropy_ptr[h] = head_ent / Float32(obs_window)

    parallelize[compute_head](n_heads)
