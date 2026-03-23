"""
sliding_window_attn.mojo — Sliding-window local attention Mojo kernel.

Wave 58b — MojoSlidingWindowAttn

Eliminates the double Python for-loop (for h in heads: for t in T:) in
subgen_attn.py and nsa_attn.py _sliding_window_attn():

    For each (head, token) pair: softmax(Q[h,t] @ K[h,lo:hi].T * scale) @ V[h,lo:hi]
    where lo = max(0, t - window_size + 1), hi = t + 1  (causal sliding window)

Specialization:
    @parameter on window_size ∈ {64, 128, 256}, head_dim ∈ {64, 128}
    parallelize over (n_heads * T) independent (head, token) tasks
    Vectorized Q·K dot product per window slot, SIMD attn×V accumulate

Reference:
    Nawrot et al. (arXiv:2402.06082, 2024) — SubGen sliding-window attention.
"""

from algorithm import parallelize, vectorize
from math import exp
from sys.info import simdwidthof

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn sliding_window_forward[
    window_size: Int,
    head_dim: Int,
](
    Q: DTypePointer[float32],    # (n_heads, T, head_dim)
    K: DTypePointer[float32],
    V: DTypePointer[float32],
    out: DTypePointer[float32],  # (n_heads, T, head_dim)
    n_heads: Int,
    T: Int,
    scale: Float32,
) -> None:
    """Causal sliding-window attention: each token attends to window_size past tokens.

    Parallelizes over (n_heads * T) independent (head, token) tasks.
    """
    alias simd_w = simdwidthof[float32]()
    let total_tasks = n_heads * T

    @parameter
    fn process_task(task_id: Int):
        let h = task_id // T
        let t = task_id % T
        let lo = max(0, t - window_size + 1)
        let hi = t + 1
        let ws = hi - lo

        # Score window Q[h,t] @ K[h,lo:hi].T
        var scores = InlineArray[Float32, window_size](0.0)
        var row_max: Float32 = -1.0e38
        let qr = (h * T + t) * head_dim

        for ki in range(ws):
            let kr = (h * T + lo + ki) * head_dim
            var dot: Float32 = 0.0

            @parameter
            fn dot_v[w: Int](j: Int):
                dot += (
                    SIMD[float32, w].load(Q, qr + j)
                    * SIMD[float32, w].load(K, kr + j)
                ).reduce_add()

            vectorize[dot_v, simd_w](head_dim)
            let s = dot * scale
            scores[ki] = s
            if s > row_max:
                row_max = s

        # Softmax
        var row_sum: Float32 = 0.0
        for ki in range(ws):
            scores[ki] = exp(scores[ki] - row_max)
            row_sum += scores[ki]
        let inv = 1.0 / (row_sum + 1.0e-9)

        # Accumulate attn × V into out[h, t]
        let or_ = (h * T + t) * head_dim
        for ki in range(ws):
            let a = scores[ki] * inv
            let vr = (h * T + lo + ki) * head_dim

            @parameter
            fn acc_v[w: Int](j: Int):
                SIMD[float32, w].store(
                    out, or_ + j,
                    SIMD[float32, w].load(out, or_ + j)
                    + a * SIMD[float32, w].load(V, vr + j),
                )

            vectorize[acc_v, simd_w](head_dim)

    parallelize[process_task](total_tasks)
