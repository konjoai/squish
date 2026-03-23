# squish/kernels/mojo/kernels/green_kv_score.mojo
# GreenKV per-head KV-cache importance scoring.
#
# For each head, computes mean softmax attention weights over a sliding
# observation window of query vectors against the full key cache.
# Parallelises over heads; SIMD-vectorised dot-product per query-key pair.
#
# Reference: Zhang et al., "GreenKV: Achieving KV Cache Compression with
# Nearly-Lossless Accuracy for Large Language Models," 2024.

from algorithm import parallelize, vectorize
import math


fn green_kv_score_kernel(
    q_ptr: UnsafePointer[Float32],    # (n_heads * obs_window * head_dim,) FP32
    k_ptr: UnsafePointer[Float32],    # (n_heads * seq_len * head_dim,)    FP32
    out_ptr: UnsafePointer[Float32],  # (n_heads * seq_len,)               FP32
    n_heads: Int,
    obs_window: Int,
    seq_len: Int,
    head_dim: Int,
):
    """Compute mean per-token softmax attention scores over obs_window."""
    alias SIMD_W = 8
    var scale = Float32(1.0) / math.sqrt(Float32(head_dim))

    @parameter
    fn score_head(h: Int):
        var q_head_off = h * obs_window * head_dim
        var k_head_off = h * seq_len * head_dim
        var out_head_off = h * seq_len

        # Accumulate mean scores across obs_window
        for qp in range(obs_window):
            var q_off = q_head_off + qp * head_dim

            # Compute logits[s] = q[qp] · k[s] * scale for all s
            var max_logit = Float32(-1.0e30)
            for s in range(seq_len):
                var k_off = k_head_off + s * head_dim
                var dot = Float32(0.0)

                @parameter
                fn dot_elem[width: Int](d: Int):
                    @parameter
                    for i in range(width):
                        dot += q_ptr[q_off + d + i] * k_ptr[k_off + d + i]

                vectorize[dot_elem, SIMD_W](head_dim)
                var logit = dot * scale
                if logit > max_logit:
                    max_logit = logit

                # Store logit temporarily in output (will normalise next)
                out_ptr[out_head_off + s] += logit  # placeholder accumulate

            # Softmax normalisation (re-pass)
            var sum_exp = Float32(0.0)
            for s in range(seq_len):
                var e = math.exp(out_ptr[out_head_off + s] / Float32(obs_window) - max_logit)
                out_ptr[out_head_off + s] = e
                sum_exp += e
            var inv_sum = Float32(1.0) / (sum_exp + Float32(1e-9))
            for s in range(seq_len):
                out_ptr[out_head_off + s] *= inv_sum / Float32(obs_window)

    parallelize[score_head](n_heads)
