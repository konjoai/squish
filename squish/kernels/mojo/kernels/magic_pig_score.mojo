# squish/kernels/mojo/kernels/magic_pig_score.mojo
# MagicPIG LSH-bucketed attention score computation kernel.
#
# Parallelises over H heads.
# Reference: He et al., arXiv 2410.16179, 2024.

from algorithm import parallelize
from math import sqrt


fn magic_pig_score_kernel(
    q_ptr:    UnsafePointer[Float32],
    k_ptr:    UnsafePointer[Float32],
    v_ptr:    UnsafePointer[Float32],
    out_ptr:  UnsafePointer[Float32],
    h:        Int,
    tq:       Int,
    seq_len:  Int,
    d:        Int,
):
    # Parallel over heads; sequential query loop per head; softmax + GEMV over V.
    @parameter
    fn process_head(hi: Int):
        var inv_scale = Float32(1.0) / sqrt(Float32(d))
        var logits = UnsafePointer[Float32].alloc(seq_len)
        for qi in range(tq):
            for si in range(seq_len):
                var dot = Float32(0.0)
                for j in range(d):
                    dot += q_ptr[hi * tq * d + qi * d + j] * k_ptr[hi * seq_len * d + si * d + j]
                logits[si] = dot * inv_scale
            var max_l = logits[0]
            for si in range(1, seq_len):
                if logits[si] > max_l:
                    max_l = logits[si]
            var sum_exp = Float32(0.0)
            for si in range(seq_len):
                var x = logits[si] - max_l
                var ex = Float32(1.0)
                var term = Float32(1.0)
                for tk in range(1, 12):
                    term = term * x / Float32(tk)
                    ex += term
                ex = ex if ex > Float32(0.0) else Float32(0.0)
                logits[si] = ex
                sum_exp += ex
            if sum_exp < Float32(1e-8):
                sum_exp = Float32(1e-8)
            for si in range(seq_len):
                logits[si] = logits[si] / sum_exp
            for j in range(d):
                out_ptr[hi * tq * d + qi * d + j] = Float32(0.0)
            for si in range(seq_len):
                for j in range(d):
                    out_ptr[hi * tq * d + qi * d + j] += logits[si] * v_ptr[hi * seq_len * d + si * d + j]
        logits.free()

    parallelize[process_head](h)
