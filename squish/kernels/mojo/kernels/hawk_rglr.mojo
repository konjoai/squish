# squish/kernels/mojo/kernels/hawk_rglr.mojo
# Hawk Real-Gated Linear Recurrence (RGLR) scan kernel.
#
# Implements: h_t = f_t ⊙ h_{t-1} + i_t ⊙ x_t
#   f_t = exp(-exp(λ) * softplus(dt_t)),  i_t = sqrt(1 - f_t²)
# Parallelises over d_state channels.
#
# Reference: De et al., "Griffin." arXiv 2402.19427 / NeurIPS 2024.

from algorithm import parallelize, vectorize
from math import exp, log, sqrt


fn hawk_rglr_kernel(
    x_ptr: UnsafePointer[Float32],      # (T * d_state) input
    dt_ptr: UnsafePointer[Float32],     # (T * d_state) delta-time
    lam_ptr: UnsafePointer[Float32],    # (d_state,) log-eigenvalues
    h_ptr: UnsafePointer[Float32],      # (d_state,) state — updated in place
    out_ptr: UnsafePointer[Float32],    # (T * d_state) output
    t_len: Int,
    d_state: Int,
):
    for t in range(t_len):
        let off = t * d_state

        @parameter
        fn update_channel(i: Int):
            let dt_val = dt_ptr[off + i]
            # softplus(dt) = log(1 + exp(dt))
            let sp = log(1.0 + exp(dt_val))
            let decay = exp(-exp(lam_ptr[i]) * sp)
            let decay_sq = decay * decay
            let i_gate = sqrt(Float32(1.0) - decay_sq if decay_sq < 1.0 else Float32(0.0))
            let new_h = decay * h_ptr[i] + i_gate * x_ptr[off + i]
            h_ptr[i] = new_h
            out_ptr[off + i] = new_h

        parallelize[update_channel](d_state)
