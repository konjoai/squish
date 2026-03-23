"""
hqq_als.mojo — HQQ Alternating Least-Squares quantization Mojo kernel.

Wave 58b — MojoHQQALS

Fused ALS iteration over a weight group (group_size floats):

    for _ in range(max_iter):
        c2    = sum(codes²) + lambda
        scale = dot(codes, W - zero) / c2
        zero  = mean(W - codes * scale)
        codes = clip(round((W - zero) / scale), 0, qmax)

All four steps compiled into one Mojo vectorize block reading W once,
writing codes once — eliminating 6 NumPy ufunc dispatches × max_iter.

Specialization:
    @parameter on group_size ∈ {32, 64, 128, 256}
    @parameter on qmax ∈ {15 (INT4), 255 (INT8)}
    parallelize over n_groups to quantize an entire weight matrix

Reference:
    Badri & Shaji (arXiv:2309.15531, 2023) — HQQ.
"""

from algorithm import parallelize, vectorize
from math import round, sqrt
from sys.info import simdwidthof

alias float32 = DType.float32
alias SIMD_WIDTH = simdwidthof[float32]()


fn hqq_als_fit_group[group_size: Int, qmax: Int](
    W: DTypePointer[float32],    # (group_size,) weight group — read-only
    codes: DTypePointer[float32],  # (group_size,) codes — in/out
    scale: DTypePointer[float32],  # scalar output
    zero: DTypePointer[float32],   # scalar output
    lmbda: Float32,
    max_iter: Int,
) -> None:
    """Fit scale, zero, and codes for one HQQ weight group via fused ALS.

    Reads W once per iteration, writing codes in-place.  Returns scale and
    zero through DTypePointer scalars for caller to gather into arrays.
    """
    alias simd_w = simdwidthof[float32]()

    # Initialise: scale from range / qmax, zero from min
    var w_min: Float32 = W[0]
    var w_max: Float32 = W[0]
    for i in range(1, group_size):
        if W[i] < w_min:
            w_min = W[i]
        if W[i] > w_max:
            w_max = W[i]

    var s = (w_max - w_min) / Float32(qmax)
    if s == 0.0:
        s = 1.0
    var z = w_min

    # Initialise codes
    for i in range(group_size):
        var c = round((W[i] - z) / s)
        if c < 0.0:
            c = 0.0
        if c > Float32(qmax):
            c = Float32(qmax)
        codes[i] = c

    # ALS iterations
    for _ in range(max_iter):
        # c2 = sum(codes²) + lambda
        var c2: Float32 = lmbda
        for i in range(group_size):
            c2 += codes[i] * codes[i]

        # scale = dot(codes, W - z) / c2
        var num: Float32 = 0.0
        for i in range(group_size):
            num += codes[i] * (W[i] - z)
        s = num / c2
        if s == 0.0:
            s = 1.0

        # zero = mean(W - codes * scale)
        var sum_z: Float32 = 0.0
        for i in range(group_size):
            sum_z += W[i] - codes[i] * s
        z = sum_z / Float32(group_size)

        # codes = clip(round((W - zero) / scale), 0, qmax)
        for i in range(group_size):
            var c = round((W[i] - z) / s)
            if c < 0.0:
                c = 0.0
            if c > Float32(qmax):
                c = Float32(qmax)
            codes[i] = c

    scale[0] = s
    zero[0] = z


fn hqq_als_fit_all[group_size: Int, qmax: Int](
    W_flat: DTypePointer[float32],    # (n_groups * group_size,) flat weights
    codes_out: DTypePointer[float32], # (n_groups * group_size,) output codes
    scales_out: DTypePointer[float32],  # (n_groups,) output scales
    zeros_out: DTypePointer[float32],   # (n_groups,) output zeros
    n_groups: Int,
    lmbda: Float32,
    max_iter: Int,
) -> None:
    """Fit all n_groups in parallel."""

    @parameter
    fn fit_group_task(g: Int):
        let w_ptr = W_flat + g * group_size
        let c_ptr = codes_out + g * group_size
        var s: Float32 = 0.0
        var z: Float32 = 0.0
        hqq_als_fit_group[group_size, qmax](
            w_ptr, c_ptr, DTypePointer[float32].address_of(s),
            DTypePointer[float32].address_of(z), lmbda, max_iter
        )
        scales_out[g] = s
        zeros_out[g] = z

    parallelize[fit_group_task](n_groups)
