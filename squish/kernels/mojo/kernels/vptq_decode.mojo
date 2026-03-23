"""
vptq_decode.mojo — VPTQ/AQLM codebook gather decode Mojo kernel.

Wave 58b — MojoVPTQDecode

Replaces centroids[indices] fancy-indexing in vptq.py and the AQLM
multi-codebook gather loop `for m in range(n_codebooks): out += cb[m][idx[m]]`
with a SIMD gather of group_size floats per group:

    for g in range(N):
        out[g, :] += centroids[indices[g], :]    (SIMD load group_size floats)

Specialization:
    @parameter on group_size ∈ {2, 4, 8, 16}
    parallelize over N groups
    SIMD[DType.float32, group_size].load for compile-time unrolled gather

Reference:
    Liu et al. (NeurIPS 2024) — VPTQ (arXiv:2409.17066).
    Tseng et al. (ICLR 2024) — AQLM (arXiv:2401.06118).
"""

from algorithm import parallelize, vectorize
from sys.info import simdwidthof

alias float32 = DType.float32


fn vptq_decode[group_size: Int](
    indices: DTypePointer[DType.int32],  # (N,)
    centroids: DTypePointer[float32],    # (K, group_size)
    out: DTypePointer[float32],          # (N, group_size) — output or accumulate
    N: Int,
    K: Int,
    accumulate: Bool,
) -> None:
    """Gather group_size floats from centroids for each index in parallel.

    Args:
        indices:    (N,) int32 centroid index per group.
        centroids:  (K, group_size) codebook entries.
        out:        (N, group_size) output buffer.
        N:          Number of groups.
        K:          Codebook size (number of centroids).
        accumulate: If True, add to existing out values (multi-codebook).
    """

    @parameter
    fn decode_group(g: Int):
        let idx = min(max(indices[g], 0), K - 1)
        let c_ptr = centroids + idx * group_size
        let o_ptr = out + g * group_size

        if accumulate:
            # For residual codebook: accumulate gather
            let loaded = SIMD[float32, group_size].load(c_ptr)
            let existing = SIMD[float32, group_size].load(o_ptr)
            SIMD[float32, group_size].store(o_ptr, existing + loaded)
        else:
            SIMD[float32, group_size].store(o_ptr,
                SIMD[float32, group_size].load(c_ptr))

    parallelize[decode_group](N)


fn vptq_decode_scaled[group_size: Int](
    indices: DTypePointer[DType.int32],  # (N,)
    centroids: DTypePointer[float32],    # (K, group_size)
    col_scales: DTypePointer[float32],   # (group_size,) per-column scales
    out: DTypePointer[float32],          # (N, group_size)
    N: Int,
    K: Int,
) -> None:
    """Gather + fused column scale application (VPTQ col_scale step)."""

    @parameter
    fn decode_scaled_group(g: Int):
        let idx = min(max(indices[g], 0), K - 1)
        let c_ptr = centroids + idx * group_size
        let o_ptr = out + g * group_size
        let s_ptr = col_scales

        let loaded = SIMD[float32, group_size].load(c_ptr)
        let scales = SIMD[float32, group_size].load(s_ptr)
        SIMD[float32, group_size].store(o_ptr, loaded * scales)

    parallelize[decode_scaled_group](N)
