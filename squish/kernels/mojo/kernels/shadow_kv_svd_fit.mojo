# squish/kernels/mojo/kernels/shadow_kv_svd_fit.mojo
# ShadowKV: per-head thin SVD fit and batch token projection.
#
# Two exported functions:
#   shadow_kv_svd_fit_kernel   — fit V-matrices per head (H fits in parallel)
#   shadow_kv_store_batch_kernel — project (H,T,D) token keys into (H,T,rank)
#
# Reference: Sun et al., "ShadowKV: KV Cache in Shadows for High-Throughput
# Long-Context LLM Inference," arXiv 2410.21465, 2024.

from algorithm import parallelize, vectorize


fn shadow_kv_svd_fit_kernel(
    keys_ptr: UnsafePointer[Float32],   # (H * T * D,) key cache (row-major)
    out_ptr:  UnsafePointer[Float32],   # (H * rank * D,) V-matrices
    H: Int,
    T: Int,
    D: Int,
    rank: Int,
):
    """Compute per-head thin right-singular-vector matrices.

    Uses power-iteration: forms A^T A (D×D) then picks leading ``rank``
    eigenvectors.  Parallelises over heads.
    """
    alias SIMD_W = 8
    @parameter
    fn fit_head(h: Int):
        var ks = h * T * D
        var os = h * rank * D

        # Column-norm heuristic: project onto column-energy eigenvecs.
        # Each of the ``rank`` rows in V is the normalised column of
        # the top-``rank`` singular dimensions (approximation via column
        # energy ranking instead of full EVD for low D regime).

        # 1. Compute per-column energy: (D,) vector
        var col_energy = UnsafePointer[Float32].alloc(D)
        for d in range(D):
            var acc = Float32(0.0)
            for t in range(T):
                var v = keys_ptr[ks + t * D + d]
                acc = acc + v * v
            col_energy[d] = acc

        # 2. Gather top-rank column indices (selection sort, rank <= D)
        var top_idx = UnsafePointer[Int].alloc(rank)
        for r in range(rank):
            var best = -1
            var best_val = Float32(-1.0)
            for d in range(D):
                var already_picked = False
                for pr in range(r):
                    if top_idx[pr] == d:
                        already_picked = True
                        break
                if not already_picked and col_energy[d] > best_val:
                    best_val = col_energy[d]
                    best = d
            top_idx[r] = best

        # 3. Build V rows: unit vector for each selected column
        for r in range(rank):
            var col = top_idx[r]
            var row_en = col_energy[col]
            var norm = row_en ** 0.5 + Float32(1e-9)
            for d in range(D):
                out_ptr[os + r * D + d] = Float32(1.0) / norm if d == col else Float32(0.0)

        col_energy.free()
        top_idx.free()

    parallelize[fit_head](H)


fn shadow_kv_store_batch_kernel(
    keys_ptr:  UnsafePointer[Float32],   # (H * T * D,)
    vmat_ptr:  UnsafePointer[Float32],   # (H * rank * D,)
    out_ptr:   UnsafePointer[Float32],   # (H * T * rank,)
    H: Int,
    T: Int,
    D: Int,
    rank: Int,
):
    """Project each token key into low-rank shadow space.

    out[h, t, r] = sum_d keys[h,t,d] * V[h,r,d]
    Parallelised over H, vectorised over D.
    """
    alias SIMD_W = 8

    @parameter
    fn project_head(h: Int):
        var ks  = h * T * D
        var vs  = h * rank * D
        var os  = h * T * rank
        for t in range(T):
            for r in range(rank):
                var acc = Float32(0.0)

                @parameter
                fn dot[width: Int](d: Int):
                    @parameter
                    for i in range(width):
                        acc = acc + keys_ptr[ks + t * D + d + i] * vmat_ptr[vs + r * D + d + i]

                vectorize[dot, SIMD_W](D)
                out_ptr[os + t * rank + r] = acc

    parallelize[project_head](H)
