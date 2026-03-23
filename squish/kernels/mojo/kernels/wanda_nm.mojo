# squish/kernels/mojo/kernels/wanda_nm.mojo
# Wanda N:M structured sparsity: importance scoring + mask generation.
#
# importance_kernel: element-wise |W| × rms, parallelised over rows.
# mask_kernel:       N:M block mask — keep top-N per M-column block,
#                    parallelised over rows with vectorised partial sort.
#
# Reference: Sun et al., "A Simple and Effective Pruning Approach for
# Large Language Models." ICLR 2024.

from algorithm import parallelize, vectorize


fn wanda_nm_importance_kernel(
    w_ptr: UnsafePointer[Float32],     # (rows * cols,)  FP32 weight matrix (row-major)
    rms_ptr: UnsafePointer[Float32],   # (cols,)          FP32 per-column activation RMS
    out_ptr: UnsafePointer[Float32],   # (rows * cols,)  FP32 importance output
    rows: Int,
    cols: Int,
):
    """Compute per-element Wanda importance: |W[r,c]| × rms[c]."""

    alias SIMD_W = 8

    @parameter
    fn compute_row(row: Int):
        var row_off = row * cols

        @parameter
        fn process_block[width: Int](c: Int):
            @parameter
            for i in range(width):
                var idx = row_off + c + i
                var v = w_ptr[idx]
                out_ptr[idx] = v if v >= 0.0 else -v
                out_ptr[idx] = out_ptr[idx] * rms_ptr[c + i]

        vectorize[process_block, SIMD_W](cols)

    parallelize[compute_row](rows)


fn wanda_nm_mask_kernel(
    imp_ptr: UnsafePointer[Float32],   # (rows * cols,) FP32 importance
    out_ptr: UnsafePointer[UInt8],     # (rows * cols,) UInt8 mask output
    rows: Int,
    cols: Int,
    n: Int,
    m: Int,
):
    """Generate N:M structured sparsity mask.

    For each row, for each column block of size m, keeps the top-n
    entries by importance value (1 = keep, 0 = prune).
    """
    var n_blocks = (cols + m - 1) // m

    @parameter
    fn process_row(row: Int):
        var row_off = row * cols
        for bi in range(n_blocks):
            var col_start = bi * m
            var col_end = col_start + m
            if col_end > cols:
                col_end = cols
            var block_w = col_end - col_start

            # Identify top-n indices in this block via selection
            # (small m, so linear scan is fine)
            var n_keep = n if n <= block_w else block_w
            for keep_i in range(n_keep):
                var best_idx = col_start
                var best_val = imp_ptr[row_off + col_start]
                for c in range(col_start + 1, col_end):
                    var v = imp_ptr[row_off + c]
                    if v > best_val:
                        best_val = v
                        best_idx = c
                out_ptr[row_off + best_idx] = 1
                # Mask out selected entry so it isn't picked again
                imp_ptr[row_off + best_idx] = -1.0e30

    parallelize[process_row](rows)
