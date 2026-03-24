/*
 * sparse_gemv.metal — Cluster-Masked Sparse GEMV for Squish Wave 66
 *
 * Implements a sparse "cluster-skip + dot-product" kernel for the decode
 * inference step (seq_len == 1) on FFN layers whose neurons have been
 * reordered by co-activation cluster (see cluster_reorder.py).
 *
 * The DejaVu / PowerInfer observation is that 40–65 % of SwiGLU FFN
 * neurons produce near-zero activations for any given token.  After
 * calibration the neurons are grouped into n_clusters clusters; a
 * lightweight linear predictor (sparsity_predictor.py) tells the runtime
 * which clusters are active *before* the GEMV.  This kernel reads that
 * per-cluster active mask and skips entire column ranges for inactive
 * clusters — zero weight bytes loaded for those clusters.
 *
 * Correctness relies on ClusterReorder.reorder() having sorted both W_up /
 * W_gate columns and W_down rows into the same cluster order, so the
 * dense-equivalent result is reproduced exactly for active clusters and
 * zero contribution is lost for inactive clusters.
 *
 * Buffer layout
 * ─────────────
 *   buffer(0) weights_flat    : float32 weight matrix, row-major
 *                               Shape (n_rows × n_cols) after cluster reorder.
 *                               Columns belonging to cluster c span
 *                               [cluster_offsets[c], cluster_offsets[c+1]).
 *   buffer(1) cluster_offsets : uint32 array, length = n_clusters + 1
 *                               Column offsets in the reordered layout.
 *                               cluster_offsets[n_clusters] == n_cols.
 *   buffer(2) active_mask     : uchar array, length = n_clusters
 *                               active_mask[c] == 1  → cluster c is active
 *                               active_mask[c] == 0  → skip cluster c
 *   buffer(3) input_vec       : float32 input vector, length n_cols
 *   buffer(4) output          : float32 output vector, length n_rows
 *   buffer(5) params          : SparseGEMVParams struct
 *
 * Dispatch
 * ────────
 *   grid    : (n_rows, 1, 1)   — one threadgroup per output row
 *   threads : (256, 1, 1)      — 256 threads per threadgroup
 *
 * Expected savings
 * ────────────────
 *   With ~50 % of clusters predicted inactive, roughly half the weight bytes
 *   are never loaded from device memory, cutting FFN memory-bandwidth cost
 *   by ~half.  Predicate overhead is O(n_clusters) per row, negligible
 *   compared to the GEMV itself.
 *
 * Limitations
 * ───────────
 *   • Weights are float32 (uncompressed baseline; TCA-TBE integration is a
 *     future wave).
 *   • Only the up-projection (W_up) and gate-projection (W_gate) benefit
 *     from sparsity; W_down is always fully evaluated.  Callers should
 *     invoke this kernel separately for W_up and W_gate.
 *   • The cluster skip is per-output-row granularity but since all rows
 *     share the *same* active mask the cluster skips are coherent across
 *     rows, ensuring good warp/simd-group efficiency.
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constant uint THREADS_PER_TG = 256;

// ---------------------------------------------------------------------------
// Parameter struct (matches Python-side struct.pack layout)
// ---------------------------------------------------------------------------

struct SparseGEMVParams {
    uint n_rows;       // rows in weight matrix (output dimension)
    uint n_cols;       // columns in weight matrix (input / neuron dimension)
    uint n_clusters;   // number of co-activation clusters
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/*
 * sparse_gemv_f32
 *
 * Each threadgroup computes one output element output[row].
 * Threads cooperate over the *active* column ranges only; inactive
 * cluster ranges are skipped via early-continue.
 *
 *   row  = threadgroup index in the grid (== output row index)
 *   tid  = thread index within threadgroup (0 … 255)
 *
 * Thread i accumulates columns  c_start + i, c_start + i + 256, …
 * for each active cluster [c_start, c_end).
 */
kernel void sparse_gemv_f32(
    device const float*            weights_flat   [[buffer(0)]],
    device const uint*             cluster_offsets [[buffer(1)]],
    device const uchar*            active_mask     [[buffer(2)]],
    device const float*            input_vec       [[buffer(3)]],
    device       float*            output          [[buffer(4)]],
    constant     SparseGEMVParams& params          [[buffer(5)]],
    threadgroup  float*            tg_accum        [[threadgroup(0)]],
    uint                           tid             [[thread_position_in_threadgroup]],
    uint                           row             [[threadgroup_position_in_grid]]
)
{
    // Guard against over-dispatched threadgroups.
    if (row >= params.n_rows) { return; }

    // Base pointer for this row's weight slice.
    device const float* w_row = weights_flat + (uint64_t)row * params.n_cols;

    // Each thread accumulates its own partial sum.
    float partial = 0.0f;

    for (uint c = 0; c < params.n_clusters; ++c) {
        // Skip inactive clusters — no weight or input bytes loaded.
        if (active_mask[c] == 0) { continue; }

        uint col_start = cluster_offsets[c];
        uint col_end   = cluster_offsets[c + 1];

        // Stride over the active cluster's column range.
        for (uint col = col_start + tid; col < col_end; col += THREADS_PER_TG) {
            partial += w_row[col] * input_vec[col];
        }
    }

    // Store thread partial sum into threadgroup shared memory.
    tg_accum[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction: halving tree within threadgroup.
    for (uint stride = THREADS_PER_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_accum[tid] += tg_accum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes the fully-reduced dot product for this row.
    if (tid == 0) {
        output[row] = tg_accum[0];
    }
}

// ---------------------------------------------------------------------------
// Dense fallback (all clusters active) — used for correctness testing and
// layers where the predictor is disabled.
// ---------------------------------------------------------------------------

/*
 * dense_gemv_f32
 *
 * Standard dense GEMV.  Same dispatch shape as sparse_gemv_f32 so the
 * caller can switch kernels without changing the grid/threadgroup sizes.
 */
kernel void dense_gemv_f32(
    device const float*            weights_flat   [[buffer(0)]],
    device const float*            input_vec      [[buffer(1)]],
    device       float*            output         [[buffer(2)]],
    constant     SparseGEMVParams& params         [[buffer(3)]],
    threadgroup  float*            tg_accum       [[threadgroup(0)]],
    uint                           tid            [[thread_position_in_threadgroup]],
    uint                           row            [[threadgroup_position_in_grid]]
)
{
    if (row >= params.n_rows) { return; }

    device const float* w_row = weights_flat + (uint64_t)row * params.n_cols;

    float partial = 0.0f;
    for (uint col = tid; col < params.n_cols; col += THREADS_PER_TG) {
        partial += w_row[col] * input_vec[col];
    }

    tg_accum[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS_PER_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_accum[tid] += tg_accum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = tg_accum[0];
    }
}
