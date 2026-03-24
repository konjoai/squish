/*
 * fused_int4_gemm.metal — Tiled INT4 fused GEMM for Squish Wave 67
 *
 * Implements the prefill path for compute-bound batch execution (seq_len ≥ 4).
 * Extends the fused INT4 approach from fused_int4_gemv.metal to a full tiled
 * GEMM without a BF16 staging buffer for the weight tile.  Activations are
 * staged in threadgroup float16 shared memory (activation tile only, never
 * the weight tile), keeping threadgroup occupancy high.
 *
 * Architecture overview
 * ─────────────────────
 * The kernel uses a standard outer-product tile decomposition:
 *
 *   For each K-tile (inner dimension block TILE_K wide):
 *     1. Load TILE_M × TILE_K INT4-packed weight bytes from device memory.
 *     2. Unpack INT4 → FP32 (scale+zero applied per group, in-register).
 *     3. Load TILE_K × TILE_N activation values into threadgroup FP16 scratch.
 *     4. Perform TILE_M × TILE_N outer-product accumulation into FP32 registers.
 *     5. Repeat.
 *   Write FP32 output tile.
 *
 * No BF16/FP16 weight staging — weights stay INT4 until dequantised inside
 * the multiply loop.  This halves the effective weight bytes in device memory
 * compared with the BF16-staging path.
 *
 * Tile dimensions
 * ───────────────
 *   TILE_M   = 64  — output rows per threadgroup (weight matrix rows)
 *   TILE_N   = 16  — output columns per threadgroup (sequence steps)
 *   TILE_K   = 64  — inner dimension per tile step
 *   THREADS  = TILE_M (64 threads x 1-D grid; each thread owns one output row)
 *
 * Buffer layout
 * ─────────────
 *   buffer(0) weights_packed : uint8[n_rows * n_cols / 2]  INT4-packed row-major
 *   buffer(1) scales         : float[n_rows * n_groups]    per-group FP32 scales
 *   buffer(2) zeros          : float[n_rows * n_groups]    per-group FP32 zeros
 *   buffer(3) input_mat      : float[n_cols * seq_len]     row-major activations
 *                              Layout: input_mat[col, tok] = input[col + tok*n_cols]
 *   buffer(4) output_mat     : float[n_rows * seq_len]     row-major
 *   buffer(5) params         : FusedInt4GEMMParams
 *
 * Dispatch
 * ────────
 *   grid    : (ceil(n_rows / TILE_M), ceil(seq_len / TILE_N), 1)
 *   threads : (THREADS, 1, 1)  — THREADS = TILE_M = 64
 *
 * Activation scratch
 * ──────────────────
 *   threadgroup half act_tile[TILE_K][TILE_N]
 *   Size: 64 × 16 × 2 = 2 KB — 16× within the 32 KB threadgroup budget.
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Tile constants
// ---------------------------------------------------------------------------

constant uint TILE_M   = 64;   // output rows per threadgroup
constant uint TILE_N   = 16;   // output columns per threadgroup (seq steps)
constant uint TILE_K   = 64;   // inner dimension per GEMM tile step
constant uint THREADS  = 64;   // threads per threadgroup = TILE_M

// ---------------------------------------------------------------------------
// Parameter struct
// ---------------------------------------------------------------------------

struct FusedInt4GEMMParams {
    uint n_rows;       // weight matrix rows (= output features)
    uint n_cols;       // weight matrix columns (= input features); must be even
    uint seq_len;      // number of input tokens (GEMM N dimension)
    uint group_size;   // INT4 quantisation group size (typically 32 or 64)
};

// ---------------------------------------------------------------------------
// Fused INT4 GEMM kernel
// ---------------------------------------------------------------------------

kernel void fused_int4_gemm(
    device  const uint8_t*             weights_packed  [[buffer(0)]],
    device  const float*               scales          [[buffer(1)]],
    device  const float*               zeros           [[buffer(2)]],
    device  const float*               input_mat       [[buffer(3)]],
    device        float*               output_mat      [[buffer(4)]],
    constant FusedInt4GEMMParams&      params          [[buffer(5)]],
    threadgroup  half*                 act_tile        [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],  // x=row_tile, y=col_tile
    uint  tid    [[thread_index_in_threadgroup]])    // 0 … TILE_M-1
{
    const uint row_tile = tg_pos.x;
    const uint col_tile = tg_pos.y;

    const uint row_base = row_tile * TILE_M;
    const uint col_base = col_tile * TILE_N;

    const uint row = row_base + tid;

    const uint n_rows     = params.n_rows;
    const uint n_cols     = params.n_cols;
    const uint seq_len    = params.seq_len;
    const uint group_size = params.group_size;
    const uint n_groups   = (n_cols + group_size - 1) / group_size;
    const uint packed_row_stride = n_cols / 2;

    // Per-thread output accumulators: one per TILE_N output column
    float acc[TILE_N];
    for (uint n = 0; n < TILE_N; ++n) {
        acc[n] = 0.0f;
    }

    // Tile over the K (inner) dimension
    for (uint k_base = 0; k_base < n_cols; k_base += TILE_K) {
        uint k_end = min(k_base + TILE_K, n_cols);

        // --- Load activation tile into threadgroup memory ---
        // act_tile[k - k_base][n] = input_mat[k, col_base + n]
        // Using all TILE_M threads to cooperatively load TILE_K × TILE_N floats.
        for (uint k = k_base + tid; k < k_end; k += THREADS) {
            for (uint n = 0; n < TILE_N; ++n) {
                uint col_idx = col_base + n;
                half val = (col_idx < seq_len) ? half(input_mat[k * seq_len + col_idx]) : 0.0h;
                act_tile[(k - k_base) * TILE_N + n] = val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Fused INT4 decode + outer-product accumulate ---
        if (row < n_rows) {
            for (uint k_pair = k_base; k_pair < k_end; k_pair += 2) {
                uint packed_idx  = row * packed_row_stride + k_pair / 2;
                uint8_t packed   = weights_packed[packed_idx];
                uint w0_int      = (packed >> 4) & 0xFu;
                uint w1_int      =  packed       & 0xFu;

                uint g0 = k_pair       / group_size;
                uint g1 = (k_pair + 1) / group_size;
                float s0 = scales[row * n_groups + g0];
                float z0 = zeros [row * n_groups + g0];
                float s1 = scales[row * n_groups + g1];
                float z1 = zeros [row * n_groups + g1];

                float w0 = s0 * float(w0_int) + z0;
                float w1 = s1 * float(w1_int) + z1;

                uint local_k0 = k_pair       - k_base;
                uint local_k1 = (k_pair + 1) - k_base;

                for (uint n = 0; n < TILE_N; ++n) {
                    acc[n] += w0 * float(act_tile[local_k0 * TILE_N + n]);
                    if ((k_pair + 1) < k_end) {
                        acc[n] += w1 * float(act_tile[local_k1 * TILE_N + n]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Write output tile ---
    if (row < n_rows) {
        for (uint n = 0; n < TILE_N; ++n) {
            uint col_idx = col_base + n;
            if (col_idx < seq_len) {
                output_mat[row * seq_len + col_idx] = acc[n];
            }
        }
    }
}
