/*
 * fused_int4_gemv.metal — Fused INT4 decode+accumulate GEMV for Squish Wave 67
 *
 * Implements a single-pass vector-matrix multiply where INT4-packed weight
 * bytes are read, unpacked in-register, scaled, and accumulated without ever
 * writing a BF16/FP16 staging buffer to device or threadgroup memory.
 *
 * Design goal
 * ───────────
 * The previous Squish INT4 path dequantised weights to a BF16 staging tensor
 * in shared memory before running the matmul — each weight byte was read
 * twice.  This kernel eliminates that second pass: 2 × INT4 values are
 * extracted per byte via bitwise shift-and-mask, scaled by the group's FP32
 * scale+zero, and accumulated directly in a FP32 register lane.
 *
 * Format
 * ──────
 * Weight packing (matches squish.convert INT4 asymmetric MSE format):
 *   packed_byte[i] = (w[2i] << 4) | (w[2i+1] & 0xF)
 *   w0 = (packed_byte >> 4) & 0xF   — high nibble
 *   w1 =  packed_byte       & 0xF   — low nibble
 * Values are asymmetric unsigned 4-bit: dequantised as
 *   w_float = scale * w_int + zero
 * where scale and zero are FP32 per group.
 *
 * Buffer layout
 * ─────────────
 *   buffer(0) weights_packed : uint8[n_rows * n_cols / 2]
 *                              INT4-packed row-major, high nibble first.
 *                              n_cols MUST be even.
 *   buffer(1) scales         : float[n_rows * n_groups]   per-group FP32 scales
 *   buffer(2) zeros          : float[n_rows * n_groups]   per-group FP32 zeros
 *   buffer(3) input_vec      : float[n_cols]              FP32 input vector
 *   buffer(4) output         : float[n_rows]              FP32 output vector
 *   buffer(5) params         : FusedInt4GEMVParams
 *
 * Dispatch
 * ────────
 *   1-D grid  : (n_rows, 1, 1)    one threadgroup per output row
 *   threads   : (THREADS_PER_TG, 1, 1)   128 threads per threadgroup
 *
 * Shared memory
 * ─────────────
 *   threadgroup float accum_scratch[THREADS_PER_TG]
 *   Size: 128 × 4 = 512 bytes — well within the 32 KB threadgroup budget.
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constant uint THREADS_PER_TG = 128;

// ---------------------------------------------------------------------------
// Parameter struct
// ---------------------------------------------------------------------------

struct FusedInt4GEMVParams {
    uint n_rows;       // weight matrix rows (= output dim)
    uint n_cols;       // weight matrix columns (= input dim); must be even
    uint group_size;   // INT4 quantisation group size (typically 32 or 64)
};

// ---------------------------------------------------------------------------
// GEMV kernel — one threadgroup per output row
// ---------------------------------------------------------------------------

kernel void fused_int4_gemv(
    device const uint8_t*           weights_packed  [[buffer(0)]],
    device const float*             scales          [[buffer(1)]],
    device const float*             zeros           [[buffer(2)]],
    device const float*             input_vec       [[buffer(3)]],
    device       float*             output          [[buffer(4)]],
    constant FusedInt4GEMVParams&   params          [[buffer(5)]],
    threadgroup  float*             accum_scratch   [[threadgroup(0)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]])
{
    if (tg_id >= params.n_rows) {
        return;
    }

    const uint row        = tg_id;
    const uint n_cols     = params.n_cols;
    const uint group_size = params.group_size;
    const uint n_groups   = (n_cols + group_size - 1) / group_size;

    // Each INT4-packed byte holds 2 weights; packed row stride in bytes
    const uint packed_row_stride = n_cols / 2;  // n_cols is always even

    float partial = 0.0f;

    // Each thread accumulates columns {tid, tid + THREADS_PER_TG, ...}
    // Elements are processed in pairs (two per packed byte).
    for (uint col_pair = tid; col_pair < n_cols / 2; col_pair += THREADS_PER_TG) {
        uint col0 = col_pair * 2;
        uint col1 = col0 + 1;

        // Load packed byte
        uint8_t packed = weights_packed[row * packed_row_stride + col_pair];
        uint w0_int = (packed >> 4) & 0xFu;
        uint w1_int =  packed       & 0xFu;

        // Fetch scale/zero for each column's group
        uint g0 = col0 / group_size;
        uint g1 = col1 / group_size;
        float scale0 = scales[row * n_groups + g0];
        float zero0  = zeros [row * n_groups + g0];
        float scale1 = scales[row * n_groups + g1];
        float zero1  = zeros [row * n_groups + g1];

        // Dequantise in-register: w_float = scale * w_int + zero
        float w0 = scale0 * float(w0_int) + zero0;
        float w1 = scale1 * float(w1_int) + zero1;

        // Accumulate dot product
        partial += w0 * input_vec[col0];
        partial += w1 * input_vec[col1];
    }

    // Parallel reduction across the threadgroup
    accum_scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction — assumes THREADS_PER_TG is a power of two
    for (uint stride = THREADS_PER_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            accum_scratch[tid] += accum_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = accum_scratch[0];
    }
}

// ---------------------------------------------------------------------------
// Batched GEMV — processes multiple input vectors in one dispatch
//
// Buffer layout (same as above):
//   buffer(3) input_vecs : float[batch_size × n_cols]  row-major
//   buffer(4) output     : float[batch_size × n_rows]  row-major
//   buffer(5) params     : FusedInt4GEMVParams (same; batch_size encoded
//                          via grid.y dispatch dimension)
// Dispatch: (n_rows, batch_size, 1) threadgroups × (THREADS_PER_TG, 1, 1)
// ---------------------------------------------------------------------------

kernel void fused_int4_gemv_batched(
    device const uint8_t*           weights_packed  [[buffer(0)]],
    device const float*             scales          [[buffer(1)]],
    device const float*             zeros           [[buffer(2)]],
    device const float*             input_vecs      [[buffer(3)]],
    device       float*             output          [[buffer(4)]],
    constant FusedInt4GEMVParams&   params          [[buffer(5)]],
    threadgroup  float*             accum_scratch   [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],  // x=row, y=batch
    uint  tid    [[thread_index_in_threadgroup]])
{
    uint row       = tg_pos.x;
    uint batch_idx = tg_pos.y;

    if (row >= params.n_rows) {
        return;
    }

    const uint n_cols     = params.n_cols;
    const uint group_size = params.group_size;
    const uint n_groups   = (n_cols + group_size - 1) / group_size;
    const uint packed_row_stride = n_cols / 2;

    const uint input_base  = batch_idx * n_cols;
    const uint output_base = batch_idx * params.n_rows;

    float partial = 0.0f;

    for (uint col_pair = tid; col_pair < n_cols / 2; col_pair += THREADS_PER_TG) {
        uint col0 = col_pair * 2;
        uint col1 = col0 + 1;

        uint8_t packed = weights_packed[row * packed_row_stride + col_pair];
        uint w0_int = (packed >> 4) & 0xFu;
        uint w1_int =  packed       & 0xFu;

        uint g0 = col0 / group_size;
        uint g1 = col1 / group_size;
        float scale0 = scales[row * n_groups + g0];
        float zero0  = zeros [row * n_groups + g0];
        float scale1 = scales[row * n_groups + g1];
        float zero1  = zeros [row * n_groups + g1];

        float w0 = scale0 * float(w0_int) + zero0;
        float w1 = scale1 * float(w1_int) + zero1;

        partial += w0 * input_vecs[input_base + col0];
        partial += w1 * input_vecs[input_base + col1];
    }

    accum_scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS_PER_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            accum_scratch[tid] += accum_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[output_base + row] = accum_scratch[0];
    }
}
