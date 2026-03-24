/*
 * lut_int2_gemv.metal — INT2 LUT-GEMM GEMV for Squish Wave 67
 *
 * Implements the LUT-GEMM decode path for INT2 (2-bit) weight quantisation,
 * following Park et al. (NeurIPS 2024).  A 256-entry FP16 lookup table is
 * pre-loaded into threadgroup memory, fully replacing all dequantisation
 * multiplies with table lookups.
 *
 * Overview
 * ────────
 * For each output row the kernel:
 *   1. Populates the 256-entry threadgroup LUT from the row's codebook
 *      (group_size code vectors, each 4 entries wide = 4 × INT2 indices
 *       packed in 1 byte).
 *   2. Iterates over packed INT2 bytes: one byte holds 4 weights
 *      (2 bits each, indices 0–3).
 *   3. Looks up each INT2 index in the LUT and accumulates the dot product.
 *
 * INT2 packing format
 * ───────────────────
 * Each byte encodes 4 weights at 2 bits each:
 *   packed_byte = (w3 << 6) | (w2 << 4) | (w1 << 2) | w0
 *   w0 = (packed_byte)       & 0x3   — bits [1:0]
 *   w1 = (packed_byte >> 2)  & 0x3   — bits [3:2]
 *   w2 = (packed_byte >> 4)  & 0x3   — bits [5:4]
 *   w3 = (packed_byte >> 6)  & 0x3   — bits [7:6]
 *
 * LUT construction
 * ────────────────
 * The LUT codebook for each row is stored in `codebook` buffer as FP16:
 *   shape: [n_rows, n_cb_entries]   n_cb_entries = group_size * 4 (= 4 LUT
 *          entries per group.  For group_size=64 → 256-entry LUT = 512 bytes.
 *
 * The LUT is indexed by the raw 2-bit INT2 index (0–3) per element, offset
 * by the group's base position in the codebook.  For group index `g` and
 * column `c` within the group:
 *   lut_idx = g * 4 + int2_value
 *
 * Threadgroup memory
 * ──────────────────
 *   threadgroup half  lut_buf[LUT_SIZE]   — 256 × 2 = 512 bytes
 *   threadgroup float accum_scratch[THREADS_PER_TG]  — 128 × 4 = 512 bytes
 *   Total: 1 KB — 32× within the Metal 32 KB budget.
 *
 * Buffer layout
 * ─────────────
 *   buffer(0) weights_packed : uint8[n_rows * n_cols / 4]
 *                              INT2-packed row-major, 4 per byte, LSB first.
 *                              n_cols MUST be a multiple of 4.
 *   buffer(1) codebook       : half[n_rows * n_cb_entries]
 *                              FP16 LUT codebook, 4 entries per group.
 *                              n_cb_entries = (n_cols / group_size) * 4
 *   buffer(2) input_vec      : float[n_cols]   FP32 input vector
 *   buffer(3) output         : float[n_rows]   FP32 output vector
 *   buffer(4) params         : LutInt2GEMVParams
 *
 * Dispatch
 * ────────
 *   1-D grid  : (n_rows, 1, 1)         one threadgroup per output row
 *   threads   : (THREADS_PER_TG, 1, 1) 128 threads
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constant uint THREADS_PER_TG_LUT = 128;   // threads per threadgroup
constant uint LUT_SIZE            = 256;  // maximum LUT entries (256 × 2 bytes)

// ---------------------------------------------------------------------------
// Parameter struct
// ---------------------------------------------------------------------------

struct LutInt2GEMVParams {
    uint n_rows;        // weight matrix rows (= output dim)
    uint n_cols;        // weight matrix columns (= input dim); multiple of 4
    uint group_size;    // quantisation group size (e.g. 64, 128)
};

// ---------------------------------------------------------------------------
// INT2 GEMV kernel — one threadgroup per output row
// ---------------------------------------------------------------------------

kernel void lut_int2_gemv(
    device  const uint8_t*          weights_packed  [[buffer(0)]],
    device  const half*             codebook        [[buffer(1)]],
    device  const float*            input_vec       [[buffer(2)]],
    device        float*            output          [[buffer(3)]],
    constant LutInt2GEMVParams&     params          [[buffer(4)]],
    threadgroup half*               lut_buf         [[threadgroup(0)]],
    threadgroup float*              accum_scratch   [[threadgroup(1)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]])
{
    if (tg_id >= params.n_rows) {
        return;
    }

    const uint row        = tg_id;
    const uint n_cols     = params.n_cols;
    const uint group_size = params.group_size;
    const uint n_groups   = n_cols / group_size;
    const uint n_cb_entries = n_groups * 4;

    // --- Step 1: Cooperatively load the LUT for this row into threadgroup ---
    // codebook for row `row` starts at row * n_cb_entries
    for (uint i = tid; i < n_cb_entries && i < LUT_SIZE; i += THREADS_PER_TG_LUT) {
        lut_buf[i] = codebook[row * n_cb_entries + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Step 2: LUT-decode and accumulate ---
    // weights_packed: n_cols / 4 bytes per row (4 INT2 weights per byte)
    const uint packed_row_stride = n_cols / 4;

    float partial = 0.0f;

    // Each thread processes every THREADS_PER_TG-th packed byte
    for (uint byte_idx = tid; byte_idx < packed_row_stride; byte_idx += THREADS_PER_TG_LUT) {
        // 4 columns encoded in one byte
        uint col_base          = byte_idx * 4;
        uint8_t packed         = weights_packed[row * packed_row_stride + byte_idx];

        // Extract 4 INT2 values (2 bits each)
        uint idx0 =  packed       & 0x3u;
        uint idx1 = (packed >> 2) & 0x3u;
        uint idx2 = (packed >> 4) & 0x3u;
        uint idx3 = (packed >> 6) & 0x3u;

        // Group for this column range (all 4 cols share the same group if
        // group_size ≥ 4, which is always true in practice)
        uint g = col_base / group_size;

        // LUT lookup: lut_idx = g * 4 + int2_value
        float w0 = float(lut_buf[g * 4 + idx0]);
        float w1 = float(lut_buf[g * 4 + idx1]);
        float w2 = float(lut_buf[g * 4 + idx2]);
        float w3 = float(lut_buf[g * 4 + idx3]);

        // Accumulate dot product
        partial += w0 * input_vec[col_base    ];
        partial += w1 * input_vec[col_base + 1];
        partial += w2 * input_vec[col_base + 2];
        partial += w3 * input_vec[col_base + 3];
    }

    // --- Step 3: Parallel reduction ---
    accum_scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS_PER_TG_LUT / 2; stride > 0; stride >>= 1) {
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
// Batched INT2 LUT-GEMV — processes multiple input vectors
//
// Dispatch: (n_rows, batch_size, 1) threadgroups
// buffer(2) input_vecs : float[batch_size × n_cols]  row-major
// buffer(3) output     : float[batch_size × n_rows]  row-major
// ---------------------------------------------------------------------------

kernel void lut_int2_gemv_batched(
    device  const uint8_t*          weights_packed  [[buffer(0)]],
    device  const half*             codebook        [[buffer(1)]],
    device  const float*            input_vecs      [[buffer(2)]],
    device        float*            output          [[buffer(3)]],
    constant LutInt2GEMVParams&     params          [[buffer(4)]],
    threadgroup half*               lut_buf         [[threadgroup(0)]],
    threadgroup float*              accum_scratch   [[threadgroup(1)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]])
{
    uint row       = tg_pos.x;
    uint batch_idx = tg_pos.y;

    if (row >= params.n_rows) {
        return;
    }

    const uint n_cols       = params.n_cols;
    const uint group_size   = params.group_size;
    const uint n_groups     = n_cols / group_size;
    const uint n_cb_entries = n_groups * 4;
    const uint packed_row_stride = n_cols / 4;

    const uint input_base  = batch_idx * n_cols;
    const uint output_base = batch_idx * params.n_rows;

    // Load LUT
    for (uint i = tid; i < n_cb_entries && i < LUT_SIZE; i += THREADS_PER_TG_LUT) {
        lut_buf[i] = codebook[row * n_cb_entries + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial = 0.0f;

    for (uint byte_idx = tid; byte_idx < packed_row_stride; byte_idx += THREADS_PER_TG_LUT) {
        uint col_base  = byte_idx * 4;
        uint8_t packed = weights_packed[row * packed_row_stride + byte_idx];

        uint idx0 =  packed       & 0x3u;
        uint idx1 = (packed >> 2) & 0x3u;
        uint idx2 = (packed >> 4) & 0x3u;
        uint idx3 = (packed >> 6) & 0x3u;

        uint g = col_base / group_size;
        float w0 = float(lut_buf[g * 4 + idx0]);
        float w1 = float(lut_buf[g * 4 + idx1]);
        float w2 = float(lut_buf[g * 4 + idx2]);
        float w3 = float(lut_buf[g * 4 + idx3]);

        partial += w0 * input_vecs[input_base + col_base    ];
        partial += w1 * input_vecs[input_base + col_base + 1];
        partial += w2 * input_vecs[input_base + col_base + 2];
        partial += w3 * input_vecs[input_base + col_base + 3];
    }

    accum_scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS_PER_TG_LUT / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            accum_scratch[tid] += accum_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[output_base + row] = accum_scratch[0];
    }
}
