/*
 * zip_gemv.metal — Fused ZipGEMV for Squish Wave 65 (TCA-TBE decode path)
 *
 * Implements a fused "decompress + dot-product" kernel for the decode
 * inference step (seq_len == 1).  One output element is produced per
 * threadgroup.  Each thread in the group handles 64 consecutive weight
 * elements, decoding their TCA-TBE representation and accumulating the
 * partial dot product.  Partial sums are reduced to a scalar in shared
 * memory before the first thread writes the result.
 *
 * Buffer layout
 * ─────────────
 *   buffer(0) weights_packed   : packed TCA-TBE blocks, row-major
 *                                Each block is exactly ZIP_BLOCK_BYTES bytes
 *                                (flags 1B + header 3B + sign 16B + range 16B
 *                                 + mantissa N*4/8 B + spill_count 2B + spills)
 *                                Blocks are stored sequentially per weight-row.
 *   buffer(1) block_offsets    : uint32 array, length = n_rows * n_block_cols
 *                                byte offset of each block in weights_packed
 *   buffer(2) input_vec        : float32 input vector, length n_cols
 *   buffer(3) output           : float32 output vector, length n_rows
 *   buffer(4) params           : ZipGEMVParams struct (n_rows, n_cols, n_block_cols, mantissa_bits)
 *
 * Dispatch
 * ────────
 *   grid    : (n_rows, 1, 1)   — one threadgroup per output element (row)
 *   threads : (256, 1, 1)      — 256 threads per threadgroup
 *
 * Nomenclature
 * ────────────
 *   BLOCK_SIZE      128 elements per TCA-TBE block
 *   THREADS_PER_TG  256
 *   ELEMS_PER_TH    64   (== BLOCK_SIZE * 2 / THREADS_PER_TG; each thread
 *                         covers half a block so two threads fully cover one
 *                         128-element block)
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

constant uint BLOCK_SIZE    = 128;   // elements per TCA-TBE block
constant uint THREADS_PER_TG = 256;
constant uint ELEMS_PER_TH  = BLOCK_SIZE / 2;  // 64: two threads per block

// Raw-block flag stored in the first byte of a serialised TCA-TBE block.
constant uchar IS_RAW_FLAG  = 0x01;

// Mantissa bits stored in each block's bitmap (must match Python codec).
// Overridable via function constant if needed; default 4.
constant uint MANTISSA_BITS [[function_constant(0)]] = 4;

// ---------------------------------------------------------------------------
// Parameter struct (matches Python-side struct.pack layout)
// ---------------------------------------------------------------------------

struct ZipGEMVParams {
    uint n_rows;          // rows in weight matrix (= output dim)
    uint n_cols;          // columns in weight matrix (= input dim)
    uint n_block_cols;    // number of TCA-TBE blocks per row = ceil(n_cols / 128)
    uint mantissa_bits;   // mantissa bits per element (typically 4)
};

// ---------------------------------------------------------------------------
// Inline decoder helpers
// ---------------------------------------------------------------------------

/*
 * Reconstruct a BF16 value (returned as float) from sign, exponent, and
 * mantissa components.  The mantissa argument holds the top `mantissa_bits`
 * bits in its LSBs; remaining mantissa bits are zero (lossless within the
 * stored precision, which matches the Python encoder).
 */
inline float decode_bf16_element(
    uchar sign_bit,
    uchar exponent,
    uchar mantissa_top,   // top mantissa_bits bits
    uint  mant_bits
) {
    uint shift = 7u - mant_bits;
    uint mantissa7 = (uint)mantissa_top << shift;                // 7-bit mantissa
    uint raw = ((uint)sign_bit << 15) | ((uint)exponent << 7) | mantissa7;
    // Reinterpret the lower 16 bits as BF16, then widen to float.
    // Metal has no native bf16 type, so we place the BF16 bits in the high
    // half of a float32 by shifting left 16 bits.
    uint f32_bits = raw << 16;
    return as_type<float>(f32_bits);
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

kernel void zip_gemv(
    device const uchar*  weights_packed  [[buffer(0)]],
    device const uint*   block_offsets   [[buffer(1)]],
    device const float*  input_vec       [[buffer(2)]],
    device float*        output          [[buffer(3)]],
    constant ZipGEMVParams& params       [[buffer(4)]],
    threadgroup float*   tg_accum        [[threadgroup(0)]],  // THREADS_PER_TG floats
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
)
{
    // Bounds guard.
    if (row >= params.n_rows) return;

    float partial = 0.0f;

    // Each thread is responsible for a contiguous run of elements spanning
    // (conceptually) ELEMS_PER_TH = 64 positions in the weight row.
    // Because BLOCK_SIZE = 128 and two threads handle one block, we map:
    //   thread t  →  block  b = t / 2
    //                half   h = t % 2   (0 = first 64 elems, 1 = last 64)
    // We iterate over blocks in strides of THREADS_PER_TG/2 = 128.

    for (uint b_start = tid / 2; b_start < params.n_block_cols; b_start += THREADS_PER_TG / 2) {
        uint block_idx = row * params.n_block_cols + b_start;
        uint byte_offset = block_offsets[block_idx];
        device const uchar* blk = weights_packed + byte_offset;

        uchar flags  = blk[0];
        uint  half   = tid & 1u;   // 0 or 1
        uint  elem_base = b_start * BLOCK_SIZE + half * ELEMS_PER_TH;

        if (flags & IS_RAW_FLAG) {
            // ── Raw fallback block ──────────────────────────────────────
            // Header: 1B flags + 2B n_elements (little-endian)
            uint n_elements = (uint)blk[1] | ((uint)blk[2] << 8);
            device const uchar* raw_words = blk + 3;

            for (uint e = 0; e < ELEMS_PER_TH; ++e) {
                uint col = elem_base + e;
                if (col >= params.n_cols || col >= n_elements) break;
                uint raw16 = (uint)raw_words[col * 2] | ((uint)raw_words[col * 2 + 1] << 8);
                uint f32_bits = raw16 << 16;
                float w = as_type<float>(f32_bits);
                partial += w * input_vec[col];
            }
        } else {
            // ── Compressed TCA-TBE block ─────────────────────────────
            // Header: flags(1) + e_mode(1) + e_lo(1) + e_hi(1) = 4B
            uchar e_mode = blk[1];
            // e_lo_offset / e_hi_offset unused for decode (only e_mode needed
            // for in-range elements; out-of-range come from spill).
            device const uchar* sign_base  = blk + 4;
            device const uchar* range_base = sign_base  + 16;
            uint n_mant_bytes = BLOCK_SIZE * params.mantissa_bits / 8u;
            device const uchar* mant_base  = range_base + 16;
            device const uchar* spill_hdr  = mant_base  + n_mant_bytes;
            uint spill_count = (uint)spill_hdr[0] | ((uint)spill_hdr[1] << 8);
            device const uchar* spill_base = spill_hdr + 2;

            uint spill_idx = 0;

            // First pass: count out-of-range elements before our range to
            // find the correct spill offset for this thread's elements.
            for (uint e = 0; e < half * ELEMS_PER_TH; ++e) {
                uint byte_i = e / 8u;
                uint bit_i  = e % 8u;
                uchar in_r = (range_base[byte_i] >> bit_i) & 1u;
                if (!in_r) ++spill_idx;
            }

            // Decode this thread's 64 elements.
            for (uint e = 0; e < ELEMS_PER_TH; ++e) {
                uint col = elem_base + e;
                if (col >= params.n_cols) break;

                uint abs_e = half * ELEMS_PER_TH + e;
                uint byte_i = abs_e / 8u;
                uint bit_i  = abs_e % 8u;

                uchar sign_bit = (sign_base[byte_i]  >> bit_i) & 1u;
                uchar in_range = (range_base[byte_i] >> bit_i) & 1u;

                float w;
                if (in_range) {
                    // Reconstruct from bitmaps.
                    uint mant_bit_offset = abs_e * params.mantissa_bits;
                    uint mant_byte       = mant_bit_offset / 8u;
                    uint mant_bit_shift  = mant_bit_offset % 8u;
                    // Extract mantissa_bits bits starting at mant_bit_shift.
                    uint combined = (uint)mant_base[mant_byte];
                    if (mant_bit_shift + params.mantissa_bits > 8u)
                        combined |= (uint)mant_base[mant_byte + 1] << 8u;
                    uchar mantissa_top = (uchar)((combined >> mant_bit_shift) & ((1u << params.mantissa_bits) - 1u));

                    w = decode_bf16_element(sign_bit, e_mode, mantissa_top, params.mantissa_bits);
                } else {
                    // Restore from spill.
                    if (spill_idx < spill_count) {
                        uint raw16 = (uint)spill_base[spill_idx * 2]
                                   | ((uint)spill_base[spill_idx * 2 + 1] << 8);
                        uint f32_bits = raw16 << 16;
                        w = as_type<float>(f32_bits);
                        ++spill_idx;
                    } else {
                        w = 0.0f;  // spill underflow — should not occur with valid data
                    }
                }

                partial += w * input_vec[col];
            }
        }
    }

    // ── Thread-group reduction ──────────────────────────────────────────────
    tg_accum[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint s = THREADS_PER_TG / 2; s > 0; s >>= 1) {
        if (tid < s) tg_accum[tid] += tg_accum[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) output[row] = tg_accum[0];
}
