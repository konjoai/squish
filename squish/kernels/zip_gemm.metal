/*
 * zip_gemm.metal — Decoupled ZipGEMM for Squish Wave 65 (TCA-TBE prefill path)
 *
 * Implements a two-kernel pipeline for the prefill inference step
 * (seq_len > 1).  Rather than a fused decompress+GEMM (which would serialize
 * decode inside the multiply loop), we separate the work into:
 *
 *   Kernel 1 — zip_decompress_tile
 *     Each threadgroup decompresses one 64×128 weight tile from TCA-TBE
 *     packed bytes into float16 in device memory (a pre-allocated scratch
 *     buffer).  The tile is TILE_ROWS=64 consecutive weight rows ×
 *     TILE_COLS=128 elements == one TCA-TBE block column.
 *
 *   Kernel 2 — zip_gemm_tile
 *     Standard tiled GEMM: reads the decompressed float16 weight tiles from
 *     the scratch buffer and performs the matrix multiply against the input
 *     activation matrix.
 *
 * Buffer layout (kernel 1 — zip_decompress_tile)
 * ────────────────────────────────────────────
 *   buffer(0)  weights_packed  : packed TCA-TBE blocks row-major
 *   buffer(1)  block_offsets   : uint32[n_rows * n_block_cols] byte offsets
 *   buffer(2)  weight_scratch  : half[n_rows * n_cols] output scratch buffer
 *   buffer(3)  params          : ZipDecompParams
 *
 * Buffer layout (kernel 2 — zip_gemm_tile)
 * ────────────────────────────────────────
 *   buffer(0)  weight_scratch  : half[n_rows * n_cols] (filled by kernel 1)
 *   buffer(1)  input_mat       : float[seq_len * n_cols] row-major activations
 *   buffer(2)  output_mat      : float[n_rows * seq_len] row-major output
 *   buffer(3)  params          : ZipGEMMParams
 *
 * Dispatch (kernel 1)
 * ───────────────────
 *   grid    : (ceil(n_block_cols / BLK_COLS_PER_TG),
 *              ceil(n_rows / TILE_ROWS), 1)
 *   threads : (128, 1, 1)
 *
 * Dispatch (kernel 2)
 * ───────────────────
 *   grid    : (ceil(seq_len / GEMM_TILE_N), ceil(n_rows / GEMM_TILE_M), 1)
 *   threads : (GEMM_TILE_N, GEMM_TILE_M, 1)  — square threadgroup blocks
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Tile dimensions
// ---------------------------------------------------------------------------

constant uint BLOCK_SIZE        = 128;   // elements per TCA-TBE block
constant uint TILE_ROWS         = 64;    // rows per decompression tile
constant uint DECOMP_THREADS    = 128;   // threads per decompress threadgroup
constant uint GEMM_TILE_M       = 16;    // output rows per GEMM tile
constant uint GEMM_TILE_N       = 16;    // output cols (seq steps) per GEMM tile
constant uint GEMM_TILE_K       = 32;    // inner dimension per GEMM tile step

constant uchar IS_RAW_FLAG      = 0x01;

// ---------------------------------------------------------------------------
// Parameter structs
// ---------------------------------------------------------------------------

struct ZipDecompParams {
    uint n_rows;          // weight matrix rows
    uint n_cols;          // weight matrix columns
    uint n_block_cols;    // ceil(n_cols / BLOCK_SIZE)
    uint mantissa_bits;   // typically 4
};

struct ZipGEMMParams {
    uint n_rows;    // weight rows (output features)
    uint n_cols;    // weight cols (input features)
    uint seq_len;   // sequence length (input rows)
};

// ---------------------------------------------------------------------------
// Shared inline decompressor (same logic as zip_gemv)
// ---------------------------------------------------------------------------

inline half decode_bf16_as_half(
    uchar sign_bit,
    uchar exponent,
    uchar mantissa_top,
    uint  mant_bits
) {
    uint shift    = 7u - mant_bits;
    uint mantissa = (uint)mantissa_top << shift;
    uint raw16    = ((uint)sign_bit << 15) | ((uint)exponent << 7) | mantissa;
    uint f32      = raw16 << 16;
    float val     = as_type<float>(f32);
    return half(val);
}

// ---------------------------------------------------------------------------
// Kernel 1 — zip_decompress_tile
// ---------------------------------------------------------------------------

/*
 * Each threadgroup decompresses one 64×128 tile from packed TCA-TBE into the
 * float16 weight scratch buffer.  The tile corresponds to TILE_ROWS=64
 * consecutive output rows and BLOCK_SIZE=128 consecutive input columns
 * starting at block column `bg.x`.
 *
 * Thread assignments inside the threadgroup:
 *   tid 0..127 each handle one row within the tile (DECOMP_THREADS == TILE_ROWS).
 *   Each thread decodes all 128 elements in its assigned row.
 */
kernel void zip_decompress_tile(
    device const uchar*          weights_packed  [[buffer(0)]],
    device const uint*           block_offsets   [[buffer(1)]],
    device half*                 weight_scratch  [[buffer(2)]],
    constant ZipDecompParams&    params          [[buffer(3)]],
    uint2 bg  [[threadgroup_position_in_grid]],   // (block_col_group, row_tile)
    uint  tid [[thread_position_in_threadgroup]]
)
{
    uint row_tile_base  = bg.y * TILE_ROWS;
    uint row            = row_tile_base + tid;
    uint block_col      = bg.x;   // one block per kernel call in x dimension

    if (row >= params.n_rows || block_col >= params.n_block_cols) return;

    uint col_base    = block_col * BLOCK_SIZE;
    uint block_idx   = row * params.n_block_cols + block_col;
    uint byte_offset = block_offsets[block_idx];
    device const uchar* blk = weights_packed + byte_offset;

    uchar flags = blk[0];

    if (flags & IS_RAW_FLAG) {
        // ── Raw block ──────────────────────────────────────────────────
        uint n_elements = (uint)blk[1] | ((uint)blk[2] << 8);
        device const uchar* raw_words = blk + 3;

        for (uint e = 0; e < BLOCK_SIZE; ++e) {
            uint col = col_base + e;
            if (col >= params.n_cols || e >= n_elements) {
                if (col < params.n_cols) weight_scratch[row * params.n_cols + col] = half(0.f);
                continue;
            }
            uint raw16 = (uint)raw_words[e * 2] | ((uint)raw_words[e * 2 + 1] << 8);
            uint f32   = raw16 << 16;
            weight_scratch[row * params.n_cols + col] = half(as_type<float>(f32));
        }
    } else {
        // ── TCA-TBE compressed block ────────────────────────────────
        uchar e_mode     = blk[1];
        device const uchar* sign_base  = blk + 4;
        device const uchar* range_base = sign_base + 16;
        uint n_mant_bytes = BLOCK_SIZE * params.mantissa_bits / 8u;
        device const uchar* mant_base  = range_base + 16;
        device const uchar* spill_hdr  = mant_base + n_mant_bytes;
        uint spill_count = (uint)spill_hdr[0] | ((uint)spill_hdr[1] << 8);
        device const uchar* spill_base = spill_hdr + 2;

        uint spill_idx = 0;

        for (uint e = 0; e < BLOCK_SIZE; ++e) {
            uint col = col_base + e;
            if (col >= params.n_cols) {
                break;
            }
            uint byte_i  = e / 8u;
            uint bit_i   = e % 8u;

            uchar sign_bit = (sign_base[byte_i]  >> bit_i) & 1u;
            uchar in_range = (range_base[byte_i] >> bit_i) & 1u;

            half w;
            if (in_range) {
                uint mant_bit_offset = e * params.mantissa_bits;
                uint mant_byte       = mant_bit_offset / 8u;
                uint mant_shift      = mant_bit_offset % 8u;
                uint combined        = (uint)mant_base[mant_byte];
                if (mant_shift + params.mantissa_bits > 8u)
                    combined |= (uint)mant_base[mant_byte + 1] << 8u;
                uchar mant_top = (uchar)((combined >> mant_shift) & ((1u << params.mantissa_bits) - 1u));
                w = decode_bf16_as_half(sign_bit, e_mode, mant_top, params.mantissa_bits);
            } else {
                if (spill_idx < spill_count) {
                    uint raw16 = (uint)spill_base[spill_idx * 2]
                               | ((uint)spill_base[spill_idx * 2 + 1] << 8);
                    uint f32   = raw16 << 16;
                    w = half(as_type<float>(f32));
                    ++spill_idx;
                } else {
                    w = half(0.f);
                }
            }

            weight_scratch[row * params.n_cols + col] = w;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 — zip_gemm_tile
// ---------------------------------------------------------------------------

/*
 * Standard tiled matrix multiply:
 *   C[n_rows × seq_len]  =  W[n_rows × n_cols]  ×  X[n_cols × seq_len]
 *
 * where W is the decompressed float16 weight matrix and X is the float32
 * activation matrix (transposed: X[n_cols][seq_len] so threads walk columns).
 *
 * Tile (GEMM_TILE_M × GEMM_TILE_N) resides in threadgroup memory.
 * Inner loop iterates over K in steps of GEMM_TILE_K.
 */
kernel void zip_gemm_tile(
    device const half*          weight_scratch  [[buffer(0)]],
    device const float*         input_mat       [[buffer(1)]],
    device float*               output_mat      [[buffer(2)]],
    constant ZipGEMMParams&     params          [[buffer(3)]],
    threadgroup half*           tg_W            [[threadgroup(0)]],  // GEMM_TILE_M × GEMM_TILE_K
    threadgroup float*          tg_X            [[threadgroup(1)]],  // GEMM_TILE_K × GEMM_TILE_N
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint2 tid2  [[thread_position_in_threadgroup]]
)
{
    uint out_col = tgpos.x * GEMM_TILE_N + tid2.x;  // seq step index
    uint out_row = tgpos.y * GEMM_TILE_M + tid2.y;  // weight row index

    float acc = 0.0f;

    for (uint k_base = 0; k_base < params.n_cols; k_base += GEMM_TILE_K) {
        // Load W tile: thread (ty, tx) loads element (ty, k_base + tx).
        uint w_col = k_base + tid2.x;
        if (out_row < params.n_rows && w_col < params.n_cols)
            tg_W[tid2.y * GEMM_TILE_K + tid2.x] = weight_scratch[out_row * params.n_cols + w_col];
        else
            tg_W[tid2.y * GEMM_TILE_K + tid2.x] = half(0.f);

        // Load X tile: thread (ty, tx) loads element (k_base + ty, tx).
        uint x_row = k_base + tid2.y;
        if (x_row < params.n_cols && out_col < params.seq_len)
            tg_X[tid2.y * GEMM_TILE_N + tid2.x] = input_mat[x_row * params.seq_len + out_col];
        else
            tg_X[tid2.y * GEMM_TILE_N + tid2.x] = 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate.
        for (uint k = 0; k < GEMM_TILE_K; ++k) {
            acc += float(tg_W[tid2.y * GEMM_TILE_K + k]) * tg_X[k * GEMM_TILE_N + tid2.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (out_row < params.n_rows && out_col < params.seq_len) {
        output_mat[out_row * params.seq_len + out_col] = acc;
    }
}
