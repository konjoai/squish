/*
 * astc_gemv.metal — ASTC texture-sampled GEMV for Squish Wave 64
 *
 * Implements a fused "hardware-ASTC-decompress + dot-product" kernel for the
 * decode inference step (seq_len == 1).  Weight tensors are stored as ASTC
 * 6×6 HDR textures (MTLPixelFormatASTC_6x6_HDR); the Metal GPU hardware
 * decompresses each 6×6 block in-flight with zero software overhead.
 *
 * The kernel computes:   output[gid] = Σ_k  W[gid, k] * input[k]
 *
 * where W is stored as an ASTC HDR texture with:
 *   width   = ceil(n_cols / 6) * 6   (padded to block boundary)
 *   height  = ceil(n_rows / 6) * 6   (padded to block boundary)
 *
 * Each texel is accessed as texture.sample(nearest, pixel_coord) → float4.
 * Only the .r channel is used (for a scalar weight matrix).  The sampler
 * uses ``filter::nearest`` and ``coord::pixel`` so texel (x, y) maps
 * exactly to weight column x, output row y.
 *
 * Buffer / texture layout
 * ───────────────────────
 *   texture(0) weight_tex     : ASTC HDR 6×6 2D texture; float4 per texel
 *                               (.r channel = weight value, .gba = 0)
 *   buffer(0)  input_vec      : float32 input vector, length n_cols
 *   buffer(1)  output         : float32 output vector, length n_rows
 *   buffer(2)  params         : ASTCGEMVParams (n_rows, n_cols)
 *
 * Dispatch
 * ────────
 *   1-D grid (n_rows, 1, 1)   — one thread per output row
 *   threads   (1,     1, 1)   — trivially parallel; tune to (64,1,1) if
 *                               occupancy profiling reveals benefit
 *
 * ASTC HDR note
 * ─────────────
 * MTLPixelFormatASTC_6x6_HDR decompresses to float16 into the shader
 * register.  Sampling into a float4 variable automatically promotes to
 * float32 on modern Metal hardware (sf16→sf32 conversion is free on A-series
 * and M-series GPUs).  The read values are exact up to the original fp16
 * quantisation applied during ASTC encoding.
 *
 * Scale-table note
 * ────────────────
 * The NumPy simulation encoder stores a per-block scale table in the ASTC
 * payload (see ASTCEncodeResult.scale_table).  The Metal path does NOT need
 * the scale table because the ASTC HDR format encodes the scale information
 * directly in the block header — the GPU handles it transparently.  The
 * scale table is retained in the squizd payload only for the CPU decode path.
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Parameter struct (matches Python-side struct.pack layout)
// ---------------------------------------------------------------------------

struct ASTCGEMVParams {
    uint n_rows;   // number of output rows  (= out_features)
    uint n_cols;   // number of input columns (= in_features)
};

// ---------------------------------------------------------------------------
// Nearest-neighbour sampler with pixel coordinates
// ---------------------------------------------------------------------------

constexpr sampler nearest_pixel(
    coord::pixel,
    filter::nearest,
    address::clamp_to_edge
);

// ---------------------------------------------------------------------------
// GEMV kernel — one thread per output row
// ---------------------------------------------------------------------------

kernel void astc_gemv(
    texture2d<float, access::sample> weight_tex [[texture(0)]],
    device const float* input_vec               [[buffer(0)]],
    device       float* output                  [[buffer(1)]],
    constant ASTCGEMVParams& params             [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.n_rows) {
        return;
    }

    float acc = 0.0f;

    for (uint k = 0; k < params.n_cols; ++k) {
        // Pixel coordinate: column = k, row = gid
        // Metal uses (x=column, y=row) — add 0.5 for centre-of-texel
        float2 coord = float2(float(k) + 0.5f, float(gid) + 0.5f);
        float4 texel = weight_tex.sample(nearest_pixel, coord);
        acc += texel.r * input_vec[k];
    }

    output[gid] = acc;
}

// ---------------------------------------------------------------------------
// Batched GEMV — grid (n_rows, batch_size, 1), one thread per (row, batch)
// ---------------------------------------------------------------------------

kernel void astc_gemv_batched(
    texture2d<float, access::sample> weight_tex [[texture(0)]],
    device const float* input_vecs              [[buffer(0)]],  // [batch_size, n_cols]
    device       float* output                  [[buffer(1)]],  // [batch_size, n_rows]
    constant ASTCGEMVParams& params             [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])      // x=row, y=batch
{
    uint out_row   = gid.x;
    uint batch_idx = gid.y;

    if (out_row >= params.n_rows) {
        return;
    }

    uint input_base  = batch_idx * params.n_cols;
    uint output_base = batch_idx * params.n_rows;

    float acc = 0.0f;

    for (uint k = 0; k < params.n_cols; ++k) {
        float2 coord = float2(float(k) + 0.5f, float(out_row) + 0.5f);
        float4 texel = weight_tex.sample(nearest_pixel, coord);
        acc += texel.r * input_vecs[input_base + k];
    }

    output[output_base + out_row] = acc;
}
