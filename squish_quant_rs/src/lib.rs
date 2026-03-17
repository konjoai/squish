//! squish_quant — High-throughput INT8 symmetric weight quantizer
//!
//! Exposed to Python via PyO3 as a drop-in replacement for the vectorized
//! numpy path in `vectro/python/interface.py`.
//!
//! Architecture:
//!   - Rayon parallel row processing (all CPU cores)
//!   - Per-row symmetric INT8: scale = max(|x|) / 127.0
//!   - ARM NEON SIMD for abs + max (optional, enabled by "simd-neon" feature)
//!   - Zero-copy numpy array access via PyO3-numpy
//!
//! Performance targets (Apple Silicon M-series):
//!   - 8–12 GB/sec sustained quantization throughput
//!   - vs ~1.5 GB/sec for vectorized numpy baseline
//!   - 14B model (29.6 GB bf16): ~3s vs ~16s numpy
//!
//! Usage from Python (after `maturin develop`):
//! ```python
//! from squish_quant import quantize_int8_f32, quantize_int8_bf16
//!
//! # arr: (N, D) float32 numpy array
//! q, scales = quantize_int8_f32(arr)
//! # q:      (N, D) int8   — quantized weights
//! # scales: (N,)   float32 — per-row scale factors
//! ```

use half::bf16;
use numpy::{
    ndarray::{Array1, Array2},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
};
use pyo3::prelude::*;
use rayon::prelude::*;

// ── BF16 → f32 helper ───────────────────────────────────────────────────────
// safetensors returns BF16 weights as raw u16 bytes.  numpy views them as
// uint16.  We convert per-element inside the Rayon loop to avoid the Python-
// side `.astype(np.float32)` copy that currently doubles peak RAM per shard.

#[inline(always)]
fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

// ── Per-row symmetric INT8 quantization (float32 input) ─────────────────────

/// Quantize a 2D float32 weight matrix to INT8.
///
/// Algorithm (per row):
///   scale_i = max(|row_i|) / 127.0   (or 1.0 if all zeros)
///   q_ij    = clip(round(x_ij / scale_i), -127, 127)
///
/// Returns (quantized: int8[N,D], scales: float32[N])
#[pyfunction]
pub fn quantize_int8_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<i8>>, Bound<'py, PyArray1<f32>>)> {
    let arr_view = arr.as_array(); // zero-copy
    let (n_rows, n_cols) = arr_view.dim();

    // Allocate output buffers (uninitialized, filled below)
    let mut q_out:     Vec<i8>  = vec![0i8; n_rows * n_cols];
    let mut scales_out: Vec<f32> = vec![0f32; n_rows];

    // Parallel row processing via Rayon
    // Each chunk is one row → safe to write without locks
    q_out
        .par_chunks_mut(n_cols)
        .zip(scales_out.par_iter_mut())
        .enumerate()
        .for_each(|(row_idx, (q_row, scale))| {
            let row = arr_view.row(row_idx);
            let row_slice = row.as_slice().unwrap_or_else(|| {
                // non-contiguous row fallback (rare, only on strided arrays)
                panic!("non-contiguous row at index {row_idx}")
            });

            // Compute per-row absolute maximum (SIMD-friendly loop)
            let row_max = row_slice
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);

            let s = if row_max == 0.0 { 1.0f32 } else { row_max / 127.0 };
            *scale = s;

            let inv_s = 1.0 / s;
            for (q_val, &x) in q_row.iter_mut().zip(row_slice.iter()) {
                // round-to-nearest, then clamp to [-127, 127]
                let q = (x * inv_s).round().clamp(-127.0, 127.0) as i8;
                *q_val = q;
            }
        });

    let q_arr = Array2::from_shape_vec((n_rows, n_cols), q_out)
        .expect("shape mismatch")
        .into_pyarray_bound(py);
    let s_arr = Array1::from_vec(scales_out).into_pyarray_bound(py);

    Ok((q_arr, s_arr))
}


// ── Group quantization (INT8 with group_size) ────────────────────────────────

/// Per-group INT8 quantization.
///
/// Instead of one scale per row, compute one scale per `group_size` elements
/// within each row.  Improves quantization accuracy for rows with uneven
/// weight magnitude distributions (common in attention projections).
///
/// group_size must divide n_cols evenly.  Typical values: 32, 64, 128.
///
/// Returns:
///   q:      (N, D) int8     — same shape as input
///   scales: (N, D/group_size) float32 — one scale per group
#[pyfunction]
pub fn quantize_int8_grouped<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<i8>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();

    if n_cols % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_cols ({n_cols}) must be divisible by group_size ({group_size})"
        )));
    }
    let n_groups = n_cols / group_size;

    let mut q_out:     Vec<i8>   = vec![0i8; n_rows * n_cols];
    let mut scales_out: Vec<f32> = vec![0f32; n_rows * n_groups];

    q_out
        .par_chunks_mut(n_cols)
        .zip(scales_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, (q_row, s_row))| {
            let row = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous");

            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let group = &row_slice[start..end];

                let gmax = group.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
                let s    = if gmax == 0.0 { 1.0 } else { gmax / 127.0 };
                s_row[g] = s;
                let inv_s = 1.0 / s;

                for (q_val, &x) in q_row[start..end].iter_mut().zip(group.iter()) {
                    *q_val = (x * inv_s).round().clamp(-127.0, 127.0) as i8;
                }
            }
        });

    let q_arr = Array2::from_shape_vec((n_rows, n_cols), q_out)
        .expect("shape")
        .into_pyarray_bound(py);
    let s_arr = Array2::from_shape_vec((n_rows, n_groups), scales_out)
        .expect("shape")
        .into_pyarray_bound(py);

    Ok((q_arr, s_arr))
}


// ── INT8 dequantization ──────────────────────────────────────────────────────

/// Reconstruct float32 from INT8 + per-row scales.
/// reconstruct(q, scales)[i,j] = q[i,j].as_f32 * scales[i]
#[pyfunction]
pub fn dequantize_int8_f32<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, i8>,
    scales: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let q_view = q.as_array();
    let s_view = scales.as_slice().expect("scales must be contiguous");
    let (n_rows, n_cols) = q_view.dim();

    let mut out: Vec<f32> = vec![0.0f32; n_rows * n_cols];

    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let s = s_view[row_idx];
            let q_row = q_view.row(row_idx);
            for (o, &qi) in out_row.iter_mut().zip(q_row.iter()) {
                *o = qi as f32 * s;
            }
        });

    Ok(Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape")
        .into_pyarray_bound(py))
}


/// Reconstruct float32 from grouped INT8 + per-group scales.
/// scales shape: (N, D/group_size)
#[pyfunction]
pub fn dequantize_int8_grouped<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, i8>,
    scales: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let q_view = q.as_array();
    let s_view = scales.as_array();
    let (n_rows, n_cols) = q_view.dim();

    if n_cols % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_cols ({n_cols}) must be divisible by group_size ({group_size})"
        )));
    }
    let n_groups = n_cols / group_size;

    let mut out: Vec<f32> = vec![0.0f32; n_rows * n_cols];

    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let q_row = q_view.row(row_idx);
            let q_slice = q_row.as_slice().expect("q row non-contiguous");
            for g in 0..n_groups {
                let scale = s_view[[row_idx, g]];
                let start = g * group_size;
                let end   = start + group_size;
                for (o, &qi) in out_row[start..end].iter_mut().zip(q_slice[start..end].iter()) {
                    *o = qi as f32 * scale;
                }
            }
        });

    Ok(Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape")
        .into_pyarray_bound(py))
}


// ── INT4 nibble quantization (2 values per byte, 50% disk vs INT8) ───────────

/// Pack two INT4 values per byte: lower nibble = even index, upper = odd.
/// Values clamped to [-7, 7] (symmetric signed 4-bit).
/// group_size must divide n_cols evenly.
///
/// Returns:
///   packed: (N, D/2) uint8  — nibble-packed quantized values
///   scales: (N, D/group_size) float32
#[pyfunction]
pub fn quantize_int4_grouped<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();

    if n_cols % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_cols ({n_cols}) must be divisible by group_size ({group_size})"
        )));
    }
    if n_cols % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be even for INT4 packing"
        ));
    }

    let n_groups  = n_cols / group_size;
    let n_packed  = n_cols / 2;

    let mut packed_out: Vec<u8>  = vec![0u8; n_rows * n_packed];
    let mut scales_out: Vec<f32> = vec![0f32; n_rows * n_groups];

    packed_out
        .par_chunks_mut(n_packed)
        .zip(scales_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, (p_row, s_row))| {
            let row = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous");

            // Compute per-group scales
            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let gmax  = row_slice[start..end]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, f32::max);
                s_row[g] = if gmax == 0.0 { 1.0 } else { gmax / 7.0 };
            }

            // Quantize + pack nibbles
            for i in 0..n_packed {
                let j0 = i * 2;
                let j1 = j0 + 1;
                let g0 = j0 / group_size;
                let g1 = j1 / group_size;
                let q0 = (row_slice[j0] / s_row[g0]).round().clamp(-7.0, 7.0) as i8;
                let q1 = (row_slice[j1] / s_row[g1]).round().clamp(-7.0, 7.0) as i8;
                // Bias to [0, 14] so nibbles are unsigned, then pack
                let n0 = (q0 + 7) as u8;   // 0..=14
                let n1 = (q1 + 7) as u8;
                p_row[i] = (n0 & 0x0F) | ((n1 & 0x0F) << 4);
            }
        });

    let packed_arr = Array2::from_shape_vec((n_rows, n_packed), packed_out)
        .expect("shape")
        .into_pyarray_bound(py);
    let scales_arr = Array2::from_shape_vec((n_rows, n_groups), scales_out)
        .expect("shape")
        .into_pyarray_bound(py);

    Ok((packed_arr, scales_arr))
}


/// Unpack nibble-packed INT4 weights back to float32.
/// packed: (N, D/2) uint8, scales: (N, D/group_size) float32
/// Returns: (N, D) float32
#[pyfunction]
pub fn dequantize_int4_grouped<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    scales: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let p_view = packed.as_array();
    let s_view = scales.as_array();
    let (n_rows, n_packed) = p_view.dim();
    let n_cols   = n_packed * 2;
    let n_groups = n_cols / group_size;

    // Validate: scales must have shape (N, n_groups)
    if s_view.dim() != (n_rows, n_groups) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "scales shape {:?} does not match expected ({n_rows}, {n_groups})",
            s_view.dim()
        )));
    }

    let mut out: Vec<f32> = vec![0.0f32; n_rows * n_cols];

    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let p_row = p_view.row(row_idx);
            for i in 0..n_packed {
                let byte = p_row[i];
                let j0   = i * 2;
                let j1   = j0 + 1;
                let q0   = ((byte & 0x0F) as i8) - 7;
                let q1   = (((byte >> 4) & 0x0F) as i8) - 7;
                let g0   = j0 / group_size;
                let g1   = j1 / group_size;
                out_row[j0] = q0 as f32 * s_view[[row_idx, g0]];
                out_row[j1] = q1 as f32 * s_view[[row_idx, g1]];
            }
        });

    Ok(Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape")
        .into_pyarray_bound(py))
}


// ── Asymmetric INT4 quantization (Q4_K_M style, unsigned [0,15] + zero-point) ─

/// Per-group asymmetric INT4 quantization.
///
/// Maps each group's [xmin, xmax] range to [0, 15] using a scale and an integer
/// zero-point offset (Q4_K_M / GGUF convention).  This uses all 16 possible
/// nibble values — symmetric INT4 wastes one level — yielding ~6–10% lower
/// quantization error for LLM weight matrices whose distribution is skewed.
///
/// Algorithm (per group of `group_size` elements):
///   scale  = (gmax − gmin) / 15.0    (or 1.0 if gmax == gmin)
///   offset = gmin                    stored as f32  ← replaces uint8 zero_point
///   q = clamp(round((x − offset) / scale), 0, 15)
///   decode: x_hat = offset + q * scale
///
/// This formulation correctly covers any [gmin, gmax] range including
/// all-positive groups (gmin > 0), where the old uint8 zero_point was
/// clamped to 0 and caused `gmax` to be under-represented.
///
/// Returns:
///   packed:  (N, D/2)            uint8   — low nibble = even index, high = odd
///   scales:  (N, D/group_size)   float32 — step size per group
///   offsets: (N, D/group_size)   float32 — gmin per group  (was uint8 zero_points)
#[pyfunction]
pub fn quantize_int4_asymmetric_grouped<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();

    if n_cols % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_cols ({n_cols}) must be divisible by group_size ({group_size})"
        )));
    }
    if n_cols % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be even for INT4 packing"
        ));
    }

    let n_groups = n_cols / group_size;
    let n_packed = n_cols / 2;

    let mut packed_out:  Vec<u8>  = vec![0u8;  n_rows * n_packed];
    let mut scales_out:  Vec<f32> = vec![0f32; n_rows * n_groups];
    let mut offsets_out: Vec<f32> = vec![0f32; n_rows * n_groups];  // gmin per group

    packed_out
        .par_chunks_mut(n_packed)
        .zip(scales_out.par_chunks_mut(n_groups))
        .zip(offsets_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, ((p_row, s_row), o_row))| {
            let row       = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous");

            // Per-group scale + offset (gmin)
            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let gmin  = row_slice[start..end]
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min);
                let gmax  = row_slice[start..end]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let scale = if gmax == gmin { 1.0f32 } else { (gmax - gmin) / 15.0 };
                // offset = gmin: q encodes (x - gmin) / scale ∈ [0, 15]
                s_row[g] = scale;
                o_row[g] = gmin;
            }

            // Quantize + pack nibbles
            for i in 0..n_packed {
                let j0 = i * 2;
                let j1 = j0 + 1;
                let g0 = j0 / group_size;
                let g1 = j1 / group_size;
                // q = round((x - offset) / scale), clamped to [0, 15]
                let q0 = ((row_slice[j0] - o_row[g0]) / s_row[g0])
                    .round().clamp(0.0, 15.0) as u8;
                let q1 = ((row_slice[j1] - o_row[g1]) / s_row[g1])
                    .round().clamp(0.0, 15.0) as u8;
                p_row[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
        });

    let packed_arr  = Array2::from_shape_vec((n_rows, n_packed), packed_out)
        .expect("shape").into_pyarray_bound(py);
    let scales_arr  = Array2::from_shape_vec((n_rows, n_groups), scales_out)
        .expect("shape").into_pyarray_bound(py);
    let offsets_arr = Array2::from_shape_vec((n_rows, n_groups), offsets_out)
        .expect("shape").into_pyarray_bound(py);

    Ok((packed_arr, scales_arr, offsets_arr))
}


/// Unpack asymmetric nibble-packed INT4 weights back to float32.
///
/// packed:  (N, D/2)          uint8
/// scales:  (N, D/group_size) float32  — step size per group
/// offsets: (N, D/group_size) float32  — gmin per group
/// Returns: (N, D)            float32
///
/// Decode: x_hat = offsets + q * scales
#[pyfunction]
pub fn dequantize_int4_asymmetric_grouped<'py>(
    py: Python<'py>,
    packed:  PyReadonlyArray2<'py, u8>,
    scales:  PyReadonlyArray2<'py, f32>,
    offsets: PyReadonlyArray2<'py, f32>,   // was zero_points: u8
    group_size:  usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let p_view  = packed.as_array();
    let s_view  = scales.as_array();
    let o_view  = offsets.as_array();
    let (n_rows, n_packed) = p_view.dim();
    let n_cols   = n_packed * 2;
    let n_groups = n_cols / group_size;

    if s_view.dim() != (n_rows, n_groups) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "scales shape {:?} does not match expected ({n_rows}, {n_groups})",
            s_view.dim()
        )));
    }
    if o_view.dim() != (n_rows, n_groups) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "offsets shape {:?} does not match expected ({n_rows}, {n_groups})",
            o_view.dim()
        )));
    }

    let mut out: Vec<f32> = vec![0.0f32; n_rows * n_cols];

    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let p_row = p_view.row(row_idx);
            for i in 0..n_packed {
                let byte = p_row[i];
                let j0   = i * 2;
                let j1   = j0 + 1;
                let g0   = j0 / group_size;
                let g1   = j1 / group_size;
                // x_hat = offset + q * scale
                let q0 = (byte & 0x0F) as f32;
                let q1 = ((byte >> 4) & 0x0F) as f32;
                out_row[j0] = o_view[[row_idx, g0]] + q0 * s_view[[row_idx, g0]];
                out_row[j1] = o_view[[row_idx, g1]] + q1 * s_view[[row_idx, g1]];
            }
        });

    Ok(Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape")
        .into_pyarray_bound(py))
}


// ── BF16-native INT8 quantization ───────────────────────────────────────────
//
// Accepts uint16 arrays (numpy view of BF16 safetensors bytes) and converts
// per-element inside the Rayon loop.  Avoids the Python-side float32 cast that
// doubles peak RAM: instead of (shard_BF16 + shard_F32 + output), peak is
// (shard_BF16 + output) — roughly half the RAM of the f32 path.
//
// Python usage:
//   # arr_bf16 is a numpy uint16 view of the raw BF16 safetensors bytes
//   q, scales = quantize_int8_bf16(arr_bf16)

/// INT8 quantization of a BF16 weight matrix supplied as uint16 (raw bit pattern).
///
/// Input:  (N, D) uint16  — raw BF16 bits from safetensors  
/// Output: ((N, D) int8, (N,) float32)
#[pyfunction]
pub fn quantize_int8_bf16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, u16>,
) -> PyResult<(Bound<'py, PyArray2<i8>>, Bound<'py, PyArray1<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();

    let mut q_out:      Vec<i8>  = vec![0i8;  n_rows * n_cols];
    let mut scales_out: Vec<f32> = vec![0f32; n_rows];

    q_out
        .par_chunks_mut(n_cols)
        .zip(scales_out.par_iter_mut())
        .enumerate()
        .for_each(|(row_idx, (q_row, scale))| {
            let row = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous row");

            // Pass 1: abs-max in f32
            let row_max = row_slice
                .iter()
                .map(|&bits| bf16_to_f32(bits).abs())
                .fold(0.0f32, f32::max);

            let s = if row_max == 0.0 { 1.0f32 } else { row_max / 127.0 };
            *scale = s;
            let inv_s = 1.0 / s;

            // Pass 2: quantize
            for (q_val, &bits) in q_row.iter_mut().zip(row_slice.iter()) {
                let x = bf16_to_f32(bits);
                *q_val = (x * inv_s).round().clamp(-127.0, 127.0) as i8;
            }
        });

    let q_arr = Array2::from_shape_vec((n_rows, n_cols), q_out)
        .expect("shape").into_pyarray_bound(py);
    let s_arr = Array1::from_vec(scales_out).into_pyarray_bound(py);
    Ok((q_arr, s_arr))
}

/// Per-group INT8 quantization of a BF16 weight matrix supplied as uint16.
///
/// Input:  (N, D) uint16, group_size  
/// Output: ((N, D) int8, (N, D/group_size) float32)
#[pyfunction]
pub fn quantize_int8_grouped_bf16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, u16>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<i8>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();

    if n_cols % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_cols ({n_cols}) must be divisible by group_size ({group_size})"
        )));
    }
    let n_groups = n_cols / group_size;

    let mut q_out:      Vec<i8>  = vec![0i8;  n_rows * n_cols];
    let mut scales_out: Vec<f32> = vec![0f32; n_rows * n_groups];

    q_out
        .par_chunks_mut(n_cols)
        .zip(scales_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, (q_row, s_row))| {
            let row = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous");

            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let gmax  = row_slice[start..end]
                    .iter()
                    .map(|&bits| bf16_to_f32(bits).abs())
                    .fold(0.0f32, f32::max);
                let s    = if gmax == 0.0 { 1.0 } else { gmax / 127.0 };
                s_row[g] = s;
                let inv_s = 1.0 / s;
                for (q_val, &bits) in q_row[start..end].iter_mut().zip(row_slice[start..end].iter()) {
                    let x = bf16_to_f32(bits);
                    *q_val = (x * inv_s).round().clamp(-127.0, 127.0) as i8;
                }
            }
        });

    let q_arr = Array2::from_shape_vec((n_rows, n_cols), q_out)
        .expect("shape").into_pyarray_bound(py);
    let s_arr = Array2::from_shape_vec((n_rows, n_groups), scales_out)
        .expect("shape").into_pyarray_bound(py);
    Ok((q_arr, s_arr))
}

/// Per-group asymmetric INT4 quantization of a BF16 weight matrix as uint16.
///
/// Avoids the float32 intermediate copy: BF16 bits are converted per-element
/// inside the Rayon loop.  Peak RAM = shard_BF16 (1×) + nibble output (0.25×).
///
/// Input:  (N, D) uint16, group_size  
/// Output: ((N, D/2) uint8 packed, (N, D/gs) float32 scales, (N, D/gs) float32 offsets)
#[pyfunction]
pub fn quantize_int4_asymmetric_bf16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, u16>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();

    if n_cols % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_cols ({n_cols}) must be divisible by group_size ({group_size})"
        )));
    }
    if n_cols % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be even for INT4 packing"
        ));
    }

    let n_groups = n_cols / group_size;
    let n_packed = n_cols / 2;

    let mut packed_out:  Vec<u8>  = vec![0u8;  n_rows * n_packed];
    let mut scales_out:  Vec<f32> = vec![0f32; n_rows * n_groups];
    let mut offsets_out: Vec<f32> = vec![0f32; n_rows * n_groups];

    packed_out
        .par_chunks_mut(n_packed)
        .zip(scales_out.par_chunks_mut(n_groups))
        .zip(offsets_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, ((p_row, s_row), o_row))| {
            let row       = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous");

            // Per-group scale + gmin offset
            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let mut gmin = f32::INFINITY;
                let mut gmax = f32::NEG_INFINITY;
                for &bits in &row_slice[start..end] {
                    let v = bf16_to_f32(bits);
                    if v < gmin { gmin = v; }
                    if v > gmax { gmax = v; }
                }
                let scale = if gmax == gmin { 1.0f32 } else { (gmax - gmin) / 15.0 };
                s_row[g] = scale;
                o_row[g] = gmin;
            }

            // Quantize + pack nibbles
            for i in 0..n_packed {
                let j0 = i * 2;
                let j1 = j0 + 1;
                let g0 = j0 / group_size;
                let g1 = j1 / group_size;
                let x0 = bf16_to_f32(row_slice[j0]);
                let x1 = bf16_to_f32(row_slice[j1]);
                let q0 = ((x0 - o_row[g0]) / s_row[g0]).round().clamp(0.0, 15.0) as u8;
                let q1 = ((x1 - o_row[g1]) / s_row[g1]).round().clamp(0.0, 15.0) as u8;
                p_row[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
            }
        });

    let packed_arr  = Array2::from_shape_vec((n_rows, n_packed), packed_out)
        .expect("shape").into_pyarray_bound(py);
    let scales_arr  = Array2::from_shape_vec((n_rows, n_groups), scales_out)
        .expect("shape").into_pyarray_bound(py);
    let offsets_arr = Array2::from_shape_vec((n_rows, n_groups), offsets_out)
        .expect("shape").into_pyarray_bound(py);
    Ok((packed_arr, scales_arr, offsets_arr))
}


// ── PyO3 module registration ─────────────────────────────────────────────────

#[pymodule]
fn squish_quant(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantize_int8_f32,                   m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int8_grouped,               m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int8_f32,                 m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int8_grouped,             m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int4_grouped,               m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int4_grouped,             m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int4_asymmetric_grouped,    m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int4_asymmetric_grouped,  m)?)?;
    // BF16-native paths (accept uint16 numpy arrays — raw bf16 bits from safetensors)
    // These avoid the Python-side .astype(float32) copy, halving peak RAM per shard.
    m.add_function(wrap_pyfunction!(quantize_int8_bf16,                  m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int8_grouped_bf16,          m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int4_asymmetric_bf16,       m)?)?;
    Ok(())
}
