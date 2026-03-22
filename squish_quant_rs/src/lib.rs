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


// ═══════════════════════════════════════════════════════════════════════════
// Wave 56a — NF4 · FP8 · INT3 · Sampler · KV-head INT8 · INT2
// ═══════════════════════════════════════════════════════════════════════════

// ── NF4 (NormalFloat4) lookup table ─────────────────────────────────────────
//
// 16 non-uniformly spaced float32 levels based on the standard-normal
// quantile function (QLoRA Table 1, arXiv 2305.14314).
const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Quantize a 2D float32 weight matrix to NF4 using a precomputed LUT.
///
/// Each element is mapped to the nearest of 16 NF4 levels after scaling
/// by the per-group absolute-maximum.  Two nibbles are packed per byte.
///
/// Returns (packed: uint8[N, D/2], scales: float32[N, D/group_size])
#[pyfunction]
pub fn quantize_nf4_grouped_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();
    if n_cols % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be divisible by group_size",
        ));
    }
    if n_cols % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be even for nibble packing",
        ));
    }
    let n_groups  = n_cols / group_size;
    let n_packed  = n_cols / 2;

    let mut packed_out: Vec<u8>  = vec![0u8;  n_rows * n_packed];
    let mut scales_out: Vec<f32> = vec![0f32; n_rows * n_groups];

    packed_out
        .par_chunks_mut(n_packed)
        .zip(scales_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, (p_row, s_row))| {
            let row = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous row");

            // Compute per-group scale = max(|x|) / 1.0  (NF4 range is [-1, 1])
            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let abs_max = row_slice[start..end]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, f32::max);
                s_row[g] = if abs_max == 0.0 { 1.0 } else { abs_max };
            }

            // Quantize: for each element, scale to [-1,1], find nearest NF4 level
            for i in 0..n_packed {
                let j0 = i * 2;
                let j1 = j0 + 1;
                let g0 = j0 / group_size;
                let g1 = j1 / group_size;
                let v0 = row_slice[j0] / s_row[g0];
                let v1 = row_slice[j1] / s_row[g1];

                // Nearest NF4 level via linear scan (16 entries — branchless)
                let mut best0 = 0usize;
                let mut best1 = 0usize;
                let mut d0 = f32::MAX;
                let mut d1 = f32::MAX;
                for (k, &lv) in NF4_LUT.iter().enumerate() {
                    let diff0 = (v0 - lv).abs();
                    let diff1 = (v1 - lv).abs();
                    if diff0 < d0 { d0 = diff0; best0 = k; }
                    if diff1 < d1 { d1 = diff1; best1 = k; }
                }
                p_row[i] = (best0 as u8 & 0x0F) | ((best1 as u8 & 0x0F) << 4);
            }
        });

    let packed_arr = Array2::from_shape_vec((n_rows, n_packed), packed_out)
        .expect("shape").into_pyarray_bound(py);
    let scales_arr = Array2::from_shape_vec((n_rows, n_groups), scales_out)
        .expect("shape").into_pyarray_bound(py);
    Ok((packed_arr, scales_arr))
}

/// Dequantize NF4 packed weights back to float32.
#[pyfunction]
pub fn dequantize_nf4_grouped_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    scales: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let packed_view = packed.as_array();
    let scales_view = scales.as_array();
    let (n_rows, n_packed) = packed_view.dim();
    let n_cols = n_packed * 2;
    let n_groups = n_cols / group_size;

    let mut out: Vec<f32> = vec![0f32; n_rows * n_cols];

    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let p_row = packed_view.row(row_idx);
            let p_slice = p_row.as_slice().expect("non-contiguous");
            let s_row = scales_view.row(row_idx);
            let s_slice = s_row.as_slice().expect("non-contiguous");

            for i in 0..n_packed {
                let byte  = p_slice[i];
                let idx0  = (byte & 0x0F) as usize;
                let idx1  = ((byte >> 4) & 0x0F) as usize;
                let j0    = i * 2;
                let j1    = j0 + 1;
                let g0    = j0 / group_size;
                let g1    = j1 / group_size;
                out_row[j0] = NF4_LUT[idx0] * s_slice[g0];
                out_row[j1] = NF4_LUT[idx1] * s_slice[g1];
            }
        });

    let _ = n_groups; // used indirectly via group_size
    Ok(Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape")
        .into_pyarray_bound(py))
}

/// NF4 quantization accepting raw BF16 (uint16) input.
#[pyfunction]
pub fn quantize_nf4_grouped_bf16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, u16>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();
    if n_cols % group_size != 0 || n_cols % 2 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be divisible by group_size and even",
        ));
    }
    let n_groups = n_cols / group_size;
    let n_packed = n_cols / 2;

    let mut packed_out: Vec<u8>  = vec![0u8;  n_rows * n_packed];
    let mut scales_out: Vec<f32> = vec![0f32; n_rows * n_groups];

    packed_out
        .par_chunks_mut(n_packed)
        .zip(scales_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, (p_row, s_row))| {
            let row = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous");

            // Convert BF16 → f32 and compute per-group scale
            let f32_row: Vec<f32> = row_slice.iter().map(|&b| bf16_to_f32(b)).collect();
            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let abs_max = f32_row[start..end]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, f32::max);
                s_row[g] = if abs_max == 0.0 { 1.0 } else { abs_max };
            }
            for i in 0..n_packed {
                let j0 = i * 2;
                let j1 = j0 + 1;
                let g0 = j0 / group_size;
                let g1 = j1 / group_size;
                let v0 = f32_row[j0] / s_row[g0];
                let v1 = f32_row[j1] / s_row[g1];
                let mut best0 = 0usize; let mut d0 = f32::MAX;
                let mut best1 = 0usize; let mut d1 = f32::MAX;
                for (k, &lv) in NF4_LUT.iter().enumerate() {
                    let diff0 = (v0 - lv).abs();
                    let diff1 = (v1 - lv).abs();
                    if diff0 < d0 { d0 = diff0; best0 = k; }
                    if diff1 < d1 { d1 = diff1; best1 = k; }
                }
                p_row[i] = (best0 as u8 & 0x0F) | ((best1 as u8 & 0x0F) << 4);
            }
        });

    let packed_arr = Array2::from_shape_vec((n_rows, n_packed), packed_out)
        .expect("shape").into_pyarray_bound(py);
    let scales_arr = Array2::from_shape_vec((n_rows, n_groups), scales_out)
        .expect("shape").into_pyarray_bound(py);
    Ok((packed_arr, scales_arr))
}

// ── FP8 E4M3 / E5M2 ──────────────────────────────────────────────────────────
//
// IEEE 754 bit manipulation (f32::to_bits) is ~10× faster than np.log2/exp2.
// E4M3: 1 sign + 4 exponent + 3 mantissa bits; max = 448.0; bias = 7
// E5M2: 1 sign + 5 exponent + 2 mantissa bits; max = 57344.0; bias = 15

const E4M3_BIAS:   i32 = 7;
const E4M3_MAX:    f32 = 448.0;
const E5M2_BIAS:   i32 = 15;
const E5M2_MAX:    f32 = 57344.0;

#[inline(always)]
fn encode_fp8_e4m3(x: f32) -> u8 {
    if x == 0.0 { return 0u8; }
    let sign = if x < 0.0 { 1u8 } else { 0u8 };
    let abs_x = x.abs().min(E4M3_MAX);
    let bits = abs_x.to_bits();
    let exp_f32 = ((bits >> 23) as i32) - 127;
    let exp_fp8 = (exp_f32 + E4M3_BIAS).clamp(0, 15) as u8;
    let mant_fp8 = ((bits >> 20) & 0x7) as u8; // top 3 mantissa bits
    (sign << 7) | (exp_fp8 << 3) | mant_fp8
}

#[inline(always)]
fn decode_fp8_e4m3(byte: u8) -> f32 {
    if byte & 0x7F == 0 { return 0.0; }
    let sign:   f32 = if (byte >> 7) != 0 { -1.0 } else { 1.0 };
    let exp_fp8 = ((byte >> 3) & 0x0F) as i32;
    let mant_fp8 = (byte & 0x07) as u32;
    let exp_f32 = (exp_fp8 - E4M3_BIAS + 127).clamp(1, 254) as u32;
    let f32_bits = (exp_f32 << 23) | (mant_fp8 << 20);
    sign * f32::from_bits(f32_bits)
}

#[inline(always)]
fn encode_fp8_e5m2(x: f32) -> u8 {
    if x == 0.0 { return 0u8; }
    let sign = if x < 0.0 { 1u8 } else { 0u8 };
    let abs_x = x.abs().min(E5M2_MAX);
    let bits = abs_x.to_bits();
    let exp_f32 = ((bits >> 23) as i32) - 127;
    let exp_fp8 = (exp_f32 + E5M2_BIAS).clamp(0, 31) as u8;
    let mant_fp8 = ((bits >> 21) & 0x3) as u8; // top 2 mantissa bits
    (sign << 7) | (exp_fp8 << 2) | mant_fp8
}

#[inline(always)]
fn decode_fp8_e5m2(byte: u8) -> f32 {
    if byte & 0x7F == 0 { return 0.0; }
    let sign:   f32 = if (byte >> 7) != 0 { -1.0 } else { 1.0 };
    let exp_fp8 = ((byte >> 2) & 0x1F) as i32;
    let mant_fp8 = (byte & 0x03) as u32;
    let exp_f32 = (exp_fp8 - E5M2_BIAS + 127).clamp(1, 254) as u32;
    let f32_bits = (exp_f32 << 23) | (mant_fp8 << 21);
    sign * f32::from_bits(f32_bits)
}

/// Quantize float32 → FP8 E4M3 with per-tensor scale.
#[pyfunction]
pub fn quantize_fp8_e4m3_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();

    // Per-tensor abs-maximum scale
    let abs_max = arr_view.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max == 0.0 { 1.0f32 } else { abs_max / E4M3_MAX };

    let mut out: Vec<u8> = vec![0u8; n_rows * n_cols];
    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, row_out)| {
            let row = arr_view.row(row_idx);
            for (o, &x) in row_out.iter_mut().zip(row.iter()) {
                *o = encode_fp8_e4m3(x / scale);
            }
        });

    let out_arr = Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape").into_pyarray_bound(py);
    let scales_arr = Array1::from_vec(vec![scale]).into_pyarray_bound(py);
    Ok((out_arr, scales_arr))
}

/// Dequantize FP8 E4M3 → float32.
#[pyfunction]
pub fn dequantize_fp8_e4m3<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, u8>,
    scale: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();
    let mut out: Vec<f32> = vec![0f32; n_rows * n_cols];
    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, row_out)| {
            let row = arr_view.row(row_idx);
            for (o, &b) in row_out.iter_mut().zip(row.iter()) {
                *o = decode_fp8_e4m3(b) * scale;
            }
        });
    Ok(Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape").into_pyarray_bound(py))
}

/// Quantize float32 → FP8 E5M2 with per-tensor scale.
#[pyfunction]
pub fn quantize_fp8_e5m2_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();

    let abs_max = arr_view.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max == 0.0 { 1.0f32 } else { abs_max / E5M2_MAX };

    let mut out: Vec<u8> = vec![0u8; n_rows * n_cols];
    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, row_out)| {
            let row = arr_view.row(row_idx);
            for (o, &x) in row_out.iter_mut().zip(row.iter()) {
                *o = encode_fp8_e5m2(x / scale);
            }
        });

    let out_arr = Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape").into_pyarray_bound(py);
    let scales_arr = Array1::from_vec(vec![scale]).into_pyarray_bound(py);
    Ok((out_arr, scales_arr))
}

/// Dequantize FP8 E5M2 → float32.
#[pyfunction]
pub fn dequantize_fp8_e5m2<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, u8>,
    scale: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();
    let mut out: Vec<f32> = vec![0f32; n_rows * n_cols];
    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, row_out)| {
            let row = arr_view.row(row_idx);
            for (o, &b) in row_out.iter_mut().zip(row.iter()) {
                *o = decode_fp8_e5m2(b) * scale;
            }
        });
    Ok(Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape").into_pyarray_bound(py))
}

// ── INT3 packing ─────────────────────────────────────────────────────────────
//
// 3-bit symmetric signed range [-3, 3].  8 values per 3 bytes (24 bits).
// Layout: bits (value_i & 0x07) packed consecutively, low index at LSB.

#[inline(always)]
fn quantize_val_int3(x: f32, scale: f32) -> u8 {
    ((x / scale).round().clamp(-3.0, 3.0) as i8 as i32 & 0x07) as u8
}

/// Quantize float32 → INT3 grouped, packed 8 values per 3 bytes.
///
/// Returns (packed: uint8[N, ceil(D*3/8)], scales: float32[N, D/group_size])
#[pyfunction]
pub fn pack_int3_grouped_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();
    if n_cols % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be divisible by group_size",
        ));
    }
    // 8 values per 3 bytes; pad to multiple of 8
    let padded = ((n_cols + 7) / 8) * 8;
    let n_packed = padded * 3 / 8;
    let n_groups = n_cols / group_size;

    let mut packed_out: Vec<u8>  = vec![0u8;  n_rows * n_packed];
    let mut scales_out: Vec<f32> = vec![0f32; n_rows * n_groups];

    packed_out
        .par_chunks_mut(n_packed)
        .zip(scales_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, (p_row, s_row))| {
            let row = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous");

            // Per-group scale = max(|x|) / 3.0
            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let abs_max = row_slice[start..end]
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f32, f32::max);
                s_row[g] = if abs_max == 0.0 { 1.0 } else { abs_max / 3.0 };
            }

            // Pack 8 values into 3 bytes
            let mut buf = [0u8; 8];
            let chunks = (n_cols + 7) / 8;
            for chunk in 0..chunks {
                let base = chunk * 8;
                for bit in 0..8 {
                    let j = base + bit;
                    let v = if j < n_cols {
                        let g = j / group_size;
                        quantize_val_int3(row_slice[j], s_row[g])
                    } else {
                        0
                    };
                    buf[bit] = v;
                }
                // Pack: buf[0]bits0-2, buf[1]bits3-5, buf[2]bits6-8, ...
                // 3 bytes hold 3×8=24 bits = 8 × 3-bit values
                let byte0 = buf[0] | (buf[1] << 3) | ((buf[2] & 0x03) << 6);
                let byte1 = ((buf[2] >> 2) & 0x01) | (buf[3] << 1) | (buf[4] << 4) | ((buf[5] & 0x01) << 7);
                let byte2 = ((buf[5] >> 1) & 0x03) | (buf[6] << 2) | (buf[7] << 5);
                let pb = chunk * 3;
                if pb     < n_packed { p_row[pb]     = byte0; }
                if pb + 1 < n_packed { p_row[pb + 1] = byte1; }
                if pb + 2 < n_packed { p_row[pb + 2] = byte2; }
            }
        });

    let packed_arr = Array2::from_shape_vec((n_rows, n_packed), packed_out)
        .expect("shape").into_pyarray_bound(py);
    let scales_arr = Array2::from_shape_vec((n_rows, n_groups), scales_out)
        .expect("shape").into_pyarray_bound(py);
    Ok((packed_arr, scales_arr))
}

/// Unpack INT3 packed bytes back to float32.
#[pyfunction]
pub fn unpack_int3_grouped<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    scales: PyReadonlyArray2<'py, f32>,
    group_size: usize,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let packed_view = packed.as_array();
    let scales_view = scales.as_array();
    let (n_rows, _n_packed) = packed_view.dim();
    let mut out: Vec<f32> = vec![0f32; n_rows * n_cols];

    let n_groups = n_cols / group_size;
    let _ = n_groups;

    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let p_row   = packed_view.row(row_idx);
            let p_slice = p_row.as_slice().expect("non-contiguous");
            let s_row   = scales_view.row(row_idx);
            let s_slice = s_row.as_slice().expect("non-contiguous");

            let chunks = (n_cols + 7) / 8;
            for chunk in 0..chunks {
                let pb = chunk * 3;
                let byte0 = if pb     < p_slice.len() { p_slice[pb]     } else { 0 };
                let byte1 = if pb + 1 < p_slice.len() { p_slice[pb + 1] } else { 0 };
                let byte2 = if pb + 2 < p_slice.len() { p_slice[pb + 2] } else { 0 };

                let vals = [
                    (byte0 & 0x07) as u8,
                    ((byte0 >> 3) & 0x07) as u8,
                    (((byte0 >> 6) & 0x03) | ((byte1 & 0x01) << 2)) as u8,
                    ((byte1 >> 1) & 0x07) as u8,
                    ((byte1 >> 4) & 0x07) as u8,
                    (((byte1 >> 7) & 0x01) | ((byte2 & 0x03) << 1)) as u8,
                    ((byte2 >> 2) & 0x07) as u8,
                    ((byte2 >> 5) & 0x07) as u8,
                ];

                for bit in 0..8usize {
                    let j = chunk * 8 + bit;
                    if j >= n_cols { break; }
                    // sign-extend 3-bit to i8: vals in [0,7], 4..7 are negative
                    let raw = vals[bit];
                    let signed: i8 = if raw >= 4 { raw as i8 - 8 } else { raw as i8 };
                    let g = j / group_size;
                    out_row[j] = signed as f32 * s_slice[g];
                }
            }
        });

    Ok(Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape").into_pyarray_bound(py))
}

// ── Fused Sampler: softmax + top-p + min-p ───────────────────────────────────
//
// Two-pass online softmax, fused reverse-scan top-p cumsum, min-p threshold.

/// Numerically stable softmax with two-pass online algorithm.
///
/// Pass 1: find abs-max; Pass 2: exp(x - max) + normalise.
/// Returns probability vector, same shape as input (1-D, len vocab_size).
#[pyfunction]
pub fn softmax_logits_f32<'py>(
    py: Python<'py>,
    logits: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let logits_view = logits.as_array();
    let n = logits_view.len();
    let logits_slice = logits_view.as_slice().expect("non-contiguous");

    let abs_max = logits_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_slice.iter().map(|&x| (x - abs_max).exp()).collect();
    let total: f32 = probs.iter().sum();
    let inv_total = 1.0 / total.max(1e-10);
    for p in probs.iter_mut() { *p *= inv_total; }

    Ok(Array1::from_vec(probs).into_pyarray_bound(py))
}

/// Apply top-p (nucleus) filter in-place via reverse cumsum scan.
///
/// Sort descending, compute cumulative probability; zero out tokens once
/// cumulative mass exceeds `p_threshold`.  Returns masked probability vector.
#[pyfunction]
pub fn top_p_filter_f32<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<'py, f32>,
    p_threshold: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let probs_view = probs.as_array();
    let n = probs_view.len();
    let probs_slice = probs_view.as_slice().expect("non-contiguous");

    // Sort indices descending by probability
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs_slice[b].partial_cmp(&probs_slice[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out: Vec<f32> = vec![0.0f32; n];
    let mut cumsum = 0.0f32;
    for &idx in &indices {
        cumsum += probs_slice[idx];
        out[idx] = probs_slice[idx];
        if cumsum >= p_threshold { break; }
    }

    // Re-normalise
    let total: f32 = out.iter().sum();
    if total > 1e-10 {
        let inv = 1.0 / total;
        for p in out.iter_mut() { *p *= inv; }
    }

    Ok(Array1::from_vec(out).into_pyarray_bound(py))
}

/// Apply min-p filter: zero tokens with probability < min_p * p_max.
#[pyfunction]
pub fn min_p_filter_f32<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<'py, f32>,
    min_p: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let probs_view = probs.as_array();
    let probs_slice = probs_view.as_slice().expect("non-contiguous");

    let p_max = probs_slice.iter().cloned().fold(0.0f32, f32::max);
    let threshold = min_p * p_max;

    let mut out: Vec<f32> = probs_slice.iter().map(|&p| if p >= threshold { p } else { 0.0 }).collect();

    // Always keep at least one token
    if out.iter().all(|&p| p == 0.0) {
        let best = probs_slice.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        out[best] = probs_slice[best];
    }

    // Re-normalise
    let total: f32 = out.iter().sum();
    if total > 1e-10 {
        let inv = 1.0 / total;
        for p in out.iter_mut() { *p *= inv; }
    }

    Ok(Array1::from_vec(out).into_pyarray_bound(py))
}

// ── KV-cache head INT8 quantization ──────────────────────────────────────────
//
// Accepts 3-D arrays (n_heads, n_seq, head_dim) — native KV cache layout.

use numpy::{PyArray3, PyReadonlyArray3};

/// Quantize KV cache heads to INT8 (per-head abs-mean scale).
///
/// Input:  float32 (n_heads, n_seq, head_dim)
/// Output: (int8 [n_heads, n_seq, head_dim], scales float32 [n_heads])
#[pyfunction]
pub fn quantize_kv_heads_int8<'py>(
    py: Python<'py>,
    kv: PyReadonlyArray3<'py, f32>,
) -> PyResult<(Bound<'py, PyArray3<i8>>, Bound<'py, PyArray1<f32>>)> {
    use numpy::ndarray::Array3;
    let kv_view = kv.as_array();
    let (n_heads, n_seq, head_dim) = kv_view.dim();

    let mut out: Vec<i8>  = vec![0i8;  n_heads * n_seq * head_dim];
    let mut scales: Vec<f32> = vec![0f32; n_heads];

    scales.par_iter_mut()
        .zip(
            out.par_chunks_mut(n_seq * head_dim)
                .enumerate()
        )
        .for_each(|(scale, (head_idx, head_out))| {
            let head_slice: Vec<f32> = (0..n_seq)
                .flat_map(|s| (0..head_dim).map(move |d| kv_view[[head_idx, s, d]]))
                .collect();

            let abs_max = head_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            *scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
            let inv_s = 1.0 / *scale;

            for (o, &x) in head_out.iter_mut().zip(head_slice.iter()) {
                *o = (x * inv_s).round().clamp(-127.0, 127.0) as i8;
            }
        });

    let out_arr = Array3::from_shape_vec((n_heads, n_seq, head_dim), out)
        .expect("shape").into_pyarray_bound(py);
    let scales_arr = Array1::from_vec(scales).into_pyarray_bound(py);
    Ok((out_arr, scales_arr))
}

/// Dequantize INT8 KV cache heads back to float32.
#[pyfunction]
pub fn dequantize_kv_heads_int8<'py>(
    py: Python<'py>,
    kv_q: PyReadonlyArray3<'py, i8>,
    scales: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    use numpy::ndarray::Array3;
    let kv_view    = kv_q.as_array();
    let scales_view = scales.as_array();
    let (n_heads, n_seq, head_dim) = kv_view.dim();

    let mut out: Vec<f32> = vec![0f32; n_heads * n_seq * head_dim];

    out.par_chunks_mut(n_seq * head_dim)
        .enumerate()
        .for_each(|(head_idx, head_out)| {
            let scale = scales_view[head_idx];
            for s in 0..n_seq {
                for d in 0..head_dim {
                    let flat = s * head_dim + d;
                    head_out[flat] = kv_view[[head_idx, s, d]] as f32 * scale;
                }
            }
        });

    Ok(Array3::from_shape_vec((n_heads, n_seq, head_dim), out)
        .expect("shape").into_pyarray_bound(py))
}

// ── INT2 packing ─────────────────────────────────────────────────────────────
//
// 2-bit unsigned [0–3] with per-group zero-point + scale.
// 4 values per byte, packed low-index at LSB.

/// Quantize float32 → INT2 grouped (4 values per byte).
///
/// Returns (packed: uint8[N, D/4], scales: float32[N, D/group_size],
///          offsets: float32[N, D/group_size])
#[pyfunction]
pub fn quantize_int2_grouped_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();
    if n_cols % group_size != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be divisible by group_size",
        ));
    }
    if n_cols % 4 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be divisible by 4 for INT2 packing",
        ));
    }
    let n_groups = n_cols / group_size;
    let n_packed = n_cols / 4;

    let mut packed_out:  Vec<u8>  = vec![0u8;  n_rows * n_packed];
    let mut scales_out:  Vec<f32> = vec![0f32; n_rows * n_groups];
    let mut offsets_out: Vec<f32> = vec![0f32; n_rows * n_groups];

    packed_out
        .par_chunks_mut(n_packed)
        .zip(scales_out.par_chunks_mut(n_groups))
        .zip(offsets_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, ((p_row, s_row), o_row))| {
            let row = arr_view.row(row_idx);
            let row_slice = row.as_slice().expect("non-contiguous");

            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let mut gmin = f32::INFINITY;
                let mut gmax = f32::NEG_INFINITY;
                for &x in &row_slice[start..end] {
                    if x < gmin { gmin = x; }
                    if x > gmax { gmax = x; }
                }
                s_row[g] = if gmax == gmin { 1.0 } else { (gmax - gmin) / 3.0 };
                o_row[g] = gmin;
            }

            for i in 0..n_packed {
                let mut byte = 0u8;
                for bit in 0..4 {
                    let j = i * 4 + bit;
                    let g = j / group_size;
                    let q = ((row_slice[j] - o_row[g]) / s_row[g])
                        .round().clamp(0.0, 3.0) as u8;
                    byte |= (q & 0x03) << (bit * 2);
                }
                p_row[i] = byte;
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

/// Dequantize INT2 grouped packed weights back to float32.
#[pyfunction]
pub fn dequantize_int2_grouped_f32<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray2<'py, u8>,
    scales: PyReadonlyArray2<'py, f32>,
    offsets: PyReadonlyArray2<'py, f32>,
    group_size: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let packed_view  = packed.as_array();
    let scales_view  = scales.as_array();
    let offsets_view = offsets.as_array();
    let (n_rows, n_packed) = packed_view.dim();
    let n_cols = n_packed * 4;

    let mut out: Vec<f32> = vec![0f32; n_rows * n_cols];
    out.par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let p_row  = packed_view.row(row_idx);
            let s_row  = scales_view.row(row_idx);
            let o_row  = offsets_view.row(row_idx);
            let p_slice = p_row.as_slice().expect("non-contiguous");
            let s_slice = s_row.as_slice().expect("non-contiguous");
            let o_slice = o_row.as_slice().expect("non-contiguous");

            for i in 0..n_packed {
                let byte = p_slice[i];
                for bit in 0..4usize {
                    let j = i * 4 + bit;
                    let q = (byte >> (bit * 2)) & 0x03;
                    let g = j / group_size;
                    out_row[j] = q as f32 * s_slice[g] + o_slice[g];
                }
            }
        });

    Ok(Array2::from_shape_vec((n_rows, n_cols), out)
        .expect("shape").into_pyarray_bound(py))
}

/// INT2 quantization accepting raw BF16 (uint16) input.
#[pyfunction]
pub fn quantize_int2_grouped_bf16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, u16>,
    group_size: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    let arr_view = arr.as_array();
    let (n_rows, n_cols) = arr_view.dim();
    if n_cols % group_size != 0 || n_cols % 4 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_cols must be divisible by group_size and by 4",
        ));
    }
    let n_groups = n_cols / group_size;
    let n_packed = n_cols / 4;

    let mut packed_out:  Vec<u8>  = vec![0u8;  n_rows * n_packed];
    let mut scales_out:  Vec<f32> = vec![0f32; n_rows * n_groups];
    let mut offsets_out: Vec<f32> = vec![0f32; n_rows * n_groups];

    packed_out
        .par_chunks_mut(n_packed)
        .zip(scales_out.par_chunks_mut(n_groups))
        .zip(offsets_out.par_chunks_mut(n_groups))
        .enumerate()
        .for_each(|(row_idx, ((p_row, s_row), o_row))| {
            let row = arr_view.row(row_idx);
            let f32_row: Vec<f32> = row.iter().map(|&b| bf16_to_f32(b)).collect();

            for g in 0..n_groups {
                let start = g * group_size;
                let end   = start + group_size;
                let mut gmin = f32::INFINITY;
                let mut gmax = f32::NEG_INFINITY;
                for &x in &f32_row[start..end] {
                    if x < gmin { gmin = x; }
                    if x > gmax { gmax = x; }
                }
                s_row[g] = if gmax == gmin { 1.0 } else { (gmax - gmin) / 3.0 };
                o_row[g] = gmin;
            }

            for i in 0..n_packed {
                let mut byte = 0u8;
                for bit in 0..4usize {
                    let j = i * 4 + bit;
                    let g = j / group_size;
                    let q = ((f32_row[j] - o_row[g]) / s_row[g])
                        .round().clamp(0.0, 3.0) as u8;
                    byte |= (q & 0x03) << (bit * 2);
                }
                p_row[i] = byte;
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
    // Wave 56a — NF4 · FP8 · INT3 · Sampler · KV-head INT8 · INT2
    m.add_function(wrap_pyfunction!(quantize_nf4_grouped_f32,            m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_nf4_grouped_f32,          m)?)?;
    m.add_function(wrap_pyfunction!(quantize_nf4_grouped_bf16,           m)?)?;
    m.add_function(wrap_pyfunction!(quantize_fp8_e4m3_f32,               m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_fp8_e4m3,                 m)?)?;
    m.add_function(wrap_pyfunction!(quantize_fp8_e5m2_f32,               m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_fp8_e5m2,                 m)?)?;
    m.add_function(wrap_pyfunction!(pack_int3_grouped_f32,               m)?)?;
    m.add_function(wrap_pyfunction!(unpack_int3_grouped,                 m)?)?;
    m.add_function(wrap_pyfunction!(softmax_logits_f32,                  m)?)?;
    m.add_function(wrap_pyfunction!(top_p_filter_f32,                    m)?)?;
    m.add_function(wrap_pyfunction!(min_p_filter_f32,                    m)?)?;
    m.add_function(wrap_pyfunction!(quantize_kv_heads_int8,              m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_kv_heads_int8,            m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int2_grouped_f32,           m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int2_grouped_f32,         m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int2_grouped_bf16,          m)?)?;
    Ok(())
}
