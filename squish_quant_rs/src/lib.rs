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


// ─────────────────────────────────────────────────────────────────────────────
// Wave 57a — Entropy Codec · PQ Accelerate · GRU Cell · Batch CosSim · SwiGLU · Randomized SVD
// ─────────────────────────────────────────────────────────────────────────────

// ── rANS + Huffman Entropy Codec ─────────────────────────────────────────────

/// Build a CDF table from an array of symbol frequencies.
/// Returns a [u32; 256] CDF where cdf[s] = sum of freqs[0..s].
fn build_cdf(freqs: &[u32; 256]) -> [u32; 256] {
    let mut cdf = [0u32; 256];
    let mut acc = 0u32;
    for i in 0..255 {
        cdf[i] = acc;
        acc += freqs[i];
    }
    cdf[255] = acc;
    cdf
}

/// rANS encode: symbols -> bytes via state machine over [u32; 256] CDF.
/// `freqs` is a length-256 array of symbol frequencies (sum = M).
#[pyfunction]
fn rans_encode<'py>(
    py: Python<'py>,
    symbols: PyReadonlyArray1<u8>,
    freqs: PyReadonlyArray1<u32>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let syms = symbols.as_slice()?;
    let freq_slice = freqs.as_slice()?;
    if freq_slice.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("freqs must have length 256"));
    }
    let mut freq_arr = [0u32; 256];
    freq_arr.copy_from_slice(freq_slice);
    let cdf = build_cdf(&freq_arr);
    let m: u32 = freq_arr.iter().sum();
    if m == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("sum of freqs must be > 0"));
    }

    // rANS encode in reverse order
    let mut state: u64 = 1u64 << 31;
    let mut out_bytes: Vec<u8> = Vec::with_capacity(syms.len() + 16);
    for &sym in syms.iter().rev() {
        let fs = freq_arr[sym as usize] as u64;
        let cs = cdf[sym as usize] as u64;
        let m64 = m as u64;
        // renorm: push low bytes until state is in [fs * L_upper, fs * L_upper*256)
        let l_upper: u64 = 1 << 23;
        while state >= fs * l_upper {
            out_bytes.push((state & 0xFF) as u8);
            state >>= 8;
        }
        state = (state / fs) * m64 + cs + (state % fs);
    }
    // encode final state (4 bytes, little-endian)
    for i in 0..4 {
        out_bytes.push(((state >> (i * 8)) & 0xFF) as u8);
    }
    out_bytes.reverse();
    Ok(PyArray1::from_vec_bound(py, out_bytes))
}

/// rANS decode: bytes -> symbols.
#[pyfunction]
fn rans_decode<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<u8>,
    freqs: PyReadonlyArray1<u32>,
    n_symbols: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let data_slice = data.as_slice()?;
    let freq_slice = freqs.as_slice()?;
    if freq_slice.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("freqs must have length 256"));
    }
    let mut freq_arr = [0u32; 256];
    freq_arr.copy_from_slice(freq_slice);
    let cdf = build_cdf(&freq_arr);
    let m: u32 = freq_arr.iter().sum();
    if m == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("sum of freqs must be > 0"));
    }

    if data_slice.len() < 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("data too short"));
    }
    let mut pos = 0usize;
    let mut state: u64 = 0;
    for i in 0..4usize {
        state |= (data_slice[pos] as u64) << (i * 8);
        pos += 1;
    }

    // Build inverse CDF lookup: cumfreq -> symbol
    let mut inv_cdf = vec![0u8; m as usize];
    for sym in 0..256usize {
        let start = cdf[sym] as usize;
        let end = if sym < 255 { cdf[sym + 1] as usize } else { m as usize };
        for k in start..end {
            if k < inv_cdf.len() {
                inv_cdf[k] = sym as u8;
            }
        }
    }

    let mut out = Vec::with_capacity(n_symbols);
    let m64 = m as u64;
    for _ in 0..n_symbols {
        let slot = (state % m64) as usize;
        let sym = inv_cdf[slot.min(inv_cdf.len().saturating_sub(1))];
        out.push(sym);
        let fs = freq_arr[sym as usize] as u64;
        let cs = cdf[sym as usize] as u64;
        state = fs * (state / m64) + slot as u64 - cs;
        // renorm: read bytes until state is in [L_lower, L_lower * 256)
        let l_lower: u64 = 1 << 23;
        while state < l_lower && pos < data_slice.len() {
            state = (state << 8) | (data_slice[pos] as u64);
            pos += 1;
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Canonical Huffman encode: symbols+codebook -> packed bit-string as bytes.
/// `code_words` and `code_lens` are 256-element arrays.
#[pyfunction]
fn huffman_encode<'py>(
    py: Python<'py>,
    symbols: PyReadonlyArray1<u8>,
    code_words: PyReadonlyArray1<u32>,
    code_lens: PyReadonlyArray1<u8>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let syms = symbols.as_slice()?;
    let cw = code_words.as_slice()?;
    let cl = code_lens.as_slice()?;
    if cw.len() != 256 || cl.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("code_words and code_lens must have length 256"));
    }

    let mut out: Vec<u8> = Vec::with_capacity(syms.len() / 2 + 8);
    let mut bit_buf: u64 = 0;
    let mut bits_used: u32 = 0;
    for &sym in syms {
        let word = cw[sym as usize] as u64;
        let len = cl[sym as usize] as u32;
        bit_buf = (bit_buf << len) | word;
        bits_used += len;
        while bits_used >= 8 {
            bits_used -= 8;
            out.push(((bit_buf >> bits_used) & 0xFF) as u8);
        }
    }
    // flush remaining bits
    if bits_used > 0 {
        out.push(((bit_buf << (8 - bits_used)) & 0xFF) as u8);
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Canonical Huffman decode: packed bytes -> symbols.
/// `code_words` and `code_lens` must describe a prefix-free code (256-symbol alphabet).
#[pyfunction]
fn huffman_decode<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<u8>,
    code_words: PyReadonlyArray1<u32>,
    code_lens: PyReadonlyArray1<u8>,
    n_symbols: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let data_slice = data.as_slice()?;
    let cw = code_words.as_slice()?;
    let cl = code_lens.as_slice()?;
    if cw.len() != 256 || cl.len() != 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("code_words and code_lens must have length 256"));
    }

    let mut out = Vec::with_capacity(n_symbols);
    let mut bit_buf: u64 = 0;
    let mut bits_avail: u32 = 0;
    let mut byte_pos = 0usize;

    let fill = |bit_buf: &mut u64, bits_avail: &mut u32, byte_pos: &mut usize| {
        while *bits_avail < 32 && *byte_pos < data_slice.len() {
            *bit_buf = (*bit_buf << 8) | (data_slice[*byte_pos] as u64);
            *bits_avail += 8;
            *byte_pos += 1;
        }
    };

    for _ in 0..n_symbols {
        fill(&mut bit_buf, &mut bits_avail, &mut byte_pos);
        // linear scan for matching code
        let mut matched = false;
        for sym in 0..256usize {
            let len = cl[sym] as u32;
            if len == 0 || len > bits_avail { continue; }
            let shift = bits_avail - len;
            let extracted = (bit_buf >> shift) & ((1u64 << len) - 1);
            if extracted == cw[sym] as u64 {
                out.push(sym as u8);
                bits_avail -= len;
                bit_buf &= (1u64 << bits_avail) - 1;
                matched = true;
                break;
            }
        }
        if !matched {
            // push a zero symbol and continue (graceful degradation)
            out.push(0u8);
        }
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

// ── PQ Accelerate (K-Means + ADC) ────────────────────────────────────────────

/// Fit K-means centroids over `(N, D)` float32 data using Rayon parallel distance computation.
/// Returns `(K, D)` centroids.
#[pyfunction]
fn pq_kmeans_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f32>,
    n_clusters: usize,
    n_iter: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let arr = data.as_array();
    let (n, d) = arr.dim();
    if n_clusters == 0 || n_clusters > n {
        return Err(pyo3::exceptions::PyValueError::new_err("n_clusters out of range"));
    }

    // K-means++ seeding: choose first centroid uniformly, then by distance probability
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);
    centroids.push(arr.row(0).to_vec());
    let mut rng_state: u64 = 0xDEADBEEF_CAFEF00D;
    let lcg_next = |s: &mut u64| -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    };

    for _ in 1..n_clusters {
        let dists: Vec<f32> = (0..n).map(|i| {
            let row = arr.row(i);
            centroids.iter().map(|c| {
                row.iter().zip(c.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum::<f32>()
            }).fold(f32::INFINITY, f32::min)
        }).collect();
        let total: f32 = dists.iter().sum();
        let threshold = (lcg_next(&mut rng_state) as f64 / u64::MAX as f64) as f32 * total;
        let mut cum = 0.0f32;
        let mut chosen = n - 1;
        for (i, &d_sq) in dists.iter().enumerate() {
            cum += d_sq;
            if cum >= threshold { chosen = i; break; }
        }
        centroids.push(arr.row(chosen).to_vec());
    }

    // Lloyd iterations
    let mut assignments = vec![0usize; n];
    for _ in 0..n_iter {
        // Assign
        assignments.par_iter_mut().enumerate().for_each(|(i, a)| {
            let row = arr.row(i);
            let mut best_dist = f32::INFINITY;
            let mut best_k = 0;
            for (k, c) in centroids.iter().enumerate() {
                let dist: f32 = row.iter().zip(c.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum();
                if dist < best_dist { best_dist = dist; best_k = k; }
            }
            *a = best_k;
        });
        // Update
        let mut sums = vec![vec![0.0f32; d]; n_clusters];
        let mut counts = vec![0usize; n_clusters];
        for (i, &k) in assignments.iter().enumerate() {
            let row = arr.row(i);
            for (s, &v) in sums[k].iter_mut().zip(row.iter()) { *s += v; }
            counts[k] += 1;
        }
        for k in 0..n_clusters {
            if counts[k] > 0 {
                let inv = 1.0 / counts[k] as f32;
                for v in sums[k].iter_mut() { *v *= inv; }
                centroids[k] = sums[k].clone();
            }
        }
    }

    let flat: Vec<f32> = centroids.into_iter().flatten().collect();
    let arr2 = Array2::from_shape_vec((n_clusters, d), flat).expect("shape");
    Ok(arr2.into_pyarray_bound(py))
}

/// PQ encode: assign each row of `(N, D)` data to the nearest centroid in `(K, D)`.
/// Returns `(N,)` u8 code array (supports K ≤ 256).
#[pyfunction]
fn pq_encode_batch<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f32>,
    centroids: PyReadonlyArray2<f32>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let data_arr = data.as_array();
    let cent_arr = centroids.as_array();
    let (n, _d) = data_arr.dim();
    let (k, _) = cent_arr.dim();
    if k > 256 {
        return Err(pyo3::exceptions::PyValueError::new_err("n_clusters must be ≤ 256"));
    }

    let codes: Vec<u8> = (0..n).into_par_iter().map(|i| {
        let row = data_arr.row(i);
        let mut best_dist = f32::INFINITY;
        let mut best_k = 0u8;
        for ki in 0..k {
            let c = cent_arr.row(ki);
            let dist: f32 = row.iter().zip(c.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum();
            if dist < best_dist { best_dist = dist; best_k = ki as u8; }
        }
        best_k
    }).collect();

    Ok(PyArray1::from_vec_bound(py, codes))
}

/// PQ ADC search: for each query `(M, D/M)`, compute distances to all N codes
/// using precomputed LUT `(M, K)` and return `(N,)` total distance float32 array.
/// `codes` has shape `(N, M)` and `lut` has shape `(M, K)`.
#[pyfunction]
fn pq_adc_search<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray2<u8>,
    lut: PyReadonlyArray2<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let codes_arr = codes.as_array();
    let lut_arr = lut.as_array();
    let (n, m) = codes_arr.dim();
    let (lm, _k) = lut_arr.dim();
    if m != lm {
        return Err(pyo3::exceptions::PyValueError::new_err("codes M dim must match lut M dim"));
    }

    let dists: Vec<f32> = (0..n).into_par_iter().map(|i| {
        let mut total = 0.0f32;
        for mi in 0..m {
            let c = codes_arr[(i, mi)] as usize;
            total += lut_arr[(mi, c)];
        }
        total
    }).collect();

    Ok(PyArray1::from_vec_bound(py, dists))
}

// ── Fused GRU Cell ────────────────────────────────────────────────────────────

/// Fused GRU step: accepts pre-multiplied gates_x and gates_h `(3 * hidden_dim,)` float32.
/// Computes reset, update, candidate gates and output h_new in one Rayon SIMD pass.
/// Returns h_new `(hidden_dim,)` float32.
#[pyfunction]
fn gru_step_f32<'py>(
    py: Python<'py>,
    gates_x: PyReadonlyArray1<f32>,
    gates_h: PyReadonlyArray1<f32>,
    h_prev: PyReadonlyArray1<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let gx = gates_x.as_slice()?;
    let gh = gates_h.as_slice()?;
    let hp = h_prev.as_slice()?;
    let total = gx.len();
    if total % 3 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("gates_x length must be divisible by 3"));
    }
    if gh.len() != total || hp.len() != total / 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("gates_h and h_prev dimension mismatch"));
    }
    let hd = total / 3;

    let sigmoid = |x: f32| -> f32 { 1.0 / (1.0 + (-x).exp()) };

    let mut h_new = vec![0.0f32; hd];
    h_new.par_iter_mut().enumerate().for_each(|(i, out)| {
        let r = sigmoid(gx[i] + gh[i]);
        let z = sigmoid(gx[hd + i] + gh[hd + i]);
        let n = (gx[2 * hd + i] + r * gh[2 * hd + i]).tanh();
        *out = (1.0 - z) * n + z * hp[i];
    });

    Ok(PyArray1::from_vec_bound(py, h_new))
}

// ── Batched Cosine Similarity ─────────────────────────────────────────────────

/// Compute `(T_a, T_b)` cosine similarity matrix from `(T_a, D)` and `(T_b, D)` float32.
/// Fused: row norms and dot products computed in one Rayon pass over T_a rows.
#[pyfunction]
fn batched_cosine_similarity_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f32>,
    b: PyReadonlyArray2<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let (ta, da) = a_arr.dim();
    let (tb, db) = b_arr.dim();
    if da != db {
        return Err(pyo3::exceptions::PyValueError::new_err("a and b must have same embedding dim"));
    }

    // Precompute b norms
    let b_norms: Vec<f32> = (0..tb).map(|j| {
        let row = b_arr.row(j);
        let sq: f32 = row.iter().map(|&x| x * x).sum();
        sq.sqrt().max(1e-12)
    }).collect();

    let flat: Vec<f32> = (0..ta).into_par_iter().flat_map(|i| {
        let a_row = a_arr.row(i);
        let a_norm: f32 = a_row.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
        (0..tb).map(|j| {
            let b_row = b_arr.row(j);
            let dot: f32 = a_row.iter().zip(b_row.iter()).map(|(&x, &y)| x * y).sum();
            dot / (a_norm * b_norms[j])
        }).collect::<Vec<f32>>()
    }).collect();

    let out = Array2::from_shape_vec((ta, tb), flat).expect("shape");
    Ok(out.into_pyarray_bound(py))
}

// ── SwiGLU / SiLU ────────────────────────────────────────────────────────────

/// Fused SiLU element-wise: returns `x * sigmoid(x)` for each element of a `(N,)` array.
#[pyfunction]
fn silu_f32<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let xs = x.as_slice()?;
    let out: Vec<f32> = xs.par_iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

/// Fused SwiGLU: `gate * sigmoid(gate) * up` element-wise.
/// `gate` and `up` must have the same length `(N,)`.
#[pyfunction]
fn swiglu_f32<'py>(
    py: Python<'py>,
    gate: PyReadonlyArray1<f32>,
    up: PyReadonlyArray1<f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let gs = gate.as_slice()?;
    let us = up.as_slice()?;
    if gs.len() != us.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("gate and up must have the same length"));
    }
    let out: Vec<f32> = gs.par_iter().zip(us.par_iter())
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

// ── Randomized SVD ────────────────────────────────────────────────────────────

/// Randomized SVD of `(m, n)` float32 matrix A.
/// Returns (U `(m, rank)`, S `(rank,)`, Vt `(rank, n)`).
/// Uses: Gaussian sketch Ω `(n, rank+oversample)`, Y = A×Ω, QR(Y), thin SVD of Q^T×A.
#[pyfunction]
fn randomized_svd_f32<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f32>,
    rank: usize,
    n_oversamples: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
    let a_arr = a.as_array();
    let (m, n) = a_arr.dim();
    let k = (rank + n_oversamples).min(m.min(n));

    // Gaussian sketch Ω: (n, k) using LCG PRNG
    let mut omega = vec![0.0f32; n * k];
    let mut rng: u64 = 0x123456789ABCDEF0;
    for v in omega.iter_mut() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Box-Muller using two LCG values
        let u1 = (rng >> 11) as f64 / (1u64 << 53) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
        *v = ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
    }

    // Y = A × Ω: (m, k)
    let mut y_flat = vec![0.0f32; m * k];
    y_flat.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        let a_row = a_arr.row(i);
        for j in 0..k {
            row[j] = a_row.iter().enumerate().map(|(l, &v)| v * omega[l * k + j]).sum();
        }
    });

    // QR decomposition of Y via Gram-Schmidt: Q is (m, k)
    let mut q_cols: Vec<Vec<f32>> = Vec::with_capacity(k);
    for j in 0..k {
        let mut col: Vec<f32> = (0..m).map(|i| y_flat[i * k + j]).collect();
        for prev in q_cols.iter() {
            let dot: f32 = col.iter().zip(prev.iter()).map(|(&a, &b)| a * b).sum();
            for (c, &p) in col.iter_mut().zip(prev.iter()) { *c -= dot * p; }
        }
        let norm: f32 = col.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in col.iter_mut() { *v /= norm; }
        }
        q_cols.push(col);
    }

    // B = Q^T × A: (k, n)
    let mut b_flat = vec![0.0f32; k * n];
    b_flat.par_chunks_mut(n).enumerate().for_each(|(j, row)| {
        let q_col = &q_cols[j];
        for l in 0..n {
            row[l] = q_col.iter().enumerate().map(|(i, &q)| q * a_arr[(i, l)]).sum();
        }
    });

    // Thin SVD of B (k, n) via one-sided Jacobi for small k
    // For simplicity, use power iteration + eigendecomposition of B×B^T (k×k)
    // Compute B × B^T: (k, k) symmetric
    let mut bbt = vec![0.0f32; k * k];
    for i in 0..k {
        for j in i..k {
            let dot: f32 = (0..n).map(|l| b_flat[i * n + l] * b_flat[j * n + l]).sum();
            bbt[i * k + j] = dot;
            bbt[j * k + i] = dot;
        }
    }

    // Extract eigenvalues/vectors of BBT via Jacobi iterations
    let mut eig_vecs: Vec<Vec<f32>> = (0..k).map(|i| {
        let mut e = vec![0.0f32; k];
        e[i] = 1.0;
        e
    }).collect();
    let mut eig_vals: Vec<f32> = (0..k).map(|i| bbt[i * k + i]).collect();

    for _ in 0..30 {
        for p in 0..k {
            for q in (p + 1)..k {
                let a_pp = eig_vals[p];
                let a_qq = eig_vals[q];
                let a_pq = {
                    let mut val = 0.0f32;
                    for l in 0..k {
                        val += eig_vecs[p][l] * (0..k).map(|m2| bbt[l * k + m2] * eig_vecs[q][m2]).sum::<f32>();
                    }
                    val
                };
                if a_pq.abs() < 1e-8 { continue; }
                let tau = (a_qq - a_pp) / (2.0 * a_pq);
                let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) }
                        else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                eig_vals[p] = a_pp - t * a_pq;
                eig_vals[q] = a_qq + t * a_pq;
                let ep = eig_vecs[p].clone();
                let eq = eig_vecs[q].clone();
                for l in 0..k {
                    eig_vecs[p][l] = c * ep[l] - s * eq[l];
                    eig_vecs[q][l] = s * ep[l] + c * eq[l];
                }
            }
        }
    }

    // Sort by descending eigenvalue
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&a, &b| eig_vals[b].partial_cmp(&eig_vals[a]).unwrap_or(std::cmp::Ordering::Equal));

    let actual_rank = rank.min(k);
    let mut s_out = vec![0.0f32; actual_rank];
    let mut u_out = vec![0.0f32; m * actual_rank];
    let mut vt_out = vec![0.0f32; actual_rank * n];

    for ri in 0..actual_rank {
        let i = idx[ri];
        let sv = eig_vals[i].max(0.0).sqrt();
        s_out[ri] = sv;
        // U[:,ri] = Q × eig_vecs[i]
        let ev = &eig_vecs[i];
        for row in 0..m {
            u_out[row * actual_rank + ri] = q_cols.iter().enumerate().map(|(j, qc)| qc[row] * ev[j]).sum();
        }
        // Vt[ri,:] = B^T × eig_vecs[i] / sv (= V[:,ri]^T)
        if sv > 1e-10 {
            let inv_sv = 1.0 / sv;
            for col in 0..n {
                vt_out[ri * n + col] =
                    (0..k).map(|j| b_flat[j * n + col] * ev[j]).sum::<f32>() * inv_sv;
            }
        }
    }

    let u_arr = Array2::from_shape_vec((m, actual_rank), u_out).expect("shape");
    let s_arr = Array1::from(s_out);
    let vt_arr = Array2::from_shape_vec((actual_rank, n), vt_out).expect("shape");

    Ok((
        u_arr.into_pyarray_bound(py),
        s_arr.into_pyarray_bound(py),
        vt_arr.into_pyarray_bound(py),
    ))
}

// ── Wave 58a: VectorKMeans ─────────────────────────────────────────────────

/// K-means++ initialisation + Lloyd iterations for vector codebook fitting.
///
/// Returns centroids as a (n_clusters, dim) float32 array.
#[pyfunction]
fn vector_kmeans_fit_f32(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    n_clusters: usize,
    n_iter: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let data = data.as_array();
    let (n, d) = (data.nrows(), data.ncols());
    if n_clusters == 0 || n_clusters > n {
        return Err(pyo3::exceptions::PyValueError::new_err("n_clusters out of range"));
    }
    // K-means++ seeding: pick first = row 0; remaining = farthest from current set
    let mut centroids: Vec<f32> = Vec::with_capacity(n_clusters * d);
    centroids.extend_from_slice(data.row(0).as_slice().unwrap_or(&[]));

    for _ in 1..n_clusters {
        let current_k = centroids.len() / d;
        let dists: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = data.row(i);
                (0..current_k)
                    .map(|c| {
                        let c_start = c * d;
                        row.iter()
                            .zip(&centroids[c_start..c_start + d])
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>()
                    })
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        let best = dists
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        centroids.extend_from_slice(data.row(best).as_slice().unwrap_or(&[]));
    }

    let mut centroids_arr = Array2::<f32>::from_shape_vec((n_clusters, d), centroids)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    // Lloyd iterations
    for _ in 0..n_iter {
        // E-step: parallel nearest-centroid assignment
        let assignments: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = data.row(i);
                (0..n_clusters)
                    .min_by(|&a, &b| {
                        let da: f32 = row
                            .iter()
                            .zip(centroids_arr.row(a))
                            .map(|(x, c)| (x - c) * (x - c))
                            .sum();
                        let db: f32 = row
                            .iter()
                            .zip(centroids_arr.row(b))
                            .map(|(x, c)| (x - c) * (x - c))
                            .sum();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            })
            .collect();

        // M-step: recompute centroids
        let mut sums = vec![0.0f32; n_clusters * d];
        let mut counts = vec![0usize; n_clusters];
        for (i, &ci) in assignments.iter().enumerate() {
            counts[ci] += 1;
            for (j, &v) in data.row(i).iter().enumerate() {
                sums[ci * d + j] += v;
            }
        }
        for c in 0..n_clusters {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..d {
                    centroids_arr[[c, j]] = sums[c * d + j] * inv;
                }
            }
        }
    }

    Ok(centroids_arr.into_pyarray_bound(py).unbind())
}

/// Assign each input vector to its nearest centroid (argmin distance).
///
/// Returns (N,) int32 index array.
#[pyfunction]
fn vector_kmeans_assign_f32(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
    centroids: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray1<i32>>> {
    let data = data.as_array();
    let centroids = centroids.as_array();
    let n = data.nrows();
    let k = centroids.nrows();

    let assignments: Vec<i32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = data.row(i);
            (0..k)
                .min_by(|&a, &b| {
                    let da: f32 = row
                        .iter()
                        .zip(centroids.row(a))
                        .map(|(x, c)| (x - c) * (x - c))
                        .sum();
                    let db: f32 = row
                        .iter()
                        .zip(centroids.row(b))
                        .map(|(x, c)| (x - c) * (x - c))
                        .sum();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0) as i32
        })
        .collect();

    Ok(Array1::from(assignments).into_pyarray_bound(py).unbind())
}

/// Reconstruct vectors from centroid indices.
///
/// Returns (N, D) float32 array where output[i] = centroids[indices[i]].
#[pyfunction]
fn vector_kmeans_reconstruct_f32(
    py: Python<'_>,
    indices: PyReadonlyArray1<i32>,
    centroids: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let indices = indices.as_array();
    let centroids = centroids.as_array();
    let n = indices.len();
    let (k, d) = (centroids.nrows(), centroids.ncols());

    let mut out = Array2::<f32>::zeros((n, d));
    for (i, &idx) in indices.iter().enumerate() {
        let ci = (idx as usize).min(k.saturating_sub(1));
        for j in 0..d {
            out[[i, j]] = centroids[[ci, j]];
        }
    }
    Ok(out.into_pyarray_bound(py).unbind())
}

// ── Wave 58a: FP6 BitPack ──────────────────────────────────────────────────

/// Encode a flat float32 array into FP6 packed bytes (4 FP6 values → 3 bytes).
///
/// FP6 layout: 1 sign + exp_bits exponent + man_bits mantissa (must sum to 5).
/// Input length must be a multiple of 4.
#[pyfunction]
fn fp6_encode_f32(
    _py: Python<'_>,
    data: PyReadonlyArray1<f32>,
    exp_bits: u32,
    man_bits: u32,
) -> PyResult<Vec<u8>> {
    let data = data.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("array must be contiguous")
    })?;
    if data.len() % 4 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("length must be a multiple of 4"));
    }
    if 1 + exp_bits + man_bits != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err("exp_bits + man_bits must equal 5"));
    }

    let exp_bias = (1u32 << (exp_bits - 1)).saturating_sub(1);
    let max_exp = (1u32 << exp_bits) - 1;
    let man_mask = (1u32 << man_bits) - 1;

    let encode_val = |v: f32| -> u8 {
        if v == 0.0 || v.is_nan() { return 0; }
        let sign: u8 = if v < 0.0 { 1 } else { 0 };
        let bits = v.abs().to_bits();
        let f32_exp = ((bits >> 23) & 0xFF) as i32;
        let f32_man = bits & 0x007F_FFFF;
        let rebias: i32 = (f32_exp - 127) + exp_bias as i32;
        let (enc_exp, enc_man);
        if rebias <= 0 {
            enc_exp = 0u8; enc_man = 0u8;
        } else if rebias >= max_exp as i32 {
            enc_exp = max_exp as u8; enc_man = man_mask as u8;
        } else {
            enc_exp = rebias as u8;
            enc_man = (f32_man >> (23 - man_bits)) as u8;
        }
        (sign << 5) | ((enc_exp & ((1 << exp_bits) - 1)) << man_bits) | (enc_man & man_mask as u8)
    };

    let n_packs = data.len() / 4;
    let mut out = vec![0u8; n_packs * 3];
    data.chunks_exact(4)
        .zip(out.chunks_exact_mut(3))
        .for_each(|(chunk, dst)| {
            let a = encode_val(chunk[0]) as u32;
            let b = encode_val(chunk[1]) as u32;
            let c = encode_val(chunk[2]) as u32;
            let e = encode_val(chunk[3]) as u32;
            let packed: u32 = (a << 18) | (b << 12) | (c << 6) | e;
            dst[0] = (packed >> 16) as u8;
            dst[1] = (packed >> 8) as u8;
            dst[2] = packed as u8;
        });
    Ok(out)
}

/// Decode FP6 packed bytes back to float32 (3 bytes → 4 FP6 values).
///
/// Input must be a multiple of 3 bytes.
#[pyfunction]
fn fp6_decode_f32(
    py: Python<'_>,
    packed: Vec<u8>,
    exp_bits: u32,
    man_bits: u32,
) -> PyResult<Py<PyArray1<f32>>> {
    if packed.len() % 3 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("packed length must be a multiple of 3"));
    }
    if 1 + exp_bits + man_bits != 6 {
        return Err(pyo3::exceptions::PyValueError::new_err("exp_bits + man_bits must equal 5"));
    }

    let exp_bias = (1u32 << (exp_bits - 1)).saturating_sub(1);
    let man_mask = (1u32 << man_bits) - 1;
    let exp_mask = (1u32 << exp_bits) - 1;

    let decode_val = |bits6: u32| -> f32 {
        let sign = (bits6 >> 5) & 1;
        let enc_exp = (bits6 >> man_bits) & exp_mask;
        let enc_man = bits6 & man_mask;
        if enc_exp == 0 && enc_man == 0 { return 0.0; }
        let f32_exp = ((enc_exp as i32 - exp_bias as i32) + 127).clamp(0, 254) as u32;
        let f32_man = enc_man << (23 - man_bits);
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_man)
    };

    let n_vals = packed.len() / 3 * 4;
    let mut out = vec![0.0f32; n_vals];
    packed.chunks_exact(3)
        .zip(out.chunks_exact_mut(4))
        .for_each(|(src, dst)| {
            let p = ((src[0] as u32) << 16) | ((src[1] as u32) << 8) | (src[2] as u32);
            dst[0] = decode_val((p >> 18) & 0x3F);
            dst[1] = decode_val((p >> 12) & 0x3F);
            dst[2] = decode_val((p >> 6) & 0x3F);
            dst[3] = decode_val(p & 0x3F);
        });

    Ok(Array1::from(out).into_pyarray_bound(py).unbind())
}

// ── Wave 58a: AWQ Channel ──────────────────────────────────────────────────

/// Accumulate per-channel absolute mean from a calibration batch.
///
/// Returns `(updated_abs_mean, new_count)` where abs_mean is `(in_features,)` float32.
#[pyfunction]
fn awq_channel_abs_mean_f32(
    py: Python<'_>,
    batch: PyReadonlyArray2<f32>,
    accumulator: PyReadonlyArray1<f32>,
    count: usize,
) -> PyResult<(Py<PyArray1<f32>>, usize)> {
    let batch = batch.as_array();
    let acc = accumulator.as_array();
    let (batch_size, in_features) = (batch.nrows(), batch.ncols());
    if acc.len() != in_features {
        return Err(pyo3::exceptions::PyValueError::new_err("accumulator length mismatch"));
    }
    // Parallel column-wise abs-sum accumulate
    let new_sum: Vec<f32> = (0..in_features)
        .into_par_iter()
        .map(|j| acc[j] + batch.column(j).iter().map(|v| v.abs()).sum::<f32>())
        .collect();
    let new_count = count + batch_size;
    let inv = 1.0 / new_count as f32;
    let mean_arr: Array1<f32> = new_sum.iter().map(|&s| s * inv).collect::<Vec<_>>().into();
    Ok((mean_arr.into_pyarray_bound(py).unbind(), new_count))
}

/// Compute AWQ channel scales: `clip(abs_mean, 1e-4, ∞) ** alpha`.
///
/// Returns `(in_features,)` float32 scale vector.
#[pyfunction]
fn awq_compute_scales_f32(
    py: Python<'_>,
    abs_mean: PyReadonlyArray1<f32>,
    alpha: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let abs_mean = abs_mean.as_array();
    let scales: Array1<f32> = abs_mean
        .iter()
        .map(|&v| v.max(1.0e-4_f32).powf(alpha))
        .collect::<Vec<_>>()
        .into();
    Ok(scales.into_pyarray_bound(py).unbind())
}

// ── Wave 58a: Model Merge ──────────────────────────────────────────────────

/// Spherical linear interpolation (SLERP) between two flat weight vectors.
#[pyfunction]
fn slerp_f32(
    py: Python<'_>,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
    t: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let a = a.as_array();
    let b = b.as_array();
    if a.len() != b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("a and b must match in length"));
    }
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0e-10);
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt().max(1.0e-10);
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x / norm_a) * (y / norm_b)).sum::<f32>();
    let dot = dot.clamp(-1.0, 1.0);
    let theta = dot.acos();
    let out: Vec<f32> = if theta.abs() < 1.0e-6 {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai * (1.0 - t) + bi * t).collect()
    } else {
        let sin_theta = theta.sin();
        let scale_a = ((1.0 - t) * theta).sin() / sin_theta;
        let scale_b = (t * theta).sin() / sin_theta;
        a.iter().zip(b.iter()).map(|(ai, bi)| ai * scale_a + bi * scale_b).collect()
    };
    Ok(Array1::from(out).into_pyarray_bound(py).unbind())
}

/// DARE: mask weight deltas with Bernoulli(density) and rescale.
///
/// `output[i] = base[i] + delta[i] * mask[i] / density` where mask ~ Bernoulli(density).
#[pyfunction]
fn dare_merge_f32(
    py: Python<'_>,
    base: PyReadonlyArray1<f32>,
    delta: PyReadonlyArray1<f32>,
    density: f32,
    seed: u64,
) -> PyResult<Py<PyArray1<f32>>> {
    let base = base.as_array();
    let delta = delta.as_array();
    if base.len() != delta.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("base and delta must match"));
    }
    if !(density > 0.0 && density <= 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("density must be in (0, 1]"));
    }
    let scale = 1.0 / density;
    // Fast deterministic per-element Bernoulli via Murmur-style hash
    let out: Vec<f32> = base
        .iter()
        .zip(delta.iter())
        .enumerate()
        .map(|(i, (&b_val, &d_val))| {
            let mut x = seed.wrapping_add((i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407));
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51AFD7ED558CCDu64);
            x ^= x >> 33;
            x = x.wrapping_mul(0xC4CEB9FE1A85EC53u64);
            x ^= x >> 33;
            let u = (x >> 11) as f32 / (1u64 << 53) as f32;
            if u < density { b_val + d_val * scale } else { b_val }
        })
        .collect();
    Ok(Array1::from(out).into_pyarray_bound(py).unbind())
}

/// TIES merge: top-trim, majority-sign election, masked mean over multiple deltas.
///
/// `deltas` shape `(n_models, n_params)`.  Returns merged flat weight vector.
#[pyfunction]
fn ties_merge_f32(
    py: Python<'_>,
    base: PyReadonlyArray1<f32>,
    deltas: PyReadonlyArray2<f32>,
    trim_fraction: f32,
    t: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let base = base.as_array();
    let deltas = deltas.as_array();
    let (n_models, n_params) = deltas.dim();
    if base.len() != n_params {
        return Err(pyo3::exceptions::PyValueError::new_err("base/deltas shape mismatch"));
    }
    // Per-model magnitude threshold (trim_fraction fraction cutoff)
    let thresholds: Vec<f32> = (0..n_models).map(|m| {
        let mut abs_vals: Vec<f32> = deltas.row(m).iter().map(|x| x.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = ((n_params as f32 * trim_fraction).ceil() as usize).min(n_params.saturating_sub(1));
        abs_vals[k]
    }).collect();
    // Sign sum across trimmed deltas
    let mut sign_sum = vec![0.0f32; n_params];
    for m in 0..n_models {
        let thresh = thresholds[m];
        for (j, &v) in deltas.row(m).iter().enumerate() {
            if v.abs() >= thresh { sign_sum[j] += v.signum(); }
        }
    }
    // Masked mean: include delta only if sign matches majority
    let mut masked_sum = vec![0.0f32; n_params];
    let mut masked_cnt = vec![0usize; n_params];
    for m in 0..n_models {
        let thresh = thresholds[m];
        for (j, &v) in deltas.row(m).iter().enumerate() {
            if v.abs() >= thresh && v.signum() == sign_sum[j].signum() {
                masked_sum[j] += v;
                masked_cnt[j] += 1;
            }
        }
    }
    let out: Vec<f32> = base
        .iter()
        .zip(masked_sum.iter())
        .zip(masked_cnt.iter())
        .map(|((&b_val, &s), &c)| if c > 0 { b_val + t * s / c as f32 } else { b_val })
        .collect();
    Ok(Array1::from(out).into_pyarray_bound(py).unbind())
}

// ── Wave 58a: MoE Bincount ─────────────────────────────────────────────────

/// Expert frequency bincount + normalize for MoE load balancing.
///
/// Returns `(n_experts,)` float32 frequency fraction array.
#[pyfunction]
fn moe_bincount_f32(
    py: Python<'_>,
    assignments: PyReadonlyArray1<i32>,
    n_experts: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let assignments = assignments.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("assignments must be contiguous")
    })?;
    if assignments.is_empty() || n_experts == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("empty assignments or n_experts"));
    }
    let batch_size = assignments.len();
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (batch_size + n_threads - 1) / n_threads;
    // Parallel chunk-local histograms, then reduce
    let local_counts: Vec<Vec<u32>> = assignments
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut hist = vec![0u32; n_experts];
            for &e in chunk {
                let idx = (e as usize).min(n_experts - 1);
                hist[idx] += 1;
            }
            hist
        })
        .collect();
    let mut counts = vec![0u32; n_experts];
    for lc in &local_counts {
        for (i, &c) in lc.iter().enumerate() { counts[i] += c; }
    }
    let inv = 1.0 / batch_size as f32;
    let freqs: Array1<f32> = counts.iter().map(|&c| c as f32 * inv).collect::<Vec<_>>().into();
    Ok(freqs.into_pyarray_bound(py).unbind())
}

/// Top-K expert selection from router logits.
///
/// Returns `(batch_size, k)` int32 array of top-k expert indices (sorted by score desc).
#[pyfunction]
fn moe_top_k_f32(
    py: Python<'_>,
    logits: PyReadonlyArray2<f32>,
    k: usize,
) -> PyResult<Py<PyArray2<i32>>> {
    let logits = logits.as_array();
    let (batch_size, n_experts) = (logits.nrows(), logits.ncols());
    let k = k.min(n_experts);
    // Collect per-row top-k in order (parallel map → Vec<Vec<i32>> → flatten)
    let rows: Vec<Vec<i32>> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let row = logits.row(i);
            let mut idx: Vec<usize> = (0..n_experts).collect();
            if k < n_experts {
                idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
                    row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let mut top = idx[..k].to_vec();
            top.sort_by(|&a, &b| {
                row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            top.into_iter().map(|x| x as i32).collect()
        })
        .collect();
    let flat: Vec<i32> = rows.into_iter().flatten().collect();
    let arr = Array2::from_shape_vec((batch_size, k), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

// ── Wave 58a: Online SGD ───────────────────────────────────────────────────

/// Fused logistic regression step: computes `y_hat = sigmoid(w·x)` and `error = y - y_hat`.
///
/// Returns `(y_hat, error)` as a Python tuple of f32 scalars.
#[pyfunction]
fn logistic_step_f32(
    _py: Python<'_>,
    weights: PyReadonlyArray1<f32>,
    features: PyReadonlyArray1<f32>,
    label: f32,
) -> PyResult<(f32, f32)> {
    let w = weights.as_array();
    let x = features.as_array();
    if w.len() != x.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("weights/features length mismatch"));
    }
    let dot: f32 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
    let y_hat = 1.0 / (1.0 + (-dot).exp());
    let error = label - y_hat;
    Ok((y_hat, error))
}

/// SGD weight update: `w += lr * error * x` (gradient ascent toward label).
///
/// Returns updated weights as `(n_features,)` float32 array.
#[pyfunction]
fn sgd_weight_update_f32(
    py: Python<'_>,
    weights: PyReadonlyArray1<f32>,
    features: PyReadonlyArray1<f32>,
    lr: f32,
    error: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let w = weights.as_array();
    let x = features.as_array();
    if w.len() != x.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("weights/features length mismatch"));
    }
    let scale = lr * error;
    let updated: Array1<f32> = w
        .iter()
        .zip(x.iter())
        .map(|(&wi, &xi)| wi + scale * xi)
        .collect::<Vec<_>>()
        .into();
    Ok(updated.into_pyarray_bound(py).unbind())
}

// ── Wave 59a — GPTQ Column Solve · QuaRot Group · Calib Scale · Flash Decode · BF16 Cast · Sparse Act GEMV ──

/// GPTQ block-parallel column-wise weight quantization with Hessian-diagonal
/// error propagation.
///
/// `weight`: `(rows, cols)` f32; `h_diag`: `(cols,)` Hessian diagonal;
/// `q_max`: maximum quantized value (e.g. 7.0 for INT4 symmetric);
/// `block_size`: number of columns per GPTQ block.
///
/// Returns `(codes_flat: (rows*cols,) i32, scales: (cols,) f32)`.
#[pyfunction]
fn gptq_column_solve_f32(
    py: Python<'_>,
    weight: PyReadonlyArray2<f32>,
    h_diag: PyReadonlyArray1<f32>,
    q_max: f32,
    block_size: usize,
) -> PyResult<(PyObject, Py<PyArray1<f32>>)> {
    let w = weight.as_array();
    let h = h_diag.as_array();
    let (rows, cols) = (w.shape()[0], w.shape()[1]);
    if h.len() != cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "h_diag length must equal cols",
        ));
    }
    let mut w_work: Vec<f32> = w.iter().copied().collect();
    let mut codes_out: Vec<i32> = vec![0i32; rows * cols];
    let mut scales_out: Vec<f32> = vec![0.0f32; cols];
    let bs = block_size.max(1);
    let n_blocks = (cols + bs - 1) / bs;
    for b in 0..n_blocks {
        let col_start = b * bs;
        let col_end = (col_start + bs).min(cols);
        for j in col_start..col_end {
            let abs_max: f32 = (0..rows)
                .map(|i| w_work[i * cols + j].abs())
                .fold(0.0f32, f32::max);
            let scale = if abs_max < 1e-9 { 1.0f32 } else { abs_max / q_max };
            scales_out[j] = scale;
            for i in 0..rows {
                let v = w_work[i * cols + j] / scale;
                let code = v.round().clamp(-q_max, q_max) as i32;
                codes_out[i * cols + j] = code;
                let err = w_work[i * cols + j] - (code as f32 * scale);
                let h_j = h[j].abs().max(1e-6f32);
                for k in (j + 1)..col_end {
                    w_work[i * cols + k] += err * (h[k] / h_j);
                }
            }
        }
    }
    let codes_arr = Array1::from(codes_out).into_pyarray_bound(py).unbind();
    let scales_arr = Array1::from(scales_out).into_pyarray_bound(py).unbind();
    Ok((codes_arr.into(), scales_arr))
}

/// QuaRot group-parallel quantization: symmetric or asymmetric INT quantization
/// over rotated weight groups.
///
/// Returns `(codes_flat: (rows*cols,) i32, scales: (n_groups,) f32, zeros: (n_groups,) f32)`.
#[pyfunction]
fn quarot_group_quant_f32(
    py: Python<'_>,
    weight: PyReadonlyArray2<f32>,
    group_size: usize,
    q_max: f32,
    symmetric: bool,
) -> PyResult<(PyObject, Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    let w = weight.as_array();
    let (rows, cols) = (w.shape()[0], w.shape()[1]);
    let gs = group_size.max(1);
    let n_groups = (cols + gs - 1) / gs;
    let w_flat: Vec<f32> = w.iter().copied().collect();
    let mut codes_out: Vec<i32> = vec![0i32; rows * cols];
    let mut scales_out: Vec<f32> = vec![1.0f32; n_groups];
    let mut zeros_out: Vec<f32> = vec![0.0f32; n_groups];
    for g in 0..n_groups {
        let c_start = g * gs;
        let c_end = (c_start + gs).min(cols);
        let vals: Vec<f32> = (0..rows)
            .flat_map(|i| (c_start..c_end).map(move |j| w_flat[i * cols + j]))
            .collect();
        if symmetric {
            let abs_max = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = if abs_max < 1e-9 { 1.0f32 } else { abs_max / q_max };
            scales_out[g] = scale;
            for i in 0..rows {
                for j in c_start..c_end {
                    let code = (w_flat[i * cols + j] / scale)
                        .round()
                        .clamp(-q_max, q_max) as i32;
                    codes_out[i * cols + j] = code;
                }
            }
        } else {
            let vmin = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let vmax = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = (vmax - vmin).max(1e-9);
            let scale = range / (2.0 * q_max);
            let zero = -vmin / scale;
            scales_out[g] = scale;
            zeros_out[g] = zero;
            for i in 0..rows {
                for j in c_start..c_end {
                    let code = (w_flat[i * cols + j] / scale + zero)
                        .round()
                        .clamp(0.0, 2.0 * q_max) as i32;
                    codes_out[i * cols + j] = code;
                }
            }
        }
    }
    let codes_arr = Array1::from(codes_out).into_pyarray_bound(py).unbind();
    let scales_arr = Array1::from(scales_out).into_pyarray_bound(py).unbind();
    let zeros_arr = Array1::from(zeros_out).into_pyarray_bound(py).unbind();
    Ok((codes_arr.into(), scales_arr, zeros_arr))
}

/// QuaRot group dequantization: reconstruct f32 weights from codes, scales, zeros.
///
/// Returns `(rows * cols,)` f32 flat array.
#[pyfunction]
fn quarot_group_dequant_f32(
    py: Python<'_>,
    codes: PyReadonlyArray1<i32>,
    scales: PyReadonlyArray1<f32>,
    zeros: PyReadonlyArray1<f32>,
    rows: usize,
    cols: usize,
    group_size: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let c = codes.as_array();
    let s = scales.as_array();
    let z = zeros.as_array();
    let gs = group_size.max(1);
    let n_elem = rows * cols;
    if c.len() != n_elem {
        return Err(pyo3::exceptions::PyValueError::new_err("codes length mismatch"));
    }
    let out: Vec<f32> = (0..rows)
        .flat_map(|i| {
            (0..cols).map(move |j| {
                let g = j / gs;
                (c[i * cols + j] as f32 - z[g]) * s[g]
            })
        })
        .collect();
    Ok(Array1::from(out).into_pyarray_bound(py).unbind())
}

/// Calibration absmax scale: per-channel absolute maximum over `(N, C)` activations.
///
/// Returns `(C,)` float32 scale array.
#[pyfunction]
fn calib_absmax_f32(
    py: Python<'_>,
    acts: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    let a = acts.as_array();
    let (n_samples, n_ch) = (a.shape()[0], a.shape()[1]);
    let flat: Vec<f32> = a.iter().copied().collect();
    let scales: Vec<f32> = (0..n_ch)
        .into_par_iter()
        .map(|c| {
            (0..n_samples)
                .map(|i| flat[i * n_ch + c].abs())
                .fold(0.0f32, f32::max)
        })
        .collect();
    Ok(Array1::from(scales).into_pyarray_bound(py).unbind())
}

/// Calibration percentile scale: per-channel `p`-th percentile of absolute activations.
///
/// Returns `(C,)` float32 scale array.
#[pyfunction]
fn calib_percentile_f32(
    py: Python<'_>,
    acts: PyReadonlyArray2<f32>,
    percentile: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let a = acts.as_array();
    let (n_samples, n_ch) = (a.shape()[0], a.shape()[1]);
    let flat: Vec<f32> = a.iter().copied().collect();
    let p = percentile.clamp(0.0, 100.0);
    let scales: Vec<f32> = (0..n_ch)
        .into_par_iter()
        .map(|c| {
            let mut col: Vec<f32> = (0..n_samples).map(|i| flat[i * n_ch + c].abs()).collect();
            col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((p / 100.0 * col.len().saturating_sub(1) as f32) as usize)
                .min(col.len().saturating_sub(1));
            col.get(idx).copied().unwrap_or(0.0)
        })
        .collect();
    Ok(Array1::from(scales).into_pyarray_bound(py).unbind())
}

/// Calibration ACIQ scale: per-channel Gaussian optimal clip over `(N, C)` activations.
///
/// `n_levels`: number of quantization levels (e.g. 256 for INT8).
/// Returns `(C,)` float32 scale array.
#[pyfunction]
fn calib_aciq_f32(
    py: Python<'_>,
    acts: PyReadonlyArray2<f32>,
    n_levels: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let a = acts.as_array();
    let (n_samples, n_ch) = (a.shape()[0], a.shape()[1]);
    let flat: Vec<f32> = a.iter().copied().collect();
    // ACIQ Gaussian: clip ≈ alpha(b) * sigma; alpha(b) ≈ sqrt(2*ln(n_samples)+1)
    let alpha_factor: f32 = (2.0 * (n_samples as f32).ln() + 1.0)
        .sqrt()
        .max(1.0);
    let _ = n_levels; // used by caller to select quantizer; sigma is the key output
    let scales: Vec<f32> = (0..n_ch)
        .into_par_iter()
        .map(|c| {
            let col: Vec<f32> = (0..n_samples).map(|i| flat[i * n_ch + c]).collect();
            // Welford online mean + variance
            let mut mean = 0.0f32;
            let mut m2 = 0.0f32;
            for (k, &v) in col.iter().enumerate() {
                let delta = v - mean;
                mean += delta / (k + 1) as f32;
                m2 += delta * (v - mean);
            }
            let variance = if n_samples > 1 { m2 / (n_samples - 1) as f32 } else { 0.0 };
            let sigma = variance.sqrt();
            (sigma * alpha_factor).max(1e-6)
        })
        .collect();
    Ok(Array1::from(scales).into_pyarray_bound(py).unbind())
}

/// Flash-decode per-split attention: Rayon parallel GEMV + online softmax per head.
///
/// `q`: `(n_heads, head_dim)` f32.
/// `k_split`: `(n_kv_heads * split_len, head_dim)` f32 — caller reshapes 3D → 2D.
/// `v_split`: `(n_kv_heads * split_len, head_dim)` f32.
/// `n_kv_heads`, `split_len`: shape parameters for indexing into k_split / v_split.
/// `gqa_group`: n_heads / n_kv_heads.
///
/// Returns `(output_flat (n_heads*head_dim,) f32, lse (n_heads,) f32, max_score (n_heads,) f32)`.
#[pyfunction]
fn flash_decode_split_f32(
    py: Python<'_>,
    q: PyReadonlyArray2<f32>,
    k_split: PyReadonlyArray2<f32>,
    v_split: PyReadonlyArray2<f32>,
    n_kv_heads: usize,
    split_len: usize,
    gqa_group: usize,
) -> PyResult<(PyObject, Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    let q_a = q.as_array();
    let k_a = k_split.as_array();
    let v_a = v_split.as_array();
    let n_heads = q_a.shape()[0];
    let head_dim = q_a.shape()[1];
    let g = gqa_group.max(1);
    let q_flat: Vec<f32> = q_a.iter().copied().collect();
    let k_flat: Vec<f32> = k_a.iter().copied().collect();
    let v_flat: Vec<f32> = v_a.iter().copied().collect();
    // kv indexing: k_flat[kv_h * split_len * head_dim + t * head_dim + d]
    let results: Vec<(Vec<f32>, f32, f32)> = (0..n_heads)
        .into_par_iter()
        .map(|h| {
            let kv_h = (h / g).min(n_kv_heads.saturating_sub(1));
            // scores[t] = dot(q[h, :], k[kv_h, t, :])
            let mut scores: Vec<f32> = (0..split_len)
                .map(|t| {
                    let q_off = h * head_dim;
                    let k_off = kv_h * split_len * head_dim + t * head_dim;
                    (0..head_dim)
                        .map(|d| q_flat[q_off + d] * k_flat[k_off + d])
                        .sum()
                })
                .collect();
            // Scale by 1/sqrt(head_dim)
            let scale = (head_dim as f32).sqrt().recip();
            for s in scores.iter_mut() { *s *= scale; }
            // Online softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
            let lse = max_s + exp_sum.ln();
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp() / exp_sum;
            }
            // output_h = sum_t(w[t] * v[kv_h, t, :])
            let mut out_h = vec![0.0f32; head_dim];
            for t in 0..split_len {
                let v_off = kv_h * split_len * head_dim + t * head_dim;
                let wt = scores[t];
                for d in 0..head_dim {
                    out_h[d] += wt * v_flat[v_off + d];
                }
            }
            (out_h, lse, max_s)
        })
        .collect();
    let mut output = vec![0.0f32; n_heads * head_dim];
    let mut lse_out = vec![0.0f32; n_heads];
    let mut max_out = vec![0.0f32; n_heads];
    for (h, (out_h, lse, ms)) in results.into_iter().enumerate() {
        let off = h * head_dim;
        output[off..off + head_dim].copy_from_slice(&out_h);
        lse_out[h] = lse;
        max_out[h] = ms;
    }
    let out_arr = Array1::from(output).into_pyarray_bound(py).unbind();
    let lse_arr = Array1::from(lse_out).into_pyarray_bound(py).unbind();
    let max_arr = Array1::from(max_out).into_pyarray_bound(py).unbind();
    Ok((out_arr.into(), lse_arr, max_arr))
}

/// BF16 → FP32 conversion: accepts raw BF16 bits as `Vec<u16>`, returns `(N,)` float32 array.
///
/// BF16 bit layout: sign(1) + exp(8) + mantissa(7).
/// Conversion: `f32::from_bits((bf16_bits as u32) << 16)`.
#[pyfunction]
fn bf16_to_f32_vec(py: Python<'_>, bf16_bits: Vec<u16>) -> PyResult<Py<PyArray1<f32>>> {
    let out: Vec<f32> = bf16_bits
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    Ok(Array1::from(out).into_pyarray_bound(py).unbind())
}

/// FP32 → BF16 conversion: returns raw BF16 bits as `Vec<u16>`.
///
/// Uses the `half` crate's `bf16::from_f32` (round-to-nearest-even).
#[pyfunction]
fn f32_to_bf16_vec(bf16_values: PyReadonlyArray1<f32>) -> PyResult<Vec<u16>> {
    use half::bf16;
    let vals = bf16_values.as_array();
    let out: Vec<u16> = vals.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
    Ok(out)
}

/// Sparse activation GEMV: `output = W @ activation` skipping zero (or near-zero) activations.
///
/// Builds a non-zero index list via `|act[i]| > threshold`, then accumulates
/// a compressed dot-product per output row via Rayon column parallelism.
///
/// `weight`: `(out_features, in_features)` f32; `activation`: `(in_features,)` f32.
/// Returns `(out_features,)` float32 array.
#[pyfunction]
fn sparse_act_gemv_f32(
    py: Python<'_>,
    weight: PyReadonlyArray2<f32>,
    activation: PyReadonlyArray1<f32>,
    threshold: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let w = weight.as_array();
    let a = activation.as_array();
    let (out_feat, in_feat) = (w.shape()[0], w.shape()[1]);
    if a.len() != in_feat {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "activation length must equal in_features",
        ));
    }
    let act_flat: Vec<f32> = a.iter().copied().collect();
    let w_flat: Vec<f32> = w.iter().copied().collect();
    let nz_indices: Vec<usize> = (0..in_feat)
        .filter(|&i| act_flat[i].abs() > threshold)
        .collect();
    let output: Vec<f32> = (0..out_feat)
        .into_par_iter()
        .map(|o| {
            let row_off = o * in_feat;
            nz_indices
                .iter()
                .map(|&i| w_flat[row_off + i] * act_flat[i])
                .sum()
        })
        .collect();
    Ok(Array1::from(output).into_pyarray_bound(py).unbind())
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
    // Wave 57a — Entropy Codec · PQ Accelerate · GRU Cell · Batch CosSim · SwiGLU · Randomized SVD
    m.add_function(wrap_pyfunction!(rans_encode,                         m)?)?;
    m.add_function(wrap_pyfunction!(rans_decode,                         m)?)?;
    m.add_function(wrap_pyfunction!(huffman_encode,                      m)?)?;
    m.add_function(wrap_pyfunction!(huffman_decode,                      m)?)?;
    m.add_function(wrap_pyfunction!(pq_kmeans_fit,                       m)?)?;
    m.add_function(wrap_pyfunction!(pq_encode_batch,                     m)?)?;
    m.add_function(wrap_pyfunction!(pq_adc_search,                       m)?)?;
    m.add_function(wrap_pyfunction!(gru_step_f32,                        m)?)?;
    m.add_function(wrap_pyfunction!(batched_cosine_similarity_f32,       m)?)?;
    m.add_function(wrap_pyfunction!(silu_f32,                            m)?)?;
    m.add_function(wrap_pyfunction!(swiglu_f32,                          m)?)?;
    m.add_function(wrap_pyfunction!(randomized_svd_f32,                  m)?)?;
    // Wave 58a — VectorKMeans · FP6BitPack · AWQChannel · ModelMerge · MoEBincount · OnlineSGD
    m.add_function(wrap_pyfunction!(vector_kmeans_fit_f32,               m)?)?;
    m.add_function(wrap_pyfunction!(vector_kmeans_assign_f32,            m)?)?;
    m.add_function(wrap_pyfunction!(vector_kmeans_reconstruct_f32,       m)?)?;
    m.add_function(wrap_pyfunction!(fp6_encode_f32,                      m)?)?;
    m.add_function(wrap_pyfunction!(fp6_decode_f32,                      m)?)?;
    m.add_function(wrap_pyfunction!(awq_channel_abs_mean_f32,            m)?)?;
    m.add_function(wrap_pyfunction!(awq_compute_scales_f32,              m)?)?;
    m.add_function(wrap_pyfunction!(slerp_f32,                           m)?)?;
    m.add_function(wrap_pyfunction!(dare_merge_f32,                      m)?)?;
    m.add_function(wrap_pyfunction!(ties_merge_f32,                      m)?)?;
    m.add_function(wrap_pyfunction!(moe_bincount_f32,                    m)?)?;
    m.add_function(wrap_pyfunction!(moe_top_k_f32,                       m)?)?;
    m.add_function(wrap_pyfunction!(logistic_step_f32,                   m)?)?;
    m.add_function(wrap_pyfunction!(sgd_weight_update_f32,               m)?)?;
    // Wave 59a — GPTQColumnSolve · QuaRotGroup · CalibScale · FlashDecodeKernel · BF16Cast · SparseActGEMV
    m.add_function(wrap_pyfunction!(gptq_column_solve_f32,               m)?)?;
    m.add_function(wrap_pyfunction!(quarot_group_quant_f32,              m)?)?;
    m.add_function(wrap_pyfunction!(quarot_group_dequant_f32,            m)?)?;
    m.add_function(wrap_pyfunction!(calib_absmax_f32,                    m)?)?;
    m.add_function(wrap_pyfunction!(calib_percentile_f32,                m)?)?;
    m.add_function(wrap_pyfunction!(calib_aciq_f32,                      m)?)?;
    m.add_function(wrap_pyfunction!(flash_decode_split_f32,              m)?)?;
    m.add_function(wrap_pyfunction!(bf16_to_f32_vec,                     m)?)?;
    m.add_function(wrap_pyfunction!(f32_to_bf16_vec,                     m)?)?;
    m.add_function(wrap_pyfunction!(sparse_act_gemv_f32,                 m)?)?;
    Ok(())
}
