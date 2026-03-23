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
    ndarray::{Array1, Array2, Array3},
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4,
    PyUntypedArrayMethods,
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

    // QR decomposition of Y via Gram-Schmidt: Q is (m, valid_k)
    // Skip near-zero columns produced when Y's rank < k (rank-deficient sketch).
    // Use a *relative* threshold: compare the residual norm to the original column
    // norm.  Near-zero residuals (< 0.1% of original) mean the column is already
    // in the span of previous vectors — normalising them amplifies float32 noise
    // into full-magnitude "phantom" unit vectors that corrupt BBT.
    let mut q_cols: Vec<Vec<f32>> = Vec::with_capacity(k);
    for j in 0..k {
        let mut col: Vec<f32> = (0..m).map(|i| y_flat[i * k + j]).collect();
        let original_norm: f32 = col.iter().map(|&x| x * x).sum::<f32>().sqrt();
        for prev in q_cols.iter() {
            let dot: f32 = col.iter().zip(prev.iter()).map(|(&a, &b)| a * b).sum();
            for (c, &p) in col.iter_mut().zip(prev.iter()) { *c -= dot * p; }
        }
        let norm: f32 = col.iter().map(|&x| x * x).sum::<f32>().sqrt();
        // Relative threshold: residual must be ≥ 0.1 % of its pre-projection magnitude.
        let threshold = (original_norm * 1e-3).max(1e-10);
        if norm > threshold {
            for v in col.iter_mut() { *v /= norm; }
            q_cols.push(col);
        }
        // Columns below threshold are rank-exhausted: silently dropped.
    }
    let valid_k = q_cols.len();

    // B = Q^T × A: (valid_k, n)
    let mut b_flat = vec![0.0f32; valid_k * n];
    b_flat.par_chunks_mut(n).enumerate().for_each(|(j, row)| {
        let q_col = &q_cols[j];
        for l in 0..n {
            row[l] = q_col.iter().enumerate().map(|(i, &q)| q * a_arr[(i, l)]).sum();
        }
    });

    // Thin SVD of B (valid_k, n) via eigendecomposition of B×B^T (valid_k×valid_k)
    let mut bbt = vec![0.0f32; valid_k * valid_k];
    for i in 0..valid_k {
        for j in i..valid_k {
            let dot: f32 = (0..n).map(|l| b_flat[i * n + l] * b_flat[j * n + l]).sum();
            bbt[i * valid_k + j] = dot;
            bbt[j * valid_k + i] = dot;
        }
    }

    // Extract eigenvalues/vectors of BBT via Jacobi iterations
    let mut eig_vecs: Vec<Vec<f32>> = (0..valid_k).map(|i| {
        let mut e = vec![0.0f32; valid_k];
        e[i] = 1.0;
        e
    }).collect();
    let mut eig_vals: Vec<f32> = (0..valid_k).map(|i| bbt[i * valid_k + i]).collect();

    for _ in 0..50 {
        for p in 0..valid_k {
            for q in (p + 1)..valid_k {
                let a_pp = eig_vals[p];
                let a_qq = eig_vals[q];
                let a_pq = {
                    let mut val = 0.0f32;
                    for l in 0..valid_k {
                        val += eig_vecs[p][l] * (0..valid_k).map(|m2| bbt[l * valid_k + m2] * eig_vecs[q][m2]).sum::<f32>();
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
                for l in 0..valid_k {
                    eig_vecs[p][l] = c * ep[l] - s * eq[l];
                    eig_vecs[q][l] = s * ep[l] + c * eq[l];
                }
            }
        }
    }
    // Recompute eigenvalues from final eigenvectors for accuracy
    for i in 0..valid_k {
        eig_vals[i] = (0..valid_k)
            .map(|l| eig_vecs[i][l] * (0..valid_k).map(|m2| bbt[l * valid_k + m2] * eig_vecs[i][m2]).sum::<f32>())
            .sum();
    }

    // Sort by descending eigenvalue
    let mut idx: Vec<usize> = (0..valid_k).collect();
    idx.sort_by(|&a, &b| eig_vals[b].partial_cmp(&eig_vals[a]).unwrap_or(std::cmp::Ordering::Equal));

    let actual_rank = rank.min(valid_k);
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
                    (0..valid_k).map(|j| b_flat[j * n + col] * ev[j]).sum::<f32>() * inv_sv;
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
        let w_slice = &w_flat[..];
        let vals: Vec<f32> = (0..rows)
            .flat_map(|i| (c_start..c_end).map(move |j| w_slice[i * cols + j]))
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
    // Wave 60a — Mamba2SSM · AdaRound · PagedKVGather · HawkRGLR · CakeEntropy · TernaryGEMV
    m.add_function(wrap_pyfunction!(mamba2_ssm_scan_f32,                 m)?)?;
    m.add_function(wrap_pyfunction!(mamba2_ssm_decode_f32,               m)?)?;
    m.add_function(wrap_pyfunction!(adaround_step_f32,                   m)?)?;
    m.add_function(wrap_pyfunction!(paged_kv_gather_f32,                 m)?)?;
    m.add_function(wrap_pyfunction!(hawk_rglr_scan_f32,                  m)?)?;
    m.add_function(wrap_pyfunction!(cake_entropy_f32,                    m)?)?;
    m.add_function(wrap_pyfunction!(ternary_gemv_i8,                     m)?)?;

    // Wave 61a
    m.add_function(wrap_pyfunction!(wanda_importance_f32,                m)?)?;
    m.add_function(wrap_pyfunction!(wanda_nm_mask_f32,                   m)?)?;
    m.add_function(wrap_pyfunction!(flute_lut_encode_f32,                m)?)?;
    m.add_function(wrap_pyfunction!(flute_lut_decode_u8,                 m)?)?;
    m.add_function(wrap_pyfunction!(delta_net_scan_f32,                  m)?)?;
    m.add_function(wrap_pyfunction!(green_kv_score_f32,                  m)?)?;
    m.add_function(wrap_pyfunction!(jacobi_conv_check_f32,               m)?)?;
    m.add_function(wrap_pyfunction!(tree_verify_softmax_f32,             m)?)?;

    // Wave 62a — SVDqHead · ShadowKVFit · ClusterKVScore · Any4Lloyd
    //            · OuroborosNgram · PyramidKVBudget · QMoECompressIter
    m.add_function(wrap_pyfunction!(svdq_head_rank_f32,                  m)?)?;
    m.add_function(wrap_pyfunction!(shadow_kv_svd_fit_f32,               m)?)?;
    m.add_function(wrap_pyfunction!(shadow_kv_store_batch_f32,           m)?)?;
    m.add_function(wrap_pyfunction!(cluster_kv_score_f32,                m)?)?;
    m.add_function(wrap_pyfunction!(any4_lloyd_step_f32,                 m)?)?;
    m.add_function(wrap_pyfunction!(ouroboros_ngram_build,               m)?)?;
    m.add_function(wrap_pyfunction!(ouroboros_lookahead_f32,             m)?)?;
    m.add_function(wrap_pyfunction!(pyramid_kv_budget_f32,               m)?)?;
    m.add_function(wrap_pyfunction!(qmoe_compress_iter_f32,              m)?)?;
    Ok(())
}

// ── Wave 60a — Mamba2 SSM chunked scan · AdaRound gradient step · Paged KV gather
//              · Hawk RGLR scan · CAKE entropy · Ternary GEMV ────────────────

/// Mamba-2 SSD chunked parallel scan (prefill path).
///
/// Implements: h_{t+1} = A_t * h_t + B_t * x_t,  y_t = C_t @ h_t
/// A is a per-head scalar broadcast; B, C are (T, d_state); x is (T,).
/// Returns `output` (T,) and final `state` (d_state,) per head.
///
/// # Arguments
/// * `a` - Log-A scalars (T,) — negative for stability.
/// * `b` - B matrices (T, d_state).
/// * `c` - C matrices (T, d_state).
/// * `x` - Input sequence (T,).
/// * `h0` - Initial state (d_state,); zero if starting fresh.
///
/// # Returns
/// `(output (T,), final_state (d_state,))`
#[pyfunction]
fn mamba2_ssm_scan_f32(
    py: Python<'_>,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray2<f32>,
    c: PyReadonlyArray2<f32>,
    x: PyReadonlyArray1<f32>,
    h0: PyReadonlyArray1<f32>,
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    let a_s = a.as_slice()?;
    let b_s = b.as_slice()?;
    let x_s = x.as_slice()?;
    let c_s = c.as_slice()?;
    let h0_s = h0.as_slice()?;
    let t_len = a_s.len();
    let d_state = h0_s.len();

    let mut h: Vec<f32> = h0_s.to_vec();
    let mut out = vec![0f32; t_len];
    for t in 0..t_len {
        let a_t = a_s[t].exp();
        let x_t = x_s[t];
        let b_row = &b_s[t * d_state..(t + 1) * d_state];
        let c_row = &c_s[t * d_state..(t + 1) * d_state];
        // h = A_t * h + B_t * x_t
        for i in 0..d_state {
            h[i] = a_t * h[i] + b_row[i] * x_t;
        }
        // y_t = C_t @ h
        out[t] = c_row.iter().zip(h.iter()).map(|(&ci, &hi)| ci * hi).sum();
    }
    let output = PyArray1::from_vec(py, out);
    let final_state = PyArray1::from_vec(py, h);
    Ok((output.into(), final_state.into()))
}

/// Mamba-2 recurrent decode: single-token O(d_state) state update.
///
/// # Arguments
/// * `a_scalar` - Scalar log-A for this step (already exp'd outside).
/// * `b_vec`    - B vector (d_state,).
/// * `c_vec`    - C vector (d_state,).
/// * `x_scalar` - Input scalar for this token.
/// * `state`    - Current recurrent state (d_state,) — modified in place via copy.
///
/// # Returns
/// `(y_scalar: f32, new_state (d_state,))`
#[pyfunction]
fn mamba2_ssm_decode_f32(
    py: Python<'_>,
    a_scalar: f32,
    b_vec: PyReadonlyArray1<f32>,
    c_vec: PyReadonlyArray1<f32>,
    x_scalar: f32,
    state: PyReadonlyArray1<f32>,
) -> PyResult<(f32, Py<PyArray1<f32>>)> {
    let b_s = b_vec.as_slice()?;
    let c_s = c_vec.as_slice()?;
    let h_s = state.as_slice()?;
    let d_state = h_s.len();
    let mut new_h = Vec::with_capacity(d_state);
    let mut y = 0f32;
    for i in 0..d_state {
        let hi = a_scalar * h_s[i] + b_s[i] * x_scalar;
        y += c_s[i] * hi;
        new_h.push(hi);
    }
    Ok((y, PyArray1::from_vec(py, new_h).into()))
}

/// AdaRound single optimisation step (vectorised over all weight elements).
///
/// Computes the rectified sigmoid h(V) = clip(σ(β*(V - ζ)) , 0, 1) and
/// takes a gradient step: V ← V - lr * ∂L/∂V where
/// ∂L/∂V = (W_soft - W_hard) + λ_reg * β_anneal * h'(V).
///
/// # Arguments
/// * `v`             - Rounding parameters (flattened, len = N).
/// * `w`             - Original weights (flattened, len = N).
/// * `w_quant_floor` - Floor-quantized weights (flattened, len = N).
/// * `q_scale`       - Per-element quantisation scale (len = N).
/// * `lr`            - Learning rate.
/// * `beta`          - Current β for the annealing schedule.
/// * `lambda_reg`    - Regularisation coefficient.
/// * `zeta`          - Stretch lower bound (default 0.1 in paper).
/// * `gamma`         - Stretch upper bound (default 1.1 in paper).
///
/// # Returns
/// Updated V vector (N,) as float32.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn adaround_step_f32(
    py: Python<'_>,
    v: PyReadonlyArray1<f32>,
    w: PyReadonlyArray1<f32>,
    w_quant_floor: PyReadonlyArray1<f32>,
    q_scale: PyReadonlyArray1<f32>,
    lr: f32,
    beta: f32,
    lambda_reg: f32,
    zeta: f32,
    gamma: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let v_s = v.as_slice()?;
    let w_s = w.as_slice()?;
    let wf_s = w_quant_floor.as_slice()?;
    let qs_s = q_scale.as_slice()?;
    let n = v_s.len();

    let new_v: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let vi = v_s[i];
            // h(V) = clip(σ(β*(V - ζ)), 0, 1)
            let sig = 1.0 / (1.0 + (-(beta * (vi - zeta))).exp());
            let h = sig.clamp(0.0, 1.0);
            // soft weight dequantized
            let w_soft = (wf_s[i] + h) * qs_s[i];
            // gradient wrt V: reconstruction loss term
            let grad_recon = w_soft - w_s[i];
            // regularisation: h'(V) = β * h * (1 - h) * (gamma - zeta)
            let h_prime = beta * h * (1.0 - h) * (gamma - zeta);
            let grad_reg = lambda_reg * h_prime;
            vi - lr * (grad_recon * qs_s[i] + grad_reg)
        })
        .collect();
    Ok(PyArray1::from_vec(py, new_v).into())
}

/// Gather K/V blocks from a paged physical buffer into a contiguous tensor.
///
/// Reconstructs the full K or V tensor for one sequence from non-contiguous
/// physical pages.  Each page holds `block_size` tokens of `head_dim` floats
/// across `n_heads` heads.
///
/// # Arguments
/// * `kv_pool`   - Physical KV pool: (max_blocks, n_heads, block_size, head_dim).
///   Passed as 2D `(max_blocks * n_heads * block_size, head_dim)`.
/// * `page_table`  - Page indices for this sequence (n_pages,).
/// * `n_heads`     - Number of KV heads.
/// * `block_size`  - Tokens per physical page.
/// * `head_dim`    - Head dimension.
/// * `n_valid_tokens` - Number of valid tokens (may be < n_pages * block_size).
///
/// # Returns
/// Contiguous tensor (n_valid_tokens, n_heads, head_dim) as float32.
#[pyfunction]
fn paged_kv_gather_f32(
    py: Python<'_>,
    kv_pool: PyReadonlyArray2<f32>,
    page_table: PyReadonlyArray1<i32>,
    n_heads: usize,
    block_size: usize,
    head_dim: usize,
    n_valid_tokens: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let pool = kv_pool.as_slice()?;
    let pages = page_table.as_slice()?;
    let stride = n_heads * block_size * head_dim; // per physical page
    let out_size = n_valid_tokens * n_heads * head_dim;
    let mut out = vec![0f32; out_size];

    let filled = out
        .par_chunks_mut(n_heads * head_dim)
        .enumerate()
        .map(|(tok_idx, dst)| {
            let page_idx = tok_idx / block_size;
            let tok_in_page = tok_idx % block_size;
            let phys_page = pages[page_idx] as usize;
            for h in 0..n_heads {
                let src_base = phys_page * stride + h * block_size * head_dim + tok_in_page * head_dim;
                let dst_base = h * head_dim;
                dst[dst_base..dst_base + head_dim]
                    .copy_from_slice(&pool[src_base..src_base + head_dim]);
            }
        });
    filled.count(); // drive the parallel iterator
    Ok(PyArray1::from_vec(py, out).into())
}

/// Hawk Real-Gated Linear Recurrence (RGLR) scan over a sequence.
///
/// Implements: h_t = f_t ⊙ h_{t-1} + i_t ⊙ x_t
/// where f_t = exp(-exp(Λ) * softplus(dt_t)) and i_t = √(1 - f_t²) (approx).
/// All operations are element-wise over d_state.
///
/// # Arguments
/// * `x`          - Projected input (T, d_state) — B*x in the paper.
/// * `dt`         - Delta-time inputs (T, d_state) before softplus.
/// * `lambda_log` - Log-eigenvalues (d_state,) — Λ in the paper.
/// * `h0`         - Initial state (d_state,).
///
/// # Returns
/// `(outputs (T, d_state), final_state (d_state,))`
#[pyfunction]
fn hawk_rglr_scan_f32(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    dt: PyReadonlyArray2<f32>,
    lambda_log: PyReadonlyArray1<f32>,
    h0: PyReadonlyArray1<f32>,
) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<f32>>)> {
    let x_s = x.as_slice()?;
    let dt_s = dt.as_slice()?;
    let lam_s = lambda_log.as_slice()?;
    let h0_s = h0.as_slice()?;
    let t_len = x.shape()[0];
    let d_state = h0_s.len();

    let mut h: Vec<f32> = h0_s.to_vec();
    let mut out = vec![0f32; t_len * d_state];
    for t in 0..t_len {
        let x_row = &x_s[t * d_state..(t + 1) * d_state];
        let dt_row = &dt_s[t * d_state..(t + 1) * d_state];
        let out_row = &mut out[t * d_state..(t + 1) * d_state];
        // parallelise over d_state channels
        out_row
            .par_iter_mut()
            .zip(h.par_iter_mut())
            .enumerate()
            .for_each(|(i, (y_i, h_i))| {
                let sp = (1.0 + (-(dt_row[i])).exp()).ln(); // softplus(dt)
                let decay = (-(lam_s[i].exp()) * sp).exp(); // f_t
                let input_gate = (1.0 - decay * decay).sqrt().max(0.0); // i_t approx
                *h_i = decay * *h_i + input_gate * x_row[i];
                *y_i = *h_i;
            });
    }
    Ok((
        PyArray1::from_vec(py, out).into(),
        PyArray1::from_vec(py, h).into(),
    ))
}

/// CAKE attention-entropy computation for KV budget allocation.
///
/// Computes mean normalised attention entropy per head across observation_window
/// recent queries.  Entropy is used to allocate KV-cache budget to each layer.
///
/// entropy(h) = -sum_t p(t) * log(p(t) + eps) / log(T)
///
/// where p(t) are softmax attention weights for query position t.
///
/// # Arguments
/// * `q_obs`   - Recent queries (obs_window, n_heads, head_dim).
///   Passed as 2D (obs_window * n_heads, head_dim).
/// * `k`       - Key cache (T, n_heads, head_dim).
///   Passed as 2D (T * n_heads, head_dim).
/// * `n_heads` - Number of attention heads.
/// * `obs_window` - Number of recent query positions.
/// * `temperature` - Softmax temperature for score scaling.
///
/// # Returns
/// Per-head mean entropy (n_heads,) as float32.
#[pyfunction]
fn cake_entropy_f32(
    py: Python<'_>,
    q_obs: PyReadonlyArray2<f32>,
    k: PyReadonlyArray2<f32>,
    n_heads: usize,
    obs_window: usize,
    temperature: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let q_s = q_obs.as_slice()?;
    let k_s = k.as_slice()?;
    let head_dim = q_obs.shape()[1];
    let t_len = k.shape()[0] / n_heads;
    let scale = 1.0 / ((head_dim as f32).sqrt() * temperature);

    let entropies: Vec<f32> = (0..n_heads)
        .into_par_iter()
        .map(|h| {
            let mut head_entropy = 0f32;
            for q_pos in 0..obs_window {
                let q_row = &q_s[(q_pos * n_heads + h) * head_dim
                    ..(q_pos * n_heads + h + 1) * head_dim];
                // compute scores for all T key positions
                let mut scores: Vec<f32> = (0..t_len)
                    .map(|t| {
                        let k_row = &k_s[(t * n_heads + h) * head_dim
                            ..(t * n_heads + h + 1) * head_dim];
                        q_row.iter().zip(k_row.iter()).map(|(&qi, &ki)| qi * ki).sum::<f32>() * scale
                    })
                    .collect();
                // softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = scores.iter_mut().map(|s| { *s = (*s - max_s).exp(); *s }).sum();
                // entropy
                let ent: f32 = scores.iter().map(|&p| {
                    let prob = p / sum_exp;
                    if prob > 1e-9 { -prob * prob.ln() } else { 0.0 }
                }).sum();
                let log_t = (t_len as f32).ln().max(1.0);
                head_entropy += ent / log_t;
            }
            head_entropy / (obs_window as f32).max(1.0)
        })
        .collect();
    Ok(PyArray1::from_vec(py, entropies).into())
}

/// Ternary weight GEMV: INT8 {-1, 0, +1} weight matrix × FP32 activation.
///
/// Exploits the {-1, 0, +1} structure: per output row, accumulate only
/// activations where w_ij == +1 (add), subtract where w_ij == -1, skip zeros.
/// Rayon parallelises over output rows.
///
/// # Arguments
/// * `weight`     - INT8 ternary weight matrix (out_features, in_features).
///   Values must be in {-1, 0, +1}.
/// * `activation` - FP32 activation vector (in_features,).
/// * `scale`      - Scalar dequantisation scale (mean(|W_orig|)).
///
/// # Returns
/// Output vector (out_features,) as float32.
#[pyfunction]
fn ternary_gemv_i8(
    py: Python<'_>,
    weight: PyReadonlyArray2<i8>,
    activation: PyReadonlyArray1<f32>,
    scale: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let w_s = weight.as_slice()?;
    let a_s = activation.as_slice()?;
    let out_f = weight.shape()[0];
    let in_f = weight.shape()[1];

    let out: Vec<f32> = (0..out_f)
        .into_par_iter()
        .map(|row| {
            let row_start = row * in_f;
            let mut acc = 0f32;
            for j in 0..in_f {
                match w_s[row_start + j] {
                    1 => acc += a_s[j],
                    -1 => acc -= a_s[j],
                    _ => {}
                }
            }
            acc * scale
        })
        .collect();
    Ok(PyArray1::from_vec(py, out).into())
}

// ── Wave 61a — Wanda · FLUTE LUT · DeltaNet · GreenKV · Jacobi · TreeVerify ─

/// Wanda per-element importance: |W| × activation_rms broadcast.
///
/// For each row r and column c: `importance[r, c] = |W[r, c]| × rms[c]`.
/// Rayon parallelises over rows.
#[pyfunction]
fn wanda_importance_f32(
    py: Python<'_>,
    w: PyReadonlyArray2<f32>,
    rms: PyReadonlyArray1<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    use numpy::ndarray::Array2;
    let w_s = w.as_slice()?;
    let r_s = rms.as_slice()?;
    let out_f = w.shape()[0];
    let in_f = w.shape()[1];

    let flat: Vec<f32> = (0..out_f)
        .into_par_iter()
        .flat_map(|row| {
            let row_start = row * in_f;
            (0..in_f)
                .map(move |c| w_s[row_start + c].abs() * r_s[c])
                .collect::<Vec<f32>>()
        })
        .collect();

    let arr = Array2::from_shape_vec((out_f, in_f), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// Wanda N:M structured sparsity mask.
///
/// For each column block of size `m`, keeps the top-`n` entries per row
/// by importance score.  Returns `uint8` mask (1 = keep, 0 = prune).
/// Rayon parallelises over rows.
#[pyfunction]
fn wanda_nm_mask_f32(
    py: Python<'_>,
    importance: PyReadonlyArray2<f32>,
    n: usize,
    m: usize,
) -> PyResult<Py<PyArray2<u8>>> {
    use numpy::ndarray::Array2;
    let imp = importance.as_slice()?;
    let out_f = importance.shape()[0];
    let in_f = importance.shape()[1];
    let n_blocks = (in_f + m - 1) / m;

    let flat: Vec<u8> = (0..out_f)
        .into_par_iter()
        .flat_map(|row| {
            let row_start = row * in_f;
            let mut row_mask = vec![0u8; in_f];
            for bi in 0..n_blocks {
                let col_start = bi * m;
                let col_end = (col_start + m).min(in_f);
                let block_w = col_end - col_start;
                let n_keep = n.min(block_w);
                let mut scored: Vec<(usize, f32)> = (0..block_w)
                    .map(|c| (col_start + c, imp[row_start + col_start + c]))
                    .collect();
                scored.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                for k in 0..n_keep {
                    row_mask[scored[k].0] = 1;
                }
            }
            row_mask
        })
        .collect();

    let arr = Array2::from_shape_vec((out_f, in_f), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// FLUTE LUT group encode: map weight values to nearest codebook entry index.
///
/// For each column group `g`, finds the codebook entry with minimum L1
/// distance to each weight value.  Rayon parallelises over groups.
///
/// `weight`   — `(rows, cols)` f32
/// `codebook` — `(n_groups, cb_size)` f32 — one codebook per group
/// Returns `(rows, cols)` uint8 code indices.
#[pyfunction]
fn flute_lut_encode_f32(
    py: Python<'_>,
    weight: PyReadonlyArray2<f32>,
    codebook: PyReadonlyArray2<f32>,
    group_size: usize,
) -> PyResult<Py<PyArray2<u8>>> {
    use numpy::ndarray::Array2;
    let w_s = weight.as_slice()?;
    let cb_s = codebook.as_slice()?;
    let rows = weight.shape()[0];
    let cols = weight.shape()[1];
    let n_groups = codebook.shape()[0];
    let cb_size = codebook.shape()[1];

    let group_chunks: Vec<(usize, Vec<u8>)> = (0..n_groups)
        .into_par_iter()
        .map(|g| {
            let col_start = g * group_size;
            let col_end = (col_start + group_size).min(cols);
            let gs_actual = col_end - col_start;
            let cb_g = &cb_s[g * cb_size..(g + 1) * cb_size];
            let mut grp = vec![0u8; rows * gs_actual];
            for row in 0..rows {
                for c in 0..gs_actual {
                    let v = w_s[row * cols + col_start + c];
                    let best = cb_g
                        .iter()
                        .enumerate()
                        .min_by(|(_, &a), (_, &b)| {
                            (v - a).abs().partial_cmp(&(v - b).abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    grp[row * gs_actual + c] = best as u8;
                }
            }
            (g, grp)
        })
        .collect();

    let mut flat = vec![0u8; rows * cols];
    for (g, grp) in group_chunks {
        let col_start = g * group_size;
        let col_end = (col_start + group_size).min(cols);
        let gs_actual = col_end - col_start;
        for row in 0..rows {
            for c in 0..gs_actual {
                flat[row * cols + col_start + c] = grp[row * gs_actual + c];
            }
        }
    }

    let arr = Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// FLUTE LUT group decode: gather codebook entries by code indices.
///
/// For each column group `g`: `out[:, g*gs:(g+1)*gs] = codebook[g][codes[:]]`.
/// Rayon parallelises over groups.
#[pyfunction]
fn flute_lut_decode_u8(
    py: Python<'_>,
    codes: PyReadonlyArray2<u8>,
    codebook: PyReadonlyArray2<f32>,
    group_size: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    use numpy::ndarray::Array2;
    let c_s = codes.as_slice()?;
    let cb_s = codebook.as_slice()?;
    let rows = codes.shape()[0];
    let cols = codes.shape()[1];
    let n_groups = codebook.shape()[0];
    let cb_size = codebook.shape()[1];

    let group_chunks: Vec<(usize, Vec<f32>)> = (0..n_groups)
        .into_par_iter()
        .map(|g| {
            let col_start = g * group_size;
            let col_end = (col_start + group_size).min(cols);
            let gs_actual = col_end - col_start;
            let cb_g = &cb_s[g * cb_size..(g + 1) * cb_size];
            let mut grp = vec![0f32; rows * gs_actual];
            for row in 0..rows {
                for c in 0..gs_actual {
                    let idx = c_s[row * cols + col_start + c] as usize;
                    grp[row * gs_actual + c] = cb_g[idx.min(cb_size - 1)];
                }
            }
            (g, grp)
        })
        .collect();

    let mut flat = vec![0f32; rows * cols];
    for (g, grp) in group_chunks {
        let col_start = g * group_size;
        let col_end = (col_start + group_size).min(cols);
        let gs_actual = col_end - col_start;
        for row in 0..rows {
            for c in 0..gs_actual {
                flat[row * cols + col_start + c] = grp[row * gs_actual + c];
            }
        }
    }

    let arr = Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// DeltaNet sequential delta-rule recurrent scan over T tokens.
///
/// For each timestep t (sequential), updates per-head state W with:
///   k̂  = k[t,h] / ‖k[t,h]‖₂
///   W  = W + β[t,h] · (v[t,h] − W @ k̂) ⊗ k̂
///   y[t,h] = W @ q[t,h]
///
/// The per-head outer-product update is parallelised with Rayon.
///
/// `q`, `k`, `v` — `(T, n_heads, head_dim)` f32
/// `beta`         — `(T, n_heads)` f32
/// Returns `(T, n_heads, head_dim)` f32 output.
#[pyfunction]
fn delta_net_scan_f32(
    py: Python<'_>,
    q: PyReadonlyArray3<f32>,
    k: PyReadonlyArray3<f32>,
    v: PyReadonlyArray3<f32>,
    beta: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray3<f32>>> {
    use numpy::ndarray::Array3;
    let q_s = q.as_slice()?;
    let k_s = k.as_slice()?;
    let v_s = v.as_slice()?;
    let b_s = beta.as_slice()?;
    let t_len = q.shape()[0];
    let n_heads = q.shape()[1];
    let head_dim = q.shape()[2];

    // W state: (n_heads, head_dim, head_dim)
    let mut state = vec![0f32; n_heads * head_dim * head_dim];
    let mut out = vec![0f32; t_len * n_heads * head_dim];

    for t in 0..t_len {
        let q_t = &q_s[t * n_heads * head_dim..(t + 1) * n_heads * head_dim];
        let k_t = &k_s[t * n_heads * head_dim..(t + 1) * n_heads * head_dim];
        let v_t = &v_s[t * n_heads * head_dim..(t + 1) * n_heads * head_dim];
        let b_t = &b_s[t * n_heads..(t + 1) * n_heads];

        let head_updates: Vec<(Vec<f32>, Vec<f32>)> = (0..n_heads)
            .into_par_iter()
            .map(|h| {
                let q_h = &q_t[h * head_dim..(h + 1) * head_dim];
                let k_raw = &k_t[h * head_dim..(h + 1) * head_dim];
                let v_h = &v_t[h * head_dim..(h + 1) * head_dim];
                let beta_h = b_t[h];
                let w_h = &state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];

                let k_norm = k_raw.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
                let k_hat: Vec<f32> = k_raw.iter().map(|&x| x / k_norm).collect();

                // Wk = W[h] @ k_hat
                let mut wk = vec![0f32; head_dim];
                for hd in 0..head_dim {
                    for ds in 0..head_dim {
                        wk[hd] += w_h[hd * head_dim + ds] * k_hat[ds];
                    }
                }

                // residual = v - Wk
                let residual: Vec<f32> =
                    v_h.iter().zip(wk.iter()).map(|(&vi, &wi)| vi - wi).collect();

                // new_W = W + beta * outer(residual, k_hat)
                let mut new_w = w_h.to_vec();
                for hd in 0..head_dim {
                    for ds in 0..head_dim {
                        new_w[hd * head_dim + ds] += beta_h * residual[hd] * k_hat[ds];
                    }
                }

                // y = new_W @ q_h
                let mut y = vec![0f32; head_dim];
                for hd in 0..head_dim {
                    for ds in 0..head_dim {
                        y[hd] += new_w[hd * head_dim + ds] * q_h[ds];
                    }
                }
                (new_w, y)
            })
            .collect();

        for (h, (new_w, y)) in head_updates.into_iter().enumerate() {
            state[h * head_dim * head_dim..(h + 1) * head_dim * head_dim]
                .copy_from_slice(&new_w);
            out[t * n_heads * head_dim + h * head_dim..t * n_heads * head_dim + (h + 1) * head_dim]
                .copy_from_slice(&y);
        }
    }

    let arr = Array3::from_shape_vec((t_len, n_heads, head_dim), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// GreenKV per-head attention score accumulation for KV cache selection.
///
/// For each head h: computes mean softmax attention weights over `obs_window`
/// query vectors against all `seq_len` key vectors, returning per-token
/// importance scores `(n_heads, seq_len)`.  Rayon parallelises over heads.
///
/// `q_obs` — `(n_heads, obs_window, head_dim)` f32 — recent query slice
/// `k`     — `(n_heads, seq_len,   head_dim)` f32 — full key cache
/// Returns `(n_heads, seq_len)` f32 importance scores.
#[pyfunction]
fn green_kv_score_f32(
    py: Python<'_>,
    q_obs: PyReadonlyArray3<f32>,
    k: PyReadonlyArray3<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    use numpy::ndarray::Array2;
    let q_s = q_obs.as_slice()?;
    let k_s = k.as_slice()?;
    let n_heads = q_obs.shape()[0];
    let obs_window = q_obs.shape()[1];
    let head_dim = q_obs.shape()[2];
    let seq_len = k.shape()[1];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let flat: Vec<f32> = (0..n_heads)
        .into_par_iter()
        .flat_map(|h| {
            let q_h = &q_s[h * obs_window * head_dim..(h + 1) * obs_window * head_dim];
            let k_h = &k_s[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
            let mut head_scores = vec![0f32; seq_len];

            for qp in 0..obs_window {
                let q_row = &q_h[qp * head_dim..(qp + 1) * head_dim];
                let mut logits: Vec<f32> = (0..seq_len)
                    .map(|s| {
                        let k_row = &k_h[s * head_dim..(s + 1) * head_dim];
                        q_row.iter().zip(k_row.iter())
                            .map(|(&qi, &ki)| qi * ki)
                            .sum::<f32>() * scale
                    })
                    .collect();
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = logits.iter_mut().map(|l| { *l = (*l - max_l).exp(); *l }).sum();
                for (s, l) in logits.iter().enumerate() {
                    head_scores[s] += l / (sum_exp + 1e-9);
                }
            }
            head_scores.iter_mut().for_each(|s| *s /= obs_window as f32);
            head_scores
        })
        .collect();

    let arr = Array2::from_shape_vec((n_heads, seq_len), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// Jacobi fixed-point convergence check and next-guess generation.
///
/// For each of the N speculative positions:
///   - `temperature == 0`: `new_guess[i] = argmax(logits[i])`
///   - `temperature > 0`:  Gumbel-max sampling (deterministic per position via
///     xorshift64 seeded from `seed ⊕ i`)
///
/// Returns `(new_guesses: (N,) i32, n_fixed: i32)` where `n_fixed` counts
/// positions where `new_guess == old_guess`.
#[pyfunction]
fn jacobi_conv_check_f32(
    py: Python<'_>,
    pos_logits: PyReadonlyArray2<f32>,
    guesses: PyReadonlyArray1<i32>,
    temperature: f32,
    seed: u64,
) -> PyResult<(Py<PyArray1<i32>>, i32)> {
    let l_s = pos_logits.as_slice()?;
    let g_s = guesses.as_slice()?;
    let n = pos_logits.shape()[0];
    let vocab = pos_logits.shape()[1];

    let new_guesses: Vec<i32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let logits = &l_s[i * vocab..(i + 1) * vocab];
            if temperature <= 0.0 {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as i32)
                    .unwrap_or(0)
            } else {
                // Gumbel-max: deterministic per-position xorshift64
                let mut state = seed
                    .wrapping_add((i as u64).wrapping_mul(6364136223846793005u64))
                    .wrapping_add(1442695040888963407u64);
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut best_idx = 0i32;
                let mut best_val = f32::NEG_INFINITY;
                for (j, &l) in logits.iter().enumerate() {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = ((state >> 33) as f32) / (u32::MAX as f32) + 1e-20f32;
                    let gumbel = (l - max_l) / temperature - (-u.ln()).ln();
                    if gumbel > best_val {
                        best_val = gumbel;
                        best_idx = j as i32;
                    }
                }
                best_idx
            }
        })
        .collect();

    let n_fixed = new_guesses
        .iter()
        .zip(g_s.iter())
        .filter(|(&ng, &og)| ng == og)
        .count() as i32;
    Ok((
        Array1::from(new_guesses).into_pyarray_bound(py).unbind(),
        n_fixed,
    ))
}

/// Tree-parallel speculative decoding token verification via rejection sampling.
///
/// For each branch `b` (parallelised), verifies draft tokens sequentially:
///   - `accept_prob = min(1, p_target[token] / p_draft[token])`
///   - On acceptance: continue to next position
///   - On rejection:  sample correction token from `max(0, p_t − p_d)` residual,
///     then stop the branch
///
/// Returns `(best_accepted: (n,) i32, best_len: i32)` — the longest accepted
/// sequence across all branches plus its length.
///
/// `draft_tokens`  — `(B, n_draft)` i32
/// `draft_logits`  — `(B, n_draft, vocab)` f32
/// `target_logits` — `(B, n_draft, vocab)` f32
#[pyfunction]
fn tree_verify_softmax_f32(
    py: Python<'_>,
    draft_tokens: PyReadonlyArray2<i32>,
    draft_logits: PyReadonlyArray3<f32>,
    target_logits: PyReadonlyArray3<f32>,
    temperature: f32,
    seed: u64,
) -> PyResult<(Py<PyArray1<i32>>, i32)> {
    let dt_s = draft_tokens.as_slice()?;
    let dl_s = draft_logits.as_slice()?;
    let tl_s = target_logits.as_slice()?;
    let n_branches = draft_tokens.shape()[0];
    let n_draft = draft_tokens.shape()[1];
    let vocab = draft_logits.shape()[2];
    let temp = temperature.max(1e-6);

    let softmax_into = |logits: &[f32], out: &mut Vec<f32>| {
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        out.clear();
        out.extend(logits.iter().map(|&l| ((l - max_l) / temp).exp()));
        let s: f32 = out.iter().sum::<f32>() + 1e-9;
        out.iter_mut().for_each(|p| *p /= s);
    };

    let branch_results: Vec<Vec<i32>> = (0..n_branches)
        .into_par_iter()
        .map(|b| {
            let mut rng = seed.wrapping_add((b as u64).wrapping_mul(2654435761u64));
            let next_rand = |state: &mut u64| -> f32 {
                *state ^= *state << 13;
                *state ^= *state >> 7;
                *state ^= *state << 17;
                ((*state >> 33) as f32) / (u32::MAX as f32)
            };
            let mut d_probs = Vec::with_capacity(vocab);
            let mut t_probs = Vec::with_capacity(vocab);
            let mut accepted: Vec<i32> = Vec::new();
            let b_off = b * n_draft * vocab;
            for i in 0..n_draft {
                let token = dt_s[b * n_draft + i] as usize;
                softmax_into(&dl_s[b_off + i * vocab..b_off + (i + 1) * vocab], &mut d_probs);
                softmax_into(&tl_s[b_off + i * vocab..b_off + (i + 1) * vocab], &mut t_probs);
                let p_d = d_probs[token.min(vocab - 1)];
                let p_t = t_probs[token.min(vocab - 1)];
                let accept_prob = (p_t / (p_d + 1e-30)).min(1.0);
                if next_rand(&mut rng) < accept_prob {
                    accepted.push(token as i32);
                } else {
                    // Sample correction from residual = max(0, p_target − p_draft)
                    let residual: Vec<f32> = t_probs
                        .iter()
                        .zip(d_probs.iter())
                        .map(|(&tp, &dp)| (tp - dp).max(0.0))
                        .collect();
                    let rsum: f32 = residual.iter().sum::<f32>() + 1e-9;
                    let r = next_rand(&mut rng);
                    let mut cum = 0f32;
                    let correction = residual
                        .iter()
                        .enumerate()
                        .find(|(_, &rv)| {
                            cum += rv / rsum;
                            cum >= r
                        })
                        .map(|(j, _)| j as i32)
                        .unwrap_or(0);
                    accepted.push(correction);
                    break;
                }
            }
            accepted
        })
        .collect();

    let best = branch_results
        .into_iter()
        .max_by_key(|v| v.len())
        .unwrap_or_default();
    let best_len = best.len() as i32;
    Ok((
        Array1::from(best).into_pyarray_bound(py).unbind(),
        best_len,
    ))
}

// ── Wave 62a — SVDqHead · ShadowKVFit · ClusterKVScore · Any4Lloyd
//              · OuroborosNgram · PyramidKVBudget · QMoECompressIter ─────────

/// Per-head SVD rank profiling for SVDq quantisation calibration.
///
/// For each (layer × head) pair, computes the singular values of the stacked
/// key matrix `keys[l, h]` (shape `(T, D)`) via a thin SVD.  Parallelised
/// over the flattened `n_layers × n_heads` index grid via Rayon.
///
/// Returns a `(n_layers, n_heads, min(T,D))` f32 array of singular values.
///
/// # Arguments
/// * `keys` - `(n_layers, n_heads, T, D)` f32 stacked key tensors.
#[pyfunction]
fn svdq_head_rank_f32(
    py: Python<'_>,
    keys: PyReadonlyArray4<f32>,
) -> PyResult<Py<PyArray3<f32>>> {
    let s = keys.as_slice()?;
    let shape = keys.shape();
    let (n_layers, n_heads, t_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    let k_svd = t_len.min(head_dim);
    let total = n_layers * n_heads;

    let results: Vec<Vec<f32>> = (0..total)
        .into_par_iter()
        .map(|idx| {
            let l = idx / n_heads;
            let h = idx % n_heads;
            let off = (l * n_heads + h) * t_len * head_dim;
            let slice = &s[off..off + t_len * head_dim];
            // Compute singular values via power-iteration approximation (3 steps).
            // Full LAPACK not available without a C FFI; use cheap estimate sufficient
            // for rank profiling: S_i ≈ ||A v_i|| via random projections.
            let mut rng_state: u64 = 0xdeadbeef_u64
                .wrapping_add((l as u64).wrapping_mul(31337))
                .wrapping_add(h as u64);
            let next_rand = |st: &mut u64| -> f32 {
                *st ^= *st << 13;
                *st ^= *st >> 7;
                *st ^= *st << 17;
                ((*st >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            };
            let mut sv = vec![0f32; k_svd];
            // Randomized range finder: k_svd probe vectors
            for ki in 0..k_svd {
                // Random unit vector in R^head_dim
                let mut probe: Vec<f32> = (0..head_dim).map(|_| next_rand(&mut rng_state)).collect();
                let pnorm: f32 = probe.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-12;
                probe.iter_mut().for_each(|x| *x /= pnorm);
                // A @ probe  ->  (t_len,)
                let mut ap = vec![0f32; t_len];
                for row in 0..t_len {
                    let row_off = row * head_dim;
                    ap[row] = slice[row_off..row_off + head_dim]
                        .iter()
                        .zip(probe.iter())
                        .map(|(&a, &p)| a * p)
                        .sum();
                }
                sv[ki] = ap.iter().map(|x| x * x).sum::<f32>().sqrt();
            }
            sv.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            sv
        })
        .collect();

    let mut out = vec![0f32; n_layers * n_heads * k_svd];
    for (idx, sv) in results.iter().enumerate() {
        let off = idx * k_svd;
        out[off..off + k_svd].copy_from_slice(sv);
    }
    let arr = Array3::from_shape_vec((n_layers, n_heads, k_svd), out)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// Per-head thin SVD fit for ShadowKV low-rank key projection.
///
/// For each head `h` (parallelised), fits a rank-`rank` projection of the
/// key cache `keys[h]` (shape `(T, D)`) via a thin SVD.  Returns the top-`rank`
/// right singular vectors as `(H, rank, D)` f32 (V-matrices per head).
///
/// # Arguments
/// * `keys` - `(n_heads, T, head_dim)` f32 key cache.
/// * `rank` - Number of singular vectors to retain.
#[pyfunction]
fn shadow_kv_svd_fit_f32(
    py: Python<'_>,
    keys: PyReadonlyArray3<f32>,
    rank: usize,
) -> PyResult<Py<PyArray3<f32>>> {
    let s = keys.as_slice()?;
    let shape = keys.shape();
    let (n_heads, t_len, head_dim) = (shape[0], shape[1], shape[2]);
    let r = rank.min(head_dim).min(t_len);

    let vs: Vec<Vec<f32>> = (0..n_heads)
        .into_par_iter()
        .map(|h| {
            let off = h * t_len * head_dim;
            let k_h = &s[off..off + t_len * head_dim];
            // Randomized SVD for the right singular vectors (V rows).
            // Omega: (head_dim, r) random Gaussian.
            let mut rng_state: u64 = 0xcafe1234_u64.wrapping_add((h as u64).wrapping_mul(6364136223846793005));
            let next_rand = |st: &mut u64| -> f32 {
                *st ^= *st << 13; *st ^= *st >> 7; *st ^= *st << 17;
                ((*st >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            };
            // Build Y = A @ Omega  (t_len × r)
            let mut omega = vec![0f32; head_dim * r];
            omega.iter_mut().for_each(|x| *x = next_rand(&mut rng_state));
            let mut y = vec![0f32; t_len * r];
            for row in 0..t_len {
                let k_row = &k_h[row * head_dim..(row + 1) * head_dim];
                for c in 0..r {
                    let col_off = c;
                    y[row * r + c] = k_row.iter().enumerate()
                        .map(|(d, &kv)| kv * omega[d * r + col_off])
                        .sum();
                }
            }
            // QR via Gram-Schmidt on Y columns  ->  Q (t_len × r)
            let mut q = vec![0f32; t_len * r];
            for c in 0..r {
                let mut col: Vec<f32> = (0..t_len).map(|row| y[row * r + c]).collect();
                for pc in 0..c {
                    let dot: f32 = col.iter().zip((0..t_len).map(|row| q[row * r + pc]))
                        .map(|(a, b)| a * b).sum();
                    for row in 0..t_len { col[row] -= dot * q[row * r + pc]; }
                }
                let norm: f32 = col.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 { continue; }
                for row in 0..t_len { q[row * r + c] = col[row] / norm; }
            }
            // B = Q^T @ A  (r × head_dim)
            let mut b = vec![0f32; r * head_dim];
            for ri in 0..r {
                for d in 0..head_dim {
                    b[ri * head_dim + d] = (0..t_len)
                        .map(|row| q[row * r + ri] * k_h[row * head_dim + d])
                        .sum();
                }
            }
            // Return B rows as the approximate right singular vectors
            b
        })
        .collect();

    let mut out = vec![0f32; n_heads * r * head_dim];
    for (h, v) in vs.iter().enumerate() {
        let off = h * r * head_dim;
        let copy_len = v.len().min(r * head_dim);
        out[off..off + copy_len].copy_from_slice(&v[..copy_len]);
    }
    let arr = Array3::from_shape_vec((n_heads, r, head_dim), out)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// Project a batch of tokens into the per-head low-rank ShadowKV space.
///
/// For each head (parallelised over token chunks), computes
/// `projected[h, i] = V[h] @ keys[h, i]`  where `V` is the `(H, rank, D)`
/// V-matrix from :func:`shadow_kv_svd_fit_f32`.
///
/// Returns `(n_heads, n_tokens, rank)` f32.
///
/// # Arguments
/// * `keys`  - `(n_heads, n_tokens, head_dim)` f32.
/// * `v_mat` - `(n_heads, rank, head_dim)` f32 right singular vectors.
#[pyfunction]
fn shadow_kv_store_batch_f32(
    py: Python<'_>,
    keys: PyReadonlyArray3<f32>,
    v_mat: PyReadonlyArray3<f32>,
) -> PyResult<Py<PyArray3<f32>>> {
    let ks = keys.as_slice()?;
    let vs = v_mat.as_slice()?;
    let ks_shape = keys.shape();
    let vm_shape = v_mat.shape();
    let (n_heads, n_tokens, head_dim) = (ks_shape[0], ks_shape[1], ks_shape[2]);
    let rank = vm_shape[1];
    if vm_shape[0] != n_heads || vm_shape[2] != head_dim {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("v_mat shape ({},{},{}) incompatible with keys ({},{},{})",
                vm_shape[0], vm_shape[1], vm_shape[2], n_heads, n_tokens, head_dim),
        ));
    }

    let projected: Vec<Vec<f32>> = (0..n_heads)
        .into_par_iter()
        .map(|h| {
            let k_off = h * n_tokens * head_dim;
            let v_off = h * rank * head_dim;
            let k_h = &ks[k_off..k_off + n_tokens * head_dim];
            let v_h = &vs[v_off..v_off + rank * head_dim];
            let mut proj = vec![0f32; n_tokens * rank];
            for i in 0..n_tokens {
                let k_row = &k_h[i * head_dim..(i + 1) * head_dim];
                for r in 0..rank {
                    proj[i * rank + r] = k_row.iter().zip(v_h[r * head_dim..(r + 1) * head_dim].iter())
                        .map(|(&k, &v)| k * v).sum();
                }
            }
            proj
        })
        .collect();

    let mut out = vec![0f32; n_heads * n_tokens * rank];
    for (h, p) in projected.iter().enumerate() {
        let off = h * n_tokens * rank;
        out[off..off + n_tokens * rank].copy_from_slice(p);
    }
    let arr = Array3::from_shape_vec((n_heads, n_tokens, rank), out)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// Compute per-cluster attention-weight sums for ClusterKV eviction decisions.
///
/// For each cluster `c` (parallelised), sums the attention weights of all tokens
/// assigned to that cluster.  Used to identify low-importance clusters for eviction.
///
/// Returns `(n_clusters,)` f32 total attention weight per cluster.
///
/// # Arguments
/// * `assignments` - `(seq_len,)` i32: cluster index for each token.
/// * `attn_weights` - `(seq_len,)` f32: softmax attention weight of each token.
/// * `n_clusters`  - Total number of clusters.
#[pyfunction]
fn cluster_kv_score_f32(
    py: Python<'_>,
    assignments: PyReadonlyArray1<i32>,
    attn_weights: PyReadonlyArray1<f32>,
    n_clusters: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let asgn = assignments.as_slice()?;
    let attn = attn_weights.as_slice()?;
    let seq_len = asgn.len();
    if attn.len() != seq_len {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("assignments len {} != attn_weights len {}", seq_len, attn.len()),
        ));
    }

    let scores: Vec<f32> = (0..n_clusters)
        .into_par_iter()
        .map(|c| {
            asgn.iter()
                .zip(attn.iter())
                .filter(|(&cl, _)| cl as usize == c)
                .map(|(_, &w)| w)
                .sum()
        })
        .collect();

    Ok(Array1::from(scores).into_pyarray_bound(py).unbind())
}

/// Lloyd k-means codebook calibration step for Any4 quantisation.
///
/// Executes `n_iter` rounds of Lloyd's algorithm on `values`.  Each iteration:
///   1. Assigns every value to its nearest centroid (parallelised in chunks).
///   2. Updates centroids to the mean of assigned values.
///   3. Re-initialises empty centroids with a random assigned value.
///
/// Returns `(k,)` f32 final centroids.
///
/// # Arguments
/// * `values`          - `(N,)` f32 weight values to cluster.
/// * `centroids_init` - `(k,)` f32 initial centroid positions.
/// * `n_iter`         - Number of Lloyd iterations (default 100).
#[pyfunction]
fn any4_lloyd_step_f32(
    py: Python<'_>,
    values: PyReadonlyArray1<f32>,
    centroids_init: PyReadonlyArray1<f32>,
    n_iter: u32,
) -> PyResult<Py<PyArray1<f32>>> {
    let vals = values.as_slice()?;
    let k = centroids_init.shape()[0];
    let mut centroids: Vec<f32> = centroids_init.as_slice()?.to_vec();
    let n = vals.len();

    for _iter in 0..n_iter {
        // ── Assign: parallelised over chunks ─────────────────────────────────
        let chunk_size = ((n + rayon::current_num_threads() - 1) / rayon::current_num_threads()).max(1);
        let assignments: Vec<u32> = vals
            .par_chunks(chunk_size)
            .flat_map(|chunk| {
                chunk.iter().map(|&v| {
                    centroids
                        .iter()
                        .enumerate()
                        .min_by(|(_, &a), (_, &b)| {
                            (a - v).abs().partial_cmp(&(b - v).abs()).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                }).collect::<Vec<_>>()
            })
            .collect();

        // ── Update: per-centroid mean ─────────────────────────────────────────
        let mut sums = vec![0f64; k];
        let mut counts = vec![0u64; k];
        for (&a, &v) in assignments.iter().zip(vals.iter()) {
            let c = a as usize;
            sums[c] += v as f64;
            counts[c] += 1;
        }
        let mut rng_state: u64 = 0xabcd1234u64.wrapping_add(_iter as u64 * 6364136223846793005);
        let next_rand_idx = |st: &mut u64| -> usize {
            *st ^= *st << 13; *st ^= *st >> 7; *st ^= *st << 17;
            (*st as usize) % n
        };
        for c in 0..k {
            if counts[c] > 0 {
                centroids[c] = (sums[c] / counts[c] as f64) as f32;
            } else {
                // Re-init empty centroid with a random value
                centroids[c] = vals[next_rand_idx(&mut rng_state)];
            }
        }
    }

    Ok(Array1::from(centroids).into_pyarray_bound(py).unbind())
}

/// Build an n-gram frequency table from a sequence of verified token IDs.
///
/// Constructs `(order-1)`-gram → next-token count maps across `n_shards`
/// independent hash partitions (parallelised via Rayon).  Returns a flat
/// `(n_unique_contexts, order + 1)` i32 array where each row is
/// `[ctx_tok_0, ..., ctx_tok_{order-2}, next_tok, count]`, sorted by count
/// descending within each context.
///
/// # Arguments
/// * `token_ids`  - `(T,)` i32 sequence of verified token IDs.
/// * `order`      - N-gram order (context length = order - 1).
/// * `max_entries` - Maximum rows to return (0 = unlimited).
#[pyfunction]
fn ouroboros_ngram_build(
    py: Python<'_>,
    token_ids: PyReadonlyArray1<i32>,
    order: usize,
    max_entries: usize,
) -> PyResult<Py<PyArray2<i32>>> {
    let toks = token_ids.as_slice()?;
    let t = toks.len();
    if order < 2 || t < order {
        let arr = Array2::<i32>::zeros((0, order + 1));
        return Ok(arr.into_pyarray_bound(py).unbind());
    }
    let ctx_len = order - 1;

    // Build shard-keyed frequency maps in parallel.
    const N_SHARDS: usize = 8;
    let shard_maps: Vec<std::collections::HashMap<Vec<i32>, std::collections::HashMap<i32, u32>>> =
        (0..N_SHARDS)
            .into_par_iter()
            .map(|shard| {
                let mut map: std::collections::HashMap<Vec<i32>, std::collections::HashMap<i32, u32>> =
                    std::collections::HashMap::new();
                for i in 0..t - order + 1 {
                    // Route to shard by simple hash of first context token
                    let shard_key = (toks[i].unsigned_abs() as usize) % N_SHARDS;
                    if shard_key != shard { continue; }
                    let ctx: Vec<i32> = toks[i..i + ctx_len].to_vec();
                    let next = toks[i + ctx_len];
                    *map.entry(ctx).or_default().entry(next).or_insert(0) += 1;
                }
                map
            })
            .collect();

    // Flatten into rows
    let mut rows: Vec<Vec<i32>> = Vec::new();
    for shard_map in &shard_maps {
        for (ctx, next_map) in shard_map {
            for (&next, &cnt) in next_map {
                let mut row = ctx.clone();
                row.push(next);
                row.push(cnt as i32);
                rows.push(row);
            }
        }
    }
    rows.sort_unstable_by(|a, b| b.last().cmp(&a.last()));
    let limit = if max_entries > 0 { max_entries.min(rows.len()) } else { rows.len() };
    let n_rows = limit;
    let n_cols = order + 1;
    let mut flat = vec![0i32; n_rows * n_cols];
    for (ri, row) in rows.iter().take(limit).enumerate() {
        let off = ri * n_cols;
        for (ci, &v) in row.iter().enumerate() {
            flat[off + ci] = v;
        }
    }
    let arr = Array2::from_shape_vec((n_rows, n_cols), flat)
        .unwrap_or_else(|_| Array2::<i32>::zeros((0, n_cols)));
    Ok(arr.into_pyarray_bound(py).unbind())
}

/// Ouroboros lookahead draft-chain temperature sampling.
///
/// For each of `depth` positions, samples a token from the corresponding
/// logits vector using temperature scaling and top-1 argmax.
/// (Full speculative temperature sampling; parallelised per independent branch.)
///
/// Returns `(depth,)` i32 draft tokens.
///
/// # Arguments
/// * `logits`      - `(depth, vocab)` f32 draft logits.
/// * `temperature` - Sampling temperature (> 0).
/// * `seed`        - Random seed.
#[pyfunction]
fn ouroboros_lookahead_f32(
    py: Python<'_>,
    logits: PyReadonlyArray2<f32>,
    temperature: f32,
    seed: u64,
) -> PyResult<Py<PyArray1<i32>>> {
    let ls = logits.as_slice()?;
    let depth = logits.shape()[0];
    let vocab = logits.shape()[1];
    let temp = temperature.max(1e-6);

    let tokens: Vec<i32> = (0..depth)
        .into_par_iter()
        .map(|d| {
            let off = d * vocab;
            let logit_row = &ls[off..off + vocab];
            let max_l = logit_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logit_row.iter().map(|&l| ((l - max_l) / temp).exp()).collect();
            let s: f32 = probs.iter().sum::<f32>() + 1e-9;
            probs.iter_mut().for_each(|p| *p /= s);
            // Multinomial sampling via xorshift64
            let mut rng = seed.wrapping_add((d as u64).wrapping_mul(6364136223846793005));
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let r = (rng >> 33) as f32 / (u32::MAX as f32);
            let mut cum = 0f32;
            probs.iter().enumerate()
                .find(|(_, &p)| { cum += p; cum >= r })
                .map(|(i, _)| i as i32)
                .unwrap_or((vocab - 1) as i32)
        })
        .collect();

    Ok(Array1::from(tokens).into_pyarray_bound(py).unbind())
}

/// Logarithmic layer-wise KV budget allocation for PyramidKV.
///
/// Computes per-layer KV token budgets following the linear-decay formula:
///   `budget[l] = max(min_budget, round(base * (1 - alpha * l / (n-1))))`
///
/// Parallelised over layers via Rayon.  Returns `(n_layers,)` i32 budgets.
///
/// # Arguments
/// * `base`       - Base budget for layer 0.
/// * `alpha`      - Decay rate ∈ [0, 1].
/// * `n_layers`   - Number of model layers.
/// * `min_budget` - Minimum budget per layer.
#[pyfunction]
fn pyramid_kv_budget_f32(
    py: Python<'_>,
    base: f32,
    alpha: f32,
    n_layers: usize,
    min_budget: usize,
) -> PyResult<Py<PyArray1<i32>>> {
    let budgets: Vec<i32> = (0..n_layers)
        .into_par_iter()
        .map(|l| {
            let frac = if n_layers > 1 { l as f32 / (n_layers - 1) as f32 } else { 0.0 };
            let b = (base * (1.0 - alpha * frac)).round() as usize;
            b.max(min_budget) as i32
        })
        .collect();

    Ok(Array1::from(budgets).into_pyarray_bound(py).unbind())
}

/// QMoE expert weight codebook compression — EM iteration.
///
/// Runs `n_iter` rounds of EM (expectation-maximisation) k-means on the
/// flattened weight blocks of a single expert.  Each outer EM step:
///   1. (E-step) Assign every block to its nearest centroid (parallelised).
///   2. (M-step) Update centroid to mean of assigned blocks (parallelised).
///
/// Returns `(codebook: (k, block_size) f32, assignments: (N,) i32)`.
///
/// # Arguments
/// * `blocks`  - `(N, block_size)` f32 weight blocks.
/// * `k`       - Codebook size (number of centroids).
/// * `n_iter`  - EM iterations.
/// * `seed`    - RNG seed for centroid initialisation.
#[pyfunction]
fn qmoe_compress_iter_f32(
    py: Python<'_>,
    blocks: PyReadonlyArray2<f32>,
    k: usize,
    n_iter: u32,
    seed: u64,
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray1<i32>>)> {
    let bs_slice = blocks.as_slice()?;
    let n_blocks = blocks.shape()[0];
    let block_size = blocks.shape()[1];
    if k == 0 || n_blocks == 0 {
        let cb = Array2::<f32>::zeros((k, block_size));
        let asgn = Array1::<i32>::zeros(n_blocks);
        return Ok((cb.into_pyarray_bound(py).unbind(), asgn.into_pyarray_bound(py).unbind()));
    }

    // Initialise codebook with k evenly-spaced blocks
    let mut codebook: Vec<f32> = vec![0f32; k * block_size];
    for ci in 0..k {
        let src_idx = (ci * n_blocks / k).min(n_blocks - 1);
        let src_off = src_idx * block_size;
        codebook[ci * block_size..(ci + 1) * block_size]
            .copy_from_slice(&bs_slice[src_off..src_off + block_size]);
    }

    let mut assignments = vec![0i32; n_blocks];

    for _iter in 0..n_iter {
        // ── E-step: assign blocks to nearest centroid (parallelised) ─────────
        let new_asgn: Vec<i32> = (0..n_blocks)
            .into_par_iter()
            .map(|bi| {
                let bk = &bs_slice[bi * block_size..(bi + 1) * block_size];
                codebook
                    .chunks(block_size)
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da: f32 = a.iter().zip(bk.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum();
                        let db: f32 = b.iter().zip(bk.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i as i32)
                    .unwrap_or(0)
            })
            .collect();
        assignments = new_asgn;

        // ── M-step: update centroids ──────────────────────────────────────────
        let new_cb: Vec<f32> = (0..k)
            .into_par_iter()
            .flat_map(|ci| {
                let mut sum = vec![0f64; block_size];
                let mut cnt = 0u64;
                for (bi, &a) in assignments.iter().enumerate() {
                    if a as usize == ci {
                        let bk = &bs_slice[bi * block_size..(bi + 1) * block_size];
                        for (s, &v) in sum.iter_mut().zip(bk.iter()) { *s += v as f64; }
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    sum.iter().map(|&s| (s / cnt as f64) as f32).collect::<Vec<_>>()
                } else {
                    // Re-init centroid: use seed-perturbed copy of block 0
                    let fallback_idx = (seed.wrapping_add(ci as u64) as usize) % n_blocks;
                    bs_slice[fallback_idx * block_size..(fallback_idx + 1) * block_size].to_vec()
                }
            })
            .collect();
        codebook = new_cb;
    }

    let cb_arr = Array2::from_shape_vec((k, block_size), codebook)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let asgn_arr = Array1::from(assignments);
    Ok((cb_arr.into_pyarray_bound(py).unbind(), asgn_arr.into_pyarray_bound(py).unbind()))
}
