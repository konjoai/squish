#!/usr/bin/env python3
"""
convert_weights.py

Convert a float16/bfloat16 safetensors model to Vectro INT8 compressed format.
Stores everything in a single .npz archive with a companion _manifest.json.

Storage layout inside the npz:
  {safe_key}__q      → int8 array     quantized weights (shape: n_rows × n_cols)
  {safe_key}__s      → float32 array  per-row scale factors (shape: n_rows)
  {safe_key}__shape  → int64 array    original tensor shape (for reshape on load)
  {safe_key}__pt     → float32 array  passthrough tensors stored unquantized

safe_key = tensor name with '.' replaced by '__'
Companion file: {output}_manifest.json  maps original_name -> safe_key

Usage:
    python3 convert_weights.py \\
        --model-dir ~/models/Qwen2.5-1.5B-Instruct \\
        --output ~/models/Qwen2.5-1.5B-Instruct-compressed/weights_compressed.npz \\
        [--passthrough embed_tokens lm_head] \\
        [--outlier-threshold 20.0] \\
        [--verbose]
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# AWQ scale application (imported lazily so convert.py works without awq.py)
# ---------------------------------------------------------------------------
def _apply_awq_single(
    name: str,
    arr_f32: np.ndarray,
    proj_apply: dict,
    ln_apply: dict,
) -> np.ndarray:
    """
    Apply pre-computed AWQ group scales to a single weight tensor.

    proj_apply : layer_path → group_scale
        For linear projections: ``W_awq[:, c] = W[:, c] * group_scale[c]``
    ln_apply   : full_tensor_name → group_scale
        For LayerNorm weights: ``gamma_awq[c] = gamma[c] / group_scale[c]``
    """
    if not proj_apply and not ln_apply:
        return arr_f32
    layer = name.removesuffix(".weight")
    if layer in proj_apply:
        s = proj_apply[layer]
        return arr_f32 * s[np.newaxis, :]
    if name in ln_apply:
        s = ln_apply[name]
        return arr_f32 / s
    return arr_f32


from squish.quant.quantizer import (  # noqa: E402
    QuantizationResult,
    quantize_embeddings,
    quantize_int4,
)


# ---------------------------------------------------------------------------
# Lazy helpers for optional compression backends
# ---------------------------------------------------------------------------
def _get_nf4():
    from squish.quant.nf4_quant import quantize_nf4
    return quantize_nf4


def _get_vptq():
    from squish.quant.vptq import VPTQConfig, VPTQQuantizer
    return VPTQConfig, VPTQQuantizer


def _get_dfloat11():
    from squish.quant.dfloat11 import DFloat11Compressor, DFloat11Config
    return DFloat11Config, DFloat11Compressor


def _get_quip():
    from squish.quant.quip_sharp import QuIPSharpConfig, QuIPSharpQuantizer
    return QuIPSharpConfig, QuIPSharpQuantizer


def _get_aqlm():
    from squish.quant.aqlm import AQLMConfig, AQLMQuantizer
    return AQLMConfig, AQLMQuantizer


# ---------------------------------------------------------------------------
# ─── TTY-safe line-clear helper ─────────────────────────────────────────────
def _clear_line() -> None:
    """Overwrite the current terminal line.  No-op when stdout is not a TTY."""
    if sys.stdout.isatty():  # pragma: no cover
        sys.stdout.write("\r" + " " * 80 + "\r")  # pragma: no cover
        sys.stdout.flush()  # pragma: no cover


# Spinner
# ---------------------------------------------------------------------------

class Spinner:
    """
    Background-thread snake spinner.
    Usage:
        with Spinner("Writing weights_compressed.npz"):
            slow_operation()
    Also supports manual updates:
        sp = Spinner("Quantizing")
        sp.update("layer 5/28")
        sp.stop()
    When stdout is not a TTY (e.g. piped to tee), all \r-based overwriting is
    suppressed to avoid rendering artifacts in VS Code / other terminals.
    """
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "", interval: float = 0.1):
        self._label    = label
        self._interval = interval
        self._suffix   = ""
        self._stop_evt = threading.Event()
        self._is_tty   = sys.stdout.isatty()
        self._thread   = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        if not self._is_tty:
            # Non-TTY: just block until stopped, no output from the spin thread
            self._stop_evt.wait()
            return
        i = 0  # pragma: no cover
        while not self._stop_evt.is_set():  # pragma: no cover
            frame = self._FRAMES[i % len(self._FRAMES)]  # pragma: no cover
            line  = f"\r  {frame}  {self._label}  {self._suffix}"  # pragma: no cover
            sys.stdout.write(line)  # pragma: no cover
            sys.stdout.flush()  # pragma: no cover
            self._stop_evt.wait(self._interval)  # pragma: no cover
            i += 1  # pragma: no cover
        # clear the spinner line
        sys.stdout.write("\r" + " " * (len(self._label) + len(self._suffix) + 10) + "\r")  # pragma: no cover
        sys.stdout.flush()  # pragma: no cover

    def update(self, suffix: str):
        self._suffix = suffix

    def start(self):
        self._thread.start()
        return self

    def stop(self, final_msg: str = ""):
        self._stop_evt.set()
        self._thread.join()
        if final_msg:
            print(f"  ✓  {final_msg}")

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()


def safe_key(tensor_name: str) -> str:
    """Convert dot-notation tensor name to a valid npz key (no dots)."""
    return tensor_name.replace(".", "__")


def has_outliers(arr_f32: np.ndarray, threshold: float) -> bool:
    """Return True if the tensor has strong outlier rows (per-row max/mean > threshold)."""
    row_max = np.max(np.abs(arr_f32), axis=-1)
    row_mean = np.mean(np.abs(arr_f32), axis=-1) + 1e-8
    ratio = row_max / row_mean
    return float(ratio.max()) > threshold


def _pick_int4_group_size(n_cols: int, max_group_size: int = 32) -> int:
    """Return the largest group_size ≤ max_group_size that evenly divides n_cols.

    Prefers group_size=max_group_size (the Q4_K_M standard) for higher per-group
    accuracy vs the old group_size=64; falls back through smaller powers of two
    for oddly-dimensioned tensors.  Returning n_cols as a last resort (= one
    group = per-row scale) handles degenerate edge cases.
    """
    for gs in (32, 16, 8, 4):
        if gs <= max_group_size and n_cols >= gs * 2 and n_cols % gs == 0:
            return gs
    return n_cols  # one group covering the whole row


def quantize_tensor(
    name: str,
    arr_f32: np.ndarray,
    outlier_threshold: float,
    passthrough_patterns: list[str],
    use_int4: bool = False,
    use_nf4: bool = False,
    use_vptq: bool = False,
    use_dfloat11: bool = False,
    vptq_config=None,
    use_quip: bool = False,
    quip_quantizer=None,
    use_aqlm: bool = False,
    aqlm_config=None,
    super_weight_passthrough: bool = False,
    int4_group_size: int | None = None,
) -> dict:
    """
    Quantize a single float32 tensor.

    Returns a dict of file suffixes → arrays / bytes objects:
      INT8 (default):       __q, __s, __shape
      INT4:                 __q4, __s4, __shape
      NF4:                  __nf4, __s_nf4, __shape
      VPTQ:                 __vq_idx, __vq_cb, __vq_res, __vq_rescb, __shape
      QuIP# E8:             __quip_e8, __quip_res, __quip_rot, __shape
      AQLM:                 __aqlm_idx, __aqlm_cb, __shape
      passthrough:          __pt, __shape
      DFloat11 (pt):        __pt_df11  (bytes blob, stored via np.frombuffer)
      DFloat11 (scales):    replaces __s4 with __s4_df11 (bytes blob)
    """
    original_shape = arr_f32.shape

    # --- decide whether to pass through ---
    # Use the caller-supplied threshold for outlier detection.
    # Super-weight protection (threshold=100.0 via --super-weight) already guards
    # against the truly catastrophic outlier tensors; a separate row-max/mean cap
    # at 12.0 was too aggressive and caused ~70% of LLM weight matrices to pass
    # through as FP16, negating the INT4 compression benefit.
    effective_threshold = outlier_threshold
    skip = super_weight_passthrough or any(p in name for p in passthrough_patterns)
    if not skip and arr_f32.ndim >= 2:
        flat = arr_f32.reshape(-1, arr_f32.shape[-1])
        skip = has_outliers(flat, effective_threshold)
        if skip:
            print(f"    [outlier passthrough] {name}")

    shape_arr = np.array(original_shape, dtype=np.int64)

    if skip:
        if use_dfloat11:
            import pickle
            DFloat11Config, DFloat11Compressor = _get_dfloat11()
            cfg = DFloat11Config(use_rans=True, use_context=True)
            comp = DFloat11Compressor(cfg)
            blocks = comp.compress_array(arr_f32.astype(np.float16))
            blob = pickle.dumps(blocks, protocol=4)
            return {
                "__pt_df11": np.frombuffer(blob, dtype=np.uint8).copy(),
                "__shape": shape_arr,
            }
        return {
            "__pt": arr_f32,
            "__shape": shape_arr,
        }

    # --- reshape to 2D for per-row quantization ---
    if arr_f32.ndim == 0:
        flat = arr_f32.reshape(1, 1)
    elif arr_f32.ndim == 1:
        flat = arr_f32.reshape(1, -1)
    else:
        flat = arr_f32.reshape(-1, arr_f32.shape[-1])

    # ── VPTQ: vector quantization to ~3 bpw ──────────────────────────────────
    if use_vptq:
        VPTQConfig, VPTQQuantizer = _get_vptq()
        cfg = vptq_config if vptq_config is not None else VPTQConfig()
        quant = VPTQQuantizer(cfg)
        layer = quant.compress(flat)
        # Serialize codebooks and indices as numpy arrays
        pri_cb_data = np.concatenate([
            layer.primary_cb.centroids.reshape(-1),
            np.array([layer.primary_cb.n_codebook_entries,
                      layer.primary_cb.group_size,
                      layer.primary_cb.n_fit_iters], dtype=np.float32),
        ]).astype(np.float32)
        res_cb_data = np.zeros(1, dtype=np.float32)
        res_idx_data = np.zeros(1, dtype=np.int64)
        if layer.residual_cb is not None:
            res_cb_data = np.concatenate([
                layer.residual_cb.centroids.reshape(-1),
                np.array([layer.residual_cb.n_codebook_entries,
                          layer.residual_cb.group_size,
                          layer.residual_cb.n_fit_iters], dtype=np.float32),
            ]).astype(np.float32)
            res_idx_data = layer.residual_indices.astype(np.int64)
        col_scales = layer.col_scales if layer.col_scales is not None else np.array([], dtype=np.float32)
        return {
            "__vq_idx":   layer.primary_indices.astype(np.int64),
            "__vq_cb":    pri_cb_data,
            "__vq_res":   res_idx_data,
            "__vq_rescb": res_cb_data,
            "__vq_cols":  col_scales.astype(np.float32),
            "__vq_meta":  np.array([flat.shape[0], flat.shape[1],
                                    cfg.group_size, cfg.n_codebook_entries,
                                    cfg.n_residual_entries], dtype=np.int64),
            "__shape":    shape_arr,
        }

    # ── QuIP# E8 lattice quantization ────────────────────────────────────────
    if use_quip and quip_quantizer is not None:
        layer = quip_quantizer.quantize(flat)
        rot = (
            layer.rotation_matrix
            if layer.rotation_matrix is not None
            else np.zeros((0,), dtype=np.float16)
        )
        return {
            "__quip_e8":  layer.e8_indices,          # uint8  (N,)
            "__quip_res": layer.residual_scales,      # float16 (N,)
            "__quip_rot": rot,                        # float16 (d_in, d_in) or empty
            "__shape":    shape_arr,
        }

    # ── AQLM: additive codebook quantization ─────────────────────────────────
    if use_aqlm:
        try:
            AQLMConfig, AQLMQuantizer = _get_aqlm()
            cfg = aqlm_config if aqlm_config is not None else AQLMConfig()
            quantizer = AQLMQuantizer(cfg)
            aqlm_layer = quantizer.calibrate(flat)
            # Serialise indices: (out_features, n_groups, n_codebooks)
            aqlm_idx = aqlm_layer.indices.astype(np.uint16)
            # Serialise codebooks: flat array with header
            # Layout: [scale, float(codebook_size), float(group_size), cb0_vectors..., cb1_vectors..., ...]
            n_codebooks = cfg.n_codebooks
            cb_size     = cfg.codebook_size
            gs          = cfg.group_size
            header_size = 3  # scale, codebook_size, group_size
            cb_flat = np.empty(header_size + n_codebooks * cb_size * gs, dtype=np.float32)
            cb_flat[0] = aqlm_layer.scale
            cb_flat[1] = float(cb_size)
            cb_flat[2] = float(gs)
            offset = header_size
            for m in range(n_codebooks):
                vecs = aqlm_layer.codebooks[m].vectors.reshape(-1)  # (cb_size * gs,)
                cb_flat[offset: offset + len(vecs)] = vecs
                offset += len(vecs)
            return {
                "__aqlm_idx": aqlm_idx,    # uint16 (out, n_groups, n_codebooks)
                "__aqlm_cb":  cb_flat,     # float32 (1 + n_codebooks*cb_size*gs,)
                "__shape":    shape_arr,
            }
        except Exception as _e:  # pragma: no cover
            print(f"    [AQLM] Warning: failed on {name}: {_e} — falling back to INT8")

    # ── NF4: quantize directly from FP32 (no INT8 intermediate step) ──────────
    if use_nf4:
        quantize_nf4 = _get_nf4()
        gs_nf4 = _pick_int4_group_size(flat.shape[1])
        packed, scales_nf4 = quantize_nf4(flat, group_size=gs_nf4)
        return {
            "__nf4":   packed,      # uint8 nibble-packed  (n, d//2)
            "__s_nf4": scales_nf4,  # float32              (n, d//gs_nf4)
            "__shape": shape_arr,
        }

    if use_int4:
        # INT4 nibble-packed: quantize directly from FP32 — eliminates the
        # INT8→reconstruct→INT4 double-quantization error present in the old code.
        # group_size=32 (Q4_K_M standard) gives more per-group scale resolution
        # than 64 at ~3% storage overhead; _pick_int4_group_size() handles
        # divisibility for oddly-shaped tensors.
        # Guard: nibble-packing requires an even number of columns.
        # Tensors with odd n_cols (e.g. some vision-tower projection heads) are
        # stored as FP16 passthrough rather than crashing the whole compression.
        if flat.shape[1] % 2 != 0:
            return {"__pt": arr_f32, "__shape": shape_arr}
        gs = _pick_int4_group_size(flat.shape[1], int4_group_size or 32)
        packed, scales4 = quantize_int4(flat, group_size=gs)
        out = {
            "__q4":    packed,   # uint8 nibble-packed  (n, d//2)
            "__s4":    scales4,  # float32              (n, d//gs)
            "__shape": shape_arr,
        }
        if use_dfloat11:
            import pickle
            # Entropy-compress the INT4 scales with DFloat11 for extra savings
            DFloat11Config, DFloat11Compressor = _get_dfloat11()
            cfg = DFloat11Config(use_rans=True, use_context=True)
            comp = DFloat11Compressor(cfg)
            blocks = comp.compress_array(scales4.astype(np.float16))
            blob = pickle.dumps(blocks, protocol=4)
            out["__s4_df11"] = np.frombuffer(blob, dtype=np.uint8).copy()
            del out["__s4"]
        return out

    # ── INT8 (default) ────────────────────────────────────────────────────────
    result: QuantizationResult = quantize_embeddings(flat, group_size=64)
    return {
        "__q": result.quantized,   # int8  (grouped-64 per default)
        "__s": result.scales,      # float32 (n_rows, n_groups) or (n_rows,)
        "__shape": shape_arr,
    }


def load_mlx_weights_shard(shard_path: Path) -> dict:
    """
    Load a single safetensors shard as float32 numpy arrays.
    Uses safetensors.numpy directly (CPU only — no Metal, no MLX).
    Avoids the Metal GPU timeout that occurs when loading 7B+ models.
    """
    try:
        from safetensors.numpy import load_file as st_load_numpy
        raw = st_load_numpy(str(shard_path))
        # safetensors.numpy returns {name: np.ndarray} — may be float16 or bfloat16
        out = {}
        for name, arr in raw.items():
            # Convert to float32 for Vectro quantization
            out[name] = arr.astype(np.float32)
        return out
    except Exception:
        # Fallback: use mlx on CPU (not Metal) to handle bfloat16 and other dtypes
        # that safetensors.numpy cannot parse.  Forcing mx.cpu avoids the
        # Metal GPU command-buffer timeout that occurs on large shards.
        import mlx.core as mx
        _prev_device = mx.default_device()
        mx.set_default_device(mx.cpu)  # type: ignore[arg-type]
        try:
            shard_weights = mx.load(str(shard_path))
            return {
                name: np.array(arr.astype(mx.float32))
                for name, arr in shard_weights.items()  # type: ignore[union-attr]
            }
        finally:
            mx.set_default_device(_prev_device)


def load_mlx_weights(model_dir: Path) -> dict:
    """
    Load all weights from safetensors shards as float32 numpy arrays.

    ⚠ NOTE: This loads the ENTIRE model into RAM as float32.  For models >3B this
    can easily exceed available memory.  Use process_weights_streaming() instead,
    which processes one shard at a time and writes output as it goes.

    Kept for backward-compatibility with existing callers.
    """
    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    weights = {}
    for shard in shard_files:
        print(f"  Loading {shard.name} ...")
        weights.update(load_mlx_weights_shard(shard))
    return weights


def process_weights_streaming(
    model_dir: Path,
    output_path: Path,
    passthrough_patterns: list[str],
    outlier_threshold: float,
    verbose: bool,
    awq_scales: dict | None = None,
    use_int4: bool = False,
    use_nf4: bool = False,
    use_vptq: bool = False,
    use_dfloat11: bool = False,
    vptq_config=None,
    use_quip: bool = False,
    quip_bits: int = 2,
    use_aqlm: bool = False,
    aqlm_config=None,
    use_super_weight: bool = False,
    int4_group_size: int | None = None,
) -> dict:
    """
    Streaming shard-by-shard compression — works for any model size.

    Processes one .safetensors shard at a time:
      1. Load shard (CPU numpy, no MLX/Metal)
      2. Quantize each tensor
      3. Write .npy files immediately
      4. Free the shard from RAM

    Peak RAM ≈ 2× the size of one shard (typically 2-5 GB for 7B models,
    ~2 GB for sharded 7B), regardless of total model size.
    """
    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    tensor_dir = output_path / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    manifest  = {}   # original_name → safe_key
    stats     = {"n_quantized": 0, "n_passthrough": 0,
                 "orig_f32_bytes": 0, "compressed_bytes": 0}
    total_tensors = 0

    print(f"\n  Processing {len(shard_files)} shard(s) …  (streaming — peak RAM ≈ 1 shard)")

    # ── Pre-compute AWQ group scales once (streaming-safe, per-tensor lookup) ─
    _awq_proj_apply: dict = {}
    _awq_ln_apply: dict = {}
    if awq_scales:
        try:
            from squish.quant.awq import prepare_awq_application
            _awq_proj_apply, _awq_ln_apply = prepare_awq_application(awq_scales)
        except ImportError:
            pass

    # ── Build QuIP# quantizer once; shared RNG advances across all tensors ───
    quip_quantizer = None
    if use_quip:
        QuIPSharpConfig, QuIPSharpQuantizer = _get_quip()
        quip_quantizer = QuIPSharpQuantizer(QuIPSharpConfig(scalar_bits=quip_bits), seed=42)

    # ── Initialise super-weight calibrator once (reused per shard) ───────────
    _sw_calibrator = None
    if use_super_weight:
        from squish.quant.super_weight_calibrator import (
            SuperWeightCalibrator,
            SuperWeightConfig,
        )
        _sw_calibrator = SuperWeightCalibrator(SuperWeightConfig(
            threshold=100.0,
            threshold_1d=1e9,   # 1D tensors (biases, layernorms) don't need INT4 super-weight protection
        ))

    for shard_idx, shard in enumerate(shard_files, 1):
        print(f"\n  [{shard_idx}/{len(shard_files)}] {shard.name}")
        shard_weights = load_mlx_weights_shard(shard)
        shard_tensors = len(shard_weights)

        # ── Scan for super weights within this shard ──────────────────────────
        sw_tensor_names: set[str] = set()
        if _sw_calibrator is not None:
            coords = _sw_calibrator.scan_weights(shard_weights)
            sw_tensor_names = {c.tensor_name for c in coords}
            if sw_tensor_names:
                n_sw = len(sw_tensor_names)
                listed = ", ".join(sorted(sw_tensor_names)[:4])
                overflow = f" (+{n_sw-4} more)" if n_sw > 4 else ""
                print(f"    [super-weight] protecting {n_sw} tensor(s) as FP16: {listed}{overflow}")

        sp = Spinner(f"Shard {shard_idx}/{len(shard_files)}  ({shard_tensors} tensors)").start()
        for tensor_idx, (name, arr_f32) in enumerate(shard_weights.items(), 1):
            sp.update(f"{tensor_idx}/{shard_tensors}  {name}")
            sk = safe_key(name)
            manifest[name] = sk

            # ── Phase 1.2: Apply AWQ scales before quantization ───────────
            if awq_scales:
                arr_f32 = _apply_awq_single(name, arr_f32, _awq_proj_apply, _awq_ln_apply)

            sub = quantize_tensor(
                name, arr_f32, outlier_threshold, passthrough_patterns,
                use_int4=use_int4,
                use_nf4=use_nf4,
                use_vptq=use_vptq,
                use_dfloat11=use_dfloat11,
                vptq_config=vptq_config,
                use_quip=use_quip,
                quip_quantizer=quip_quantizer,
                use_aqlm=use_aqlm,
                aqlm_config=aqlm_config,
                super_weight_passthrough=(name in sw_tensor_names),
                int4_group_size=int4_group_size,
            )

            # Write immediately — don't accumulate in RAM
            _BINARY_BLOB_SUFFIXES = {"__pt_df11", "__s4_df11"}
            for suffix, data in sub.items():
                out_path = tensor_dir / f"{sk}{suffix}.npy"
                if suffix in _BINARY_BLOB_SUFFIXES:
                    # Binary blobs stored as uint8 numpy arrays (byte-exact round-trip)
                    np.save(str(out_path), np.asarray(data, dtype=np.uint8))
                elif suffix == "__pt":
                    np.save(str(out_path), data.astype(np.float16))
                else:
                    np.save(str(out_path), data)

            orig_bytes = arr_f32.nbytes
            comp_bytes = sum(
                (tensor_dir / f"{sk}{sfx}.npy").stat().st_size
                for sfx in sub
                if not sfx.endswith("__shape")
            )
            stats["orig_f32_bytes"]   += orig_bytes
            stats["compressed_bytes"] += comp_bytes

            if "__pt" in sub or "__pt_df11" in sub:
                stats["n_passthrough"] += 1
            else:
                stats["n_quantized"] += 1

            if verbose:
                ratio = orig_bytes / max(comp_bytes, 1)
                if "__pt_df11" in sub:
                    mode = "DF11"
                elif "__pt" in sub:
                    mode = "PT"
                elif "__nf4" in sub:
                    mode = "NF4"
                elif "__vq_idx" in sub:
                    mode = "VPTQ"
                elif "__quip_e8" in sub:
                    mode = "QUIP"
                elif "__aqlm_idx" in sub:
                    mode = "AQLM"
                elif use_int4:
                    mode = "Q4"
                else:
                    mode = "Q8"
                _clear_line()
                print(f"  [{mode}] {name}: {arr_f32.shape} ratio={ratio:.2f}x")

            total_tensors += 1
            del arr_f32

        sp.stop(f"Shard {shard_idx} done  ({shard_tensors} tensors written)")
        del shard_weights

    # Write manifest
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    # Sentinel for consistency with old npy-dir reader
    (tensor_dir / ".manifest_ready").touch()

    print(f"\n  Total tensors: {total_tensors}")
    return stats


def write_npy_dir(output_dir: Path, npz_dict: dict, manifest: dict) -> int:
    """
    Write tensors as individual uncompressed .npy files for memory-mapped loading.

    Layout::
        {output_dir}/
            manifest.json          # original_name → safe_key
            tensors/
                {safe_key}__q.npy     # int8 quantized weights
                {safe_key}__s.npy     # float32 per-row scales
                {safe_key}__shape.npy # int64 original shape
                {safe_key}__pt.npy    # float16 passthrough weights

    Passthrough tensors are stored as float16:
      - Original model was bfloat16.  bf16 → f32 is lossless; f32 → f16 has MORE
        mantissa bits than bf16 (10 vs 7), so all precision from the source model
        is preserved.  Saves ~50%% disk vs float32.
    No zlib compression: the OS can mmap individual .npy files for near-zero
    decompression overhead when loading.

    Returns total bytes written.
    """
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    for key, arr in npz_dict.items():
        out_arr = arr.astype(np.float16) if key.endswith("__pt") else arr
        path = tensor_dir / f"{key}.npy"
        np.save(str(path), out_arr)
        total_bytes += path.stat().st_size
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return total_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--format",
        choices=["npz", "npy-dir"],
        default="npy-dir",
        help="Storage format: npy-dir (default, fast mmap loading) or npz (zlib-compressed)",
    )
    ap.add_argument(
        "--passthrough",
        nargs="*",
        default=[],
        metavar="PATTERN",
        help="Tensor name substrings to store as float32 without quantizing",
    )
    ap.add_argument(
        "--outlier-threshold",
        type=float,
        default=20.0,
        help="Auto-passthrough if row max/mean ratio exceeds this (default: 20)",
    )
    ap.add_argument(
        "--awq-scales",
        metavar="DIR",
        default=None,
        help="Directory of .awq.npy scale files produced by 'python3 -m squish.quant.awq'. "
             "When provided, AWQ scales are applied to each weight tensor before "
             "quantization for improved INT8 accuracy.",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--int4",
        action="store_true",
        default=False,
        help="Use INT4 nibble-packed quantization instead of INT8.  Halves disk usage "
             "(~1.5 GB for 1.5B vs ~2.9 GB INT8) at ≤2%% accuracy delta.  "
             "Requires squish_quant Rust extension (built with maturin).  "
             "Recommended for 1.5B models where every GB matters.",
    )
    ap.add_argument(
        "--nf4",
        action="store_true",
        default=False,
        help="Use NF4 (NormalFloat-4) quantization.  Better SNR than INT4 under "
             "Gaussian weight distribution — exact QLoRA codebook from Dettmers et al. "
             "(arXiv:2305.14314).  ~4 bpw, ~0.2 dB better SNR than INT4.",
    )
    ap.add_argument(
        "--vptq",
        action="store_true",
        default=False,
        help="Use VPTQ vector quantization (arXiv:2409.17066, NeurIPS 2025).  "
             "~3 bpw with near-INT4 quality.  Slower to compress but best ratio.",
    )
    ap.add_argument(
        "--vptq-codebook-size",
        type=int,
        default=256,
        metavar="N",
        help="VPTQ primary codebook size (default: 256 = 8-bit codes).",
    )
    ap.add_argument(
        "--vptq-group-size",
        type=int,
        default=8,
        metavar="N",
        help="VPTQ vector group size (default: 8 weights per vector).",
    )
    ap.add_argument(
        "--dfloat11",
        action="store_true",
        default=False,
        help="Apply DFloat11 lossless entropy compression with rANS + context model "
             "to passthrough tensors and INT4 scales.  ~0.5-1 additional bpw savings "
             "over raw quantization.  Zero quality impact (lossless).",
    )
    ap.add_argument(
        "--ultra",
        action="store_true",
        default=False,
        help="Ultra compression mode: enables --nf4 --dfloat11 and maximises entropy "
             "coding.  Right at the near-lossless INT4-class compression limit.",
    )
    ap.add_argument(
        "--quip",
        action="store_true",
        default=False,
        help="Use QuIP# trellis-coded E8 lattice quantization (arXiv:2402.04396).  "
             "Each 8-D weight block is projected onto one of 256 unit-sphere E8 "
             "codewords plus a per-block float16 scale.  Enables incoherence "
             "preprocessing via random Hadamard rotation.  ~2-3 bpw effective.",
    )
    ap.add_argument(
        "--quip-bits",
        type=int,
        default=2,
        choices=[2, 3],
        metavar="N",
        help="QuIP# scalar bits for residual scale representation (2 or 3, default: 2).",
    )
    ap.add_argument(
        "--aqlm",
        action="store_true",
        default=False,
        help="Use AQLM (Additive Quantization of Language Models, ICML 2024).  "
             "Additive codebook lookup achieves ~2-bit effective precision with "
             "beam-search calibration.  Pure numpy — no GPU required.",
    )
    ap.add_argument(
        "--aqlm-codebooks",
        type=int,
        default=2,
        metavar="M",
        help="Number of additive codebooks for AQLM (default: 2).",
    )
    ap.add_argument(
        "--aqlm-cbsize",
        type=int,
        default=16,
        metavar="K",
        help="Number of codewords per AQLM codebook (default: 16).",
    )
    ap.add_argument(
        "--super-weight",
        action="store_true",
        default=False,
        help="Protect super-weight tensors as FP16 passthrough during INT4/NF4 "
             "compression.  Scans each shard for extreme outlier elements "
             "(element/row-mean ratio > 100) before quantizing; tensors containing "
             "super-weights are stored losslessly to prevent coherence collapse.  "
             "Recommended for all INT4 conversions.",
    )
    ap.add_argument(
        "--int4-group-size",
        type=int,
        default=None,
        dest="int4_group_size",
        metavar="N",
        help="Override per-group size for INT4 quantization (must be a power of two "
             "that divides the weight matrix column count). Default: auto-select 32.",
    )
    args = ap.parse_args()

    # ── --ultra implies nf4 + dfloat11 ────────────────────────────────────────
    if args.ultra:
        args.nf4 = True
        args.dfloat11 = True

    # ── VPTQ config (built once and reused per tensor) ─────────────────────────
    vptq_config = None
    if args.vptq:
        from squish.quant.vptq import VPTQConfig
        vptq_config = VPTQConfig(
            n_codebook_entries=args.vptq_codebook_size,
            group_size=args.vptq_group_size,
        )

    # ── AQLM config (built once and reused per tensor) ─────────────────────────
    aqlm_config = None
    if args.aqlm:
        try:
            AQLMConfig, _ = _get_aqlm()
            aqlm_config = AQLMConfig(
                n_codebooks=args.aqlm_codebooks,
                codebook_size=args.aqlm_cbsize,
            )
        except ImportError:  # pragma: no cover
            print("  [AQLM] Warning: squish.quant.aqlm not found — falling back to INT8")
            args.aqlm = False

    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # For npy-dir, output_path IS the directory; manifest lives inside it.
    # For npz, manifest is a sibling file.
    if args.format == "npy-dir":
        manifest_path = str(output_path / "manifest.json")
    else:
        manifest_path = str(output_path).replace(".npz", "_manifest.json")

    if args.format == "npy-dir":
        # ── Streaming path (7B+): load one shard → quantize → write → free ─────
        # Keeps peak RAM to ~1 shard (~2 GB for 7B) instead of loading the full
        # model into GPU memory all at once (which triggers Metal GPU timeout on
        # 16 GB unified-memory machines like M3 MacBook Pro).
        print(f"\nStreaming quantization → {output_path}/tensors/")
        print("  (CPU-only shard loading — no Metal GPU, works for any model size)")

        # ── Load AWQ scales if provided ────────────────────────────────────
        awq_scales: dict = {}
        if args.awq_scales:
            try:
                from squish.quant.awq import load_awq_scales
                awq_scales = load_awq_scales(args.awq_scales)
                n_awq = len(awq_scales)
                print(f"  [AWQ] Loaded {n_awq} layer scales from {args.awq_scales}")
            except Exception as e:
                print(f"  [AWQ] Warning: could not load scales: {e}  (continuing without AWQ)")

        t0 = time.time()
        stats = process_weights_streaming(
            model_dir,
            output_path,
            args.passthrough,
            args.outlier_threshold,
            args.verbose,
            awq_scales=awq_scales,
            use_int4=args.int4,
            use_nf4=args.nf4,
            use_vptq=args.vptq,
            use_dfloat11=args.dfloat11,
            vptq_config=vptq_config,
            use_quip=args.quip,
            quip_bits=args.quip_bits,
            use_aqlm=args.aqlm,
            aqlm_config=aqlm_config,
            use_super_weight=args.super_weight,
            int4_group_size=args.int4_group_size,
        )
        elapsed = time.time() - t0

        tensor_dir = output_path / "tensors"
        disk_bytes = sum(p.stat().st_size for p in tensor_dir.glob("*.npy"))
        disk_mb = disk_bytes / 1e6
        orig_gb = stats["orig_f32_bytes"] / 1e9
        comp_gb = stats["compressed_bytes"] / 1e9
        ratio = stats["orig_f32_bytes"] / max(stats["compressed_bytes"], 1)
        disk_ratio = stats["orig_f32_bytes"] / max(disk_bytes, 1)
        n_total = stats["n_quantized"] + stats["n_passthrough"]

        print(f"\n{'='*50}")
        print("  Format:           npy-dir (streaming)")
        _int4_gs = getattr(args, "int4_group_size", None) or 32
        _mode_str = (
            "ULTRA (NF4 + DFloat11 rANS)" if args.ultra else
            "QuIP# E8 trellis-coded" if args.quip else
            "AQLM additive codebook" if args.aqlm else
            "NF4 (NormalFloat-4)" if args.nf4 else
            "VPTQ (vector quantization)" if args.vptq else
            f"INT4 nibble-packed (group-{_int4_gs})" if args.int4 else
            "INT8 per-group-64"
        )
        if args.dfloat11 and not args.ultra:
            _mode_str += " + DFloat11 entropy"
        print(f"  Quantization:     {_mode_str}")
        _quant_label = "INT4" if args.int4 else "INT8"
        print(f"  Tensors:          {n_total} total")
        print(f"    Quantized ({_quant_label}): {stats['n_quantized']}")
        print(f"    Passthrough (f16 on disk): {stats['n_passthrough']}")
        print(f"  Original (f32):   {orig_gb:.3f} GB")
        print(f"  Quantized raw:    {comp_gb:.3f} GB  ({ratio:.2f}x ratio)")
        print(f"  On-disk (npy-dir): {disk_mb:.1f} MB  ({disk_ratio:.2f}x ratio)")
        print(f"  Total time:       {elapsed:.1f}s")
        print(f"  Manifest:         {output_path / 'manifest.json'}")
        print(f"{'='*50}")

    else:
        # ── Legacy batch path (NPZ / small models) ────────────────────────────
        print(f"Loading weights from {model_dir} ...")
        weights = load_mlx_weights(model_dir)
        print(f"  {len(weights)} tensors loaded")

        print(f"\nQuantizing {len(weights)} tensors ...")
        npz_dict = {}
        manifest = {}  # original_name -> safe_key
        stats = {
            "n_quantized": 0,
            "n_passthrough": 0,
            "orig_f32_bytes": 0,
            "compressed_bytes": 0,
        }

        t0 = time.time()
        total = len(weights)
        sp = Spinner("Quantizing").start()
        for idx, (name, arr_f32) in enumerate(weights.items(), 1):
            sp.update(f"{idx}/{total}  {name}")
            sk = safe_key(name)
            manifest[name] = sk

            sub = quantize_tensor(name, arr_f32, args.outlier_threshold, args.passthrough)

            for suffix, data in sub.items():
                npz_dict[sk + suffix] = data

            orig_bytes = arr_f32.nbytes
            comp_bytes = sum(d.nbytes for k, d in sub.items() if not k.endswith("__shape"))
            stats["orig_f32_bytes"] += orig_bytes
            stats["compressed_bytes"] += comp_bytes

            if "__pt" in sub:
                stats["n_passthrough"] += 1
            else:
                stats["n_quantized"] += 1

            if args.verbose:
                ratio = orig_bytes / max(comp_bytes, 1)
                mode = "PT" if "__pt" in sub else "Q8"
                _clear_line()
                print(f"  [{mode}] {name}: {arr_f32.shape} ratio={ratio:.2f}x")

        sp.stop(f"Quantization done  ({stats['n_quantized']}Q + {stats['n_passthrough']}PT)")

        t1 = time.time()
        print(f"\nWriting {output_path} ...")
        print("  (this is the slow step — zlib single-threaded compression)")
        with Spinner(f"savez_compressed  →  {output_path.name}"):
            np.savez_compressed(str(output_path), **npz_dict)
        write_time = time.time() - t1
        print(f"  Written in {write_time:.1f}s")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        elapsed = time.time() - t0
        orig_gb = stats["orig_f32_bytes"] / 1e9
        comp_gb = stats["compressed_bytes"] / 1e9
        ratio = stats["orig_f32_bytes"] / max(stats["compressed_bytes"], 1)
        disk_bytes = output_path.stat().st_size
        disk_mb = disk_bytes / 1e6
        disk_ratio = stats["orig_f32_bytes"] / max(disk_bytes, 1)

        print(f"\n{'='*50}")
        print("  Format:           npz")
        print(f"  Tensors:          {len(weights)} total")
        print(f"    Quantized (Q8): {stats['n_quantized']}")
        print(f"    Passthrough (f16 on disk): {stats['n_passthrough']}")
        print(f"  Original (f32):   {orig_gb:.3f} GB")
        print(f"  Quantized raw:    {comp_gb:.3f} GB  ({ratio:.2f}x ratio)")
        print(f"  On-disk (zlib):   {disk_mb:.1f} MB  ({disk_ratio:.2f}x ratio)")
        print(f"  Write time:       {write_time:.1f}s")
        print(f"  Total time:       {elapsed:.1f}s")
        print(f"  Manifest:         {manifest_path}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
