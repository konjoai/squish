# AQLM — Additive Quantization of Language Models

## 1. Overview

AQLM (Additive Quantization of Language Models) is a post-training weight-compression
technique that achieves **sub-2-bit effective precision** on large language model weight
matrices by representing each group of weights as the sum of multiple discrete codeword
vectors drawn from learned codebooks.

**Reference paper:** Egiazarian et al. (2024), *AQLM: Additive Quantization of Language
Models*, ICML 2024. arXiv: [2401.06118](https://arxiv.org/abs/2401.06118).

The key result of the paper is that 2-bit AQLM can match or exceed the perplexity of
3–4 bit scalar quantisation methods (GPTQ, AWQ) on 7B+ parameter models, while reducing
model storage by 2–3× compared to INT8 and 8–16× compared to FP16.

---

## 2. Algorithm

### 2.1 Codebook Initialisation (k-means)

Each weight row is split into groups of `group_size` (D) weights. The `n_codebooks` (M)
additive codebooks are initialised sequentially:

1. Collect all residual weight-group vectors (initially the raw weight groups themselves).
2. Run **k-means++** initialisation followed by Lloyd iterations to fit a codebook of
   `codebook_size` (K) centroid vectors, each of length D.
3. Assign each weight group to its nearest centroid and subtract the selected codeword
   from the residual, exposing the error for the next codebook.
4. Repeat from step 1 for codebook m+1 using the updated residuals.

This greedy sequential strategy is a form of **residual vector quantization** (RVQ).

### 2.2 Beam-Search Residual Assignment

After codebook initialisation, the index assignment is refined with a beam search over
the codebook product space:

- A beam of width `beam_width` (B) candidate index tuples is maintained.
- At each codebook level, the B current candidates are each extended by the K nearest
  codewords; the top-B candidates by total reconstruction error are retained.
- This yields a near-optimal joint assignment across all M codebooks without the
  exponential cost of exhaustive search.

The implementation in squish uses a greedy nearest-neighbour assignment (beam_width=1
equivalent) for speed in the pure-Python path. The full beam search is activated when
`beam_width > 1` in `AQLMConfig`.

### 2.3 Why Sub-2-bit Effective Precision?

Each group of D=8 weights is encoded by M=2 indices into codebooks of size K=16 (4-bit
codes). Total index bits per group = M × log₂(K) = 2 × 4 = **8 bits for 8 weights** =
**1.0 bit/weight** for the index storage. The codebook vectors themselves add a small
fixed overhead (K × D × float16 = 16 × 8 × 16 bits = 2048 bits per codebook), which
amortises to near-zero for large matrices. The resulting effective precision is typically
**1.0–2.0 bpw** depending on the codebook configuration.

---

## 3. Squish Implementation

### 3.1 Class Overview

| Class | Role |
|---|---|
| `AQLMConfig` | Dataclass holding all hyperparameters |
| `AQLMCodebook` | Single codebook: K codewords of length D, k-means init + nearest-query |
| `AQLMLayer` | Compressed weight matrix: codebooks + index array + scale; `dequantize()` method |
| `AQLMQuantizer` | Top-level API: `compress(W)` → `AQLMLayer`, `decompress(layer)` → ndarray |

All classes live in `squish/quant/aqlm.py`. Public symbols are re-exported via
`squish.quant.aqlm.__all__`.

### 3.2 `AQLMConfig`

```python
from squish.quant.aqlm import AQLMConfig

cfg = AQLMConfig(
    n_codebooks=2,       # M additive codebooks
    codebook_size=16,    # K codewords per codebook (log2(K) bits per index)
    group_size=8,        # D weights per group vector
    n_iterations=25,     # k-means Lloyd iterations
    beam_width=8,        # beam width for residual assignment
)
```

### 3.3 `compress` / `decompress` API

```python
import numpy as np
from squish.quant.aqlm import AQLMConfig, AQLMQuantizer

# Create a random weight matrix (e.g. a transformer projection layer)
W = np.random.randn(4096, 4096).astype(np.float32)

# Compress
cfg = AQLMConfig(n_codebooks=2, codebook_size=16, group_size=8)
quantizer = AQLMQuantizer(cfg)

layer = quantizer.compress(W)          # returns AQLMLayer
print(f"bpw = {layer.compressed_bits / W.size:.3f}")

# Decompress
W_hat = quantizer.decompress(layer)    # returns float32 ndarray, shape (4096, 4096)

# Reconstruction SNR
noise = W - W_hat
snr_db = 10 * np.log10(np.mean(W**2) / np.mean(noise**2))
print(f"SNR = {snr_db:.1f} dB")
```

### 3.4 `AQLMLayer.dequantize()`

`dequantize()` reconstructs the weight matrix directly from the stored index arrays
and codebook vectors:

```python
# Manual dequantize
W_hat = layer.dequantize()
```

Internally this performs a vectorised gather across all M codebooks:

```
reconstructed[i, j] = sum_m( codebooks[m].vectors[ indices[i, j, m] ] )
W_hat[i, :] = reconstructed[i, :] * layer.scale
```

### 3.5 Model-Level Quantization

```python
from squish.quant.aqlm import AQLMConfig, quantize_model_aqlm

# model_weights is a dict[str, np.ndarray] of the model's 2-D weight tensors
compressed = quantize_model_aqlm(model_weights, config=AQLMConfig())
# compressed is a dict[str, AQLMLayer]
```

---

## 4. Compression Tradeoffs

The table below compares common quantisation methods for a **7B parameter model**
(e.g. Llama-2-7B / Qwen-2.5-7B). PPL delta is approximate change in wikitext-2
perplexity versus FP16 baseline; lower is better. Compress time is per-layer on
a single CPU core (pure Python / numpy).

| Method | BPW | Size (GB) | PPL delta | Compress time |
|---|---|---|---|---|
| AQLM 2-bit | ~2.0 | ~1.75 | +0.5–1.5 | minutes (CPU) |
| GPTQ 3-bit | 3.0 | ~2.6 | +0.3–0.8 | minutes (GPU) |
| GPTQ 4-bit | 4.0 | ~3.5 | +0.1–0.3 | minutes (GPU) |
| INT8 | 8.0 | ~7.0 | ~0.0 | seconds |
| FP16 | 16.0 | ~14.0 | 0.0 (baseline) | — |

Notes:
- AQLM bpw includes codebook storage amortised over a 7B parameter matrix collection.
- PPL delta varies by model family, calibration set, and configuration.
- GPTQ timings assume GPU-accelerated Hessian computation; squish AQLM runs on CPU.

---

## 5. Usage via CLI

AQLM compression is available as an experimental flag via the squish `compress` command:

```bash
squish compress \
    --model-dir models/Qwen2.5-7B \
    --aqlm \
    --output compressed/Qwen2.5-7B-aqlm
```

The `--aqlm` flag activates `quantize_model_aqlm` internally. The output directory
contains the compressed weight files alongside a `config.json` with the `AQLMConfig`
parameters used.

> **Note:** `--aqlm` is marked **experimental**. The CLI interface may change between
> squish releases. For production use, call `quantize_model_aqlm` directly from Python.

---

## 6. Limitations

- **Post-training only.** AQLM in squish is a pure weight-compression step with no
  activation-aware calibration (Hessian-weighted reconstruction as used in GPTQ / AWQ
  is not yet implemented). The `calib_inputs` parameter in `AQLMQuantizer.calibrate`
  is reserved for a future activation-aware path.

- **Small models suffer more PPL loss.** Models with fewer than ~3B parameters have
  shallower residual streams and less redundancy; 2-bit compression with default settings
  typically incurs +2–5 PPL on sub-3B models compared to +0.5–1.5 PPL on 7B+ models.
  For small models, prefer GPTQ 4-bit or increase `codebook_size` / `n_codebooks`.

- **Pure Python / numpy path.** The squish AQLM implementation is written in pure numpy
  for portability. Compress time scales as O(out_features × n_groups × codebook_size²)
  and can be slow for large layers (4096×4096) on CPU. A Metal / MLX accelerated path
  is planned for a future release.

- **No fused dequantize kernel.** Inference with AQLM-compressed weights currently
  dequantizes the full weight matrix before each matmul. Fused kernel support
  (index-gather + matmul in a single Metal dispatch) is tracked as a future enhancement.
