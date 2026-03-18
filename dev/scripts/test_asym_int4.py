#!/usr/bin/env python3
"""Quick SNR comparison: asymmetric INT4 vs symmetric INT4."""
import squish_quant
import numpy as np

rng = np.random.default_rng(42)
# Skewed: positive mean (common in LLM FFN/gate weights)
arr_skewed = (rng.standard_normal((256, 512)) * 0.02 + 0.003).astype(np.float32)
# Symmetric Gaussian: embedding / attention weights
arr_gaussian = rng.standard_normal((256, 512)).astype(np.float32) * 0.02

print("=== INT4 Asymmetric vs Symmetric SNR comparison (group_size=32) ===\n")
for name, arr in [("Skewed LLM weights", arr_skewed), ("Gaussian weights", arr_gaussian)]:
    packed_a, scales_a, zps_a = squish_quant.quantize_int4_asymmetric_grouped(arr, 32)
    recon_a = squish_quant.dequantize_int4_asymmetric_grouped(packed_a, scales_a, zps_a, 32)
    packed_s, scales_s = squish_quant.quantize_int4_grouped(arr, 32)
    recon_s = squish_quant.dequantize_int4_grouped(packed_s, scales_s, 32)

    snr_a = 10 * np.log10(np.mean(arr**2) / np.mean((arr - recon_a)**2))
    snr_s = 10 * np.log10(np.mean(arr**2) / np.mean((arr - recon_s)**2))
    mse_a = np.mean((arr - recon_a)**2)
    mse_s = np.mean((arr - recon_s)**2)
    mse_pct = (mse_s - mse_a) / mse_s * 100

    print(f"{name}")
    print(f"  Asymmetric: SNR={snr_a:.2f} dB  MSE={mse_a:.2e}")
    print(f"  Symmetric:  SNR={snr_s:.2f} dB  MSE={mse_s:.2e}")
    print(f"  Improvement: +{snr_a - snr_s:.2f} dB  ({mse_pct:.1f}% lower MSE)\n")

print("=== Storage overhead of asymmetric (zero_points array) ===")
rows, cols, gs = 256, 4096, 32
n_groups = cols // gs
# Symmetric:  packed=(rows, cols//2)  scales=(rows, n_groups)
sym_bytes = (rows * cols // 2) + (rows * n_groups * 4)
# Asymmetric: packed=(rows, cols//2)  scales=(rows, n_groups)  zp=(rows, n_groups)
asym_bytes = (rows * cols // 2) + (rows * n_groups * 4) + (rows * n_groups)
overhead_pct = (asym_bytes - sym_bytes) / sym_bytes * 100
print(f"  rows={rows}, cols={cols}, gs={gs}")
print(f"  Symmetric:  {sym_bytes/1024:.1f} KB")
print(f"  Asymmetric: {asym_bytes/1024:.1f} KB  (+{overhead_pct:.1f}% overhead)")
