#!/usr/bin/env python3
"""Benchmark: asymmetric INT4 plain vs MSE-clipped vs symmetric."""
import sys
sys.path.insert(0, "/Users/wscholl/squish")

import numpy as np
from squish.quant.quantizer import (
    quantize_int4_asymmetric,
    quantize_int4_asymmetric_mse,
    dequantize_int4_asymmetric,
)
import squish_quant
import time

rng = np.random.default_rng(42)

def snr_db(orig, approx):
    sig = np.mean(orig.astype(np.float32) ** 2)
    err = np.mean((orig.astype(np.float32) - approx.astype(np.float32)) ** 2)
    return float("inf") if err == 0 else 10 * np.log10(sig / max(err, 1e-30))

print("=== INT4 MSE Clipping vs Plain Asymmetric (group_size=32) ===\n")

test_cases = [
    ("Gaussian weights (std=0.02)",
     rng.standard_normal((512, 512)).astype(np.float32) * 0.02),
    ("Skewed weights (std=0.02, mean=0.003)",
     (rng.standard_normal((512, 512)) * 0.02 + 0.003).astype(np.float32)),
    ("With outliers (1% of values are 10x larger)",
     None),  # generated below
]

# Generate the outlier case
base = rng.standard_normal((512, 512)).astype(np.float32) * 0.02
mask = rng.random((512, 512)) < 0.01  # 1% outliers
outlier_arr = base.copy()
outlier_arr[mask] *= 10.0  # inflate 1% to 10x magnitude
test_cases[2] = ("With 1% outliers (10× magnitude)", outlier_arr)

for name, arr in test_cases:
    # Symmetric INT4 (baseline)
    packed_s, scales_s = squish_quant.quantize_int4_grouped(
        np.ascontiguousarray(arr, dtype=np.float32), 32
    )
    recon_s = squish_quant.dequantize_int4_grouped(packed_s, scales_s, 32)
    snr_sym = snr_db(arr, recon_s)

    # Asymmetric INT4 (plain)
    t0 = time.perf_counter()
    packed_a, scales_a, zps_a = quantize_int4_asymmetric(arr, group_size=32)
    t_plain = time.perf_counter() - t0
    recon_a = dequantize_int4_asymmetric(packed_a, scales_a, zps_a, group_size=32)
    snr_asym = snr_db(arr, recon_a)

    # Asymmetric INT4 + MSE clipping
    t0 = time.perf_counter()
    packed_m, scales_m, zps_m = quantize_int4_asymmetric_mse(arr, group_size=32, n_clip_candidates=8)
    t_mse = time.perf_counter() - t0
    recon_m = dequantize_int4_asymmetric(packed_m, scales_m, zps_m, group_size=32)
    snr_mse = snr_db(arr, recon_m)

    print(f"{name}")
    print(f"  Symmetric INT4:          {snr_sym:.2f} dB  (baseline)")
    print(f"  Asymmetric INT4:         {snr_asym:.2f} dB  (+{snr_asym-snr_sym:.2f} dB)  {t_plain*1000:.1f}ms")
    print(f"  Asymmetric INT4 + MSE:   {snr_mse:.2f} dB  (+{snr_mse-snr_sym:.2f} dB vs sym, +{snr_mse-snr_asym:.2f} dB vs plain)  {t_mse*1000:.1f}ms")
    overhead = t_mse / max(t_plain, 1e-9) - 1
    print(f"  MSE overhead:            {overhead*100:.0f}% slower than plain asymmetric\n")

# Large matrix test (7B-class)
print("=== Large matrix (4096x4096, mimicking 7B FFN weight) ===")
arr_large = rng.standard_normal((4096, 4096)).astype(np.float32) * 0.02

t0 = time.perf_counter()
packed_m, scales_m, zps_m = quantize_int4_asymmetric_mse(arr_large, group_size=32, n_clip_candidates=8)
t_large = time.perf_counter() - t0
recon_m = dequantize_int4_asymmetric(packed_m, scales_m, zps_m, group_size=32)
snr_large_mse = snr_db(arr_large, recon_m)

packed_a, scales_a, zps_a = quantize_int4_asymmetric(arr_large, group_size=32)
recon_a = dequantize_int4_asymmetric(packed_a, scales_a, zps_a, group_size=32)
snr_large_plain = snr_db(arr_large, recon_a)

print(f"  Plain asymmetric:  {snr_large_plain:.2f} dB")
print(f"  MSE-clipped:       {snr_large_mse:.2f} dB  (+{snr_large_mse-snr_large_plain:.2f} dB)")
print(f"  Time for MSE:      {t_large:.2f}s  ({4096*4096*4/1e6:.0f} MB tensor)")
