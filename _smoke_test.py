import numpy as np
from squish.quantizer import quantize_embeddings, reconstruct_embeddings, _quantize_numpy_asymmetric, _reconstruct_numpy

# Simple 2x4 test case
x = np.array([[1.0, 2.0, 3.0, 4.0],
              [-0.5, 0.5, 1.5, 2.5]], dtype=np.float32)

ra = _quantize_numpy_asymmetric(x, group_size=0)  # per-row
print(f"q={ra.quantized}")
print(f"scales={ra.scales}")
print(f"zero_points={ra.zero_points}")

xa = _reconstruct_numpy(ra)
print(f"original={x}")
print(f"reconstructed={xa}")
mse = np.mean((x - xa)**2)
print(f"MSE={mse}")
assert mse < 0.01, f"Per-row MSE too high: {mse}"

# Per-group test
ra2 = _quantize_numpy_asymmetric(x, group_size=2)
print(f"\nper-group: q={ra2.quantized}, zp={ra2.zero_points}, scales={ra2.scales}")
xa2 = _reconstruct_numpy(ra2)
print(f"original={x}")
print(f"reconstructed={xa2}")
mse2 = np.mean((x - xa2)**2)
print(f"Per-group MSE={mse2}")
assert mse2 < 0.01, f"Per-group MSE too high: {mse2}"
print("Debug passed")
