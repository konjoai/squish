import numpy as np
from squish.quantizer import quantize_embeddings, reconstruct_embeddings, mean_cosine_similarity

rng = np.random.default_rng(42)
# Shifted mean: asymmetric is better for MSE (absolute error), not cosine
x = rng.standard_normal((256, 512)).astype(np.float32) + 2.0

rs = quantize_embeddings(x, group_size=64, asymmetric=False)
xs = reconstruct_embeddings(rs)
mse_s = float(np.mean((x - xs) ** 2))
cs = mean_cosine_similarity(x, xs)

ra = quantize_embeddings(x, group_size=64, asymmetric=True)
xa = reconstruct_embeddings(ra)
mse_a = float(np.mean((x - xa) ** 2))
ca = mean_cosine_similarity(x, xa)

xcenter = rng.standard_normal((256, 512)).astype(np.float32)
rx = quantize_embeddings(xcenter, group_size=64, soft_clip_sigma=3.0)
xxs = reconstruct_embeddings(rx)
cxs = mean_cosine_similarity(xcenter, xxs)

print(f"Symmetric:  cosine={cs:.6f}  MSE={mse_s:.6f}")
print(f"Asymmetric: cosine={ca:.6f}  MSE={mse_a:.6f}  (zp shape={ra.zero_points.shape})")
print(f"Soft-clip(3s): cosine={cxs:.6f}")

assert mse_a < mse_s, f"Asymmetric MSE {mse_a} should beat symmetric {mse_s}"
assert ra.zero_points is not None
assert rs.zero_points is None
assert cxs > 0.999
print("Quantizer assertions passed")

import os; os.environ["SQUISH_OFFLINE"] = "1"
from squish.catalog import search
hits = search("qwen")
print(f"catalog search('qwen'): {len(hits)} hits -> {[e.id for e in hits[:3]]}")
assert len(hits) > 0
print("All checks PASS")
