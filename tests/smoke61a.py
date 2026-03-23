"""Quick smoke test for Wave 61a Rust functions."""
import squish_quant as sq
import numpy as np

rng = np.random.default_rng(0)
w = rng.standard_normal((4, 8)).astype(np.float32)
rms = np.ones(8, dtype=np.float32)

imp = sq.wanda_importance_f32(w, rms)
assert imp.shape == (4, 8), f"Expected (4,8), got {imp.shape}"
print(f"wanda_importance: {imp.shape}")

mask = sq.wanda_nm_mask_f32(imp, 2, 4)
assert mask.shape == (4, 8) and mask.dtype == np.uint8
print(f"wanda_nm_mask: {mask.shape} {mask.dtype}")

cb = rng.standard_normal((2, 4)).astype(np.float32)
codes = sq.flute_lut_encode_f32(w, cb, 4)
assert codes.shape == (4, 8)
print(f"flute_encode: {codes.shape}")

decoded = sq.flute_lut_decode_u8(codes, cb, 4)
assert decoded.shape == (4, 8)
print(f"flute_decode: {decoded.shape}")

q = rng.standard_normal((3, 2, 4)).astype(np.float32)
k = rng.standard_normal((3, 2, 4)).astype(np.float32)
v = rng.standard_normal((3, 2, 4)).astype(np.float32)
beta = np.full((3, 2), 0.1, dtype=np.float32)
out = sq.delta_net_scan_f32(q, k, v, beta)
assert out.shape == (3, 2, 4)
print(f"delta_net_scan: {out.shape}")

q_obs = rng.standard_normal((2, 3, 4)).astype(np.float32)
k2 = rng.standard_normal((2, 5, 4)).astype(np.float32)
scores = sq.green_kv_score_f32(q_obs, k2)
assert scores.shape == (2, 5)
print(f"green_kv_score: {scores.shape}")

logits = rng.standard_normal((4, 10)).astype(np.float32)
guesses = np.zeros(4, dtype=np.int32)
ng, nf = sq.jacobi_conv_check_f32(logits, guesses, 0.0, 42)
assert ng.shape == (4,) and isinstance(nf, int)
print(f"jacobi_conv: {ng.shape} n_fixed={nf}")

dtok = np.zeros((2, 3), dtype=np.int32)
dl = rng.standard_normal((2, 3, 10)).astype(np.float32)
tl = rng.standard_normal((2, 3, 10)).astype(np.float32)
tok, bl = sq.tree_verify_softmax_f32(dtok, dl, tl, 1.0, 42)
assert isinstance(bl, int)
print(f"tree_verify: {tok.shape} best_len={bl}")

print("ALL SMOKE TESTS PASSED")
