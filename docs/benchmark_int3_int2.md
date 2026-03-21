# INT3 vs INT2 Quantization Benchmark

Synthetic Gaussian weights (σ=0.02) at transformer-realistic shapes.  
INT3: MiLo (group_size=64, max_rank=8)  
INT2: WeightOnlyInt2Quant (group_size=64, asymmetric)  

## Per-Layer Results

| Layer | Method | BPW | SNR (dB) | Compress ratio | Compress (ms) | Decomp (ms) | Size (MB) |
| ---------------------------- | ---------- | ------ | --------- | -------------- | ------------- | ----------- | ---------- |
| attn_proj  [576×576] | INT3-MiLo | 4.89 | 13.4 | 6.55× | 271.8 | 140.8 | 0.20 |
| attn_proj  [576×576] | INT2-WOQ | 3.00 | 7.4 | 10.67× | 9.1 | 0.5 | 0.12 |
| ffn_up     [576×1536] | INT3-MiLo | 4.61 | 13.3 | 6.94× | 725.3 | 381.0 | 0.51 |
| ffn_up     [576×1536] | INT2-WOQ | 3.00 | 7.4 | 10.67× | 15.3 | 1.3 | 0.33 |
| attn_proj  [1024×1024] | INT3-MiLo | 4.50 | 13.3 | 7.11× | 933.0 | 454.1 | 0.59 |
| attn_proj  [1024×1024] | INT2-WOQ | 3.00 | 7.4 | 10.67× | 18.0 | 1.6 | 0.39 |
| ffn_up     [1024×3072] | INT3-MiLo | 4.33 | 13.2 | 7.38× | 2849.5 | 1357.5 | 1.70 |
| ffn_up     [1024×3072] | INT2-WOQ | 3.00 | 7.4 | 10.67× | 54.7 | 5.3 | 1.18 |
| attn_proj  [2048×2048] | INT3-MiLo | 4.25 | 13.2 | 7.53× | 5004.6 | 1830.3 | 2.23 |
| attn_proj  [2048×2048] | INT2-WOQ | 3.00 | 7.4 | 10.67× | 72.2 | 7.7 | 1.57 |
| ffn_up     [2048×8192] | INT3-MiLo | 4.16 | 13.2 | 7.70× | 17875.4 | 7655.9 | 8.72 |
| ffn_up     [2048×8192] | INT2-WOQ | 3.00 | 7.4 | 10.67× | 293.4 | 30.7 | 6.29 |
| attn_proj  [4096×4096] | INT3-MiLo | 4.12 | 13.2 | 7.76× | 28514.9 | 7562.5 | 8.65 |
| attn_proj  [4096×4096] | INT2-WOQ | 3.00 | 7.4 | 10.67× | 290.9 | 30.6 | 6.29 |
| ffn_up     [4096×14336] | INT3-MiLo | 4.08 | 13.2 | 7.84× | 79157.8 | 27573.9 | 29.95 |
| ffn_up     [4096×14336] | INT2-WOQ | 3.00 | 7.4 | 10.67× | 1045.1 | 109.7 | 22.02 |

## Averages

| Method     | BPW   | SNR (dB) | Compress ratio | Compress (ms) | Decomp (ms) |
| ---------- | ----- | -------- | -------------- | ------------- | ----------- |
| INT3-MiLo  | 4.37 | 13.2 | 7.35× | 16917 | 5869.5 |
| INT2-WOQ   | 3.00 | 7.4 | 10.67× | 225 | 23.4 |

## Model Projection (28-layer Qwen3-1.7B scale, FP32 baseline)

| Method     | Compressed (GB) | Original FP32 (GB) | Memory Savings |
| ---------- | --------------- | ------------------ | -------------- |
| INT3-MiLo  | 1.44 | 11.14 | 87% |
| INT2-WOQ   | 1.04 | 11.14 | 91% |
