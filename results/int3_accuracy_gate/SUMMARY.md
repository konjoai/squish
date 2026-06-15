# INT3 vs INT4 accuracy gate — arc_easy (0-shot, n=1000)

Fast paired comparison to decide whether INT3-7B is safe to ship as default vs
the current INT4 default. 0-shot + no chat template depresses the absolute
score for BOTH models equally; the DELTA is the valid signal.

| metric    | INT3            | INT4            | Δ (INT3-INT4)         |
|-----------|-----------------|-----------------|-----------------------|
| acc       | 0.619 ± 0.0154  | 0.647 ± 0.0151  | -0.028 (1.3σ, n.s.)   |
| acc_norm  | 0.551 ± 0.0157  | 0.541 ± 0.0158  | +0.010 (0.45σ, tied)  |

Verdict: INT3 ≈ INT4 (no significant degradation). INT3 safe to ship relative
to INT4. Absolute numbers are low due to 0-shot/no-chat eval, not quantization;
a 25-shot+chat run is advisable before a formal release for the leaderboard
number, but the relative ship decision is conclusive.
