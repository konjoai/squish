# Optional real-corpus drop-in

Place real `*.txt` documents here (code reviews, technical docs, Q&A) to have the
benchmark build prompts from genuine content instead of the synthetic generator.
When this directory contains `.txt` files, `corpus.Corpus` uses them verbatim
(sliced to exact token lengths); otherwise it falls back to the deterministic
varied-prose generator. Either way every generated prompt is saved with its seed
and SHA-256 under `results/benchmark_matrix/.../prompts/` for auditing.
