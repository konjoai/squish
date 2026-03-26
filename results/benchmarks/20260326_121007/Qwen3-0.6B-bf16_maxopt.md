## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 4588 | 243 | 20.3 |
| What is the time complexity of quicksort?               | 692 | 254 | 26.4 |
| Write a Python function that reverses a string.         | 255 | 253 | 29.9 |
| What causes the Northern Lights?                        | 458 | 253 | 29.7 |
| **Average** | **1498** | — | **26.6** |

_Reproduced with: `squish bench --markdown`_