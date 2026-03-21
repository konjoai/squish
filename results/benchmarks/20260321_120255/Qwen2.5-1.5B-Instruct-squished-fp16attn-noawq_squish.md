## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 596 | 50 | 20.6 |
| What is the time complexity of quicksort?               | 165 | 30 | 23.5 |
| Write a Python function that reverses a string.         | 182 | 249 | 26.1 |
| What causes the Northern Lights?                        | 185 | 187 | 26.6 |
| **Average** | **282** | — | **24.2** |

_Reproduced with: `squish bench --markdown`_