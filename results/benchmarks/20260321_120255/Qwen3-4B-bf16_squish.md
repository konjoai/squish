## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 1363 | 255 | 9.7 |
| What is the time complexity of quicksort?               | 527 | 254 | 10.1 |
| Write a Python function that reverses a string.         | 519 | 254 | 10.0 |
| What causes the Northern Lights?                        | 503 | 255 | 9.8 |
| **Average** | **728** | — | **9.9** |

_Reproduced with: `squish bench --markdown`_