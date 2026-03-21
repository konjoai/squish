## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 586 | 254 | 36.6 |
| What is the time complexity of quicksort?               | 254 | 243 | 37.0 |
| Write a Python function that reverses a string.         | 268 | 221 | 32.0 |
| What causes the Northern Lights?                        | 635 | 248 | 34.3 |
| **Average** | **436** | — | **35.0** |

_Reproduced with: `squish bench --markdown`_