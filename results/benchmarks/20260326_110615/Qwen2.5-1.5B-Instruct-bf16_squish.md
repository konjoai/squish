## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 307 | 83 | 10.6 |
| What is the time complexity of quicksort?               | 462 | 135 | 11.1 |
| Write a Python function that reverses a string.         | 881 | 139 | 9.9 |
| What causes the Northern Lights?                        | 684 | 252 | 14.0 |
| **Average** | **584** | — | **11.4** |

_Reproduced with: `squish bench --markdown`_