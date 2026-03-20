## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 2025 | 41 | 20.2 |
| What is the time complexity of quicksort?               | 1369 | 32 | 23.3 |
| Write a Python function that reverses a string.         | 5549 | 139 | 25.0 |
| What causes the Northern Lights?                        | 4896 | 127 | 25.9 |
| **Average** | **3460** | — | **23.6** |

_Reproduced with: `squish bench --markdown`_