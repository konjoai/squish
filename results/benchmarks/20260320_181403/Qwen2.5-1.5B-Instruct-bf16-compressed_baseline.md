## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 2723 | 59 | 21.7 |
| What is the time complexity of quicksort?               | 1703 | 42 | 24.7 |
| Write a Python function that reverses a string.         | 6432 | 114 | 17.7 |
| What causes the Northern Lights?                        | 6181 | 142 | 23.0 |
| **Average** | **4260** | — | **21.7** |

_Reproduced with: `squish bench --markdown`_