## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 10789 | 76 | 7.0 |
| What is the time complexity of quicksort?               | 28953 | 256 | 8.8 |
| Write a Python function that reverses a string.         | 26363 | 224 | 8.5 |
| What causes the Northern Lights?                        | 26830 | 256 | 9.5 |
| **Average** | **23234** | — | **8.5** |

_Reproduced with: `squish bench --markdown`_