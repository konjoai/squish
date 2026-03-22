## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 598 | 55 | 16.9 |
| What is the time complexity of quicksort?               | 771 | 71 | 17.2 |
| Write a Python function that reverses a string.         | 364 | 126 | 18.7 |
| What causes the Northern Lights?                        | 452 | 159 | 17.9 |
| **Average** | **546** | — | **17.7** |

_Reproduced with: `squish bench --markdown`_