## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 2296 | 60 | 7.4 |
| What is the time complexity of quicksort?               | 779 | 241 | 9.8 |
| Write a Python function that reverses a string.         | 742 | 221 | 8.9 |
| What causes the Northern Lights?                        | 647 | 248 | 10.2 |
| **Average** | **1116** | — | **9.1** |

_Reproduced with: `squish bench --markdown`_