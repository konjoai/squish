## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 5275 | 79 | 15.0 |
| What is the time complexity of quicksort?               | 8532 | 127 | 14.9 |
| Write a Python function that reverses a string.         | 7180 | 104 | 14.5 |
| What causes the Northern Lights?                        | 10587 | 148 | 14.0 |
| **Average** | **7894** | — | **14.6** |

_Reproduced with: `squish bench --markdown`_