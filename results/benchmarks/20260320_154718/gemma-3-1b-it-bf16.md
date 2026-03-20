## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 7336 | 252 | 34.3 |
| What is the time complexity of quicksort?               | 7803 | 245 | 31.4 |
| Write a Python function that reverses a string.         | 8658 | 208 | 24.0 |
| What causes the Northern Lights?                        | 9857 | 248 | 25.1 |
| **Average** | **8413** | — | **28.7** |

_Reproduced with: `squish bench --markdown`_