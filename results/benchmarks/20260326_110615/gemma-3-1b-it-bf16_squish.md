## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 299 | 255 | 18.4 |
| What is the time complexity of quicksort?               | 509 | 256 | 18.4 |
| Write a Python function that reverses a string.         | 980 | 256 | 16.2 |
| What causes the Northern Lights?                        | 1971 | 256 | 20.1 |
| **Average** | **940** | — | **18.3** |

_Reproduced with: `squish bench --markdown`_