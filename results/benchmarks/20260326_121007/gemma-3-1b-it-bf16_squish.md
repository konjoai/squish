## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 343 | 255 | 23.0 |
| What is the time complexity of quicksort?               | 465 | 256 | 23.2 |
| Write a Python function that reverses a string.         | 508 | 256 | 22.1 |
| What causes the Northern Lights?                        | 905 | 255 | 23.1 |
| **Average** | **555** | — | **22.9** |

_Reproduced with: `squish bench --markdown`_