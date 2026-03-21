## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 543 | 255 | 20.0 |
| What is the time complexity of quicksort?               | 282 | 255 | 21.1 |
| Write a Python function that reverses a string.         | 290 | 256 | 28.2 |
| What causes the Northern Lights?                        | 285 | 256 | 0.8 |
| **Average** | **350** | — | **17.5** |

_Reproduced with: `squish bench --markdown`_