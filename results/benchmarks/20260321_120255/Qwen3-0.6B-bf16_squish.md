## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 315 | 252 | 59.3 |
| What is the time complexity of quicksort?               | 132 | 254 | 59.3 |
| Write a Python function that reverses a string.         | 141 | 254 | 62.9 |
| What causes the Northern Lights?                        | 141 | 255 | 63.6 |
| **Average** | **182** | — | **61.3** |

_Reproduced with: `squish bench --markdown`_