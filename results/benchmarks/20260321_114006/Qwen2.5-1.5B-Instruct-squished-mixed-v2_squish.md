## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 548 | 66 | 22.4 |
| What is the time complexity of quicksort?               | 163 | 95 | 19.3 |
| Write a Python function that reverses a string.         | 256 | 116 | 19.9 |
| What causes the Northern Lights?                        | 503 | 160 | 19.2 |
| **Average** | **367** | — | **20.2** |

_Reproduced with: `squish bench --markdown`_