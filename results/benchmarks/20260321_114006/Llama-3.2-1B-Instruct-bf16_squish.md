## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 495 | 89 | 19.8 |
| What is the time complexity of quicksort?               | 264 | 256 | 22.1 |
| Write a Python function that reverses a string.         | 445 | 235 | 19.9 |
| What causes the Northern Lights?                        | 359 | 256 | 23.9 |
| **Average** | **391** | — | **21.4** |

_Reproduced with: `squish bench --markdown`_