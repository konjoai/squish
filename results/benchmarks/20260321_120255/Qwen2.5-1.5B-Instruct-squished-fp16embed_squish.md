## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 592 | 44 | 19.9 |
| What is the time complexity of quicksort?               | 165 | 47 | 24.8 |
| Write a Python function that reverses a string.         | 196 | 152 | 25.8 |
| What causes the Northern Lights?                        | 181 | 126 | 26.5 |
| **Average** | **284** | — | **24.3** |

_Reproduced with: `squish bench --markdown`_