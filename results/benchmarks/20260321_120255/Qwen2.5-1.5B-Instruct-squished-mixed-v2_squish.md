## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 569 | 50 | 21.0 |
| What is the time complexity of quicksort?               | 170 | 103 | 26.3 |
| Write a Python function that reverses a string.         | 192 | 228 | 23.2 |
| What causes the Northern Lights?                        | 179 | 231 | 26.5 |
| **Average** | **278** | — | **24.3** |

_Reproduced with: `squish bench --markdown`_