## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 328 | 81 | 18.2 |
| What is the time complexity of quicksort?               | 300 | 55 | 18.1 |
| Write a Python function that reverses a string.         | 399 | 233 | 16.2 |
| What causes the Northern Lights?                        | 932 | 255 | 17.3 |
| **Average** | **490** | — | **17.4** |

_Reproduced with: `squish bench --markdown`_