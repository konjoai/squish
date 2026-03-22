## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 612 | 109 | 18.1 |
| What is the time complexity of quicksort?               | 1092 | 35 | 14.4 |
| Write a Python function that reverses a string.         | 223 | 203 | 18.8 |
| What causes the Northern Lights?                        | 452 | 106 | 19.5 |
| **Average** | **595** | — | **17.7** |

_Reproduced with: `squish bench --markdown`_