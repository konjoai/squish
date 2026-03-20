## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 10052 | 58 | 5.8 |
| What is the time complexity of quicksort?               | 26869 | 237 | 8.8 |
| Write a Python function that reverses a string.         | 30691 | 216 | 7.0 |
| What causes the Northern Lights?                        | 29191 | 250 | 8.6 |
| **Average** | **24201** | — | **7.5** |

_Reproduced with: `squish bench --markdown`_