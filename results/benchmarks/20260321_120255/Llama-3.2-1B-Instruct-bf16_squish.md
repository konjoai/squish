## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 499 | 85 | 28.2 |
| What is the time complexity of quicksort?               | 169 | 159 | 31.8 |
| Write a Python function that reverses a string.         | 154 | 194 | 30.8 |
| What causes the Northern Lights?                        | 160 | 256 | 33.8 |
| **Average** | **246** | — | **31.2** |

_Reproduced with: `squish bench --markdown`_