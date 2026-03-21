## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 581 | 40 | 19.7 |
| What is the time complexity of quicksort?               | 171 | 48 | 25.1 |
| Write a Python function that reverses a string.         | 162 | 80 | 24.4 |
| What causes the Northern Lights?                        | 174 | 128 | 26.7 |
| **Average** | **272** | — | **24.0** |

_Reproduced with: `squish bench --markdown`_