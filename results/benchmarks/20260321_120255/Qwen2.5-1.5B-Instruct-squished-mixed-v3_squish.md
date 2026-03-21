## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 560 | 40 | 19.9 |
| What is the time complexity of quicksort?               | 167 | 30 | 23.8 |
| Write a Python function that reverses a string.         | 171 | 184 | 26.5 |
| What causes the Northern Lights?                        | 180 | 100 | 25.6 |
| **Average** | **270** | — | **24.0** |

_Reproduced with: `squish bench --markdown`_