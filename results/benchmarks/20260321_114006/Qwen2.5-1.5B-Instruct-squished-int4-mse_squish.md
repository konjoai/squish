## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 643 | 83 | 22.5 |
| What is the time complexity of quicksort?               | 172 | 64 | 24.8 |
| Write a Python function that reverses a string.         | 166 | 117 | 23.7 |
| What causes the Northern Lights?                        | 173 | 209 | 27.1 |
| **Average** | **289** | — | **24.5** |

_Reproduced with: `squish bench --markdown`_