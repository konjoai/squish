## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 538 | 58 | 21.1 |
| What is the time complexity of quicksort?               | 175 | 42 | 24.3 |
| Write a Python function that reverses a string.         | 186 | 212 | 25.7 |
| What causes the Northern Lights?                        | 175 | 235 | 27.0 |
| **Average** | **269** | — | **24.5** |

_Reproduced with: `squish bench --markdown`_