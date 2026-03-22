## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 594 | 47 | 20.1 |
| What is the time complexity of quicksort?               | 173 | 70 | 22.9 |
| Write a Python function that reverses a string.         | 249 | 149 | 24.9 |
| What causes the Northern Lights?                        | 241 | 51 | 23.9 |
| **Average** | **314** | — | **22.9** |

_Reproduced with: `squish bench --markdown`_