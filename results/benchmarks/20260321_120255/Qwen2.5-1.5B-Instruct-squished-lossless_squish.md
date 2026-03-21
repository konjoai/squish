## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 572 | 50 | 20.9 |
| What is the time complexity of quicksort?               | 166 | 211 | 26.4 |
| Write a Python function that reverses a string.         | 203 | 161 | 25.3 |
| What causes the Northern Lights?                        | 187 | 89 | 25.9 |
| **Average** | **282** | — | **24.6** |

_Reproduced with: `squish bench --markdown`_