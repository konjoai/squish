## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 575 | 64 | 21.2 |
| What is the time complexity of quicksort?               | 167 | 122 | 26.4 |
| Write a Python function that reverses a string.         | 190 | 89 | 23.1 |
| What causes the Northern Lights?                        | 412 | 256 | 25.1 |
| **Average** | **336** | — | **23.9** |

_Reproduced with: `squish bench --markdown`_