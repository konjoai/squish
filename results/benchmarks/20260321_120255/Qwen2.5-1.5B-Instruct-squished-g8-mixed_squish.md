## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 593 | 72 | 22.3 |
| What is the time complexity of quicksort?               | 186 | 31 | 22.8 |
| Write a Python function that reverses a string.         | 227 | 180 | 25.7 |
| What causes the Northern Lights?                        | 682 | 113 | 22.4 |
| **Average** | **422** | — | **23.3** |

_Reproduced with: `squish bench --markdown`_