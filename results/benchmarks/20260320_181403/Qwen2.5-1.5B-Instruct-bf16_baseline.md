## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 3897 | 89 | 22.8 |
| What is the time complexity of quicksort?               | 3990 | 104 | 26.1 |
| Write a Python function that reverses a string.         | 9592 | 238 | 24.8 |
| What causes the Northern Lights?                        | 9703 | 256 | 26.4 |
| **Average** | **6796** | — | **25.0** |

_Reproduced with: `squish bench --markdown`_