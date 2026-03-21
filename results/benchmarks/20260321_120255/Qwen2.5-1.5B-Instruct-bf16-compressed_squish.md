## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 686 | 73 | 20.5 |
| What is the time complexity of quicksort?               | 181 | 56 | 24.3 |
| Write a Python function that reverses a string.         | 262 | 95 | 22.2 |
| What causes the Northern Lights?                        | 200 | 115 | 20.9 |
| **Average** | **332** | — | **22.0** |

_Reproduced with: `squish bench --markdown`_