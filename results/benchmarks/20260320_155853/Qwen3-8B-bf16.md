## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 29462 | 254 | 8.6 |
| What is the time complexity of quicksort?               | 33736 | 254 | 7.5 |
| Write a Python function that reverses a string.         | 26106 | 253 | 9.7 |
| What causes the Northern Lights?                        | 24777 | 253 | 10.2 |
| **Average** | **28520** | — | **9.0** |

_Reproduced with: `squish bench --markdown`_