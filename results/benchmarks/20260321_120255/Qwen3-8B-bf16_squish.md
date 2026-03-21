## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 483 | 254 | 15.2 |
| What is the time complexity of quicksort?               | 498 | 251 | 14.3 |
| Write a Python function that reverses a string.         | 582 | 253 | 13.4 |
| What causes the Northern Lights?                        | 574 | 254 | 12.5 |
| **Average** | **535** | — | **13.8** |

_Reproduced with: `squish bench --markdown`_