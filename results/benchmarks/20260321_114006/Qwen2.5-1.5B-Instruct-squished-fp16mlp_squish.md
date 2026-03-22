## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 544 | 56 | 15.7 |
| What is the time complexity of quicksort?               | 417 | 40 | 20.9 |
| Write a Python function that reverses a string.         | 225 | 130 | 19.8 |
| What causes the Northern Lights?                        | 347 | 113 | 20.8 |
| **Average** | **383** | — | **19.3** |

_Reproduced with: `squish bench --markdown`_