## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 240 | 76 | 17.5 |
| What is the time complexity of quicksort?               | 347 | 190 | 17.9 |
| Write a Python function that reverses a string.         | 1597 | 234 | 15.1 |
| What causes the Northern Lights?                        | 833 | 256 | 17.8 |
| **Average** | **754** | — | **17.1** |

_Reproduced with: `squish bench --markdown`_