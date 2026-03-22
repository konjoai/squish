## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 326 | 253 | 60.9 |
| What is the time complexity of quicksort?               | 126 | 252 | 48.7 |
| Write a Python function that reverses a string.         | 336 | 253 | 48.2 |
| What causes the Northern Lights?                        | 258 | 253 | 42.4 |
| **Average** | **261** | — | **50.1** |

_Reproduced with: `squish bench --markdown`_