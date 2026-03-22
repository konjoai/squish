## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 656 | 80 | 16.4 |
| What is the time complexity of quicksort?               | 333 | 70 | 13.9 |
| Write a Python function that reverses a string.         | 471 | 109 | 19.2 |
| What causes the Northern Lights?                        | 641 | 143 | 19.4 |
| **Average** | **525** | — | **17.2** |

_Reproduced with: `squish bench --markdown`_