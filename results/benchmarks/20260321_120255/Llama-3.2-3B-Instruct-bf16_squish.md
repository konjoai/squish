## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 1135 | 84 | 10.9 |
| What is the time complexity of quicksort?               | 354 | 210 | 12.4 |
| Write a Python function that reverses a string.         | 472 | 238 | 11.3 |
| What causes the Northern Lights?                        | 555 | 256 | 12.0 |
| **Average** | **629** | — | **11.6** |

_Reproduced with: `squish bench --markdown`_