## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 26836 | 239 | 8.9 |
| What is the time complexity of quicksort?               | 29179 | 253 | 8.7 |
| Write a Python function that reverses a string.         | 27631 | 254 | 9.2 |
| What causes the Northern Lights?                        | 26878 | 255 | 9.5 |
| **Average** | **27631** | — | **9.1** |

_Reproduced with: `squish bench --markdown`_