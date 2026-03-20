## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 3159 | 88 | 27.8 |
| What is the time complexity of quicksort?               | 7897 | 253 | 32.0 |
| Write a Python function that reverses a string.         | 6211 | 191 | 30.7 |
| What causes the Northern Lights?                        | 7761 | 256 | 33.0 |
| **Average** | **6257** | — | **30.9** |

_Reproduced with: `squish bench --markdown`_