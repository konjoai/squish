## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 3150 | 85 | 27.0 |
| What is the time complexity of quicksort?               | 7705 | 256 | 33.2 |
| Write a Python function that reverses a string.         | 7807 | 244 | 31.2 |
| What causes the Northern Lights?                        | 7660 | 256 | 33.4 |
| **Average** | **6581** | — | **31.2** |

_Reproduced with: `squish bench --markdown`_