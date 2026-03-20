## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 3873 | 47 | 12.1 |
| What is the time complexity of quicksort?               | 16303 | 256 | 15.7 |
| Write a Python function that reverses a string.         | 11558 | 185 | 16.0 |
| What causes the Northern Lights?                        | 13275 | 239 | 18.0 |
| **Average** | **11252** | — | **15.4** |

_Reproduced with: `squish bench --markdown`_