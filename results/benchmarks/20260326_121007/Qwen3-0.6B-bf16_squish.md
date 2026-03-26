## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 210 | 200 | 27.1 |
| What is the time complexity of quicksort?               | 202 | 255 | 29.6 |
| Write a Python function that reverses a string.         | 997 | 254 | 21.0 |
| What causes the Northern Lights?                        | 833 | 254 | 26.1 |
| **Average** | **561** | — | **26.0** |

_Reproduced with: `squish bench --markdown`_