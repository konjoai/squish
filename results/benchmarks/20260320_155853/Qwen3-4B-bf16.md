## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 36578 | 255 | 7.0 |
| What is the time complexity of quicksort?               | 35495 | 255 | 7.2 |
| Write a Python function that reverses a string.         | 33420 | 254 | 7.6 |
| What causes the Northern Lights?                        | 32489 | 252 | 7.8 |
| **Average** | **34495** | — | **7.4** |

_Reproduced with: `squish bench --markdown`_