## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 31311 | 88 | 1.8 |
| What is the time complexity of quicksort?               | 919 | 219 | 5.5 |
| Write a Python function that reverses a string.         | 654 | 239 | 6.4 |
| What causes the Northern Lights?                        | 800 | 256 | 6.6 |
| **Average** | **8421** | — | **5.1** |

_Reproduced with: `squish bench --markdown`_