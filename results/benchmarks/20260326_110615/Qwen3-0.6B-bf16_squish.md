## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 760 | 186 | 27.1 |
| What is the time complexity of quicksort?               | 427 | 254 | 29.3 |
| Write a Python function that reverses a string.         | 454 | 252 | 29.5 |
| What causes the Northern Lights?                        | 443 | 254 | 31.2 |
| **Average** | **521** | — | **29.3** |

_Reproduced with: `squish bench --markdown`_