## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 2877 | 70 | 24.3 |
| What is the time complexity of quicksort?               | 6316 | 210 | 33.2 |
| Write a Python function that reverses a string.         | 6613 | 205 | 31.0 |
| What causes the Northern Lights?                        | 7793 | 255 | 32.7 |
| **Average** | **5900** | — | **30.3** |

_Reproduced with: `squish bench --markdown`_