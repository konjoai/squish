## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 7033 | 236 | 33.6 |
| What is the time complexity of quicksort?               | 8964 | 238 | 26.5 |
| Write a Python function that reverses a string.         | 9803 | 224 | 22.8 |
| What causes the Northern Lights?                        | 13186 | 245 | 18.6 |
| **Average** | **9747** | — | **25.4** |

_Reproduced with: `squish bench --markdown`_