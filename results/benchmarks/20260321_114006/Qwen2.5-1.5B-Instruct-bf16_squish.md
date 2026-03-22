## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 643 | 64 | 21.5 |
| What is the time complexity of quicksort?               | 176 | 107 | 17.4 |
| Write a Python function that reverses a string.         | 295 | 166 | 19.8 |
| What causes the Northern Lights?                        | 303 | 124 | 20.4 |
| **Average** | **354** | — | **19.8** |

_Reproduced with: `squish bench --markdown`_