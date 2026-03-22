## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 605 | 70 | 15.6 |
| What is the time complexity of quicksort?               | 297 | 37 | 21.8 |
| Write a Python function that reverses a string.         | 315 | 116 | 18.3 |
| What causes the Northern Lights?                        | 375 | 152 | 19.4 |
| **Average** | **398** | — | **18.8** |

_Reproduced with: `squish bench --markdown`_