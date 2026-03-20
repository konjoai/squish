## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 3961 | 93 | 23.5 |
| What is the time complexity of quicksort?               | 4622 | 121 | 26.2 |
| Write a Python function that reverses a string.         | 4463 | 85 | 19.0 |
| What causes the Northern Lights?                        | 10638 | 232 | 21.8 |
| **Average** | **5921** | — | **22.6** |

_Reproduced with: `squish bench --markdown`_