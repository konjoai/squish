## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 3021 | 73 | 24.1 |
| What is the time complexity of quicksort?               | 8071 | 212 | 26.3 |
| Write a Python function that reverses a string.         | 3827 | 91 | 23.8 |
| What causes the Northern Lights?                        | 9670 | 256 | 26.5 |
| **Average** | **6147** | — | **25.2** |

_Reproduced with: `squish bench --markdown`_