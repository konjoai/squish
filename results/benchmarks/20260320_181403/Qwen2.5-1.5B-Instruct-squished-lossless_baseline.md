## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 2368 | 50 | 21.1 |
| What is the time complexity of quicksort?               | 2338 | 59 | 25.2 |
| Write a Python function that reverses a string.         | 5563 | 114 | 20.5 |
| What causes the Northern Lights?                        | 8808 | 198 | 22.5 |
| **Average** | **4769** | — | **22.3** |

_Reproduced with: `squish bench --markdown`_