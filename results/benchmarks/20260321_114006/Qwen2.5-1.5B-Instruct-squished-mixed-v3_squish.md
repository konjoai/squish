## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 608 | 42 | 19.3 |
| What is the time complexity of quicksort?               | 937 | 28 | 10.1 |
| Write a Python function that reverses a string.         | 303 | 116 | 17.7 |
| What causes the Northern Lights?                        | 742 | 256 | 17.8 |
| **Average** | **648** | — | **16.2** |

_Reproduced with: `squish bench --markdown`_