## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 2986 | 67 | 22.4 |
| What is the time complexity of quicksort?               | 8172 | 170 | 20.8 |
| Write a Python function that reverses a string.         | 6378 | 146 | 22.9 |
| What causes the Northern Lights?                        | 3804 | 87 | 22.9 |
| **Average** | **5335** | — | **22.2** |

_Reproduced with: `squish bench --markdown`_