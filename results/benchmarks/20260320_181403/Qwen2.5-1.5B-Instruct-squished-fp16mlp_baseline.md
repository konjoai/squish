## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 3003 | 67 | 22.3 |
| What is the time complexity of quicksort?               | 1307 | 31 | 23.7 |
| Write a Python function that reverses a string.         | 4155 | 104 | 25.0 |
| What causes the Northern Lights?                        | 3953 | 76 | 19.2 |
| **Average** | **3105** | — | **22.6** |

_Reproduced with: `squish bench --markdown`_