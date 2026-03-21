## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 557 | 54 | 20.2 |
| What is the time complexity of quicksort?               | 182 | 30 | 23.3 |
| Write a Python function that reverses a string.         | 180 | 245 | 25.7 |
| What causes the Northern Lights?                        | 203 | 114 | 25.9 |
| **Average** | **280** | — | **23.8** |

_Reproduced with: `squish bench --markdown`_