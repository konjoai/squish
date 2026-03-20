## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 8584 | 82 | 9.6 |
| What is the time complexity of quicksort?               | 23590 | 256 | 10.9 |
| Write a Python function that reverses a string.         | 22223 | 227 | 10.2 |
| What causes the Northern Lights?                        | 21568 | 256 | 11.9 |
| **Average** | **18991** | — | **10.6** |

_Reproduced with: `squish bench --markdown`_