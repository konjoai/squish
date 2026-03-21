## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 793 | 251 | 2.3 |
| What is the time complexity of quicksort?               | 669 | 256 | 10.9 |
| Write a Python function that reverses a string.         | 670 | 8 | 0.2 |
| What causes the Northern Lights?                        | 639 | 256 | 3.8 |
| **Average** | **693** | — | **4.3** |

_Reproduced with: `squish bench --markdown`_