## Squish Benchmark — 2026-03-26

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 10101 | 254 | 10.7 |
| What is the time complexity of quicksort?               | 562 | 256 | 18.0 |
| Write a Python function that reverses a string.         | 310 | 256 | 19.5 |
| What causes the Northern Lights?                        | 570 | 256 | 20.4 |
| **Average** | **2886** | — | **17.2** |

_Reproduced with: `squish bench --markdown`_