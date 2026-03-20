## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 4259 | 254 | 59.6 |
| What is the time complexity of quicksort?               | 4068 | 252 | 61.9 |
| Write a Python function that reverses a string.         | 4222 | 254 | 60.1 |
| What causes the Northern Lights?                        | 4011 | 255 | 63.6 |
| **Average** | **4140** | — | **61.3** |

_Reproduced with: `squish bench --markdown`_