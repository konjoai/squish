## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 21054 | 254 | 12.1 |
| What is the time complexity of quicksort?               | 20396 | 251 | 12.3 |
| Write a Python function that reverses a string.         | 20461 | 253 | 12.4 |
| What causes the Northern Lights?                        | 19927 | 255 | 12.8 |
| **Average** | **20459** | — | **12.4** |

_Reproduced with: `squish bench --markdown`_