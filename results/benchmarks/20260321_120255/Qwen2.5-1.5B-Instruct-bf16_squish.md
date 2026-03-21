## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 548 | 32 | 18.5 |
| What is the time complexity of quicksort?               | 171 | 141 | 26.6 |
| Write a Python function that reverses a string.         | 180 | 117 | 24.7 |
| What causes the Northern Lights?                        | 172 | 196 | 26.9 |
| **Average** | **268** | — | **24.2** |

_Reproduced with: `squish bench --markdown`_