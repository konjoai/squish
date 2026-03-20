## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 15249 | 253 | 16.6 |
| What is the time complexity of quicksort?               | 16119 | 254 | 15.8 |
| Write a Python function that reverses a string.         | 18291 | 254 | 13.9 |
| What causes the Northern Lights?                        | 19905 | 254 | 12.8 |
| **Average** | **17391** | — | **14.7** |

_Reproduced with: `squish bench --markdown`_