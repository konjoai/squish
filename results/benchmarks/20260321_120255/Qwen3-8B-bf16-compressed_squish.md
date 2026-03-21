## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 480 | 254 | 18.5 |
| What is the time complexity of quicksort?               | 398 | 255 | 18.6 |
| Write a Python function that reverses a string.         | 417 | 254 | 17.9 |
| What causes the Northern Lights?                        | 479 | 255 | 16.4 |
| **Average** | **443** | — | **17.9** |

_Reproduced with: `squish bench --markdown`_