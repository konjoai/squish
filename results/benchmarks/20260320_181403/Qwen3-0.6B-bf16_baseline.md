## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 4267 | 254 | 59.5 |
| What is the time complexity of quicksort?               | 4066 | 253 | 62.2 |
| Write a Python function that reverses a string.         | 4111 | 254 | 61.8 |
| What causes the Northern Lights?                        | 4213 | 254 | 60.3 |
| **Average** | **4164** | — | **60.9** |

_Reproduced with: `squish bench --markdown`_