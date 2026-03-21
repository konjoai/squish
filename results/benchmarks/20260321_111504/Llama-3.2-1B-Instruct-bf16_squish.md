## Squish Benchmark — 2026-03-21

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 912 | 256 | 4.4 |
| What is the time complexity of quicksort?               | 928 | 11 | 0.8 |
| Write a Python function that reverses a string.         | 1099 | 254 | 8.1 |
| What causes the Northern Lights?                        | 936 | 256 | 4.6 |
| **Average** | **969** | — | **4.5** |

_Reproduced with: `squish bench --markdown`_