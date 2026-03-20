## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 7131 | 256 | 35.9 |
| What is the time complexity of quicksort?               | 6925 | 239 | 34.5 |
| Write a Python function that reverses a string.         | 7049 | 221 | 31.3 |
| What causes the Northern Lights?                        | 6873 | 247 | 35.9 |
| **Average** | **6995** | — | **34.4** |

_Reproduced with: `squish bench --markdown`_