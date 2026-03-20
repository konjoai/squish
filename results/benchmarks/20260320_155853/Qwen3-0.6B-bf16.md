## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 4219 | 253 | 59.9 |
| What is the time complexity of quicksort?               | 4027 | 243 | 60.3 |
| Write a Python function that reverses a string.         | 4055 | 254 | 62.6 |
| What causes the Northern Lights?                        | 4071 | 255 | 62.6 |
| **Average** | **4093** | — | **61.4** |

_Reproduced with: `squish bench --markdown`_