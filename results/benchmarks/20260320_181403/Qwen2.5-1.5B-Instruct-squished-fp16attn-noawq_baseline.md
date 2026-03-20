## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 2392 | 50 | 20.9 |
| What is the time complexity of quicksort?               | 2476 | 30 | 12.1 |
| Write a Python function that reverses a string.         | 7846 | 156 | 19.9 |
| What causes the Northern Lights?                        | 6944 | 143 | 20.6 |
| **Average** | **4914** | — | **18.4** |

_Reproduced with: `squish bench --markdown`_