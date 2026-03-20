## Squish Benchmark — 2026-03-20

> Hardware: Apple M3 · 17 GB unified memory
> Server: http://127.0.0.1:11435/v1 · 256 max tokens

| Prompt | TTFT (ms) | Tokens | Tok/s |
|--------|----------:|-------:|------:|
| Explain quantum entanglement in two sentences.          | 1930 | 37 | 19.2 |
| What is the time complexity of quicksort?               | 1175 | 27 | 23.0 |
| Write a Python function that reverses a string.         | 4982 | 126 | 25.3 |
| What causes the Northern Lights?                        | 6271 | 157 | 25.0 |
| **Average** | **3590** | — | **23.1** |

_Reproduced with: `squish bench --markdown`_