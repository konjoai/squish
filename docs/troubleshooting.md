# Troubleshooting / FAQ

This page covers the most common issues encountered when running Squish on Apple Silicon Macs.
If your issue is not listed here, open a [GitHub Discussion](https://github.com/wesleyscholl/squish/discussions)
or ask in [Discord](https://discord.gg/squish).

---

## Quick diagnostics

Run these two commands first. The answers are required context for any bug report or Discord question.

```bash
squish --version
python -c "import mlx; print(mlx.__version__)"
```

Expected output (versions will differ):

```
squish 0.9.2
0.22.0
```

Additional environment snapshot:

```bash
python --version                       # should be 3.10+
pip show squish mlx mlx-lm            # installed versions and locations
sw_vers                                # macOS version
sysctl -n hw.memsize                   # total unified memory in bytes
squish models                          # locally cached models
```

Paste the output of all of the above when filing a bug.

---

## 8 GB Mac — out of memory (OOM)

### Symptoms

- `squish run` or `squish serve` exits immediately with no error message
- Kernel logs show `jetsam` events (`log show --predicate 'eventMessage contains "jetsam"' --last 5m`)
- The macOS Activity Monitor "Memory Pressure" gauge is solidly red before inference starts
- Python raises `MemoryError` or the process is killed mid-generation

### Why it happens

Apple Silicon's unified memory is shared between the CPU, the GPU (Metal), and the OS.
On an 8 GB device, macOS reserves approximately 1.5–2 GB for the kernel and active apps,
leaving roughly 6–6.5 GB for model weights and the inference runtime.
A stock INT8 7B/8B model requires ~8 GB of weight data, which does not fit.

### Fixes

**1. Use INT4 quantisation (recommended for 8 GB Macs)**

INT4 halves the weight footprint compared to INT8.

```bash
squish pull llama3.1:8b --quant int4
squish run  llama3.1:8b --quant int4
```

!!! warning "INT4 accuracy"
    INT4 produces measurably lower quality output on models below 3B parameters.
    For 7B+ models the degradation is minor in practice.

**2. Use `--fault-tolerance` to survive partial allocation failures**

```bash
squish run llama3.1:8b --quant int4 --fault-tolerance
```

`--fault-tolerance` catches Metal allocation errors and retries with a smaller KV-cache
instead of crashing.

**3. Use `--adaptive-quant` to let Squish pick the quantisation level automatically**

```bash
squish run llama3.1:8b --adaptive-quant
```

Squish reads available unified memory at startup and selects INT4 or INT8 accordingly.

**4. Reduce the context window**

Every additional token in the context window consumes KV-cache memory.
Drop the window to 2 048 or 4 096 if your task allows it:

```bash
squish run llama3.1:8b --quant int4 --ctx 2048
```

**5. Free memory before running Squish**

Quit Safari, Xcode, Simulator, and any Electron apps. Each frees hundreds of megabytes.
You can also temporarily reduce the number of active menu-bar apps.

**6. Choose a smaller model**

3B models fit comfortably in INT8 on 8 GB devices:

```bash
squish pull llama3.2:3b
squish run  llama3.2:3b
```

### Reference: model memory requirements

| Model | INT8 (approx.) | INT4 (approx.) | Fits 8 GB? |
|---|---|---|---|
| llama3.2:1b | ~1.2 GB | ~0.7 GB | Yes (either quant) |
| llama3.2:3b | ~3.5 GB | ~2.0 GB | Yes (either quant) |
| llama3.1:8b | ~8.5 GB | ~4.5 GB | INT4 only |
| qwen3:8b | ~8.3 GB | ~4.4 GB | INT4 only |
| llama3.1:70b | ~72 GB | ~38 GB | No |

---

## Tokenizer errors

### Symptoms

- `RuntimeError: Tokenizer 'tiktoken' not found`
- `OSError: Can't load tokenizer for '<model>'`
- `ValueError: Unrecognized model in AutoTokenizer`
- `KeyError: 'tokenizer_config.json'`

### Root causes and fixes

**Missing tokenizer files in the model cache**

The model download may have been interrupted, leaving an incomplete cache entry.
Delete and re-pull:

```bash
squish rm llama3.1:8b
squish pull llama3.1:8b
```

To verify cache integrity manually:

```bash
ls ~/.squish/models/llama3.1-8b/
# Should contain: config.json  tokenizer.json  tokenizer_config.json  *.safetensors (or *.npy)
```

If any of those files are missing, the pull was incomplete. Re-pull.

**Incompatible `transformers` version**

Some tokenizers require a minimum version of `transformers`:

```bash
pip install --upgrade transformers
```

If you need to pin `transformers` for another reason, check the model card on HuggingFace
for the minimum supported version.

**`tiktoken` not installed**

GPT-style models (e.g. Qwen, some Mistral variants) use `tiktoken`:

```bash
pip install tiktoken
```

**`sentencepiece` not installed**

LLaMA-family and Gemma models use SentencePiece:

```bash
pip install sentencepiece
```

**`tokenizers` Rust extension missing**

If `pip install squish` ran in an environment without a Rust compiler and a pre-built wheel was
unavailable for your Python version, the `tokenizers` package may have fallen back to a pure-Python
stub that is missing methods:

```bash
pip install --upgrade --force-reinstall tokenizers
```

If the error persists, install Rust and rebuild:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
pip install --upgrade --force-reinstall tokenizers
```

### Checking which tokenizer a model needs

```bash
python -c "
import json, pathlib
cfg = pathlib.Path('~/.squish/models/').expanduser()
for p in cfg.rglob('tokenizer_config.json'):
    data = json.loads(p.read_text())
    print(p.parent.name, '->', data.get('tokenizer_class', 'unknown'))
"
```

---

## MLX version mismatches

### Symptoms

- `ImportError: cannot import name 'quantize' from 'mlx.core'`
- `AttributeError: module 'mlx.nn' has no attribute 'QuantizedLinear'`
- `TypeError: forward() got an unexpected keyword argument 'cache'`
- Inference completes but produces garbled / nonsensical output

### Why it happens

Squish pins a minimum MLX version because the MLX API changes frequently between minor releases.
If the installed MLX is older or newer than what Squish expects, the above errors appear.

### Check current versions

```bash
pip show squish mlx mlx-lm
```

The `Requires:` field in the `squish` output lists the accepted MLX range.

### Upgrade to the latest compatible MLX

```bash
pip install --upgrade mlx mlx-lm
```

### Pin to a specific version

If the latest MLX introduces a regression, pin to the last known-good version:

```bash
pip install "mlx==0.22.0" "mlx-lm==0.22.0"
```

Replace `0.22.0` with whichever version the Squish release notes specify.

!!! tip "Use a virtual environment"
    Running inside a `venv` or `conda` environment makes version pinning safe and reversible
    without touching your system Python:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install "squish" "mlx==0.22.0" "mlx-lm==0.22.0"
    ```

### Force-reinstall if the environment is in a broken state

```bash
pip install --upgrade --force-reinstall mlx mlx-lm squish
```

### Verify MLX works after reinstall

```bash
python -c "
import mlx.core as mx
print('MLX version:', mx.__version__)
arr = mx.array([1.0, 2.0, 3.0])
print('Basic computation:', arr * 2)
print('Metal device available:', mx.default_device())
"
```

Expected output includes `Device(gpu, 0)` confirming Metal is active.

### Known version compatibility table

| Squish version | Minimum MLX | Maximum tested MLX |
|---|---|---|
| 0.9.x | 0.18.0 | 0.22.x |
| 0.8.x | 0.16.0 | 0.19.x |
| 0.7.x | 0.14.0 | 0.17.x |

---

## Ollama port conflicts

### Symptoms

- `squish serve` exits immediately with `OSError: [Errno 48] Address already in use`
- `curl http://localhost:11435/v1/models` returns connection refused even after `squish serve` appeared to start
- Requests to Squish unexpectedly return Ollama-formatted responses

### Background

Squish listens on **port 11435** by default. Ollama listens on **port 11434** by default.
These are different ports, but conflicts can still occur if:

- You started Squish on a custom port that clashes with another process
- Ollama was reconfigured (via `OLLAMA_HOST`) to use port 11435
- Another Squish instance is already running

### Find what is using a port

```bash
# Replace 11435 with the port in question
lsof -nP -iTCP:11435 -sTCP:LISTEN
```

The output shows the PID and process name occupying the port.

### Stop the conflicting process

=== "Ollama"
    ```bash
    # Stop Ollama gracefully
    brew services stop ollama
    # or
    pkill -f ollama
    ```

=== "Another Squish instance"
    ```bash
    pkill -f "squish serve"
    # or kill by PID from lsof output
    kill <PID>
    ```

=== "Other process"
    ```bash
    # Kill by PID obtained from lsof
    kill <PID>
    ```

### Run Squish on a different port

If you need both Squish and Ollama running at the same time, give each a distinct port:

```bash
# Squish on 11435 (default)
squish serve --port 11435

# Ollama stays on 11434 (its default) — no change needed
```

If port 11435 is already taken by something you cannot stop:

```bash
squish serve --port 11436
```

Then point your OpenAI-compatible client at `http://localhost:11436/v1`.

### Prevent Ollama from auto-starting at login

```bash
brew services stop ollama
brew services disable ollama   # prevents restart on next login
```

To re-enable later:

```bash
brew services enable ollama
brew services start ollama
```

### Verify Squish is listening after startup

```bash
# Check the port is bound
lsof -nP -iTCP:11435 -sTCP:LISTEN

# Hit the health endpoint
curl -s http://localhost:11435/v1/models | python -m json.tool
```

A valid response lists the locally cached models.

---

## Common errors and solutions

| Error message | Likely cause | Fix |
|---|---|---|
| `Address already in use` on port 11435 | Another process (often a prior Squish instance) is bound to the port | `lsof -nP -iTCP:11435` → `kill <PID>` |
| `jetsam` killed process | 8 GB OOM — model weights exceeded available unified memory | Switch to `--quant int4` or use a smaller model |
| `MemoryError` during model load | Insufficient free unified memory | Close other apps; use `--quant int4 --ctx 2048` |
| `RuntimeError: Tokenizer 'tiktoken' not found` | `tiktoken` package not installed | `pip install tiktoken` |
| `OSError: Can't load tokenizer for '<model>'` | Incomplete model download | `squish rm <model>` → `squish pull <model>` |
| `KeyError: 'tokenizer_config.json'` | Missing file in model cache | Re-pull the model |
| `ImportError: cannot import name 'quantize' from 'mlx.core'` | MLX version too old | `pip install --upgrade mlx mlx-lm` |
| `AttributeError: module 'mlx.nn' has no attribute 'QuantizedLinear'` | MLX version too old | `pip install --upgrade mlx mlx-lm` |
| `TypeError: forward() got an unexpected keyword argument 'cache'` | MLX / mlx-lm API mismatch | Pin matching versions: see [MLX version mismatches](#mlx-version-mismatches) |
| Garbled or nonsensical output | MLX version mismatch producing silent wrong results | `pip install --upgrade --force-reinstall mlx mlx-lm squish` |
| `pip install squish` fails with Rust/maturin error | Rust compiler not present and no pre-built wheel available | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` then retry |
| `squish: command not found` after `pip install squish` | `~/.local/bin` not in `PATH` | Add `export PATH="$HOME/.local/bin:$PATH"` to `~/.zshrc` |
| Requests to Squish return Ollama JSON format | Client is hitting port 11434 (Ollama) instead of 11435 (Squish) | Update `base_url` to `http://localhost:11435/v1` |

---

## Still stuck?

1. Re-run the [Quick diagnostics](#quick-diagnostics) commands and copy the full output.
2. Check [open GitHub issues](https://github.com/wesleyscholl/squish/issues) — your error may already have a fix.
3. Open a [GitHub Discussion](https://github.com/wesleyscholl/squish/discussions) with the diagnostic output.
4. Join [Discord](https://discord.gg/squish) for real-time help.
