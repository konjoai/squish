# Cross-Platform Support Plan

This document captures the full plan for expanding squish beyond Apple Silicon
to Linux (CUDA, ROCm, CPU), Windows, and Cloud accelerators (TPU, Gaudi/IPU).

---

## Current State

| Platform | Status | Notes |
|----------|--------|-------|
| Apple Silicon (M1–M4) | ✅ Production | MLX backend, full feature set |
| Linux CUDA | ✅ Wired (Phase 1+2) | `create_backend()`, `torch_ops.py`, `compressed_loader_torch.py`, `server.py` Linux path |
| Linux ROCm | ❌ Not started | Requires ROCm torch wheel + Triton-ROCm |
| Linux CPU | ✅ Wired (Phase 1+2) | PyTorch CPU path via `_TorchBackend`; BF16 fallback |
| Windows | ❌ Not started | CUDA path possible; WSL2 tested; native installer TBD |
| Docker | ✅ Complete (Phase 3) | `Dockerfile.cuda`, `Dockerfile.cpu`, `docker-compose.yml` with profiles |
| Kubernetes | ❌ Not started | No Helm chart, no k8s YAML |
| Cloud TPU | ❌ Not started | JAX backend required |
| Intel Gaudi / IPU | ❌ Not started | Habana libs required |

Backend code lives in `squish/backend.py`:
```python
class _AppleBackend:   # complete & production
class _TorchBackend:   # skeleton — NOT wired into server.py
class _StubBackend:    # test-only fallback
```

---

## Phase 1 — Wire the _TorchBackend (Linux CUDA)

**Goal:** Make `squish serve` work on a Linux machine with a CUDA GPU.

### 1.1 Backend auto-detection in `server.py`

```python
# squish/server.py  (current — Apple only)
from squish.backend import _AppleBackend
backend = _AppleBackend(model_path, config)

# squish/server.py  (new — platform-detected)
from squish.backend import create_backend
backend = create_backend(model_path, config)
```

Add `create_backend()` factory to `backend.py`:
```python
def create_backend(model_path, config):
    if sys.platform == "darwin":
        return _AppleBackend(model_path, config)
    if torch.cuda.is_available():
        return _TorchBackend(model_path, config, device="cuda")
    return _TorchBackend(model_path, config, device="cpu")
```

### 1.2 Complete `_TorchBackend` implementation

Fill in `_TorchBackend`:
- `load_model()` — load safetensors with `torch.load` or `transformers.AutoModelForCausalLM`
- `generate()` — call `model.generate()` with `transformers` tokenizer
- `stream_generate()` — wrap with a `TextIteratorStreamer`
- `unload()` / `hot_reload()` — release GPU memory, reload weights

### 1.3 Compressed-weight loading for Torch backend

Currently `squish`'s `.squish` format (packed INT4 nibbles + scales) is loaded
only via `squish/mlx_ops.py`. Add:
- `squish/torch_ops.py` — mirror of `mlx_ops.py` for PyTorch tensors
- `squish/compressed_loader_torch.py` — loads `.squish` dir into a HF model

### 1.4 Modules to update

| Module | Change needed |
|--------|--------------|
| `squish/server.py` | Use `create_backend()` factory |
| `squish/backend.py` | Implement `_TorchBackend` fully |
| `squish/torch_ops.py` | New: INT4 unpack + matmul for CUDA |
| `squish/compressed_loader_torch.py` | New: loads squish dir → HF model |
| `squish/convert.py` | Add `--device cuda/cpu` flag |
| `squish/cli.py` | Pass platform hints to server |

### 1.5 Acceptance criteria
- `squish serve --model ./models/Qwen2.5-7B-Instruct` works on an A100
- `/v1/chat/completions` endpoint returns correct responses
- BF16 and INT4-compressed models both load and generate correctly
- All existing Apple tests continue to pass

---

## Phase 2 — Linux Module Compatibility

Each squish module needs review and adaptation for non-Apple platforms.

### Module status matrix

| Module | Apple | Linux/CUDA | Notes |
|--------|-------|-----------|-------|
| `squish.compress` (INT4) | ✅ | 🔶 partial | Uses `mlx.core` ops; needs torch path |
| `squish.compress` (MiLo/INT3) | ✅ | 🔶 partial | Pure Python + numpy; mostly portable |
| `squish.compress` (AQLM/INT2) | ✅ | 🔶 partial | Has torch dependency already |
| `squish.serve` | ✅ | ❌ | Needs Phase 1 wiring |
| `squish.pull` | ✅ | ✅ | Pure HTTP; already portable |
| `squish.eval` | ✅ | ❌ | Calls `mlx_lm` directly |
| `squish.bench` | ✅ | ❌ | Calls `mlx_lm` directly |
| `squish.grad` (LoRA) | ✅ | ❌ | Uses MLX autograd |
| `squish.convert` CLI | ✅ | 🔶 partial | Uses mlx for quant; torch for loading |

### 2.1 `squish.eval` (perplexity / benchmark)
- Abstract `compute_log_probs()`: mlx path vs torch path
- Introduce platform guard in `eval.py`:
  ```python
  if sys.platform == "darwin":
      from squish._eval_mlx import compute_log_probs
  else:
      from squish._eval_torch import compute_log_probs
  ```

### 2.2 `squish.grad` (LoRA fine-tuning)
- LoRA on Linux requires `peft` + `transformers` with a CUDA-capable model
- Phase 2 scope: make LoRA work with the Torch backend (CUDA/ROCm)
- Longer-term: unify grad API between MLX and torch backends

### 2.3 `squish.compress` INT4 on CUDA
- The nibble-pack kernel (`mlx_ops.py`) needs a CUDA equivalent
- Options:
  - Write a CUDA extension with `torch.utils.cpp_extension`
  - Use `bitsandbytes` 4-bit pack/unpack as reference
  - Use `torchao` INT4 kernels (Facebook)
- Recommended: `torchao` — actively maintained, supports INT4/INT8/INT2

### 2.4 Triton kernel port (optional for Phase 2)
- Any custom Triton (MOJo/triton) kernels need CUDA-target variants
- Intel Triton-ROCm enables AMD GPUs with minimal changes

---

## Phase 3 — Windows Support

### 3.1 CUDA path (NVIDIA GPU)
- Python packaging: add `[windows]` extras for CUDA torch
- CUDA 12.x torch wheel:
  ```
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
- WSL2: already works via the Linux path; officially support as a config option
- Native Windows: additional path separators / file permission fixes

### 3.2 CPU fallback (any Windows machine)
- Use `torch` CPU with bfloat16 or float32 for inference
- Performance note: INT4 on CPU with `torchao` is ~2–4× slower than GPU
- Document recommended minimum specs: 32 GB RAM for 7B models in BF16

### 3.3 Windows installer
- Use PyInstaller or Nuitka to build a standalone `squish.exe`
- Bundle model download via `squish pull` into installer wizard (InnoSetup or NSIS)
- Goal: one-click install that works offline after initial model download
- Installer artifact built in CI via `windows-latest` GitHub Actions runner

### 3.4 Windows-specific changes
| Area | Change |
|------|--------|
| File paths | Replace all `os.path.join` bare strings with `Path` objects throughout |
| Shell scripts | Provide `.bat` or `.ps1` equivalents of all `.sh` scripts |
| Symlinks | Replace symlink-based model caching with copy or hardlink-based cache |
| Process signals | Replace `signal.SIGTERM` with `win32api.SetConsoleCtrlHandler` on Windows |
| MLX | Not available on Windows — must route all gen through Torch backend |

---

## Phase 4 — Docker Containerisation ✅ COMPLETE

**Delivered:** `Dockerfile.cuda`, `Dockerfile.cpu`, `docker-compose.yml` (dual-profile),
`.dockerignore`, `[linux]` extra in `pyproject.toml`, `docker-build` CI job,
`tests/test_docker_entrypoint_unit.py` (39 tests), `SQUISH_MODEL` / `SQUISH_HOST` /
`SQUISH_PORT` env-var defaults wired into `squish.cli serve` and `squish.cli run`.

### 4.1 Base images

```
squish:base      — python:3.11-slim, deps, no model
squish:cuda      — nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
squish:cpu       — python:3.11-slim, CPU-only torch
squish:apple     — (not containerised; macOS-only MLX)
```

### 4.2 `Dockerfile.cuda`

```dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir torch==2.3.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -e ".[linux]"

COPY squish/ ./squish/
COPY scripts/ ./scripts/

EXPOSE 8080
ENTRYPOINT ["python3", "-m", "squish.cli", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
```

### 4.3 `docker-compose.yml`

```yaml
version: "3.9"

services:
  squish-server:
    image: squish:cuda
    build:
      context: .
      dockerfile: Dockerfile.cuda
    ports:
      - "8080:8080"
    volumes:
      - ${MODELS_DIR:-./models}:/models:ro
    environment:
      - SQUISH_MODEL=/models/Qwen2.5-7B-Instruct
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### 4.4 Model volume strategy
- Models are NOT bundled in the image (too large)
- Mount a host `models/` directory at `/models` inside container
- `squish pull` supports writing to any target directory: `squish pull qwen2.5-7b --dir /models`

---

## Phase 5 — Kubernetes / Helm Chart

### 5.1 Single-pod deployment (inference only)

```
squish-serve/
  Chart.yaml
  values.yaml
  templates/
    deployment.yaml
    service.yaml
    configmap.yaml
    pvc.yaml          # persistent model cache
    hpa.yaml          # horizontal pod autoscaler
```

`values.yaml` skeleton:
```yaml
image:
  repository: ghcr.io/squish-ai/squish
  tag: "latest"
  pullPolicy: IfNotPresent

model:
  id: "qwen2.5-7b"
  storageClass: "standard"
  storageSize: "50Gi"

resources:
  limits:
    nvidia.com/gpu: 1
    memory: "32Gi"
  requests:
    cpu: "4"
    memory: "16Gi"

service:
  type: ClusterIP
  port: 8080

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 4
  targetCPUUtilizationPercentage: 70
```

### 5.2 Distributed inference (tensor parallelism)

For 70B+ models, sharded across multiple GPU nodes:
- Use `torch.distributed` + NCCL for tensor parallelism
- StatefulSet with headless service for pod-to-pod communication
- Each pod holds a shard; pod-0 is the "coordinator"
- `values.yaml` flag: `distributed.enabled: true`, `distributed.worldSize: 4`

### 5.3 Autoscaling with KEDA

For GPU-aware autoscaling based on inference queue depth:
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: squish-serve-scaledobject
spec:
  scaleTargetRef:
    name: squish-serve
  minReplicaCount: 1
  maxReplicaCount: 8
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus:9090
        metricName: squish_queue_depth
        threshold: "10"
```

### 5.4 CI/CD pipeline additions
- Add `docker-build` job to `.github/workflows/ci.yml`
- Push multi-arch image (linux/amd64, linux/arm64) to `ghcr.io/squish-ai/squish`
- Chart released via `helm/chart-releaser-action`

---

## Phase 6 — Cloud Accelerators (TPU / Gaudi)

### 6.1 Google Cloud TPU (JAX backend)

TPU requires JAX (not PyTorch) for full performance. Plan:

1. Add `_JaxBackend` to `squish/backend.py`
2. `create_backend()` detects TPU: `if jax.default_backend() == "tpu"`
3. Port INT4 kernel to JAX:
   - Use `jax.lax.dot_general` with 4-bit quantised values
   - `jax.numpy` for nibble unpack
4. Use `MaxText` or `t5x`-style sharding for multi-TPU pods
5. Provide Cloud Run / GKE TPU node pool Terraform template

### 6.2 Intel Gaudi / IPU (Habana backend)

Intel Gaudi 2/3 uses the `habana_frameworks` package:
1. Add `_HabanaBackend` to `backend.py`
2. Load model onto Gaudi device: `model.to("hpu")`
3. Inference via `optimum-habana` (HuggingFace Optimum extension)
4. INT4 quant: use `neural-compressor` from Intel for weight-only quantisation

Priority: lower than CUDA/ROCm — implement only if enterprise demand justifies.

---

## Implementation Order (recommended)

| Order | Phase | Est. Effort | Unblocks | Status |
|-------|-------|------------|---------|--------|
| 1 | Wire `_TorchBackend` | 1 week | All Linux work | ✅ Done (commit 62ff229) |
| 2 | Linux CUDA module compat | 2 weeks | Docker + K8s | ✅ Done (commit 62ff229) |
| 3 | Docker + compose | 3 days | K8s, CI | ✅ Done |
| 4 | K8s / Helm chart | 1 week | Cloud deploy | ❌ Not started |
| 5 | CI multi-platform build | 3 days | Distribution | 🔶 Partial (docker-build lint job added) |
| 6 | Windows CUDA path | 1 week | Windows users | ❌ Not started |
| 7 | Windows installer | 1 week | Non-dev users | ❌ Not started |
| 8 | TPU JAX backend | 3 weeks | Cloud TPU customers | ❌ Not started |
| 9 | Gaudi / IPU | 2 weeks | Enterprise HPC | ❌ Not started |

---

## Testing Strategy

For each new platform, all of the following must pass before marking complete:

1. **Unit tests** — `pytest tests/ -m platform` (new platform marker)
2. **Integration test** — `squish serve` + `/v1/chat/completions` round-trip
3. **Compression test** — compress a 1B model and verify PPL delta < 0.5
4. **CI matrix** — GitHub Actions job on target runner (ubuntu-latest / windows-latest / self-hosted CUDA)

---

*Last updated: 2025 — initial cross-platform planning session.*
