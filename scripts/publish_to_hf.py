#!/usr/bin/env python3
"""publish_to_hf.py — Publish a squished (already-quantized) model to the
konjoai Hugging Face organization.

This script does NOT quantize. Squishing and uploading are separate steps:
compress a model first (see ``squish compress`` / ``scripts/compress_and_upload``),
then point this script at the resulting local directory.

Workflow
--------
  1. Validate the local model with a one-token inference (``squish run``).
  2. Generate an mlx_lm-compatible model card with YAML frontmatter.
  3. Dry-run (default): print the card, list files, show the target URL, exit.
     Live (--no-dry-run): create ``konjoai/<name>`` and upload the directory.

Usage
-----
  python scripts/publish_to_hf.py \\
      --local-path ~/models/Qwen2.5-7B-Instruct-int4 \\
      --hf-name Qwen2.5-7B-Instruct-squished \\
      --source-id Qwen/Qwen2.5-7B-Instruct \\
      --quant INT4 \\
      --context 128000 \\
      --base-license apache-2.0

Defaults to dry-run; add --no-dry-run to upload for real.

Exit codes: 0 success, 1 usage/input error, 2 validation/runtime error.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

_HF_ORG = "konjoai"
_VALIDATION_PROMPT = "Hello"
_VALIDATION_MAX_TOKENS = 5


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


def build_model_card(
    hf_name: str,
    source_model_id: str,
    quantization: str,
    context_length: int,
    base_license: str,
) -> str:
    """Render the model card markdown (YAML frontmatter + body)."""
    quant_tag = quantization.lower()
    return f"""---
language:
- en
license: {base_license}
tags:
- squish
- mlx
- apple-silicon
- quantized
- {quant_tag}
base_model: {source_model_id}
library_name: mlx_lm
pipeline_tag: text-generation
---

# {hf_name}

Pre-compressed for [squish](https://github.com/konjoai/squish), a local LLM
inference server for Apple Silicon.

🌐 [squish.run](https://squish.run)

## Quick Start

```bash
pip install squish-ai
squish pull {_HF_ORG}/{hf_name}
squish run {hf_name}
```

## Details
- **Base model:** {source_model_id}
- **Quantization:** {quantization}
- **Context length:** {context_length:,} tokens
- **Library:** mlx_lm 0.18+
- **Hardware:** Apple Silicon (M1/M2/M3/M4/M5)

## License

This compressed variant inherits the base model's {base_license} license. The
squish compression tooling is BUSL-1.1.

## Direct mlx_lm usage

```python
from mlx_lm import load, generate

model, tokenizer = load("{_HF_ORG}/{hf_name}")
response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
```
"""


# ---------------------------------------------------------------------------
# Local-model inspection
# ---------------------------------------------------------------------------


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def list_model_files(local_path: Path) -> tuple[list[tuple[str, int]], int]:
    """Return ([(relative_name, size_bytes), ...], total_bytes)."""
    files: list[tuple[str, int]] = []
    total = 0
    for path in sorted(local_path.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            files.append((str(path.relative_to(local_path)), size))
            total += size
    return files, total


# ---------------------------------------------------------------------------
# Pre-upload validation
# ---------------------------------------------------------------------------


def validate_model(local_path: Path) -> bool:
    """Run a one-token inference against the local model via mlx_lm.

    Returns True if validation ran and passed, False if it was skipped because
    mlx_lm is unavailable (e.g. off Apple Silicon). Raises RuntimeError if the
    model loads or generates incorrectly, so the caller can abort the upload
    rather than shipping a model that doesn't run.

    ``squish run`` is not used here: with a prompt but no live daemon it starts
    a blocking server instead of doing a one-shot generation.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("  ⚠  Not on Apple Silicon — skipping inference validation.")
        return False
    try:
        import mlx_lm  # type: ignore[import]
    except ImportError:
        print("  ⚠  mlx_lm not installed — skipping inference validation.")
        return False

    try:
        model, tokenizer = mlx_lm.load(str(local_path))
        text = mlx_lm.generate(
            model,
            tokenizer,
            prompt=_VALIDATION_PROMPT,
            max_tokens=_VALIDATION_MAX_TOKENS,
            verbose=False,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise RuntimeError(f"validation inference failed: {exc}") from exc

    if not text or not text.strip():
        raise RuntimeError("validation inference produced empty output.")
    return True


# ---------------------------------------------------------------------------
# HF auth
# ---------------------------------------------------------------------------


def _resolve_token() -> str:
    """Return an HF token from HF_TOKEN or the cached login, or raise."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        ) from exc
    cached = HfFolder.get_token()
    if cached:
        return cached
    raise RuntimeError(
        "no Hugging Face token found.\n"
        "  Set HF_TOKEN, or run: huggingface-cli login\n"
        "  Create a write token at: https://huggingface.co/settings/tokens"
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _upload(local_path: Path, repo_id: str, model_card: str, token: str) -> None:
    """Create the repo and upload the model directory + card."""
    try:
        from huggingface_hub import HfApi  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    card_path = local_path / "README.md"
    original = card_path.read_text() if card_path.exists() else None
    card_path.write_text(model_card)
    try:
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Publish squished model via publish_to_hf",
            token=token,
        )
    finally:
        if original is not None:
            card_path.write_text(original)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def publish_squished_model(
    local_model_path: str,
    hf_repo_name: str,
    source_model_id: str,
    quantization: str,
    context_length: int,
    base_license: str = "apache-2.0",
    private: bool = False,
    dry_run: bool = True,
) -> None:
    """Validate and publish a squished model to konjoai/<hf_repo_name>."""
    local_path = Path(local_model_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"local model path not found: {local_path}")

    repo_id = f"{_HF_ORG}/{hf_repo_name}"
    repo_url = f"https://huggingface.co/{repo_id}"
    model_card = build_model_card(
        hf_repo_name, source_model_id, quantization, context_length, base_license
    )
    files, total = list_model_files(local_path)

    print(f"  Local model:  {local_path}")
    print(f"  Target repo:  {repo_id}  ({'private' if private else 'public'})")
    print(f"  URL:          {repo_url}")
    print(f"  Files:        {len(files)} ({_human_size(total)})")
    for name, size in files:
        print(f"    - {name}  ({_human_size(size)})")

    print("\n  Validating with one-token inference …")
    if validate_model(local_path):
        print("  ✓  Validation inference succeeded.")

    print("\n  ── Model card ──────────────────────────────────────────")
    print(model_card)
    print("  ────────────────────────────────────────────────────────")

    if dry_run:
        print("\n  DRY RUN — no data written to Hugging Face.")
        print(f"  To publish for real: re-run with --no-dry-run (target {repo_url}).")
        return

    token = _resolve_token()
    print(f"\n  Uploading to {repo_id} …")
    _upload(local_path, repo_id, model_card, token)
    print(f"  ✓  Published: {repo_url}")
    print(
        "  Reminder: update the org README model table at "
        "https://huggingface.co/spaces/konjoai/README (manual for now)."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="publish_to_hf",
        description="Publish a squished model to the konjoai Hugging Face org.",
    )
    p.add_argument(
        "--local-path", required=True, help="Path to the local squished model directory."
    )
    p.add_argument(
        "--hf-name",
        required=True,
        help="Repo name under konjoai/, e.g. Qwen2.5-7B-Instruct-squished.",
    )
    p.add_argument(
        "--source-id", required=True, help="Base model id, e.g. Qwen/Qwen2.5-7B-Instruct."
    )
    p.add_argument("--quant", required=True, help="Quantization label, e.g. INT4 or INT3.")
    p.add_argument("--context", required=True, type=int, help="Context length in tokens.")
    p.add_argument(
        "--base-license", default="apache-2.0", help="Base model license (default: apache-2.0)."
    )
    p.add_argument(
        "--private", action="store_true", help="Create the repo as private (default: public)."
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually upload. Without this flag the script only dry-runs.",
    )
    p.set_defaults(dry_run=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        publish_squished_model(
            local_model_path=args.local_path,
            hf_repo_name=args.hf_name,
            source_model_id=args.source_id,
            quantization=args.quant,
            context_length=args.context,
            base_license=args.base_license,
            private=args.private,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"  ✗  {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"  ✗  {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
