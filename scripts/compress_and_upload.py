#!/usr/bin/env python3
"""compress_and_upload.py — Compress a BF16 model to INT4 (and optionally INT3/INT2)
and push each quantized variant to a Hugging Face repository.

Workflow per bit-width:
  1. Compress BF16 → Squish npy-dir (INT4/INT3/INT2)
  2. Run 5-prompt coherence smoke test (repetition-loop detection)
  3. Upload the resulting directory to HF as a separate model repo or sub-folder
  4. (Optional) Delete the local compressed copy after upload

Usage
-----
  python scripts/compress_and_upload.py \\
      --model ~/.cache/squish/models/Qwen3-8B-Instruct-bf16 \\
      --bits int4 int3 \\
      --hf-repo myorg/Qwen3-8B-Instruct \\
      --hf-token $HF_TOKEN \\
      --delete-local

Exit codes: 0 success, 1 usage/input error, 2 runtime/compression error.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Coherence smoke-test prompts
# ---------------------------------------------------------------------------
_SMOKE_PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is in one sentence.",
    "Write a Python function that returns the factorial of n.",
    "What is 17 × 23?",
    "Briefly describe the water cycle.",
]
_MAX_NEW_TOKENS = 80
_REPETITION_WINDOW = 6   # consecutive repeated tokens → loop signal
_REPETITION_THRESHOLD = 3  # how many repetitions count as a loop


def _detect_repetition_loop(token_ids: list[int]) -> bool:
    """Return True if the last tokens form a repetition-loop pattern."""
    if len(token_ids) < _REPETITION_WINDOW * 2:
        return False
    window = token_ids[-_REPETITION_WINDOW * 2:]
    first_half = window[:_REPETITION_WINDOW]
    second_half = window[_REPETITION_WINDOW:]
    matches = sum(a == b for a, b in zip(first_half, second_half))
    return matches >= _REPETITION_THRESHOLD


def _smoke_test(model_dir: Path, verbose: bool) -> bool:
    """Load the quantized model and run a quick coherence check.

    Returns True when all prompts pass (no repetition loops detected).
    The function avoids importing squish at module load time so that the
    script can be used even without a full squish install for unit tests.
    """
    try:
        import mlx_lm  # type: ignore[import]
    except ImportError:
        print("  ⚠  mlx_lm not installed — skipping coherence smoke test.", file=sys.stderr)
        return True

    squish_4bit = model_dir / "squish_4bit"
    load_path = squish_4bit if squish_4bit.exists() else model_dir
    try:
        model, tokenizer = mlx_lm.load(str(load_path))
    except Exception as exc:
        print(f"  ✗  Smoke test failed to load model: {exc}", file=sys.stderr)
        return False

    failed = 0
    for i, prompt in enumerate(_SMOKE_PROMPTS, 1):
        try:
            text = mlx_lm.generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=_MAX_NEW_TOKENS,
                verbose=False,
            )
        except Exception as exc:
            print(f"  ✗  Smoke prompt {i} raised: {exc}", file=sys.stderr)
            failed += 1
            continue

        # Rudimentary repetition check on the raw text
        tokens = text.split()
        if len(tokens) >= _REPETITION_WINDOW * 2:
            first = tokens[-_REPETITION_WINDOW * 2 : -_REPETITION_WINDOW]
            second = tokens[-_REPETITION_WINDOW:]
            if sum(a == b for a, b in zip(first, second)) >= _REPETITION_THRESHOLD:
                print(f"  ✗  Repetition loop detected in prompt {i}.", file=sys.stderr)
                if verbose:
                    print(f"     Output: {text[:120]!r}", file=sys.stderr)
                failed += 1

        elif verbose:
            print(f"     Prompt {i} OK: {text[:60]!r}")

    passed = len(_SMOKE_PROMPTS) - failed
    print(f"  Coherence: {passed}/{len(_SMOKE_PROMPTS)} prompts passed")
    return failed == 0


# ---------------------------------------------------------------------------
# Compression helper
# ---------------------------------------------------------------------------

def _compress(model_dir: Path, output_dir: Path, bits: str, no_awq: bool,
              awq_samples: int, calibration_data: str | None, verbose: bool) -> bool:
    """Invoke squish.convert via subprocess. Returns True on success."""
    cmd = [
        sys.executable, "-m", "squish.cli", "compress",
        str(model_dir),
        "--output", str(output_dir),
        "--format", bits,
    ]
    if no_awq:
        cmd.append("--no-awq")
    if calibration_data:
        cmd += ["--calibration-data", calibration_data]
    if verbose:
        cmd.append("--verbose")

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# HF upload helper
# ---------------------------------------------------------------------------

def _upload_to_hf(local_dir: Path, hf_repo: str, path_in_repo: str,
                  token: str | None, commit_message: str) -> None:
    """Upload a local directory to a Hugging Face repository."""
    try:
        from huggingface_hub import HfApi  # type: ignore[import]
    except ImportError:
        print("  ✗  huggingface_hub not installed.", file=sys.stderr)
        print("     Install with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)
    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True, token=token)
    except Exception as exc:
        print(f"  ✗  Could not create/verify HF repo {hf_repo}: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"  Uploading {local_dir.name} → {hf_repo}/{path_in_repo} …")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=hf_repo,
        path_in_repo=path_in_repo,
        repo_type="model",
        commit_message=commit_message,
        token=token,
    )
    print(f"  ✓  Uploaded to https://huggingface.co/{hf_repo}/tree/main/{path_in_repo}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="compress_and_upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Compress a BF16 model to one or more quantized formats and
            optionally upload each to a Hugging Face model repository.

            Example
            -------
              python scripts/compress_and_upload.py \\
                  --model ~/.cache/squish/models/Qwen3-8B-Instruct-bf16 \\
                  --bits int4 int3 \\
                  --hf-repo myorg/Qwen3-8B-Squish \\
                  --delete-local
        """),
    )
    p.add_argument("--model", required=True,
                   help="Path to the BF16 source model directory.")
    p.add_argument("--bits", nargs="+", default=["int4"],
                   choices=["int4", "int3", "int2"],
                   help="Quantization levels to produce (default: int4).")
    p.add_argument("--hf-repo",
                   help="Hugging Face repo ID, e.g. 'myorg/MyModel'. "
                        "If omitted, skips upload.")
    p.add_argument("--hf-token",
                   help="Hugging Face write token. Falls back to HF_TOKEN env var.")
    p.add_argument("--delete-local", action="store_true",
                   help="Delete the local compressed directory after a successful upload.")
    p.add_argument("--no-awq", action="store_true",
                   help="Skip AWQ calibration (faster, slight accuracy cost).")
    p.add_argument("--awq-samples", type=int, default=20,
                   help="Number of AWQ calibration samples (default: 20).")
    p.add_argument("--calibration-data",
                   help="Path to a .jsonl file with calibration prompts.")
    p.add_argument("--skip-smoke-test", action="store_true",
                   help="Skip the post-compression coherence smoke test.")
    p.add_argument("--output-dir",
                   help="Parent directory for compressed outputs. "
                        "Defaults to the source model's parent directory.")
    p.add_argument("--verbose", action="store_true",
                   help="Show verbose compression and upload output.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress informational output (errors still go to stderr).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    verbose = args.verbose and not args.quiet

    model_dir = Path(args.model).expanduser().resolve()
    if not model_dir.exists():
        print(f"  ✗  Model directory not found: {model_dir}", file=sys.stderr)
        return 1

    output_parent = Path(args.output_dir).expanduser() if args.output_dir else model_dir.parent

    import os
    hf_token: str | None = args.hf_token or os.environ.get("HF_TOKEN")

    if args.hf_repo and not hf_token:
        print("  ⚠  --hf-repo given but no HF token found.", file=sys.stderr)
        print("     Pass --hf-token or set the HF_TOKEN environment variable.", file=sys.stderr)
        return 1

    results: dict[str, str] = {}  # bits → "ok" | "failed" | "skipped"

    for bits in args.bits:
        if not args.quiet:
            print(f"\n{'='*60}")
            print(f"  Processing: {model_dir.name}  →  {bits.upper()}")
            print(f"{'='*60}\n")

        # ── Derive output directory name ─────────────────────────────────
        import re
        base_name = re.sub(
            r'-(bf16|fp16|[0-9]+bit)(-mlx)?$', '',
            model_dir.name, flags=re.IGNORECASE,
        )
        output_dir = output_parent / f"{base_name}-{bits}"

        # ── Compress ─────────────────────────────────────────────────────
        if output_dir.exists():
            if not args.quiet:
                print(f"  ℹ  {output_dir.name} already exists — skipping compression.")
        else:
            if not args.quiet:
                print(f"  Compressing to {bits.upper()} …")
            success = _compress(
                model_dir=model_dir,
                output_dir=output_dir,
                bits=bits,
                no_awq=args.no_awq,
                awq_samples=args.awq_samples,
                calibration_data=args.calibration_data,
                verbose=verbose,
            )
            if not success:
                print(f"  ✗  Compression to {bits.upper()} failed.", file=sys.stderr)
                results[bits] = "failed"
                continue
            if not args.quiet:
                print(f"  ✓  Compressed: {output_dir}")

        # ── Smoke test ────────────────────────────────────────────────────
        if not args.skip_smoke_test:
            if not args.quiet:
                print(f"  Running coherence smoke test …")
            passed = _smoke_test(output_dir, verbose=verbose)
            if not passed:
                print(
                    f"  ✗  {bits.upper()} model failed coherence test — "
                    f"skipping upload to avoid shipping a broken model.",
                    file=sys.stderr,
                )
                results[bits] = "failed-coherence"
                continue
        else:
            if not args.quiet:
                print("  Smoke test skipped (--skip-smoke-test).")

        # ── Upload ────────────────────────────────────────────────────────
        if args.hf_repo:
            path_in_repo = output_dir.name
            _upload_to_hf(
                local_dir=output_dir,
                hf_repo=args.hf_repo,
                path_in_repo=path_in_repo,
                token=hf_token,
                commit_message=f"Add {bits.upper()} quantized model via squish compress_and_upload",
            )

            # ── Delete local copy after upload ────────────────────────────
            if args.delete_local:
                shutil.rmtree(output_dir)
                if not args.quiet:
                    print(f"  Deleted local copy: {output_dir}")

        results[bits] = "ok"

    # ── Summary ───────────────────────────────────────────────────────────────
    if not args.quiet:
        print(f"\n{'='*60}")
        print("  Summary")
        print(f"{'='*60}")
        for bits, status in results.items():
            icon = "✓" if status == "ok" else "✗"
            print(f"    {icon}  {bits.upper():5}  {status}")
        print()

    failed = sum(1 for s in results.values() if s != "ok")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
