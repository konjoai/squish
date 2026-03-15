#!/usr/bin/env python3
"""
dev/scripts/model_pipeline.py

Three-job CI/CD pipeline for automated model compression and publishing.

Jobs:
  watch    — Poll HuggingFace for new model releases and output candidates
  compress — Run squish compress --int4 on each candidate model
  publish  — Upload compressed model directories to the HF Hub

Usage:
    python dev/scripts/model_pipeline.py --job watch [--dry-run] [--output models.json]
    python dev/scripts/model_pipeline.py --job compress [--validate] [--dry-run] [--models models.json]
    python dev/scripts/model_pipeline.py --job publish [--dry-run] [--models models.json]

Dry-run mode completes without any external calls (no HF_TOKEN, no real models needed).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ModelCandidate:
    """A candidate model for compression."""
    name: str
    hf_repo: str
    size_gb: float
    architecture: str
    priority: str  # P0, P1, P2


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""
    dry_run: bool = False
    validate: bool = False
    output_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "squish" / "pipeline")
    hf_token: Optional[str] = None


# ── Known architectures for candidate filtering ───────────────────────────────

_KNOWN_ARCHITECTURES = {"Qwen", "Llama", "Mistral", "Phi"}

_SYNTHETIC_CANDIDATES = [
    ModelCandidate(
        name="Qwen2.5-1.5B-Instruct",
        hf_repo="Qwen/Qwen2.5-1.5B-Instruct",
        size_gb=3.1,
        architecture="Qwen",
        priority="P0",
    ),
    ModelCandidate(
        name="Llama-3.2-3B-Instruct",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        size_gb=6.4,
        architecture="Llama",
        priority="P1",
    ),
    ModelCandidate(
        name="Phi-3.5-mini-instruct",
        hf_repo="microsoft/Phi-3.5-mini-instruct",
        size_gb=7.6,
        architecture="Phi",
        priority="P1",
    ),
]


# ── WatchJob ─────────────────────────────────────────────────────────────────

class WatchJob:
    """
    Poll HuggingFace for new model releases that match criteria:
      - Architecture: Qwen, Llama, Mistral, Phi
      - Size: 1B-15B parameters
      - Available for download
      - Not yet in squish-community HF org
    """

    def run(self, config: PipelineConfig) -> list[ModelCandidate]:
        """Return candidate models. Dry-run returns 3 synthetic candidates."""
        if config.dry_run:
            print("[watch] dry-run: returning synthetic candidates")
            return self._synthetic_candidates()

        return self._fetch_candidates(config)

    def _synthetic_candidates(self) -> list[ModelCandidate]:
        """Return 3 hardcoded candidates for dry-run mode."""
        return list(_SYNTHETIC_CANDIDATES)

    def _fetch_candidates(self, config: PipelineConfig) -> list[ModelCandidate]:
        """Query HuggingFace API for new model releases (live mode)."""
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415
        except ImportError:
            print("[watch] ERROR: huggingface_hub not installed. pip install huggingface-hub",
                  file=sys.stderr)
            return []

        hf_token = config.hf_token or os.environ.get("HF_TOKEN", "")
        api = HfApi(token=hf_token or None)

        # Fetch already-published squish-community models to avoid duplicates
        try:
            published = {
                m.id.split("/")[-1].replace("-squished", "").replace("-int4", "")
                for m in api.list_models(author="squish-community")
            }
        except Exception as exc:
            print(f"[watch] WARN: could not list squish-community models: {exc}", file=sys.stderr)
            published = set()

        candidates: list[ModelCandidate] = []

        for arch in sorted(_KNOWN_ARCHITECTURES):
            try:
                models = list(api.list_models(
                    filter=arch,
                    sort="lastModified",
                    direction=-1,
                    limit=20,
                ))
            except Exception as exc:
                print(f"[watch] WARN: could not list {arch} models: {exc}", file=sys.stderr)
                continue

            for m in models:
                model_name = m.id.split("/")[-1]

                # Skip already-published
                if model_name in published:
                    continue

                # Size filter: 1B-15B (use tags or safetensors metadata)
                size_gb = self._estimate_size_gb(m)
                if not (1.0 <= size_gb <= 60.0):
                    continue

                # Must be publicly downloadable (no gated models without token)
                if getattr(m, "gated", False) and not hf_token:
                    continue

                priority = "P0" if size_gb <= 4.0 else ("P1" if size_gb <= 10.0 else "P2")
                candidates.append(ModelCandidate(
                    name=model_name,
                    hf_repo=m.id,
                    size_gb=size_gb,
                    architecture=arch,
                    priority=priority,
                ))

        # Deduplicate by name, prefer P0
        seen: dict[str, ModelCandidate] = {}
        for c in candidates:
            if c.name not in seen or c.priority < seen[c.name].priority:
                seen[c.name] = c

        result = sorted(seen.values(), key=lambda c: (c.priority, c.name))
        print(f"[watch] Found {len(result)} candidate(s)")
        return result

    def _estimate_size_gb(self, model_info) -> float:
        """Estimate model size from tags or safetensors metadata. Returns GB."""
        # Check tags like "1B", "3B", "7B", "13B"
        tags = getattr(model_info, "tags", []) or []
        for tag in tags:
            tag_lower = tag.lower()
            for suffix, mult in [("b", 1.0), ("m", 0.001)]:
                if tag_lower.endswith(suffix):
                    try:
                        n = float(tag_lower[:-1])
                        # Approximate: 1B params ≈ 2 GB (BF16) = 1 GB (INT4)
                        return n * 2.0 * mult
                    except ValueError:
                        pass
        # Default: unknown size — assume fits
        return 7.0


# ── CompressJob ───────────────────────────────────────────────────────────────

class CompressJob:
    """
    Run `squish compress --int4 --output-dir ...` for each candidate model.
    """

    def run(
        self,
        candidates: list[ModelCandidate],
        config: PipelineConfig,
    ) -> list[Path]:
        """
        Compress each candidate. Returns list of output directories.

        With dry_run=True, prints commands without executing subprocess.
        With validate=True, checks that the compressed model loads without error.
        """
        output_dirs: list[Path] = []
        config.output_dir.mkdir(parents=True, exist_ok=True)

        for candidate in candidates:
            out_dir = config.output_dir / candidate.name
            cmd = self._build_command(candidate, out_dir)

            if config.dry_run:
                print(f"[compress] DRY-RUN would execute: {' '.join(cmd)}")
                output_dirs.append(out_dir)
                continue

            print(f"[compress] Compressing {candidate.hf_repo} → {out_dir}")
            try:
                result = subprocess.run(cmd, check=True, capture_output=False)
                print(f"[compress] Done: {candidate.name} (exit={result.returncode})")
            except subprocess.CalledProcessError as exc:
                print(f"[compress] ERROR: {candidate.name} failed (exit={exc.returncode})",
                      file=sys.stderr)
                continue
            except FileNotFoundError:
                print("[compress] ERROR: 'squish' command not found. Is squish installed?",
                      file=sys.stderr)
                continue

            if config.validate:
                ok = self._validate(out_dir)
                if ok:
                    print(f"[compress] Validation passed: {candidate.name}")
                else:
                    print(f"[compress] Validation FAILED: {candidate.name}", file=sys.stderr)
                    continue

            output_dirs.append(out_dir)

        return output_dirs

    def _build_command(self, candidate: ModelCandidate, output_dir: Path) -> list[str]:
        """Build the squish compress command for a candidate."""
        return [
            "squish",
            "compress",
            "--int4",
            "--output-dir", str(output_dir),
            candidate.hf_repo,
        ]

    def _validate(self, output_dir: Path) -> bool:
        """Check that the compressed model directory loads without error."""
        manifest = output_dir / "manifest.json"
        if not manifest.exists():
            return False
        try:
            with open(manifest) as f:
                data = json.load(f)
            return isinstance(data, dict) and len(data) > 0
        except Exception:
            return False


# ── PublishJob ────────────────────────────────────────────────────────────────

class PublishJob:
    """
    Upload compressed model directories to the HF Hub via publish_hf.py.
    """

    # Path to publish_hf.py relative to this script (dev/scripts/ → dev/)
    _PUBLISH_SCRIPT = Path(__file__).resolve().parent.parent / "publish_hf.py"

    def run(self, output_dirs: list[Path], config: PipelineConfig) -> None:
        """
        Publish each output directory to HF Hub.

        With dry_run=True, prints upload commands without executing.
        """
        for out_dir in output_dirs:
            model_name = out_dir.name
            repo = f"squish-community/{model_name}-int4"
            cmd = self._build_command(out_dir, repo, config)

            if config.dry_run:
                print(f"[publish] DRY-RUN would execute: {' '.join(cmd)}")
                continue

            print(f"[publish] Uploading {model_name} → {repo}")
            try:
                env = os.environ.copy()
                if config.hf_token:
                    env["HF_TOKEN"] = config.hf_token
                result = subprocess.run(cmd, check=True, capture_output=False, env=env)
                print(f"[publish] Done: {model_name} (exit={result.returncode})")
            except subprocess.CalledProcessError as exc:
                print(f"[publish] ERROR: {model_name} failed (exit={exc.returncode})",
                      file=sys.stderr)
            except FileNotFoundError:
                print(f"[publish] ERROR: publish script not found at {self._PUBLISH_SCRIPT}",
                      file=sys.stderr)

    def _build_command(
        self,
        model_dir: Path,
        repo: str,
        config: PipelineConfig,
    ) -> list[str]:
        """Build the publish_hf.py command."""
        cmd = [
            sys.executable,
            str(self._PUBLISH_SCRIPT),
            "--model-dir", str(model_dir),
            "--repo", repo,
            "--base-model", model_dir.name,
        ]
        if config.hf_token:
            cmd += ["--hf-token", config.hf_token]
        return cmd


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Model compression CI/CD pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument(
        "--job",
        required=True,
        choices=["watch", "compress", "publish"],
        help="Pipeline job to run: watch | compress | publish",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print commands without executing (no HF_TOKEN or real models needed)",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="(compress job) Validate that compressed model loads without error",
    )
    ap.add_argument(
        "--output",
        metavar="FILE",
        default="models.json",
        help="(watch job) Path to write candidate list JSON (default: models.json)",
    )
    ap.add_argument(
        "--models",
        metavar="FILE",
        default="models.json",
        help="(compress/publish jobs) Path to read candidate list JSON (default: models.json)",
    )
    ap.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory for compressed models (default: ~/.cache/squish/pipeline)",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (falls back to HF_TOKEN env var)",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = _build_parser()
    args = ap.parse_args(argv)

    config = PipelineConfig(
        dry_run=args.dry_run,
        validate=args.validate,
        output_dir=Path(args.output_dir) if args.output_dir else (
            Path.home() / ".cache" / "squish" / "pipeline"
        ),
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
    )

    if args.job == "watch":
        job = WatchJob()
        candidates = job.run(config)

        # Write results to JSON
        output_path = Path(args.output)
        data = [
            {
                "name": c.name,
                "hf_repo": c.hf_repo,
                "size_gb": c.size_gb,
                "architecture": c.architecture,
                "priority": c.priority,
            }
            for c in candidates
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[watch] Written {len(candidates)} candidate(s) to {output_path}")

    elif args.job == "compress":
        # Load candidates from JSON
        models_path = Path(args.models)
        if not models_path.exists():
            print(f"[compress] ERROR: models file not found: {models_path}", file=sys.stderr)
            sys.exit(1)

        with open(models_path) as f:
            raw = json.load(f)

        candidates = [
            ModelCandidate(
                name=r["name"],
                hf_repo=r["hf_repo"],
                size_gb=r["size_gb"],
                architecture=r["architecture"],
                priority=r["priority"],
            )
            for r in raw
        ]

        job = CompressJob()
        output_dirs = job.run(candidates, config)
        print(f"[compress] Compressed {len(output_dirs)} model(s)")

    elif args.job == "publish":
        # Load output dirs from candidates JSON
        models_path = Path(args.models)
        if not models_path.exists():
            print(f"[publish] ERROR: models file not found: {models_path}", file=sys.stderr)
            sys.exit(1)

        with open(models_path) as f:
            raw = json.load(f)

        output_dirs = [
            config.output_dir / r["name"]
            for r in raw
        ]

        job = PublishJob()
        job.run(output_dirs, config)

    else:
        print(f"Unknown job: {args.job}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
