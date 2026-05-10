#!/usr/bin/env python3
"""
Konjo Adversarial Review Agent — Wall 3.

Critic model: claude-opus-4-6
CRITIC_MODEL must never match the builder model. The builder (Sonnet) has blind
spots from the construction process. The critic comes in cold from a distinct
session with a different capability profile to reduce correlated failures.

Research basis:
  - Homogeneous AI review pipelines echo rather than cancel errors (arXiv 2603.25773)
  - AI agent self-review is architecturally circular: it checks code against
    itself, not against intent
  - Pass-rate benchmarks undermeasure extension robustness (SlopCodeBench, arXiv 2603.24755)

Prompt caching: the system prompt (>1024 tokens) is sent with
cache_control={"type":"ephemeral"} to exploit Anthropic's prompt cache.
Within a typical CI run the cache warms on first call and hits on any retry.

Exit codes:
  0 — APPROVED or WARNING-only (merge may proceed)
  1 — BLOCKER found (merge must not proceed)
  2 — API error or misconfiguration (treat as soft failure, warn and proceed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
CRITIC_MODEL = "claude-opus-4-6"
MAX_DIFF_CHARS = 80_000
MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

# ── System prompt (cached block) ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are the Konjo Adversarial Reviewer — an independent critic whose role is to
find flaws that the builder missed. You were NOT involved in writing this code.
You have no loyalty to the implementation choices made. Your only loyalty is to
the ten quality standards below.

Research context you should know:
- AI-assisted code produces 1.7× more logical/correctness bugs (CodeRabbit 2026)
- AI agents change tests so broken code passes instead of fixing the code (Baltes 2026)
- AI self-review shares the same training distribution as the builder — it checks
  code against itself, not against intent. You must check against intent.

THE TEN MANDATORY REVIEW QUESTIONS — answer each explicitly:

Q1  CORRECTNESS
    Does this code actually do what it claims? Identify any logical error,
    off-by-one, race condition, incorrect algorithm, or silent data corruption.
    The code compiling and tests passing is not sufficient proof of correctness.

Q2  COVERAGE BLIND SPOTS
    What input paths are untested? What would cause silent failure or wrong
    output that the test suite would not catch? Are error paths tested, or only
    happy paths? Would mutation testing survive this test suite?

Q3  DEAD CODE
    Is there any unreachable code, unused variable, imported-but-never-called
    function, or commented-out block? Zero tolerance. Not even one line.

Q4  DOCUMENTATION
    Is every public API documented? Does documentation match the implementation?
    Are complex algorithms explained with the math? Are invariants stated?

Q5  ERROR HANDLING
    Are errors propagated or swallowed? Any bare except:/catch? Any
    unwrap()/expect() outside test code? Any fallback that silently masks a
    real failure? Any error that logs only "something went wrong"?

Q6  DRY VIOLATION
    Does any block of logic appear more than once at >85% similarity across
    >10 lines anywhere in this diff or implied by it? Name files and line numbers.

Q7  COMPLEXITY AND SIZE
    Does any function exceed cognitive complexity 15? Does any function body
    exceed 50 lines? Does any file exceed 500 lines? Is there a simpler way
    to accomplish the same thing with fewer lines?

Q8  SECURITY
    Prompt injection surface? Logging of sensitive data (tokens, keys, raw
    user input)? Missing input validation at API boundaries? Hardcoded secrets?
    Missing rate limiting? Unsafe deserialization?

Q9  PERFORMANCE
    Any algorithmic regression (O(n²) where O(n log n) is straightforward)?
    Any blocking call on an async path? Any unnecessary allocation in a hot
    loop? Any benchmark that would catch a regression if this code regressed?

Q10 KONJO STANDARD — THE SHIPBUILDER TEST
    Is this code seaworthy: tested, lean, and ready to carry real load?
    What would you cut? What is here for comfort rather than function?
    If 10,000 requests hit this code simultaneously for 30 days without
    restart, would it hold? If not, what breaks first?

VERDICT RULES:
- Issue BLOCKER for: any violation of Q1, Q3, Q5 (unwrap/silent failure),
  Q8 security issues, any function >100 lines, any public API undocumented
- Issue WARNING for: Q2 partial coverage, Q6 minor duplication, Q7 complexity
  approaching limits, Q9 cold-path performance, Q10 style improvements
- Issue APPROVED only when all ten questions pass without reservation

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences:
{
  "verdict": "APPROVED" | "WARNING" | "BLOCKER",
  "summary": "one honest sentence about the overall quality",
  "questions": {
    "Q1": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "specific finding or 'Pass'"},
    "Q2": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "..."},
    "Q3": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "..."},
    "Q4": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "..."},
    "Q5": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "..."},
    "Q6": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "..."},
    "Q7": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "..."},
    "Q8": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "..."},
    "Q9": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "..."},
    "Q10": {"verdict": "PASS" | "WARN" | "BLOCK", "finding": "..."}
  },
  "blockers": ["file:line — specific blocking issue with recommended fix"],
  "warnings": ["file:line — non-blocking improvement"],
  "approved_aspects": ["what was genuinely well done, if anything"]
}

Be specific. Reference exact file paths and line numbers from the diff.
Show the correct code when issuing a BLOCKER, not just a description.
Do not approve code that violates any of the above checks. Slop does not ship.
"""


def _load_anthropic():
    """Import anthropic SDK; raise ImportError with install instructions if absent."""
    try:
        import anthropic  # noqa: PLC0415
        return anthropic
    except ImportError:
        print(
            "ERROR: anthropic package not found.\n"
            "Install: pip install anthropic\n"
            "Or: uv add anthropic",
            file=sys.stderr,
        )
        raise


def _call_api(diff_text: str, anthropic_module) -> dict:
    """Call Claude Opus with the diff; return parsed JSON verdict."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. "
            "Export it or add it to GitHub Actions secrets."
        )

    client = anthropic_module.Anthropic(api_key=api_key)

    truncated = len(diff_text) > MAX_DIFF_CHARS
    if truncated:
        diff_text = diff_text[:MAX_DIFF_CHARS]
        diff_text += (
            f"\n\n[DIFF TRUNCATED: showing first {MAX_DIFF_CHARS} chars. "
            "Focus your review on what is visible above.]"
        )

    user_content = (
        "Review this pull request diff against the ten Konjo quality standards.\n\n"
        f"<diff>\n{diff_text}\n</diff>"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=CRITIC_MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text.strip()

            # Log token usage to stderr for cost tracking
            usage = response.usage
            print(
                f"[konjo-review] tokens: input={usage.input_tokens} "
                f"output={usage.output_tokens} "
                f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)} "
                f"cache_write={getattr(usage, 'cache_creation_input_tokens', 0)}",
                file=sys.stderr,
            )

            return json.loads(raw)

        except (anthropic_module.RateLimitError, anthropic_module.APIStatusError) as exc:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"[konjo-review] API error ({exc}), retrying in {delay:.0f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                raise
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Critic returned non-JSON response. Raw output:\n{raw}"
            ) from exc

    raise RuntimeError("Exhausted retries without success")


def _render_human(result: dict) -> str:
    """Render the JSON verdict as a readable markdown report."""
    lines = ["# Konjo Adversarial Review Report\n"]
    verdict = result.get("verdict", "UNKNOWN")
    emoji = {"APPROVED": "✅", "WARNING": "⚠️", "BLOCKER": "🚫"}.get(verdict, "❓")
    lines.append(f"## Verdict: {emoji} {verdict}\n")
    lines.append(f"**Summary:** {result.get('summary', '')}\n")

    blockers = result.get("blockers", [])
    if blockers:
        lines.append("## 🚫 Blockers (must fix before merge)\n")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    warnings = result.get("warnings", [])
    if warnings:
        lines.append("## ⚠️ Warnings (advisory)\n")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Question-by-Question\n")
    emoji_map = {"PASS": "✅", "WARN": "⚠️", "BLOCK": "🚫"}
    for q, data in result.get("questions", {}).items():
        v = data.get("verdict", "?")
        finding = data.get("finding", "")
        lines.append(f"**{q}** {emoji_map.get(v, '?')} {v} — {finding}")
    lines.append("")

    approved = result.get("approved_aspects", [])
    if approved:
        lines.append("## ✅ What was done well\n")
        for a in approved:
            lines.append(f"- {a}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Konjo Adversarial Review — Wall 3 quality gate"
    )
    diff_group = parser.add_mutually_exclusive_group()
    diff_group.add_argument("--diff-file", help="Path to diff file (default: stdin)")
    diff_group.add_argument(
        "--diff", help="Diff text inline (use with <() process substitution)"
    )
    parser.add_argument(
        "--output", help="Write human-readable report to this file (default: stdout)"
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_out",
        help="Write machine-readable JSON to stdout instead of human report"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be sent to the API without calling it (for testing)"
    )
    parser.add_argument(
        "--soft-fail", action="store_true",
        help="Exit 0 even on BLOCKER verdict (report only; use during framework rollout)"
    )
    args = parser.parse_args()

    # Read diff
    if args.diff:
        diff_text = args.diff
    elif args.diff_file:
        diff_text = Path(args.diff_file).read_text()
    else:
        if sys.stdin.isatty():
            print("ERROR: provide --diff-file or pipe a diff to stdin", file=sys.stderr)
            return 2
        diff_text = sys.stdin.read()

    if not diff_text.strip():
        print("[konjo-review] Empty diff — nothing to review. Approved.", file=sys.stderr)
        return 0

    if args.dry_run:
        print("[konjo-review] DRY RUN — would send to API:", file=sys.stderr)
        print(f"  model: {CRITIC_MODEL}", file=sys.stderr)
        print(f"  diff chars: {len(diff_text)}", file=sys.stderr)
        print(f"  system prompt chars: {len(SYSTEM_PROMPT)}", file=sys.stderr)
        return 0

    try:
        anthropic = _load_anthropic()
        result = _call_api(diff_text, anthropic)
    except (ImportError, ValueError, RuntimeError) as exc:
        print(f"[konjo-review] ERROR: {exc}", file=sys.stderr)
        print("[konjo-review] Soft-failing: treating as WARNING (could not reach API)", file=sys.stderr)
        return 0  # Don't block CI on API unavailability

    verdict = result.get("verdict", "UNKNOWN")
    has_blockers = verdict == "BLOCKER" or bool(result.get("blockers"))

    if args.json_out:
        print(json.dumps(result, indent=2))
    else:
        report = _render_human(result)
        if args.output:
            Path(args.output).write_text(report)
            print(f"[konjo-review] Report written to {args.output}", file=sys.stderr)
        else:
            print(report)

    if has_blockers and not args.soft_fail:
        print(
            f"\n[konjo-review] VERDICT: {verdict} — merge blocked. Fix all blockers first.",
            file=sys.stderr,
        )
        return 1

    print(f"[konjo-review] VERDICT: {verdict}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
