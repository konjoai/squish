"""squish/squash/report.py — Human-readable compliance report generator.

Wave 15.  Reads existing attestation artifacts from a model directory and
produces a self-contained HTML document with inline CSS — zero external
dependencies.

Artifacts consumed (all optional — absent sections are omitted gracefully):
- ``squash-attest.json``           — master attestation record
- ``cyclonedx-mlbom.json``         — hash / performance metrics / CVEs
- ``squash-scan.json``             — security scanner findings
- ``squash-policy-<name>.json``    — per-policy evaluation results
- ``squash-vex-report.json``       — VEX affected-model summary

Output:
- ``squash-report.html``  (default; ``--output`` overrides)

Usage::

    from squish.squash.report import ComplianceReporter
    html = ComplianceReporter.generate_html(Path("./my-model"))
    Path("./my-model/squash-report.html").write_text(html)

    # Or write directly:
    ComplianceReporter.write(Path("./my-model"))
"""

from __future__ import annotations

import datetime
import html as html_lib
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — embedded inline so the HTML file is fully self-contained
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #222; font-size: 14px; line-height: 1.5; }
.page { max-width: 960px; margin: 32px auto; padding: 0 16px 64px; }
h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
h2 { font-size: 15px; font-weight: 600; margin: 24px 0 8px;
     padding-bottom: 4px; border-bottom: 1px solid #ddd; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 12px; font-weight: 600; }
.badge.pass { background: #d4edda; color: #155724; }
.badge.fail { background: #f8d7da; color: #721c24; }
.badge.skip { background: #e2e3e5; color: #383d41; }
.badge.warn { background: #fff3cd; color: #856404; }
table { width: 100%; border-collapse: collapse; font-size: 13px;
        background: #fff; border: 1px solid #ddd; border-radius: 4px; }
th { background: #f0f0f0; font-weight: 600; padding: 6px 10px; text-align: left; }
td { padding: 6px 10px; border-top: 1px solid #eee; vertical-align: top; }
tr:hover td { background: #fafafa; }
.card { background: #fff; border: 1px solid #ddd; border-radius: 4px;
        padding: 14px 16px; margin-bottom: 12px; }
.kv { display: flex; flex-wrap: wrap; gap: 8px 24px; }
.kv dt { font-weight: 600; min-width: 120px; color: #555; }
.kv dd { font-family: monospace; word-break: break-all; }
.finding-critical { color: #721c24; }
.finding-high { color: #856404; }
.finding-medium { color: #0c5460; }
.finding-low { color: #155724; }
footer { margin-top: 40px; font-size: 12px; color: #999; text-align: center; }
"""


class ComplianceReporter:
    """Generate HTML compliance reports from Squash attestation artifacts.

    All methods are static — the class is a namespace, not a stateful object.
    """

    @staticmethod
    def generate_html(model_dir: Path) -> str:
        """Read all attestation artifacts in *model_dir* and return HTML.

        Missing artifacts are silently skipped — the report is always
        generated even when only partial data is available.
        """
        ctx = _load_artifacts(model_dir)
        return _render_html(ctx)

    @staticmethod
    def write(model_dir: Path, output_path: Path | None = None) -> Path:
        """Generate the report and write it to *output_path*.

        Parameters
        ----------
        model_dir:
            Directory containing attestation artifacts.
        output_path:
            Destination file.  Defaults to ``<model_dir>/squash-report.html``.

        Returns
        -------
        Path
            The path where the HTML file was written.
        """
        dest = output_path or (model_dir / "squash-report.html")
        html = ComplianceReporter.generate_html(model_dir)
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(html, encoding="utf-8")
        tmp.replace(dest)
        log.debug("ComplianceReporter: wrote %s", dest)
        return dest


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict[str, Any] | None:
    """Read and parse a JSON file; return None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.debug("ComplianceReporter: could not load %s — %s", path, e)
        return None


def _load_artifacts(model_dir: Path) -> dict[str, Any]:
    """Collect all available attestation artifacts into a context dict."""
    ctx: dict[str, Any] = {"model_dir": model_dir}

    ctx["attest"] = _load_json(model_dir / "squash-attest.json")
    ctx["cdx"] = _load_json(model_dir / "cyclonedx-mlbom.json")
    ctx["scan"] = _load_json(model_dir / "squash-scan.json")
    ctx["vex"] = _load_json(model_dir / "squash-vex-report.json")

    # Enumerate all per-policy results
    policies: dict[str, dict] = {}
    for fp in sorted(model_dir.glob("squash-policy-*.json")):
        data = _load_json(fp)
        if data:
            name = fp.stem.removeprefix("squash-policy-")
            policies[name] = data
    ctx["policies"] = policies

    # Sig bundle presence (verification not run here — just detect)
    cdx_bom = model_dir / "cyclonedx-mlbom.json"
    ctx["bundle_present"] = (cdx_bom.with_name(cdx_bom.name + ".sig.json")).exists()

    return ctx


def _esc(v: Any) -> str:
    return html_lib.escape(str(v))


def _badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{_esc(text)}</span>'


def _render_html(ctx: dict[str, Any]) -> str:
    attest: dict | None = ctx.get("attest")
    cdx: dict | None = ctx.get("cdx")
    scan: dict | None = ctx.get("scan")
    vex: dict | None = ctx.get("vex")
    policies: dict = ctx.get("policies", {})
    model_dir: Path = ctx["model_dir"]

    # Determine overall pass/fail for title badge
    passed: bool | None = attest.get("passed") if attest else None
    overall_badge = (
        _badge("PASS", "pass") if passed
        else _badge("FAIL", "fail") if passed is False
        else _badge("UNKNOWN", "skip")
    )

    model_id = (
        (attest.get("model_id") or model_dir.name) if attest else model_dir.name
    )
    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    sections: list[str] = []

    # ── Section 1: Model identity ─────────────────────────────────────────────
    id_rows = [f"<dt>Model</dt><dd>{_esc(model_id)}</dd>"]
    if attest:
        id_rows.append(f"<dt>Path</dt><dd>{_esc(attest.get('model_path', ''))}</dd>")
        id_rows.append(f"<dt>Attested at</dt><dd>{_esc(attest.get('attested_at', ''))}</dd>")
        id_rows.append(f"<dt>Squash version</dt><dd>{_esc(attest.get('squash_version', ''))}</dd>")
    if cdx:
        comp = cdx.get("components", [{}])[0]
        id_rows.append(f"<dt>PURL</dt><dd>{_esc(comp.get('purl', ''))}</dd>")
        id_rows.append(
            f"<dt>Quant format</dt>"
            f"<dd>{_esc(comp.get('modelCard', {}).get('modelParameters', {}).get('quantizationScheme', ''))}</dd>"
        )

    sections.append(
        f"<h2>Model Identity</h2>"
        f"<div class='card'><dl class='kv'>{''.join(id_rows)}</dl></div>"
    )

    # ── Section 2: BOM hashes ─────────────────────────────────────────────────
    if cdx:
        comp = cdx.get("components", [{}])[0]
        hashes = comp.get("hashes", [])
        sig_row = (
            _badge("Signed", "pass") if ctx.get("bundle_present")
            else _badge("Not signed", "skip")
        )
        hash_rows = "".join(
            f"<tr><td>{_esc(h.get('alg',''))}</td>"
            f"<td><code>{_esc(h.get('content',''))}</code></td></tr>"
            for h in hashes
        )
        sections.append(
            f"<h2>BOM Hashes &amp; Signing</h2>"
            f"<div class='card'>"
            f"<p style='margin-bottom:8px'>Sigstore signature: {sig_row}</p>"
            f"<table><thead><tr><th>Algorithm</th><th>Hash</th></tr></thead>"
            f"<tbody>{hash_rows}</tbody></table></div>"
        )

    # ── Section 3: Performance metrics ───────────────────────────────────────
    if cdx:
        comp = cdx.get("components", [{}])[0]
        metrics = (
            comp.get("modelCard", {})
            .get("quantitativeAnalysis", {})
            .get("performanceMetrics", [])
        )
        if metrics:
            m_rows = "".join(
                f"<tr><td>{_esc(m.get('slice',''))}</td>"
                f"<td>{_esc(m.get('value',''))}%</td>"
                f"<td>{_esc(m.get('deltaFromBaseline',''))}</td></tr>"
                for m in metrics
            )
            sections.append(
                f"<h2>Performance Metrics</h2>"
                f"<div class='card'><table>"
                f"<thead><tr><th>Task</th><th>Score</th><th>Δ vs baseline</th></tr></thead>"
                f"<tbody>{m_rows}</tbody></table></div>"
            )

    # ── Section 4: Security scan ──────────────────────────────────────────────
    if scan:
        status = scan.get("status", "unknown")
        scan_badge = _badge(status.upper(), "pass" if status == "clean" else "fail")
        critical = scan.get("critical", 0)
        high = scan.get("high", 0)
        findings = scan.get("findings", [])
        sev_key, unk = 'severity', 'unknown'
        f_rows = "".join(
            f"<tr>"
            f"<td class='finding-{_esc(f.get(sev_key, unk))}'>"
            f"{_esc(f.get('severity','').upper())}</td>"
            f"<td>{_esc(f.get('id',''))}</td>"
            f"<td>{_esc(f.get('title',''))}</td>"
            f"<td>{_esc(f.get('file',''))}</td>"
            f"</tr>"
            for f in findings
        )
        sections.append(
            f"<h2>Security Scan</h2>"
            f"<div class='card'>"
            f"<p style='margin-bottom:8px'>"
            f"Status: {scan_badge} &nbsp;"
            f"Critical: <strong>{_esc(critical)}</strong> &nbsp;"
            f"High: <strong>{_esc(high)}</strong></p>"
            + (
                f"<table><thead><tr><th>Severity</th><th>ID</th>"
                f"<th>Title</th><th>File</th></tr></thead>"
                f"<tbody>{f_rows}</tbody></table>"
                if findings else "<p>No findings.</p>"
            )
            + "</div>"
        )

    # ── Section 5: Policy results ─────────────────────────────────────────────
    if policies:
        p_rows = "".join(
            f"<tr>"
            f"<td>{_esc(name)}</td>"
            f"<td>{_badge('PASS', 'pass') if data.get('passed') else _badge('FAIL', 'fail')}</td>"
            f"<td>{_esc(data.get('error_count', 0))}</td>"
            f"<td>{_esc(data.get('warning_count', 0))}</td>"
            f"<td>{_esc(data.get('pass_count', 0))}</td>"
            f"</tr>"
            for name, data in policies.items()
        )
        sections.append(
            f"<h2>Policy Evaluation</h2>"
            f"<div class='card'><table>"
            f"<thead><tr><th>Policy</th><th>Result</th>"
            f"<th>Errors</th><th>Warnings</th><th>Passed</th></tr></thead>"
            f"<tbody>{p_rows}</tbody></table></div>"
        )

    # ── Section 6: VEX summary ────────────────────────────────────────────────
    if vex:
        affected = vex.get("affected_models", [])
        under_inv = vex.get("under_investigation", [])
        vex_badge = _badge("CLEAN", "pass") if not affected else _badge("AFFECTED", "fail")
        sections.append(
            f"<h2>VEX Vulnerability Status</h2>"
            f"<div class='card'>"
            f"<p>Status: {vex_badge} &nbsp; "
            f"Affected: <strong>{len(affected)}</strong> &nbsp; "
            f"Under investigation: <strong>{len(under_inv)}</strong></p>"
            f"</div>"
        )

    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Squash Compliance Report — {_esc(model_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">
  <h1>Squash Compliance Report &nbsp; {overall_badge}</h1>
  <p style="color:#666; margin-bottom:8px;">
    Model: <strong>{_esc(model_id)}</strong> &nbsp;|&nbsp; Generated: {_esc(generated_at)}
  </p>
  {body}
  <footer>Generated by <strong>Squash</strong> AI-SBOM attestation engine</footer>
</div>
</body>
</html>
"""
