"""squish/install/launch_preflight.py — Launch Preflight Checks.

Wave 72: validates the runtime environment before starting the Squish server,
giving users clear, actionable diagnostics instead of opaque startup errors.

Checks performed:

1. Python ≥ 3.10
2. MLX importable (Apple Silicon ML framework)
3. Metal GPU accessible via MLX
4. Available disk space ≥ 2 GiB
5. Available RAM ≥ 4 GiB
6. Write permission on the model cache directory
7. Target server port not already in use

Usage::

    from squish.install.launch_preflight import run_preflight_checks, format_report

    report = run_preflight_checks()
    print(format_report(report))
    if report.failed:
        raise SystemExit(1)
"""

from __future__ import annotations

import enum
import os
import shutil
import socket
import sys
from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "CheckStatus",
    "PreflightCheck",
    "PreflightReport",
    "run_preflight_checks",
    "format_report",
]

# Minimum requirements
_MIN_PYTHON = (3, 10)
_MIN_DISK_GIB = 2.0
_MIN_MEMORY_GIB = 4.0
_DEFAULT_SERVER_PORT = 8080
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/squish")


class CheckStatus(enum.Enum):
    """Result of a single preflight check."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"


@dataclass
class PreflightCheck:
    """Outcome of one preflight check.

    Attributes:
        name: Short human-readable check name.
        status: :class:`CheckStatus` result.
        message: One-line summary (present after any status).
        detail: Optional extended detail for ``WARN`` or ``ERROR`` checks.
    """

    name: str
    status: CheckStatus
    message: str
    detail: Optional[str] = None


@dataclass
class PreflightReport:
    """Aggregated results from :func:`run_preflight_checks`.

    Attributes:
        checks: All check results in order.
        passed: Count of ``OK`` checks.
        warned: Count of ``WARN`` checks.
        failed: Count of ``ERROR`` checks.
    """

    checks: list[PreflightCheck] = field(default_factory=list)
    passed: int = 0
    warned: int = 0
    failed: int = 0

    @property
    def ok(self) -> bool:
        """``True`` when there are no ERROR checks."""
        return self.failed == 0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_python_version() -> PreflightCheck:
    """Verify Python ≥ 3.10."""
    current = sys.version_info[:2]
    if current >= _MIN_PYTHON:
        return PreflightCheck(
            name="Python version",
            status=CheckStatus.OK,
            message=f"Python {current[0]}.{current[1]} ✓",
        )
    return PreflightCheck(
        name="Python version",
        status=CheckStatus.ERROR,
        message=f"Python {current[0]}.{current[1]} is too old",
        detail=(
            f"Squish requires Python ≥ {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}. "
            f"Install a newer Python from python.org or via `brew install python`."
        ),
    )


def _check_mlx_available() -> PreflightCheck:
    """Verify that the MLX package can be imported."""
    try:
        import mlx.core as mx  # noqa: F401
        return PreflightCheck(
            name="MLX available",
            status=CheckStatus.OK,
            message="MLX importable ✓",
        )
    except ImportError:
        return PreflightCheck(
            name="MLX available",
            status=CheckStatus.ERROR,
            message="MLX not found",
            detail=(
                "Install MLX with: pip install mlx\n"
                "MLX requires an Apple Silicon Mac (M1/M2/M3/M4)."
            ),
        )


def _check_metal_gpu() -> PreflightCheck:
    """Verify that MLX can access Metal (Apple Silicon GPU)."""
    try:
        import mlx.core as mx
        # metal_is_available() returns True on supported hardware
        if hasattr(mx, "metal") and hasattr(mx.metal, "is_available"):
            available = mx.metal.is_available()
        else:
            # Older MLX versions — try creating a small array and checking device
            a = mx.array([1.0])
            mx.eval(a)
            available = True

        if available:
            return PreflightCheck(
                name="Metal GPU",
                status=CheckStatus.OK,
                message="Metal GPU accessible ✓",
            )
        return PreflightCheck(
            name="Metal GPU",
            status=CheckStatus.WARN,
            message="Metal GPU not available — CPU fallback will be used",
            detail=(
                "Squish runs on CPU but inference will be significantly slower. "
                "An Apple Silicon Mac (M1 or later) is recommended."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return PreflightCheck(
            name="Metal GPU",
            status=CheckStatus.WARN,
            message=f"Could not verify Metal: {exc}",
            detail="This is a non-critical warning; CPU inference may still work.",
        )


def _check_disk_space(
    path: str = _DEFAULT_CACHE_DIR,
    min_gib: float = _MIN_DISK_GIB,
) -> PreflightCheck:
    """Verify ≥ *min_gib* GiB of free disk space at *path*."""
    # Ensure the directory exists for shutil.disk_usage
    os.makedirs(path, exist_ok=True)
    try:
        usage = shutil.disk_usage(path)
        free_gib = usage.free / (1024 ** 3)
    except OSError as exc:
        return PreflightCheck(
            name="Disk space",
            status=CheckStatus.WARN,
            message=f"Could not check disk space: {exc}",
        )

    if free_gib >= min_gib:
        return PreflightCheck(
            name="Disk space",
            status=CheckStatus.OK,
            message=f"{free_gib:.1f} GiB free ✓",
        )
    if free_gib >= 0.5:
        return PreflightCheck(
            name="Disk space",
            status=CheckStatus.WARN,
            message=f"Only {free_gib:.1f} GiB free (recommended: ≥ {min_gib:.0f} GiB)",
            detail="Model files may not fit. Free up disk space or use a smaller model.",
        )
    return PreflightCheck(
        name="Disk space",
        status=CheckStatus.ERROR,
        message=f"Critically low disk space: {free_gib:.2f} GiB",
        detail=(
            f"At least {min_gib:.0f} GiB is required. "
            "Free up disk space before continuing."
        ),
    )


def _check_memory(min_gib: float = _MIN_MEMORY_GIB) -> PreflightCheck:
    """Verify ≥ *min_gib* GiB of total system memory.

    Reads from ``/proc/meminfo`` on Linux or uses ``sysctl`` on macOS.
    Returns a ``WARN`` (not ``ERROR``) because shared GPU/CPU memory on Apple
    Silicon cannot be reliably measured the same way.
    """
    total_gib: Optional[float] = None

    # macOS
    if sys.platform == "darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                total_gib = int(result.stdout.strip()) / (1024 ** 3)
        except Exception:  # noqa: BLE001
            pass

    # Linux
    if total_gib is None and os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        total_gib = kb / (1024 ** 2)
                        break
        except Exception:  # noqa: BLE001
            pass

    if total_gib is None:
        return PreflightCheck(
            name="System memory",
            status=CheckStatus.WARN,
            message="Could not determine system memory",
        )

    if total_gib >= min_gib:
        return PreflightCheck(
            name="System memory",
            status=CheckStatus.OK,
            message=f"{total_gib:.0f} GiB total ✓",
        )
    return PreflightCheck(
        name="System memory",
        status=CheckStatus.WARN,
        message=f"Only {total_gib:.1f} GiB RAM (recommended: ≥ {min_gib:.0f} GiB)",
        detail=(
            "Smaller models (≤ 4B parameters) may still work. "
            "Expect slower inference or out-of-memory errors with larger models."
        ),
    )


def _check_write_permission(path: str = _DEFAULT_CACHE_DIR) -> PreflightCheck:
    """Verify the model cache directory is writable."""
    os.makedirs(path, exist_ok=True)
    test_file = os.path.join(path, ".squish_preflight_write_test")
    try:
        with open(test_file, "w") as fh:
            fh.write("ok")
        os.remove(test_file)
        return PreflightCheck(
            name="Cache directory writable",
            status=CheckStatus.OK,
            message=f"{path} is writable ✓",
        )
    except OSError as exc:
        return PreflightCheck(
            name="Cache directory writable",
            status=CheckStatus.ERROR,
            message=f"Cannot write to {path}",
            detail=(
                f"Error: {exc}\n"
                f"Fix permissions with: chmod u+w {path}"
            ),
        )


def _check_server_port_free(port: int = _DEFAULT_SERVER_PORT) -> PreflightCheck:
    """Verify the target server port is not already bound."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            result = sock.connect_ex(("127.0.0.1", port))
        if result != 0:
            return PreflightCheck(
                name=f"Port {port} free",
                status=CheckStatus.OK,
                message=f"Port {port} is available ✓",
            )
        return PreflightCheck(
            name=f"Port {port} free",
            status=CheckStatus.WARN,
            message=f"Port {port} is already in use",
            detail=(
                f"Another process is listening on port {port}. "
                f"Start Squish on a different port with --port, "
                f"or find the conflicting process with: lsof -i :{port}"
            ),
        )
    except OSError as exc:
        return PreflightCheck(
            name=f"Port {port} free",
            status=CheckStatus.WARN,
            message=f"Could not test port {port}: {exc}",
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def run_preflight_checks(
    port: int = _DEFAULT_SERVER_PORT,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> PreflightReport:
    """Run all preflight checks and return an aggregated report.

    Args:
        port: Server port to test availability for.
        cache_dir: Model cache directory to check for disk space and permissions.

    Returns:
        :class:`PreflightReport` with all individual results and summary counts.
    """
    checks = [
        _check_python_version(),
        _check_mlx_available(),
        _check_metal_gpu(),
        _check_disk_space(path=cache_dir),
        _check_memory(),
        _check_write_permission(path=cache_dir),
        _check_server_port_free(port=port),
    ]
    passed = sum(1 for c in checks if c.status == CheckStatus.OK)
    warned = sum(1 for c in checks if c.status == CheckStatus.WARN)
    failed = sum(1 for c in checks if c.status == CheckStatus.ERROR)
    return PreflightReport(
        checks=checks,
        passed=passed,
        warned=warned,
        failed=failed,
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_ANSI_GREEN = "\033[92m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RED = "\033[91m"
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"

_STATUS_ICON = {
    CheckStatus.OK: "✓",
    CheckStatus.WARN: "⚠",
    CheckStatus.ERROR: "✗",
}
_STATUS_COLOR = {
    CheckStatus.OK: _ANSI_GREEN,
    CheckStatus.WARN: _ANSI_YELLOW,
    CheckStatus.ERROR: _ANSI_RED,
}


def format_report(report: PreflightReport, *, color: bool = True) -> str:
    """Render a preflight report as a human-readable string.

    Args:
        report: The :class:`PreflightReport` to format.
        color: If ``True`` (default), include ANSI colour codes.

    Returns:
        Multi-line string suitable for printing to a terminal.
    """
    lines: list[str] = [
        f"{_ANSI_BOLD}Squish Preflight Checks{_ANSI_RESET}" if color
        else "Squish Preflight Checks",
        "─" * 50,
    ]

    for check in report.checks:
        icon = _STATUS_ICON[check.status]
        msg = check.message
        if color:
            c = _STATUS_COLOR[check.status]
            icon_str = f"{c}{icon}{_ANSI_RESET}"
            name_str = f"{_ANSI_BOLD}{check.name:<30}{_ANSI_RESET}"
        else:
            icon_str = icon
            name_str = f"{check.name:<30}"
        lines.append(f"  {icon_str}  {name_str}  {msg}")
        if check.detail:
            for detail_line in check.detail.splitlines():
                lines.append(f"       {detail_line}")

    lines.append("─" * 50)
    summary_parts = [f"{report.passed} passed"]
    if report.warned:
        w = (
            f"{_ANSI_YELLOW}{report.warned} warnings{_ANSI_RESET}"
            if color else f"{report.warned} warnings"
        )
        summary_parts.append(w)
    if report.failed:
        f_str = (
            f"{_ANSI_RED}{report.failed} errors{_ANSI_RESET}"
            if color else f"{report.failed} errors"
        )
        summary_parts.append(f_str)
    lines.append("  " + " · ".join(summary_parts))

    if report.failed == 0 and report.warned == 0:
        ready = (
            f"  {_ANSI_GREEN}{_ANSI_BOLD}All checks passed — Squish is ready!{_ANSI_RESET}"
            if color else "  All checks passed — Squish is ready!"
        )
        lines.append(ready)
    elif report.failed > 0:
        fail_msg = (
            f"  {_ANSI_RED}{_ANSI_BOLD}Fix the errors above before starting Squish.{_ANSI_RESET}"
            if color else "  Fix the errors above before starting Squish."
        )
        lines.append(fail_msg)

    return "\n".join(lines)
