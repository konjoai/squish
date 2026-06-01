"""squish/daemon/launchagent.py — macOS LaunchAgent plist management.

Installs/removes a LaunchAgent that auto-starts squishd at user login.

Usage (via CLI)
---------------
    squish daemon install [--model <model_dir>]   # installs plist + loads agent
    squish daemon uninstall                       # unloads agent + removes plist

The plist is written to ~/Library/LaunchAgents/ai.konjo.squishd.plist.

References
----------
https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

PLIST_LABEL = "ai.konjo.squishd"
PLIST_DIR   = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH  = PLIST_DIR / f"{PLIST_LABEL}.plist"


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def plist_content(
    squishd_bin: str,
    model_dir: str = "",
    compressed_dir: str = "",
    sock_path: str = "/tmp/squish.sock",
    max_models: int = 2,
    log_path: str = "",
) -> str:
    """Generate the LaunchAgent plist XML.

    Parameters
    ----------
    squishd_bin : str
        Full path to the ``squishd`` executable.
    model_dir : str
        Model directory to pre-load at daemon start.
    compressed_dir : str
        Compressed weights dir (optional).
    sock_path : str
        Unix socket path.
    max_models : int
        Max resident models.
    log_path : str
        Path for stdout/stderr logs (defaults to ~/.squish/daemon.log).
    """
    if not log_path:
        log_path = str(Path.home() / ".squish" / "daemon.log")

    args_list = _xml_args(squishd_bin, model_dir, compressed_dir, sock_path, max_models)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{PLIST_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
{args_list}
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{log_path}</string>

    <key>StandardErrorPath</key>
    <string>{log_path}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>SQUISH_SOCK</key>
        <string>{sock_path}</string>
        <key>SQUISH_MAX_MODELS</key>
        <string>{max_models}</string>
    </dict>
</dict>
</plist>
"""


def _xml_args(
    squishd_bin: str,
    model_dir: str,
    compressed_dir: str,
    sock_path: str,
    max_models: int,
) -> str:
    def _item(v: str) -> str:
        return f"        <string>{v}</string>"

    items = [_item(squishd_bin), _item("start"), _item("--foreground")]
    if model_dir:
        items.append(_item(model_dir))
        if compressed_dir:
            items += [_item("--compressed-dir"), _item(compressed_dir)]
    items += [_item("--sock"), _item(sock_path)]
    items += [_item("--max-models"), _item(str(max_models))]
    return "\n".join(items)


def install(
    model_dir: str = "",
    compressed_dir: str = "",
    sock_path: str = "/tmp/squish.sock",
    max_models: int = 2,
    squishd_bin: str = "",
) -> str:
    """Write the plist and register it with launchctl.

    Returns the path to the installed plist.
    Raises ``RuntimeError`` on non-macOS systems.
    Raises ``FileNotFoundError`` if squishd binary is not found.
    """
    if not _is_macos():
        raise RuntimeError(
            "LaunchAgent installation is macOS-only.  "
            "On Linux, use systemd --user or a cron @reboot entry."
        )

    # Resolve squishd binary
    if not squishd_bin:
        import shutil
        squishd_bin = shutil.which("squishd") or ""
    if not squishd_bin:
        # Derive from the current Python interpreter
        import sys
        squishd_bin = str(Path(sys.executable).parent / "squishd")
    if not Path(squishd_bin).exists():
        raise FileNotFoundError(
            f"squishd binary not found at {squishd_bin}.  "
            "Run 'pip install -e .' to create the entry point."
        )

    log_path = str(Path.home() / ".squish" / "daemon.log")
    content = plist_content(
        squishd_bin=squishd_bin,
        model_dir=model_dir,
        compressed_dir=compressed_dir,
        sock_path=sock_path,
        max_models=max_models,
        log_path=log_path,
    )

    PLIST_DIR.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_text(content)

    # Load the agent (non-fatal — user may not be in a launchctl-capable session)
    try:
        subprocess.run(
            ["launchctl", "load", "-w", str(PLIST_PATH)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        # Don't abort — the plist is installed; user can load manually
        import warnings
        warnings.warn(
            f"launchctl load returned non-zero: {exc.stderr.decode().strip()}",
            stacklevel=2,
        )

    return str(PLIST_PATH)


def uninstall() -> None:
    """Unload and remove the LaunchAgent plist.

    No-op if the plist does not exist.
    Raises ``RuntimeError`` on non-macOS systems.
    """
    if not _is_macos():
        raise RuntimeError("LaunchAgent management is macOS-only.")

    if PLIST_PATH.exists():
        try:
            subprocess.run(
                ["launchctl", "unload", "-w", str(PLIST_PATH)],
                check=False,
                capture_output=True,
            )
        except Exception:
            pass
        PLIST_PATH.unlink()


def is_installed() -> bool:
    """Return True if the plist file is present."""
    return PLIST_PATH.exists()


def status() -> dict:
    """Return installation + launchctl status."""
    installed = is_installed()
    loaded    = False
    if installed and _is_macos():
        try:
            out = subprocess.check_output(
                ["launchctl", "list", PLIST_LABEL],
                stderr=subprocess.DEVNULL,
            ).decode()
            loaded = bool(out.strip())
        except subprocess.CalledProcessError:
            loaded = False
    return {
        "installed": installed,
        "loaded":    loaded,
        "plist_path": str(PLIST_PATH) if installed else None,
    }
