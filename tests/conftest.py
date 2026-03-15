"""
tests/conftest.py
Shared pytest configuration for all Squish tests.
"""
import warnings

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model", default=None,
        help="Model hint passed to squish --model  (e.g. '14b', '7b', full path)"
    )
    parser.addoption(
        "--run-hardware", action="store_true", default=False,
        help="Run @pytest.mark.hardware tests (requires Apple Silicon + MLX + a cached model)"
    )
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="Run @pytest.mark.integration tests (requires a live squish server at localhost:11435)"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hardware: require Apple Silicon + MLX + a cached model (skipped in CI unless --run-hardware is set)",
    )
    config.addinivalue_line(
        "markers",
        "integration: requires live squish serve (skipped unless --run-integration or -m integration)",
    )
    # SwigPy warnings from compiled MLX/Metal bindings — not actionable
    warnings.filterwarnings("ignore", message="builtin type SwigPy.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="builtin type swigvar.*", category=DeprecationWarning)
    # Starlette TestClient timeout kwarg deprecation — upstream issue
    warnings.filterwarnings("ignore", message="You should not use the 'timeout'.*", category=DeprecationWarning)


def pytest_collection_modifyitems(config, items):
    """Skip hardware and integration tests unless the corresponding flag is set."""
    run_hardware = config.getoption("--run-hardware")
    run_integration = config.getoption("--run-integration")
    # Also allow -m integration to opt in without the flag.
    mark_expr = getattr(config.option, "markexpr", "") or ""
    m_integration = "integration" in mark_expr

    skip_hw = pytest.mark.skip(reason="hardware test — pass --run-hardware to enable")
    skip_int = pytest.mark.skip(reason="integration test — pass --run-integration to enable")

    for item in items:
        if not run_hardware and item.get_closest_marker("hardware"):
            item.add_marker(skip_hw)
        if not run_integration and not m_integration and item.get_closest_marker("integration"):
            item.add_marker(skip_int)
