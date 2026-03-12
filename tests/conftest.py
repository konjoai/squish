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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hardware: require Apple Silicon + MLX + a cached model (skipped in CI unless --run-hardware is set)",
    )
    # SwigPy warnings from compiled MLX/Metal bindings — not actionable
    warnings.filterwarnings("ignore", message="builtin type SwigPy.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message="builtin type swigvar.*", category=DeprecationWarning)
    # Starlette TestClient timeout kwarg deprecation — upstream issue
    warnings.filterwarnings("ignore", message="You should not use the 'timeout'.*", category=DeprecationWarning)


def pytest_collection_modifyitems(config, items):
    """Skip hardware-marked tests unless --run-hardware is explicitly passed."""
    if config.getoption("--run-hardware"):
        return  # run everything
    skip_hw = pytest.mark.skip(reason="hardware test — pass --run-hardware to enable")
    for item in items:
        if item.get_closest_marker("hardware"):
            item.add_marker(skip_hw)
