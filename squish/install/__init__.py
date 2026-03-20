"""squish/install — Platform-aware dependency resolution and installation.

This package provides smart pip extras selection, install validation,
and environment health checks for cross-platform squish deployment.
"""

from squish.install.dependency_resolver import (
    DependencyGroup,
    DependencyResolver,
    DependencyResolverConfig,
    DependencyResolverStats,
    InstallSpec,
)

__all__ = [
    "DependencyGroup",
    "DependencyResolver",
    "DependencyResolverConfig",
    "DependencyResolverStats",
    "InstallSpec",
]
