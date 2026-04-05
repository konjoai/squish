"""squish.squash.integrations.ray — Ray Serve deployment decorator.

Wires Squash BOM attestation into the Ray Serve deployment lifecycle.
At :py:meth:`Deployment.bind` time the decorator:

1. Scans ``model_dir`` via :class:`~squish.squash.attest.AttestPipeline`.
2. Injects the BOM summary into the Ray Serve ``user_config`` under the key
   ``"squash_bom_summary"`` so downstream code (health-checks, metrics) can
   read it from :py:func:`ray.serve.get_deployment_status`.
3. Raises :class:`RuntimeError` if ``require_bom=True`` (default) and the
   scan fails.

Ray is an **optional** runtime dependency.  The decorator and mix-in class are
safe to *import* and *define* without Ray installed; they only require Ray at
``.bind()`` / ``.serve()`` call-time.

Usage — decorator::

    from ray import serve
    from squish.squash.integrations.ray import squash_serve

    @squash_serve(model_dir="models/my-llm", policy="eu-ai-act")
    @serve.deployment
    class MyLLM:
        async def __call__(self, request):
            ...

    app = MyLLM.bind()

Usage — mix-in base class::

    from ray import serve
    from squish.squash.integrations.ray import SquashServeDeployment

    @serve.deployment
    class MyLLM(SquashServeDeployment):
        _squash_model_dir = "models/my-llm"
        _squash_policy = "eu-ai-act"

        async def __call__(self, request):
            ...

    app = MyLLM.bind()
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SQUASH_METADATA_KEY = "squash_bom_summary"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SquashServeConfig:
    """Configuration for :func:`squash_serve` and :class:`SquashServeDeployment`."""

    model_dir: str | Path | None = None
    """Path to the model directory to validate at bind-time."""

    require_bom: bool = True
    """Raise :class:`RuntimeError` if BOM generation fails (default ``True``)."""

    policy: str | None = None
    """Policy name to enforce during bind-time validation (optional)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra key/value pairs merged into the Ray Serve ``user_config`` dict."""


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def squash_serve(
    cls: type | None = None,
    *,
    model_dir: str | Path | None = None,
    require_bom: bool = True,
    policy: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Decorate a Ray Serve deployment class with Squash BOM attestation.

    Can be used with or without arguments::

        @squash_serve                          # no args — uses class defaults
        @serve.deployment
        class MyModel: ...

        @squash_serve(model_dir="./models")    # with args
        @serve.deployment
        class MyModel: ...

    At :py:meth:`~ray.serve.Deployment.bind` time a BOM is generated for
    *model_dir* and merged into ``user_config["squash_bom_summary"]``.
    """
    config = SquashServeConfig(
        model_dir=model_dir,
        require_bom=require_bom,
        policy=policy,
        metadata=metadata or {},
    )

    def _decorate(deployment_cls: type) -> type:
        return _wrap_deployment(deployment_cls, config)

    if cls is not None:
        # Called as @squash_serve (no parentheses)
        return _decorate(cls)
    return _decorate


# ---------------------------------------------------------------------------
# Mix-in base class
# ---------------------------------------------------------------------------


class SquashServeDeployment:
    """Mix-in that auto-attests at ``.bind()`` time.

    Override class-level ``_squash_*`` attributes to configure::

        @serve.deployment
        class MyModel(SquashServeDeployment):
            _squash_model_dir = "models/my-llm"
            _squash_require_bom = True
            _squash_policy = "eu-ai-act"
    """

    _squash_model_dir: str | None = None
    _squash_require_bom: bool = True
    _squash_policy: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        config = SquashServeConfig(
            model_dir=cls._squash_model_dir,
            require_bom=cls._squash_require_bom,
            policy=cls._squash_policy,
        )
        _wrap_deployment(cls, config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wrap_deployment(deployment_cls: type, config: SquashServeConfig) -> type:
    """Patch ``.bind()`` on a Ray Serve Deployment class to run Squash first."""
    original_bind = getattr(deployment_cls, "bind", None)

    @functools.wraps(original_bind if original_bind is not None else _noop)
    def _patched_bind(*args: Any, **kwargs: Any) -> Any:
        bom_summary = _run_squash_validation(config)

        # Merge squash metadata into user_config
        user_config: dict[str, Any] = {}
        existing = kwargs.get("user_config")
        if isinstance(existing, dict):
            user_config.update(existing)
        user_config[_SQUASH_METADATA_KEY] = bom_summary
        user_config.update(config.metadata)
        kwargs["user_config"] = user_config

        if original_bind is not None:
            return original_bind(*args, **kwargs)
        return deployment_cls  # fallback: Ray not installed

    deployment_cls.bind = _patched_bind  # type: ignore[method-assign]
    return deployment_cls


def _run_squash_validation(config: SquashServeConfig) -> dict[str, Any]:
    """Run attestation pipeline and return a BOM summary dict.

    Returns ``{"validated": False, "reason": "..."}`` on non-fatal errors
    when ``config.require_bom`` is ``False``.  Raises :class:`RuntimeError`
    on failure when ``config.require_bom`` is ``True`` (default).
    """
    if config.model_dir is None:
        log.debug("squash_serve: no model_dir configured — skipping BOM validation")
        return {"validated": False, "reason": "no model_dir configured"}

    model_path = Path(config.model_dir)
    if not model_path.exists():
        msg = f"squash_serve: model_dir does not exist: {model_path}"
        if config.require_bom:
            raise RuntimeError(msg)
        log.warning(msg)
        return {"validated": False, "reason": "model_dir not found", "path": str(model_path)}

    try:
        from squish.squash.attest import AttestConfig, AttestPipeline  # noqa: PLC0415

        attest_config = AttestConfig(
            model_path=model_path,
            policies=[config.policy] if config.policy else [],
            fail_on_violation=False,  # gate is caller's responsibility
        )
        result = AttestPipeline.run(attest_config)
        summary: dict[str, Any] = {
            "validated": True,
            "model_dir": str(model_path),
            "summary": result.summary(),
        }
        if config.policy:
            summary["policy"] = config.policy
        log.info("squash_serve: BOM validated — %s", summary["summary"])
        return summary
    except Exception as exc:  # noqa: BLE001
        msg = f"squash_serve: attestation failed — {exc}"
        if config.require_bom:
            raise RuntimeError(msg) from exc
        log.warning(msg)
        return {"validated": False, "reason": str(exc)}


def _noop(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    """Placeholder used by functools.wraps when bind() is absent."""
