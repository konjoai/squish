"""squish.squash.integrations.langchain — LangChain callback for Squash.

Attests the underlying model whenever a LangChain :class:`~langchain.llms.base.LLM`
or :class:`~langchain.chat_models.base.BaseChatModel` is first loaded, then
re-evaluates policy on every generation if ``continuous_audit=True``.

Usage::

    from langchain_community.llms import LlamaCpp
    from squish.squash.integrations.langchain import SquashCallback

    callback = SquashCallback(
        model_path=Path("./llama-3.1-8b.gguf"),
        policies=["eu-ai-act"],
        fail_on_violation=True,
    )

    llm = LlamaCpp(
        model_path="./llama-3.1-8b.gguf",
        callbacks=[callback],
    )
    # First call triggers attestation; subsequent calls are no-ops unless
    # continuous_audit=True.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from squish.squash.attest import AttestConfig, AttestPipeline, AttestResult

log = logging.getLogger(__name__)


class SquashCallback:
    """LangChain callback that attests a model on first use.

    Implements the minimal LangChain callback interface (``on_llm_start``,
    ``on_chat_model_start``) without inheriting from LangChain base classes so
    that squish does not take a hard dependency on the LangChain SDK version.
    When LangChain *is* installed, the callback duck-types as a valid
    :class:`~langchain.callbacks.base.BaseCallbackHandler`.

    Parameters
    ----------
    model_path:
        Path to the model file or directory.
    policies:
        Policies to evaluate on attestation.
    fail_on_violation:
        If ``True``, raises :class:`~squish.squash.attest.AttestationViolationError`
        on policy failure.
    continuous_audit:
        If ``True``, re-runs policy evaluation on every ``on_llm_end`` call
        (expensive; intended for compliance-sensitive pipelines).
    """

    def __init__(
        self,
        model_path: Path,
        *,
        policies: list[str] | None = None,
        fail_on_violation: bool = True,
        continuous_audit: bool = False,
        **attest_kwargs,
    ) -> None:
        self._model_path = model_path
        self._policies = policies or ["enterprise-strict"]
        self._fail_on_violation = fail_on_violation
        self._continuous_audit = continuous_audit
        self._attest_kwargs = attest_kwargs
        self._result: AttestResult | None = None
        self._attested = False

    # ── LangChain callback interface (duck-typed) ─────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Attest on first LLM invocation."""
        self._maybe_attest()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        **kwargs: Any,
    ) -> None:
        """Attest on first chat model invocation."""
        self._maybe_attest()

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Re-attest after generation if continuous_audit is enabled."""
        if self._continuous_audit and self._attested:
            log.debug("Squash continuous audit: re-evaluating policy …")
            self._run_attestation()

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:  # noqa: D401
        """No-op — let LangChain handle LLM errors normally."""

    # ── Property access ───────────────────────────────────────────────────

    @property
    def last_result(self) -> AttestResult | None:
        """The most recent :class:`~squish.squash.attest.AttestResult`, or None."""
        return self._result

    # ── Private ───────────────────────────────────────────────────────────

    def _maybe_attest(self) -> None:
        if not self._attested:
            self._run_attestation()
            self._attested = True

    def _run_attestation(self) -> None:
        config = AttestConfig(
            model_path=self._model_path,
            policies=self._policies,
            fail_on_violation=self._fail_on_violation,
            **self._attest_kwargs,
        )
        self._result = AttestPipeline.run(config)
        log.info(
            "SquashCallback: attestation %s for %s",
            "passed" if self._result.passed else "FAILED",
            self._model_path,
        )


# ── Wave 46 — SquashAuditCallback ─────────────────────────────────────────────

import time as _time  # noqa: E402


class SquashAuditCallback(SquashCallback):
    """LangChain callback that attests *and* writes an audit trail.

    Extends :class:`SquashCallback` by routing every LLM invocation through
    :class:`~squish.squash.governor.AgentAuditLogger`.  Each ``llm_start``
    and ``llm_end`` event is written as an :class:`~squish.squash.governor.AuditEntry`
    with SHA-256 hashes of the prompt / response payload and the measured
    first-token latency.

    The logger defaults to the process-level singleton but callers may supply
    their own ``AgentAuditLogger`` instance for test isolation or custom log
    paths::

        from squish.squash.governor import AgentAuditLogger
        from squish.squash.integrations.langchain import SquashAuditCallback

        logger = AgentAuditLogger(log_path="/var/log/squash/audit.jsonl")
        callback = SquashAuditCallback(
            model_path=Path("./qwen3-8b-q4"),
            session_id="req-abc-123",
            audit_logger=logger,
        )
        llm = LlamaCpp(model_path="...", callbacks=[callback])

    Parameters
    ----------
    session_id:
        A stable identifier for this conversation / request.  Defaults to an
        empty string if not provided.
    audit_logger:
        Supply a custom :class:`~squish.squash.governor.AgentAuditLogger`
        instance.  If *None*, the process-level singleton is used.
    All other parameters are forwarded to :class:`SquashCallback`.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        session_id: str = "",
        audit_logger: "Any | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(model_path, **kwargs)
        self._session_id = session_id
        self._audit_logger = audit_logger
        self._start_ts: float = 0.0

    def _get_logger(self):
        if self._audit_logger is not None:
            return self._audit_logger
        from squish.squash.governor import get_audit_logger, _hash_text  # noqa: F401
        return get_audit_logger()

    # ── LangChain callback overrides ──────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Attest (first call) and log llm_start with hashed prompt."""
        self._start_ts = _time.monotonic()
        self._maybe_attest()
        try:
            from squish.squash.governor import _hash_text
            combined = "\n".join(prompts)
            self._get_logger().append(
                session_id=self._session_id,
                event_type="llm_start",
                model_id=str(self._model_path),
                input_hash=_hash_text(combined),
                metadata={"prompt_count": len(prompts)},
            )
        except Exception as exc:
            log.debug("SquashAuditCallback: llm_start log failed: %s", exc)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        **kwargs: Any,
    ) -> None:
        """Attest (first call) and log llm_start for chat model."""
        self._start_ts = _time.monotonic()
        self._maybe_attest()
        try:
            from squish.squash.governor import _hash_text
            combined = str(messages)
            self._get_logger().append(
                session_id=self._session_id,
                event_type="llm_start",
                model_id=str(self._model_path),
                input_hash=_hash_text(combined),
                metadata={"message_count": len(messages)},
            )
        except Exception as exc:
            log.debug("SquashAuditCallback: chat_start log failed: %s", exc)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log llm_end with hashed output and measured latency."""
        latency_ms = (_time.monotonic() - self._start_ts) * 1000 if self._start_ts else -1.0
        if self._continuous_audit and self._attested:
            log.debug("Squash continuous audit: re-evaluating policy …")
            self._run_attestation()
        try:
            from squish.squash.governor import _hash_text
            output_text = str(getattr(response, "generations", response))
            self._get_logger().append(
                session_id=self._session_id,
                event_type="llm_end",
                model_id=str(self._model_path),
                output_hash=_hash_text(output_text),
                latency_ms=round(latency_ms, 2),
                metadata={
                    "attestation_passed": (
                        self._result.passed if self._result else None
                    ),
                },
            )
        except Exception as exc:
            log.debug("SquashAuditCallback: llm_end log failed: %s", exc)
