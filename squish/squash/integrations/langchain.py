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
