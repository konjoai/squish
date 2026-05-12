"""squish/serving/router.py

W110 — Prompt Router: rule-based prompt classification and model routing.

When squish receives a query the router classifies the task type and
recommends which locally-available model family to serve it.  All logic
is pure Python — no ML, no model loading, no network calls.

Public API
──────────
    RouterCategory   — StrEnum of task categories
    RouterRule       — single named regex rule with priority and model hint
    RouterDecision   — frozen routing result (category + confidence + hint)
    RouterConfig     — configuration bundle (rules + fallback behaviour)
    PromptRouter     — main classifier; call .route(prompt) → RouterDecision
    get_default_router() → PromptRouter singleton factory

Algorithm
─────────
1. Validate prompt (empty → UNKNOWN, confidence 0.0).
2. Evaluate custom then default rules in descending priority order;
   first match wins → confidence 1.0.
3. If no rule matched and enable_heuristics=True: count keyword hits per
   category; highest count > 0 → confidence 0.5; all zero → UNKNOWN 0.0.
4. Return RouterDecision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import auto
from typing import Optional

# Python 3.10 ships StrEnum; guard for 3.10 where it lives in enum.
try:
    from enum import StrEnum
except ImportError:  # pragma: no cover — Python < 3.11 fallback
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Minimal StrEnum shim for Python < 3.11."""

        @staticmethod
        def _generate_next_value_(name: str, *_args: object) -> str:
            return name.lower()


__all__ = [
    "RouterCategory",
    "RouterRule",
    "RouterDecision",
    "RouterConfig",
    "PromptRouter",
    "get_default_router",
]

_MAX_REASONING_LEN = 120


class RouterCategory(StrEnum):
    """Taxonomy of prompt task types used for routing decisions."""

    CODE        = auto()
    MATH        = auto()
    CREATIVE    = auto()
    FACTUAL     = auto()
    CONVERSATION = auto()
    UNKNOWN     = auto()


# ── Keyword tables for heuristic fallback ─────────────────────────────────────
# Each entry: (category, frozenset-of-keywords).
# Evaluated only when no rule matches and enable_heuristics is True.
_HEURISTIC_KEYWORDS: list[tuple[RouterCategory, frozenset[str]]] = [
    (RouterCategory.CODE,       frozenset({"def", "class", "function", "import", "algorithm"})),
    (RouterCategory.MATH,       frozenset({"equation", "formula", "calculate", "integral", "matrix"})),
    (RouterCategory.CREATIVE,   frozenset({"write", "story", "poem", "imagine", "creative"})),
    (RouterCategory.FACTUAL,    frozenset({"what", "when", "who", "where", "why", "how"})),
]


@dataclass
class RouterRule:
    """A single named routing rule.

    Attributes
    ----------
    name:
        Unique human-readable identifier for this rule.
    category:
        The RouterCategory this rule maps to on a match.
    pattern:
        Compiled regex applied to the *lowercased* prompt.
    priority:
        Higher values are checked first; ties broken by list order.
    model_hint:
        Suggested model family string (e.g. ``"codellama"``), or ``None``
        when no specific model is preferred.
    """

    name:       str
    category:   RouterCategory
    pattern:    re.Pattern  # type: ignore[type-arg]
    priority:   int
    model_hint: Optional[str] = None


@dataclass(frozen=True)
class RouterDecision:
    """Immutable routing result returned by :meth:`PromptRouter.route`.

    Attributes
    ----------
    category:
        Classified task type.
    matched_rule:
        Name of the rule that fired, or ``None`` if heuristics/fallback decided.
    model_hint:
        Suggested model family, or ``None`` when no preference.
    confidence:
        ``1.0`` — rule matched; ``0.5`` — heuristic matched; ``0.0`` — unknown.
    reasoning:
        Human-readable explanation (≤ 120 chars).
    """

    category:     RouterCategory
    matched_rule: Optional[str]
    model_hint:   Optional[str]
    confidence:   float
    reasoning:    str

    def asdict(self) -> dict:
        """Return a plain-dict representation of this decision."""
        return {
            "category":     str(self.category),
            "matched_rule": self.matched_rule,
            "model_hint":   self.model_hint,
            "confidence":   self.confidence,
            "reasoning":    self.reasoning,
        }


@dataclass
class RouterConfig:
    """Configuration bundle for :class:`PromptRouter`.

    Attributes
    ----------
    rules:
        Custom rules evaluated *before* the built-in defaults.
    fallback_category:
        Category returned when no rule or heuristic fires
        (default: ``CONVERSATION``).
    enable_heuristics:
        When ``True`` (default), a keyword-count heuristic is tried after
        all rules fail before returning the fallback category.
    """

    rules:             list[RouterRule] = field(default_factory=list)
    fallback_category: RouterCategory  = RouterCategory.CONVERSATION
    enable_heuristics: bool            = True


def _compile(pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:  # type: ignore[type-arg]
    """Compile *pattern* with *flags*; raise ValueError on invalid syntax."""
    return re.compile(pattern, flags)


def _truncate(text: str, max_len: int = _MAX_REASONING_LEN) -> str:
    """Return *text* truncated to *max_len* chars (ellipsis if cut)."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


class PromptRouter:
    """Rule-based prompt classifier and model router.

    Parameters
    ----------
    config:
        Optional configuration; when ``None`` the default rule set is used.

    Usage::

        router = PromptRouter()
        decision = router.route("Write a Python function that sorts a list")
        print(decision.category)     # RouterCategory.CODE
        print(decision.confidence)   # 1.0
    """

    def __init__(self, config: Optional[RouterConfig] = None) -> None:
        """Initialise with *config* or the built-in default rule set."""
        self._config = config if config is not None else RouterConfig()
        default_rules = _default_rules()
        # Custom rules are evaluated before defaults (by priority merge).
        self._rules: list[RouterRule] = sorted(
            self._config.rules + default_rules,
            key=lambda r: r.priority,
            reverse=True,
        )

    def route(self, prompt: str) -> RouterDecision:
        """Classify *prompt* and return a :class:`RouterDecision`.

        Steps
        -----
        1. Empty prompt → UNKNOWN, confidence 0.0.
        2. Try rules in descending priority; first match wins (confidence 1.0).
        3. If heuristics enabled, count keyword hits per category;
           highest > 0 → confidence 0.5.
        4. Return UNKNOWN or fallback_category decision.
        """
        if not prompt or not prompt.strip():
            return RouterDecision(
                category=RouterCategory.UNKNOWN,
                matched_rule=None,
                model_hint=None,
                confidence=0.0,
                reasoning="empty prompt",
            )

        lowered = prompt.lower()

        for rule in self._rules:
            if rule.pattern.search(lowered):
                return RouterDecision(
                    category=rule.category,
                    matched_rule=rule.name,
                    model_hint=rule.model_hint,
                    confidence=1.0,
                    reasoning=_truncate(f"rule '{rule.name}' matched"),
                )

        if self._config.enable_heuristics:
            return self._heuristic_decision(lowered)

        return RouterDecision(
            category=self._config.fallback_category,
            matched_rule=None,
            model_hint=None,
            confidence=0.5,
            reasoning=_truncate(
                f"no rule matched; fallback to {self._config.fallback_category}"
            ),
        )

    def explain(self, prompt: str) -> dict:
        """Return routing metadata for *prompt* — useful for debug/introspection.

        Keys
        ----
        All keys from :meth:`RouterDecision.asdict` plus:

        ``prompt_length``
            Character count of the original prompt string.
        ``n_rules_checked``
            Number of rules evaluated (always the full rule list length).
        """
        decision = self.route(prompt)
        result = decision.asdict()
        result["prompt_length"] = len(prompt)
        result["n_rules_checked"] = len(self._rules)
        return result

    def _heuristic_decision(self, lowered: str) -> RouterDecision:
        """Apply keyword heuristics to *lowered* prompt; return RouterDecision."""
        words = set(re.findall(r"\b\w+\b", lowered))
        best_cat: Optional[RouterCategory] = None
        best_count = 0

        for category, keywords in _HEURISTIC_KEYWORDS:
            count = len(words & keywords)
            if count > best_count:
                best_count = count
                best_cat = category

        if best_cat is not None and best_count > 0:
            return RouterDecision(
                category=best_cat,
                matched_rule=None,
                model_hint=None,
                confidence=0.5,
                reasoning=_truncate(
                    f"heuristic: {best_count} {best_cat} keyword(s) matched"
                ),
            )

        return RouterDecision(
            category=RouterCategory.UNKNOWN,
            matched_rule=None,
            model_hint=None,
            confidence=0.0,
            reasoning="no rule or heuristic matched",
        )


def _default_rules() -> list[RouterRule]:
    """Return the built-in rule set, sorted descending by priority.

    Rules
    -----
    1. python-code       — Python def/class/import patterns      (priority 90)
    2. general-code      — generic function/algorithm/language    (priority 85)
    3. math-equation     — equations, formulae, integrals         (priority 80)
    4. math-calc         — explicit calculate/compute keywords     (priority 75)
    5. creative-story    — story / poem / imagine / fictional      (priority 70)
    6. creative-write    — "write me" / "draft" / composition      (priority 65)
    7. factual-qa        — wh-questions about facts/history        (priority 60)
    8. factual-explain   — explain / describe / summarise          (priority 55)
    """
    rules = [
        RouterRule(
            name="python-code",
            category=RouterCategory.CODE,
            pattern=_compile(
                r"\b(def\s+\w+|class\s+\w+|import\s+\w+|from\s+\w+\s+import"
                r"|print\(|for\s+\w+\s+in\b)"
            ),
            priority=90,
            model_hint="codellama",
        ),
        RouterRule(
            name="general-code",
            category=RouterCategory.CODE,
            pattern=_compile(
                r"\b(function|algorithm|implement|refactor|debug|bug|snippet"
                r"|compile|syntax|variable|loop|recursion|api\s+endpoint)\b"
            ),
            priority=85,
            model_hint="codellama",
        ),
        RouterRule(
            name="math-equation",
            category=RouterCategory.MATH,
            pattern=_compile(
                r"(\b(equation|formula|integral|derivative|matrix|eigenvalue"
                r"|polynomial|differential|theorem|proof)\b"
                r"|[=+\-*/^]{2,}|\d+\s*[=+\-*/^]\s*\d)"
            ),
            priority=80,
            model_hint="qwen2.5-math",
        ),
        RouterRule(
            name="math-calc",
            category=RouterCategory.MATH,
            pattern=_compile(
                r"\b(calculat|comput|solv|evaluat|simplif|factori[sz]e"
                r"|probability|statistic|vector|tensor)\b",
            ),
            priority=75,
            model_hint="qwen2.5-math",
        ),
        RouterRule(
            name="creative-story",
            category=RouterCategory.CREATIVE,
            pattern=_compile(
                r"\b(story|poem|fiction|novel|narrative|imagine|fantasy"
                r"|character|plot|dialogue|screenplay|haiku|sonnet)\b"
            ),
            priority=70,
            model_hint=None,
        ),
        RouterRule(
            name="creative-write",
            category=RouterCategory.CREATIVE,
            pattern=_compile(
                r"\b(write\s+(me\s+)?(a|an|some)|draft|compose|create\s+(a|an)"
                r"|generate\s+(a|an)\s+(poem|essay|blog|song|letter))\b"
            ),
            priority=65,
            model_hint=None,
        ),
        RouterRule(
            name="factual-qa",
            category=RouterCategory.FACTUAL,
            pattern=_compile(
                r"^(what|when|who|where|why|how|which|whose|whom)\b"
                r"|(history|capital|president|inventor|discovery|founded|born|died)\b"
            ),
            priority=60,
            model_hint=None,
        ),
        RouterRule(
            name="factual-explain",
            category=RouterCategory.FACTUAL,
            pattern=_compile(
                r"\b(explain|describe|summari[sz]e|tell\s+me\s+about"
                r"|definition\s+of|overview\s+of|what\s+is\s+the)\b"
            ),
            priority=55,
            model_hint=None,
        ),
    ]
    return sorted(rules, key=lambda r: r.priority, reverse=True)


_default_router_singleton: Optional[PromptRouter] = None


def get_default_router() -> PromptRouter:
    """Return the module-level default :class:`PromptRouter` singleton.

    The singleton is created on first call and reused thereafter.
    Thread-safety note: concurrent first-calls may create two instances;
    only one will be stored.  Both are functionally identical.
    """
    global _default_router_singleton
    if _default_router_singleton is None:
        _default_router_singleton = PromptRouter()
    return _default_router_singleton
