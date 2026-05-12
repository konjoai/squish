"""tests/test_router.py — W110: Prompt Router.

25 tests covering:
- Default rules existence
- Empty prompt → UNKNOWN
- Category matching for CODE / MATH / CREATIVE / FACTUAL
- Rule-name propagation on match
- Heuristic fallback (no rule match)
- Heuristic confidence = 0.5
- Heuristics disabled → fallback category
- UNKNOWN confidence = 0.0
- Rule match confidence = 1.0
- Case-insensitive matching
- model_hint propagation
- Priority ordering (higher-priority rule wins)
- Custom rule overrides default
- RouterDecision type check
- RouterDecision frozen (mutation raises)
- Custom fallback category
- Empty rules list uses heuristics
- explain() dict keys (category, matched_rule, model_hint, confidence, reasoning)
- explain() prompt_length
- explain() n_rules_checked
- CLI: build_parser() includes 'route' subcommand
- get_default_router() returns PromptRouter instance
"""
from __future__ import annotations

import re
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squish.serving.router import (  # noqa: E402
    PromptRouter,
    RouterCategory,
    RouterConfig,
    RouterDecision,
    RouterRule,
    _default_rules,
    get_default_router,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_rule(
    name: str,
    category: RouterCategory,
    pattern: str,
    priority: int = 50,
    model_hint: str | None = None,
) -> RouterRule:
    return RouterRule(
        name=name,
        category=category,
        pattern=re.compile(pattern, re.IGNORECASE),
        priority=priority,
        model_hint=model_hint,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. Default rules
# ═════════════════════════════════════════════════════════════════════════════


def test_router_default_rules_not_empty():
    """Built-in rule list must contain at least 8 rules."""
    rules = _default_rules()
    assert len(rules) >= 8, f"Expected ≥8 default rules, got {len(rules)}"


# ═════════════════════════════════════════════════════════════════════════════
# 2. Empty prompt
# ═════════════════════════════════════════════════════════════════════════════


def test_route_empty_prompt_returns_unknown():
    """An empty string must yield UNKNOWN."""
    router = PromptRouter()
    decision = router.route("")
    assert decision.category == RouterCategory.UNKNOWN


def test_route_whitespace_only_returns_unknown():
    """Whitespace-only prompt must yield UNKNOWN."""
    router = PromptRouter()
    decision = router.route("   \t\n")
    assert decision.category == RouterCategory.UNKNOWN


# ═════════════════════════════════════════════════════════════════════════════
# 3. Category routing via default rules
# ═════════════════════════════════════════════════════════════════════════════


def test_route_code_category_on_python_snippet():
    """'def sort_list():' must route to CODE."""
    router = PromptRouter()
    decision = router.route("def sort_list(items): return sorted(items)")
    assert decision.category == RouterCategory.CODE


def test_route_math_category_on_equation():
    """A prompt containing 'equation' must route to MATH."""
    router = PromptRouter()
    decision = router.route("Solve the differential equation dy/dx = 2x")
    assert decision.category == RouterCategory.MATH


def test_route_creative_category_on_story_prompt():
    """A story request must route to CREATIVE."""
    router = PromptRouter()
    decision = router.route("Write a short story about a robot who learns to paint")
    assert decision.category == RouterCategory.CREATIVE


def test_route_factual_category_on_question():
    """A 'What is …' question must route to FACTUAL."""
    router = PromptRouter()
    decision = router.route("What is the boiling point of water?")
    assert decision.category == RouterCategory.FACTUAL


# ═════════════════════════════════════════════════════════════════════════════
# 4. Confidence and rule-name on match
# ═════════════════════════════════════════════════════════════════════════════


def test_route_matched_rule_name_set():
    """matched_rule must be a non-empty string when a rule fires."""
    router = PromptRouter()
    decision = router.route("import numpy as np")
    assert decision.matched_rule is not None
    assert len(decision.matched_rule) > 0


def test_route_confidence_1_on_rule_match():
    """Confidence must be 1.0 when a rule matches."""
    router = PromptRouter()
    decision = router.route("def fibonacci(n): pass")
    assert decision.confidence == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 5. Heuristic fallback
# ═════════════════════════════════════════════════════════════════════════════


def test_route_no_match_uses_heuristics():
    """When no rule matches, heuristic keywords must fire for known terms.

    We bypass the default rule list by zeroing _rules directly so only
    the heuristic path is exercised.
    """
    router = PromptRouter()
    router._rules = []  # strip all rules; heuristics must decide
    decision = router.route("calculate the formula")
    assert decision.category == RouterCategory.MATH


def test_route_heuristic_confidence_is_0_5():
    """Heuristic-derived decisions must have confidence exactly 0.5."""
    router = PromptRouter()
    router._rules = []  # strip rules so the heuristic path fires
    decision = router.route("calculate the formula")
    assert decision.confidence == 0.5


def test_route_heuristics_disabled_returns_fallback():
    """When heuristics are disabled and no rule matches, return fallback_category."""
    cfg = RouterConfig(rules=[], enable_heuristics=False,
                       fallback_category=RouterCategory.CONVERSATION)
    router = PromptRouter(cfg)
    router._rules = []  # ensure no rule matches
    decision = router.route("calculate the formula")
    assert decision.category == RouterCategory.CONVERSATION


# ═════════════════════════════════════════════════════════════════════════════
# 6. UNKNOWN confidence
# ═════════════════════════════════════════════════════════════════════════════


def test_route_unknown_confidence_is_0():
    """Empty prompt must have confidence 0.0."""
    router = PromptRouter()
    decision = router.route("")
    assert decision.confidence == 0.0


def test_route_heuristic_all_zero_returns_unknown():
    """Prompt with no matching keywords must yield UNKNOWN with confidence 0.0."""
    cfg = RouterConfig(rules=[], enable_heuristics=True)
    router = PromptRouter(cfg)
    # 'xyzzy' has no overlap with any keyword set.
    decision = router.route("xyzzy plugh zorkmid")
    assert decision.category == RouterCategory.UNKNOWN
    assert decision.confidence == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 7. Case insensitivity
# ═════════════════════════════════════════════════════════════════════════════


def test_route_case_insensitive():
    """Matching must work regardless of prompt capitalisation."""
    router = PromptRouter()
    upper  = router.route("DEF MY_FUNCTION(): PASS")
    lower  = router.route("def my_function(): pass")
    assert upper.category == lower.category == RouterCategory.CODE


# ═════════════════════════════════════════════════════════════════════════════
# 8. model_hint propagation
# ═════════════════════════════════════════════════════════════════════════════


def test_route_model_hint_propagated():
    """A rule with a model_hint must surface it on the decision."""
    rule = _make_rule("hint-test", RouterCategory.CODE, r"\bpython\b",
                      priority=200, model_hint="codellama")
    cfg  = RouterConfig(rules=[rule])
    router = PromptRouter(cfg)
    decision = router.route("Write a Python function")
    assert decision.model_hint == "codellama"


# ═════════════════════════════════════════════════════════════════════════════
# 9. Priority ordering
# ═════════════════════════════════════════════════════════════════════════════


def test_route_priority_order_respected():
    """When two rules match, the one with higher priority must win."""
    low_rule  = _make_rule("low",  RouterCategory.CREATIVE, r"\btest\b", priority=10)
    high_rule = _make_rule("high", RouterCategory.MATH,     r"\btest\b", priority=99)
    cfg = RouterConfig(rules=[low_rule, high_rule], enable_heuristics=False)
    router = PromptRouter(cfg)
    decision = router.route("This is a test prompt")
    assert decision.matched_rule == "high"
    assert decision.category == RouterCategory.MATH


# ═════════════════════════════════════════════════════════════════════════════
# 10. Custom rule overrides default
# ═════════════════════════════════════════════════════════════════════════════


def test_route_custom_rule_overrides_default():
    """A custom rule with very high priority must beat built-in rules."""
    custom = _make_rule(
        "override", RouterCategory.CONVERSATION,
        r"\bdef\b",  # would normally fire CODE via default rules
        priority=999,
    )
    cfg    = RouterConfig(rules=[custom])
    router = PromptRouter(cfg)
    decision = router.route("def my_func(): pass")
    assert decision.matched_rule == "override"
    assert decision.category == RouterCategory.CONVERSATION


# ═════════════════════════════════════════════════════════════════════════════
# 11. Return type
# ═════════════════════════════════════════════════════════════════════════════


def test_route_returns_router_decision():
    """route() must return a RouterDecision instance."""
    router = PromptRouter()
    result = router.route("Hello there")
    assert isinstance(result, RouterDecision)


# ═════════════════════════════════════════════════════════════════════════════
# 12. Frozen dataclass
# ═════════════════════════════════════════════════════════════════════════════


def test_router_decision_frozen():
    """RouterDecision must be immutable — mutation must raise FrozenInstanceError."""
    decision = RouterDecision(
        category=RouterCategory.CODE,
        matched_rule="test-rule",
        model_hint=None,
        confidence=1.0,
        reasoning="test",
    )
    with pytest.raises(FrozenInstanceError):
        decision.category = RouterCategory.MATH  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════════════
# 13. RouterConfig custom fallback
# ═════════════════════════════════════════════════════════════════════════════


def test_router_config_custom_fallback():
    """RouterConfig.fallback_category must be used when heuristics are off."""
    cfg = RouterConfig(
        rules=[],
        fallback_category=RouterCategory.FACTUAL,
        enable_heuristics=False,
    )
    router = PromptRouter(cfg)
    decision = router.route("some completely neutral prompt without keywords")
    assert decision.category == RouterCategory.FACTUAL


# ═════════════════════════════════════════════════════════════════════════════
# 14. Empty rules list — heuristics still work
# ═════════════════════════════════════════════════════════════════════════════


def test_router_config_empty_rules_uses_heuristics():
    """With an empty custom rule list, heuristics must still classify correctly."""
    cfg = RouterConfig(rules=[], enable_heuristics=True)
    router = PromptRouter(cfg)
    decision = router.route("Imagine a world where dragons exist — write a poem")
    assert decision.category == RouterCategory.CREATIVE


# ═════════════════════════════════════════════════════════════════════════════
# 15. explain()
# ═════════════════════════════════════════════════════════════════════════════

_REQUIRED_EXPLAIN_KEYS = {
    "category", "matched_rule", "model_hint", "confidence", "reasoning",
    "prompt_length", "n_rules_checked",
}


def test_explain_returns_dict_with_required_keys():
    """explain() must return a dict containing all required keys."""
    router = PromptRouter()
    result = router.explain("def foo(): pass")
    assert _REQUIRED_EXPLAIN_KEYS.issubset(result.keys())


def test_explain_includes_prompt_length():
    """explain()['prompt_length'] must equal len(prompt)."""
    router = PromptRouter()
    prompt = "What is the speed of light?"
    result = router.explain(prompt)
    assert result["prompt_length"] == len(prompt)


def test_explain_includes_n_rules_checked():
    """explain()['n_rules_checked'] must equal the number of loaded rules."""
    router = PromptRouter()
    result = router.explain("Hello")
    assert isinstance(result["n_rules_checked"], int)
    assert result["n_rules_checked"] >= 8


# ═════════════════════════════════════════════════════════════════════════════
# 16. CLI integration
# ═════════════════════════════════════════════════════════════════════════════


def test_cli_route_subcommand_registered():
    """build_parser() must include a 'route' subcommand."""
    from squish.cli import build_parser
    ap = build_parser()
    # Walk subparsers to find 'route'
    subparsers_actions = [
        action for action in ap._actions
        if hasattr(action, "_name_parser_map")
    ]
    assert subparsers_actions, "No subparsers found in build_parser()"
    all_commands = list(subparsers_actions[0]._name_parser_map.keys())
    assert "route" in all_commands, (
        f"'route' not in CLI subcommands: {all_commands}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 17. Singleton factory
# ═════════════════════════════════════════════════════════════════════════════


def test_get_default_router_returns_router():
    """get_default_router() must return a PromptRouter instance."""
    router = get_default_router()
    assert isinstance(router, PromptRouter)
