from unittest.mock import patch

from earl_sdk.interactive.flows import run as run_flow
from earl_sdk.interactive.flows.run import (
    _build_scoring_dimension_groups,
    _canonical_scoring_dimension_id,
    _category_choice_label,
    _dimension_group_key,
    _group_additional_verifier_choices,
    _pick_additional_verifiers,
    _verifier_category_key,
)


def test_dimension_group_key_uses_prefix_before_double_dash() -> None:
    assert (
        _dimension_group_key("scoring-dimensions/medication-management--dosing-accuracy")
        == "medication-management"
    )


def test_canonical_scoring_dimension_adds_lumos_namespace() -> None:
    assert (
        _canonical_scoring_dimension_id("final-diagnosis--justification")
        == "scoring-dimensions/final-diagnosis--justification"
    )
    assert (
        _canonical_scoring_dimension_id("scoring-dimensions/final-diagnosis--justification")
        == "scoring-dimensions/final-diagnosis--justification"
    )


def test_final_diagnosis_group_includes_treatment_recommendation_dimensions() -> None:
    groups = _build_scoring_dimension_groups(
        [
            "final-diagnosis--data-consistency",
            "final-diagnosis--justification",
            "final-diagnosis--probability-appropriateness",
            "final-diagnosis--symptom-inclusivity",
            "first-line-treatment-recommendation--guideline-adherence",
            "first-line-treatment-recommendation--personalization",
            "first-line-treatment-recommendation--risk-benefit-communication",
        ]
    )

    assert groups["final-diagnosis"] == [
        "scoring-dimensions/final-diagnosis--data-consistency",
        "scoring-dimensions/final-diagnosis--justification",
        "scoring-dimensions/final-diagnosis--probability-appropriateness",
        "scoring-dimensions/final-diagnosis--symptom-inclusivity",
        "scoring-dimensions/first-line-treatment-recommendation--guideline-adherence",
        "scoring-dimensions/first-line-treatment-recommendation--personalization",
        "scoring-dimensions/first-line-treatment-recommendation--risk-benefit-communication",
    ]
    assert "first-line-treatment-recommendation" not in groups


def test_scoring_dimension_groups_dedupe_and_keep_other_prefixes() -> None:
    groups = _build_scoring_dimension_groups(
        [
            "communication--clarity",
            "communication--clarity",
            "communication--empathy",
            "clinical-correctness",
        ]
    )

    assert groups["communication"] == [
        "scoring-dimensions/communication--clarity",
        "scoring-dimensions/communication--empathy",
    ]
    assert groups["clinical-correctness"] == ["scoring-dimensions/clinical-correctness"]


# ── Additional-verifier category collapsing ──────────────────────────────────


_SAMPLE_CATALOG = [
    ("scoring-dimensions/adaptive-dialogue--context-recall",
     "📊 adaptive-dialogue--context-recall  [scoring]"),
    ("scoring-dimensions/adaptive-dialogue--question-management",
     "📊 adaptive-dialogue--question-management  [scoring]"),
    ("scoring-dimensions/adaptive-dialogue--state-sensitivity",
     "📊 adaptive-dialogue--state-sensitivity  [scoring]"),
    ("scoring-dimensions/clinical-correctness",
     "📊 clinical-correctness  [scoring]"),
    ("hard-gates/communication--clarity",
     "🛡  communication--clarity  [hard-gate]"),
    ("scoring-dimensions/communication--empathy",
     "📊 communication--empathy  [scoring]"),
]


def test_verifier_category_strips_namespace_and_double_dash() -> None:
    assert (
        _verifier_category_key("scoring-dimensions/adaptive-dialogue--context-recall")
        == "adaptive-dialogue"
    )
    assert (
        _verifier_category_key("hard-gates/communication--clarity")
        == "communication"
    )
    assert _verifier_category_key("scoring-dimensions/clinical-correctness") == "clinical-correctness"


def test_verifier_category_honors_dimension_group_aliases() -> None:
    """The final-diagnosis alias must also apply to the catalog list so
    customers see one bucket instead of two."""
    assert (
        _verifier_category_key(
            "scoring-dimensions/first-line-treatment-recommendation--guideline-adherence"
        )
        == "final-diagnosis"
    )


def test_group_additional_verifier_choices_buckets_by_category() -> None:
    buckets = dict(_group_additional_verifier_choices(_SAMPLE_CATALOG))
    assert set(buckets) == {"adaptive-dialogue", "clinical-correctness", "communication"}
    assert len(buckets["adaptive-dialogue"]) == 3
    assert len(buckets["communication"]) == 2  # hard-gate + scoring mixed
    assert len(buckets["clinical-correctness"]) == 1


def test_category_choice_label_aggregates_kind_icons() -> None:
    buckets = dict(_group_additional_verifier_choices(_SAMPLE_CATALOG))
    label = _category_choice_label("communication", buckets["communication"])
    assert "🛡" in label and "📊" in label
    assert "(2 verifiers)" in label
    assert "Communication" in label


def test_pick_additional_verifiers_auto_includes_singleton_categories() -> None:
    """A category with one member must not trigger a second prompt; selecting
    its row alone is enough."""
    calls: list[tuple] = []

    def fake_select_many(title, choices, min_count=1, defaults=None):
        calls.append((title, [c[0] for c in choices], defaults))
        if title == "Select verifier categories":
            return ["clinical-correctness"]
        raise AssertionError(f"unexpected second prompt: {title}")

    with patch.object(run_flow, "select_many", side_effect=fake_select_many):
        out = _pick_additional_verifiers(_SAMPLE_CATALOG)

    assert out == ["scoring-dimensions/clinical-correctness"]
    assert len(calls) == 1  # only the category prompt; no refinement step


def test_pick_additional_verifiers_pre_checks_members_for_refinement() -> None:
    """Selecting a multi-member category must open a refinement prompt with
    every member already checked. The user can then deselect freely."""
    refinement_prompts: list[tuple[str, list[str]]] = []

    def fake_select_many(title, choices, min_count=1, defaults=None):
        if title == "Select verifier categories":
            return ["adaptive-dialogue"]
        refinement_prompts.append((title, defaults or []))
        return [
            "scoring-dimensions/adaptive-dialogue--context-recall",
            "scoring-dimensions/adaptive-dialogue--state-sensitivity",
        ]

    with patch.object(run_flow, "select_many", side_effect=fake_select_many):
        out = _pick_additional_verifiers(_SAMPLE_CATALOG)

    assert len(refinement_prompts) == 1
    title, defaults = refinement_prompts[0]
    assert "adaptive-dialogue" in title.lower() or "Adaptive Dialogue" in title
    assert set(defaults) == {
        "scoring-dimensions/adaptive-dialogue--context-recall",
        "scoring-dimensions/adaptive-dialogue--question-management",
        "scoring-dimensions/adaptive-dialogue--state-sensitivity",
    }
    assert out == [
        "scoring-dimensions/adaptive-dialogue--context-recall",
        "scoring-dimensions/adaptive-dialogue--state-sensitivity",
    ]


def test_pick_additional_verifiers_empty_returns_empty() -> None:
    assert _pick_additional_verifiers([]) == []


def test_pick_additional_verifiers_user_cancels_top_level() -> None:
    with patch.object(run_flow, "select_many", return_value=[]):
        assert _pick_additional_verifiers(_SAMPLE_CATALOG) == []
