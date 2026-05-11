from earl_sdk.interactive.flows.run import (
    _build_scoring_dimension_groups,
    _canonical_scoring_dimension_id,
    _dimension_group_key,
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
