"""Unit tests for CLI patient insights formatting."""

from __future__ import annotations

import unittest

from earl_sdk.interactive.patient_insights_format import (
    format_patient_insights_markup,
    insights_from_message,
)


class TestPatientInsightsFormat(unittest.TestCase):
    def test_insights_from_message(self) -> None:
        self.assertIsNone(insights_from_message({"role": "patient", "content": "hi"}))
        self.assertIsNone(insights_from_message({"role": "patient", "metadata": {}}))
        ins = {"turn_number": 2, "thoughts": ["a", "b"]}
        got = insights_from_message({"role": "patient", "metadata": {"insights": ins}})
        self.assertEqual(got, ins)

    def test_format_markup_basic(self) -> None:
        m = format_patient_insights_markup(
            {
                "turn_number": 1,
                "internal_state": {
                    "engagement_level": 7,
                    "gut_reaction": "Uneasy",
                },
                "thoughts": ["Need to leave soon"],
            }
        )
        self.assertIsNotNone(m)
        assert m is not None
        self.assertIn("Turn", m)
        self.assertIn("Engagement", m)
        self.assertIn("Gut reaction", m)
        self.assertIn("Thoughts", m)

    def test_format_empty(self) -> None:
        self.assertIsNone(format_patient_insights_markup(None))
        self.assertIsNone(format_patient_insights_markup({}))


if __name__ == "__main__":
    unittest.main()
