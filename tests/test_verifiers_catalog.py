"""Unit tests for Lumos verifiers catalog JSON parsing."""

from __future__ import annotations

import unittest

from earl_sdk.verifiers_catalog import parse_verifiers_list_payload


class TestParseVerifiersListPayload(unittest.TestCase):
    def test_none_and_empty(self) -> None:
        self.assertEqual(parse_verifiers_list_payload(None), ([], []))
        self.assertEqual(parse_verifiers_list_payload({}), ([], []))

    def test_flat_path_strings(self) -> None:
        data = [
            "hard-gates/x",
            "scoring-dimensions/clinical-correctness",
        ]
        g, s = parse_verifiers_list_payload(data)
        self.assertEqual(g, ["hard-gates/x"])
        self.assertEqual(s, ["scoring-dimensions/clinical-correctness"])

    def test_dict_arrays(self) -> None:
        data = {
            "hard_gates": ["fabricated-ehr-data", "hard-gates/already-prefixed"],
            "scoring_dimensions": ["communication--empathy", "scoring-dimensions/clarity"],
        }
        g, s = parse_verifiers_list_payload(data)
        self.assertIn("hard-gates/fabricated-ehr-data", g)
        self.assertIn("hard-gates/already-prefixed", g)
        self.assertIn("scoring-dimensions/communication--empathy", s)
        self.assertIn("scoring-dimensions/clarity", s)

    def test_verifiers_objects_with_type(self) -> None:
        data = {
            "verifiers": [
                {"id": "hard-gates/foo", "type": "hard_gate"},
                {"verifier_id": "scoring-dimensions/bar", "kind": "scoring_dimension"},
            ]
        }
        g, s = parse_verifiers_list_payload(data)
        self.assertIn("hard-gates/foo", g)
        self.assertIn("scoring-dimensions/bar", s)

    def test_wrapped_in_data_key(self) -> None:
        inner = {"hard_gates": ["x"], "scoring_dimensions": ["y"]}
        g, s = parse_verifiers_list_payload({"data": inner})
        self.assertEqual(g, ["hard-gates/x"])
        self.assertEqual(s, ["scoring-dimensions/y"])


if __name__ == "__main__":
    unittest.main()
