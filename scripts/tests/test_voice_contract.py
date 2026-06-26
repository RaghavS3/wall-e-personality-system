#!/usr/bin/env python3
"""Focused tests for production VOICE_CONTRACT content."""

from __future__ import annotations

import re
import sys
import unittest

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import server


class ProductionVoiceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = server.VOICE_CONTRACT

    def test_brand_identity_preserved(self):
        self.assertIn("Wall-e by Rs2 Labs", self.contract)
        self.assertIn("i'm wall-e by Rs2 Labs", self.contract)

    def test_merged_005_hard_rules_present(self):
        required_phrases = (
            "OPEN LOOP CONTINUITY",
            "DRAFT FIDELITY",
            "CAPABILITY TRUTH",
            "anchor to the named loop",
            "Do not invent investigations, confirmations, or conclusions",
            "cannot access their account/portal/login",
            'broad "no browser access"',
            "ANTI-SLOP CHECK",
        )
        for phrase in required_phrases:
            self.assertIn(phrase, self.contract, msg=phrase)

    def test_anti_slop_rules_present(self):
        banned = (
            '"i hear you"',
            '"how can i help you"',
            '"as an ai"',
            "therapy cadence",
            "generic refusal closers",
        )
        for phrase in banned:
            self.assertIn(phrase, self.contract, msg=phrase)

    def test_no_experiment_metadata(self):
        forbidden = re.compile(
            r"\b(HYPOTHESIS|Phase 3\.5|successor|recipe|benchmark|scoring|tournament)\b",
            re.IGNORECASE,
        )
        self.assertIsNone(forbidden.search(self.contract))

    def test_json_output_shape_preserved(self):
        self.assertIn('"messages":', self.contract)
        self.assertIn('"memory_ops":', self.contract)
        self.assertIn('"actions":', self.contract)


if __name__ == "__main__":
    unittest.main()
