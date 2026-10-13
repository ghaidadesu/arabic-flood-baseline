"""
Offline self-test: no API key and no internet needed.
It plugs a FAKE Gemini client into baseline.py and checks the logic
(3 calls, token counting, repair, retries, error handling, data cleaning/split).
It does NOT measure how good the model is.

    python tests/test_offline.py
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import baseline
import data_utils


def reply(text, tokens_in=100, tokens_out=10, tokens_think=5):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=tokens_in,
                                       candidates_token_count=tokens_out,
                                       thoughts_token_count=tokens_think),
    )


class ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"fake API error {code}")
        self.code = code


class FakeClient:
    """Returns the scripted items one by one (a reply or an exception)."""
    def __init__(self, script):
        self.script = list(script)
        self.models = self
        self.prompts = []

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


GOOD = '{"relevance": "Related", "information_type": "Caution and advice"}'
BAD_RULE = '{"relevance": "Not related", "information_type": "Caution and advice"}'
NOT_REL = '{"relevance": "Not related", "information_type": "Not applicable"}'


class BaselineTests(unittest.TestCase):
    def setUp(self):
        baseline._make_config = lambda: None      # avoid needing google-genai
        baseline.time.sleep = lambda s: None      # do not really wait

    def run_with(self, script, tweet="تغريدة"):
        client = FakeClient(script)
        baseline.set_client(client)
        return baseline.classify_tweet(tweet), client

    def test_three_calls_and_token_totals(self):
        result, client = self.run_with([reply(GOOD), reply(GOOD), reply(GOOD)])
        self.assertTrue(result["valid"])
        self.assertEqual(result["calls"], 3)
        self.assertEqual(result["repairs"], 0)
        self.assertEqual(result["input_tokens"], 300)
        self.assertEqual(result["output_tokens"], 30)
        self.assertEqual(result["reasoning_tokens"], 15)
        self.assertEqual(result["total_tokens"], 345)
        self.assertFalse(result["first_output_violation"])
        self.assertEqual(result["relevance"], "Related")
        self.assertIn("تغريدة", client.prompts[0])

    def test_label_text_is_normalised(self):
        messy = '```json\n{"relevance": " related ", "information_type": "caution and advice"}\n```'
        result, _ = self.run_with([reply(messy), reply(GOOD), reply(GOOD)])
        self.assertTrue(result["valid"])
        self.assertEqual(result["initial"]["information_type"], "Caution and advice")

    def test_rule_violation_is_recorded_then_repaired(self):
        result, client = self.run_with([reply(BAD_RULE), reply(NOT_REL), reply(NOT_REL), reply(NOT_REL)])
        self.assertTrue(result["first_output_violation"])
        self.assertEqual(result["repairs"], 1)
        self.assertEqual(result["calls"], 4)
        self.assertTrue(result["valid"])
        self.assertIn("invalid", client.prompts[1])

    def test_garbage_after_repair_gives_error_not_crash(self):
        result, _ = self.run_with([reply("hello"), reply("still not json")])
        self.assertFalse(result["valid"])
        self.assertIn("Call 1", result["error"])
        self.assertEqual(result["calls"], 2)

    def test_temporary_error_is_retried(self):
        result, _ = self.run_with([ApiError(503), reply(GOOD), reply(GOOD), reply(GOOD)])
        self.assertTrue(result["valid"])
        self.assertEqual(result["api_retries"], 1)
        self.assertEqual(result["calls"], 3)

    def test_bad_key_is_not_retried_and_message_is_friendly(self):
        result, client = self.run_with([ApiError(403)])
        self.assertFalse(result["valid"])
        self.assertIn("API key", result["error"])
        self.assertEqual(result["api_retries"], 0)
        self.assertEqual(len(client.prompts), 1)

    def test_gives_up_after_max_attempts(self):
        result, _ = self.run_with([ApiError(503)] * 3)
        self.assertFalse(result["valid"])
        self.assertEqual(result["api_retries"], 2)

    def test_empty_input_raises(self):
        baseline.set_client(FakeClient([]))
        with self.assertRaises(ValueError):
            baseline.classify_tweet("   ")

    def test_missing_key_message(self):
        baseline.set_client(None)
        old = os.environ.pop("GEMINI_API_KEY", None)
        try:
            result = baseline.classify_tweet("تغريدة")
        finally:
            if old:
                os.environ["GEMINI_API_KEY"] = old
        self.assertFalse(result["valid"])
        self.assertIn("GEMINI_API_KEY", result["error"])

    def test_cost_is_computed_when_prices_are_set(self):
        baseline.PRICE_INPUT_PER_M, baseline.PRICE_OUTPUT_PER_M = 1.0, 2.0
        try:
            result, _ = self.run_with([reply(GOOD)] * 3)
        finally:
            baseline.PRICE_INPUT_PER_M = baseline.PRICE_OUTPUT_PER_M = None
        self.assertAlmostEqual(result["cost_usd"], (300 * 1.0 + 45 * 2.0) / 1e6)


class DataTests(unittest.TestCase):
    def test_wrapper_removed(self):
        self.assertEqual(data_utils.clean_tweet_text("['سيول الليث https://t.co/x']"), "سيول الليث https://t.co/x")
        self.assertEqual(data_utils.clean_tweet_text("['a \"b\" c']\""), 'a "b" c')

    def test_escapes_are_decoded(self):
        # the CSV really contains the two characters backslash + n, and \U0001f932 style escapes
        self.assertEqual(data_utils.clean_tweet_text("['line1\\nline2 \\U0001f932']"), "line1\nline2 \U0001f932")
        self.assertEqual(data_utils.clean_tweet_text('["it\'s fine"]'), "it's fine")

    def test_cleaning_and_split(self):
        rows = []
        for i in range(400):
            info = baseline.INFO_TYPE_LABELS[i % 7]
            rel = "Not related" if info == "Not applicable" else "Related"
            rows.append({"ID": f"[{i}]", "Tweet Text": f"['t{i}']", "Relevance": rel,
                         "Information Type": info, "Event": "x"})
        rows.append(dict(rows[0]))                                   # exact duplicate
        conflict = dict(rows[1]); conflict["Information Type"] = "Other useful information"
        rows.append(conflict)                                        # conflicting duplicate
        clean = data_utils.clean_dataset(pd.DataFrame(rows))
        self.assertEqual(len(clean), 399)                            # 400 - the conflicting ID
        self.assertEqual(clean["ID"].duplicated().sum(), 0)
        dev, ev = data_utils.split_dataset(clean, eval_size=80)
        self.assertEqual((len(dev), len(ev)), (319, 80))
        self.assertEqual(len(set(dev["ID"]) & set(ev["ID"])), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
