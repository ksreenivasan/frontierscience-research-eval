import json
import tempfile
import unittest
from pathlib import Path

from frontierscience_eval.core import (
    DATASET_SHA256,
    PILOT_GROUPS,
    SMOKE_GROUP,
    load_dataset,
    parse_verdict,
    select_subset,
    validate_dataset,
)
from frontierscience_eval.providers import build_payload


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = Path("data/research-test.jsonl")

    def test_pinned_dataset(self):
        rows = load_dataset(self.data_path)
        result = validate_dataset(rows)
        self.assertEqual(result["rows"], 60)
        self.assertEqual(result["unique_payloads"], 59)
        self.assertEqual(len(result["pilot_rows"]), 9)

    def test_subsets(self):
        rows = load_dataset(self.data_path)
        smoke = select_subset(rows, "smoke")
        pilot = select_subset(rows, "research-pilot-v1")
        self.assertEqual(smoke[0]["task_group_id"], SMOKE_GROUP)
        self.assertEqual({item["task_group_id"] for item in pilot}, set(PILOT_GROUPS))
        self.assertEqual(len({item["sample_id"] for item in pilot}), 9)

    def test_verdict_parser(self):
        self.assertEqual(parse_verdict("reason\nVERDICT: 7.5"), 7.5)
        self.assertEqual(parse_verdict("VERDICT: 2\nwork\nVERDICT: 8"), 8.0)
        for bad in ["VERDICT: 11", "score 7", "VERDICT: -1"]:
            with self.assertRaises(ValueError):
                parse_verdict(bad)

    def test_high_reasoning_payloads_have_no_tools(self):
        models = json.loads(Path("configs/direct-pilot.json").read_text())["models"]
        payloads = {model["provider"]: build_payload(model["provider"], model, "x") for model in models}
        self.assertEqual(payloads["openai"]["reasoning"]["effort"], "high")
        self.assertEqual(payloads["anthropic"]["output_config"]["effort"], "high")
        self.assertEqual(payloads["gemini"]["generationConfig"]["thinkingConfig"]["thinkingLevel"], "HIGH")
        self.assertTrue(payloads["openrouter"]["reasoning"]["enabled"])
        for payload in payloads.values():
            self.assertNotIn("tools", payload)


if __name__ == "__main__":
    unittest.main()
