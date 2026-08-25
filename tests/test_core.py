import argparse
import json
import tempfile
import unittest
from pathlib import Path

from frontierscience_eval.__main__ import command_generate, command_grade, write_manifest

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

    def test_full_selector_retains_60_rows_with_unique_ids(self):
        rows = load_dataset(self.data_path)
        full = select_subset(rows, "research-full")
        self.assertEqual(len(full), 60)
        self.assertEqual(len({item["sample_id"] for item in full}), 60)
        self.assertEqual(
            {item["source_row"] for item in full if item["task_group_id"] == "af50243e-3a60-4460-9536-f9a02c4f8eb8"},
            {6, 11},
        )
        self.assertTrue(all(item["sample_id"].endswith(f"row{item['source_row']:02d}") for item in full if item["source_row"] in {6, 11}))

    def test_full_generation_requires_explicit_launch_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(
                config=Path("configs/direct-full.json"),
                data=self.data_path,
                subset="research-full",
                artifact_root=Path(directory),
                run_id="must-not-launch",
                trials=30,
                resume_from=None,
                allow_full_run=False,
            )
            with self.assertRaisesRegex(ValueError, "explicit --allow-full-run"):
                command_generate(args)

    def test_full_judging_requires_explicit_launch_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "full"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({"subset": "research-full"}))
            args = argparse.Namespace(
                config=Path("configs/direct-full.json"),
                artifact_root=Path(directory),
                run_id="full",
                resume=True,
                allow_full_run=False,
            )
            with self.assertRaisesRegex(ValueError, "explicit --allow-full-run"):
                command_grade(args)

    def test_resume_rejects_manifest_condition_change(self):
        config = json.loads(Path("configs/direct-full.json").read_text())
        sample = select_subset(load_dataset(self.data_path), "research-full")[:1]
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_manifest(run, config, "research-full", sample, 30)
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                write_manifest(run, config, "research-full", sample, 1)

    def test_verdict_parser(self):
        self.assertEqual(parse_verdict("reason\nVERDICT: 7.5"), 7.5)
        self.assertEqual(parse_verdict("VERDICT: 2\nwork\nVERDICT: 8"), 8.0)
        self.assertEqual(parse_verdict("0"), 0.0)
        for bad in ["VERDICT: 11", "score 7", "VERDICT: -1"]:
            with self.assertRaises(ValueError):
                parse_verdict(bad)

    def test_high_reasoning_payloads_have_no_tools(self):
        full_config = json.loads(Path("configs/direct-full.json").read_text())
        self.assertEqual(full_config["subset"], "research-full")
        self.assertEqual(full_config["trials"], 30)
        models = full_config["models"]
        payloads = {model["provider"]: build_payload(model["provider"], model, "x") for model in models}
        self.assertEqual(payloads["openai"]["reasoning"]["effort"], "high")
        self.assertEqual(payloads["anthropic"]["output_config"]["effort"], "high")
        self.assertEqual(payloads["gemini"]["generationConfig"]["thinkingConfig"]["thinkingLevel"], "HIGH")
        self.assertTrue(payloads["openrouter"]["reasoning"]["enabled"])
        for payload in payloads.values():
            self.assertNotIn("tools", payload)


if __name__ == "__main__":
    unittest.main()
