import argparse
import contextlib
import io
import hashlib
import json
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from frontierscience_eval.__main__ import build_summary, command_generate, command_grade, parse_shard


class FakeProvider:
    """Deterministic stand-in for http_json that records every request body."""

    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()

    def __call__(self, url, method="GET", headers=None, payload=None, timeout=1800):
        body = None if payload is None else json.dumps(payload).encode()
        with self.lock:
            self.requests.append((url, method, json.dumps(headers, sort_keys=True), timeout, body))
        if url.endswith("/models"):
            return {"data": [{"id": "served-id"}]}
        digest = hashlib.sha256(body).hexdigest()[:16]
        if url.endswith("/chat/completions"):
            capped = int(digest, 16) % 3 == 0
            return {
                "id": f"chatcmpl-{digest}",
                "model": "served-id",
                "choices": [
                    {
                        "message": {"content": None if capped else f"answer {digest}", "reasoning": f"thinking {digest}"},
                        "finish_reason": "length" if capped else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
        return {
            "id": f"resp-{digest}",
            "model": "judge-id",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": f"VERDICT: {int(digest, 16) % 11}"}]}],
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        key_file = root / "key"
        key_file.write_text("dummy-token")
        price = {"input": 0.0, "output": 0.0}
        self.config_path = root / "config.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "label": "endpoint-model",
                            "provider": "openai_compatible",
                            "model_id": "served-id",
                            "base_url": "http://model.example/v1",
                            "key_file": str(key_file),
                            "reasoning_history": "empty",
                            "max_output_tokens": 64,
                            "price_per_million": price,
                            "request_parameters": {"temperature": 1.0, "reasoning_effort": "high"},
                        }
                    ],
                    "judge": {
                        "label": "judge",
                        "provider": "openai",
                        "model_id": "judge-id",
                        "key_file": str(key_file),
                        "max_output_tokens": 64,
                        "price_per_million": price,
                    },
                }
            )
        )
        self.root = root

    def tearDown(self):
        self.directory.cleanup()

    def run_condition(self, run_id, workers, sample_shard=None):
        fake = FakeProvider()
        with patch("frontierscience_eval.providers.http_json", fake), contextlib.redirect_stdout(io.StringIO()):
            generated = command_generate(
                argparse.Namespace(
                    config=self.config_path,
                    data=Path("data/research-test.jsonl"),
                    subset="research-pilot-v1",
                    artifact_root=self.root,
                    run_id=run_id,
                    trials=3,
                    resume_from=None,
                    allow_full_run=False,
                    sample_shard=sample_shard,
                    workers=workers,
                )
            )
            graded = command_grade(
                argparse.Namespace(
                    config=self.config_path,
                    artifact_root=self.root,
                    run_id=run_id,
                    resume=True,
                    allow_full_run=False,
                    workers=workers,
                )
            )
        self.assertEqual((generated, graded), (0, 0))
        return fake.requests

    def records(self, run_id, name):
        rows = [json.loads(line) for line in (self.root / run_id / name).read_text().splitlines()]
        keyed = {row["answer_id"]: {k: v for k, v in row.items() if k not in {"created_at", "latency_seconds"}} for row in rows}
        self.assertEqual(len(keyed), len(rows))
        return keyed

    def test_workers_send_identical_requests_and_write_identical_records(self):
        serial = self.run_condition("serial", workers=1)
        parallel = self.run_condition("parallel", workers=8)
        self.assertEqual(len(serial), 1 + 1 + 27 + 27)
        self.assertEqual(Counter(serial), Counter(parallel))
        self.assertEqual(self.records("serial", "answers.jsonl"), self.records("parallel", "answers.jsonl"))
        self.assertEqual(self.records("serial", "judgments.jsonl"), self.records("parallel", "judgments.jsonl"))
        raw = {name: {p.name: p.read_bytes() for p in (self.root / name / "raw").iterdir()} for name in ("serial", "parallel")}
        self.assertEqual(len(raw["serial"]), 54)
        self.assertEqual(raw["serial"], raw["parallel"])
        config = json.loads(self.config_path.read_text())
        summaries = [build_summary(self.root / name, config) for name in ("serial", "parallel")]
        for summary in summaries:
            summary.pop("generated_at")
            self.assertTrue(summary["completeness"]["complete"])
        self.assertEqual(summaries[0], summaries[1])

    def test_sample_shards_are_disjoint_and_cover_the_subset(self):
        self.run_condition("whole", workers=4)
        self.run_condition("shard-0", workers=4, sample_shard=(0, 2))
        self.run_condition("shard-1", workers=4, sample_shard=(1, 2))
        shards = [self.records(f"shard-{index}", "answers.jsonl") for index in range(2)]
        self.assertFalse(set(shards[0]) & set(shards[1]))
        self.assertEqual({**shards[0], **shards[1]}, self.records("whole", "answers.jsonl"))
        manifest = json.loads((self.root / "shard-1" / "manifest.json").read_text())
        self.assertEqual(manifest["sample_shard"], "1/2")
        self.assertNotIn("sample_shard", json.loads((self.root / "whole" / "manifest.json").read_text()))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_shard("2/2")


if __name__ == "__main__":
    unittest.main()
