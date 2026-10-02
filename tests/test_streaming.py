import argparse
import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from frontierscience_eval.__main__ import command_generate
from frontierscience_eval.providers import call_model


class StreamingHandler(BaseHTTPRequestHandler):
    """Chat-completions server whose model ID selects how the stream ends."""

    def log_message(self, *args):
        pass

    def send_json(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send_json({"data": [{"id": model} for model in MODELS]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        model = body["model"]
        if not body.get("stream"):
            self.send_json(
                {
                    "id": "chatcmpl-canary",
                    "model": model,
                    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        tokens = 0

        def send(delta, finish_reason=None):
            nonlocal tokens
            tokens += 1
            event = {
                "id": f"chatcmpl-{model}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 7, "completion_tokens": tokens, "total_tokens": 7 + tokens},
            }
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()

        try:
            if model == "silent":
                time.sleep(3)
            send({"role": "assistant", "reasoning": "step 1 "})
            if model == "error":
                self.wfile.write(b'data: {"error": {"message": "engine failed", "code": 500}}\n\n')
                return
            if model == "truncated":
                return
            # "slow" streams past the 1 s timeout; "stall" goes silent just before it.
            for step in range(2, {"slow": 40, "silent": 40, "stall": 5}.get(model, 2)):
                time.sleep(0.25)
                send({"reasoning": f"step {step} "})
            if model == "stall":
                time.sleep(3)
            send({"reasoning": "done"})
            send({"content": "The "})
            if model == "normal":
                send({"content": "answer"}, "stop")
            else:
                send({"content": "partial"}, "length")
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


MODELS = ["normal", "slow", "stall", "silent", "error", "truncated", "max_tokens", "context"]


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
        self.server.daemon_threads = True
        self.server.requests = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        key_file = self.root / "key"
        key_file.write_text("dummy-token")
        self.models = [
            {
                "label": model,
                "provider": "openai_compatible",
                "model_id": model,
                "base_url": f"http://127.0.0.1:{self.server.server_port}/v1",
                "key_file": str(key_file),
                "reasoning_history": "empty",
                # Each stream ends after 4 tokens, so "length" there is the output cap only when 4 were requested.
                "max_output_tokens": None if model == "context" else 4,
                "price_per_million": {"input": 0.0, "output": 0.0},
                "request_parameters": {"temperature": 1.0, "top_p": 0.95},
                "request_timeout_seconds": 1,
            }
            for model in MODELS
        ]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.directory.cleanup()

    def generate(self, streaming=True):
        judge = {
            "label": "judge",
            "provider": "openai",
            "model_id": "judge-id",
            "key_file": self.models[0]["key_file"],
            "max_output_tokens": 64,
            "price_per_million": {"input": 0.0, "output": 0.0},
        }
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps({"models": self.models, "judge": judge}))
        run_id = "streaming"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = command_generate(
                argparse.Namespace(
                    config=config_path,
                    data=Path("data/research-test.jsonl"),
                    subset="smoke",
                    artifact_root=self.root,
                    run_id=run_id,
                    trials=1,
                    resume_from=None,
                    allow_full_run=False,
                    sample_shard=None,
                    workers=len(MODELS),
                    streaming=streaming,
                )
            )
        run_dir = self.root / run_id
        answers = {
            row["model_label"]: row
            for row in map(json.loads, (run_dir / "answers.jsonl").read_text().splitlines())
        }
        errors = [json.loads(line) for line in (run_dir / "generation_errors.jsonl").read_text().splitlines()]
        return status, run_dir, answers, errors

    def test_streaming_keeps_partial_output_and_records_why_it_stopped(self):
        status, run_dir, answers, errors = self.generate()
        self.assertEqual(status, 1)
        self.assertTrue(json.loads((run_dir / "manifest.json").read_text())["streaming"])

        generation = [body for body in self.server.requests if body.get("stream")]
        self.assertEqual(len(generation), len(MODELS))
        for body in generation:
            self.assertEqual(
                body["stream_options"], {"include_usage": True, "continuous_usage_stats": True}
            )
        canaries = [body for body in self.server.requests if not body.get("stream")]
        self.assertEqual(len(canaries), len(MODELS))
        self.assertTrue(all("stream_options" not in body for body in canaries))

        normal = answers["normal"]
        self.assertEqual(normal["answer"], "The answer")
        self.assertEqual(normal["finish_reason"], "stop")
        self.assertEqual((normal["incomplete"], normal["incomplete_reason"]), (False, None))
        self.assertEqual(normal["usage"], {"input_tokens": 7, "output_tokens": 4, "reasoning_tokens": 0})
        self.assertTrue(normal["streaming"])
        raw = json.loads((run_dir / normal["raw_path"]).read_text())
        self.assertEqual(raw["choices"][0]["message"]["reasoning"], "step 1 done")

        for label in ("slow", "stall"):
            row = answers[label]
            self.assertEqual((row["incomplete"], row["incomplete_reason"]), (True, "timeout"))
            self.assertIsNone(row["finish_reason"])
            self.assertTrue(row["answer"].startswith("step 1"))
            self.assertNotIn("The", row["answer"])
            self.assertGreaterEqual(row["usage"]["output_tokens"], 1)
            # The deadline bounds the whole response, even when the stream stalls just before it.
            self.assertLess(row["latency_seconds"], 1.6)
        self.assertGreater(answers["slow"]["usage"]["output_tokens"], 1)

        for label in ("max_tokens", "context"):
            row = answers[label]
            self.assertEqual(row["answer"], "The partial")
            self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(answers["max_tokens"]["incomplete_reason"], "max_tokens")
        self.assertEqual(answers["context"]["incomplete_reason"], "context_limit")
        self.assertTrue(answers["max_tokens"]["incomplete"] and answers["context"]["incomplete"])

        # No output before the timeout, an error event or a stream cut short stays a generation
        # error and is retried, as on the non-streaming path.
        self.assertEqual(set(answers), set(MODELS) - {"silent", "error", "truncated"})
        self.assertEqual(
            sorted((row["model_label"], row["error_type"], row["error"][:26]) for row in errors),
            [
                ("error", "RuntimeError", "stream error from provider"),
                ("silent", "TimeoutError", "stream timed out after 1 s"),
                ("truncated", "RuntimeError", "stream ended without a fin"),
            ],
        )

        # The manifest records the flag, so the run cannot be resumed with it toggled.
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            self.generate(streaming=False)

    def test_streaming_requires_openai_compatible_models(self):
        with self.assertRaisesRegex(ValueError, "only openai_compatible"):
            call_model({"provider": "gemini", "key_file": "unused"}, "prompt", streaming=True)


if __name__ == "__main__":
    unittest.main()
