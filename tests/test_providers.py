import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frontierscience_eval.providers import build_payload, require_endpoint_canary


class ProviderTests(unittest.TestCase):
    def test_gemini_38_uses_high_thinking_without_legacy_sampling(self):
        payload = build_payload(
            "gemini",
            {
                "model_id": "gemini-3.8-flash",
                "max_output_tokens": 128,
            },
            "test",
        )
        generation = payload["generationConfig"]
        self.assertEqual(generation["thinkingConfig"], {"thinkingLevel": "HIGH"})
        for parameter in ("temperature", "topP", "topK", "top_p", "top_k"):
            self.assertNotIn(parameter, generation)

    @patch("frontierscience_eval.providers.http_json")
    def test_endpoint_canary_checks_catalog_then_inference(self, http_json):
        http_json.side_effect = [
            {"data": [{"id": "served-id"}]},
            {
                "id": "response-id",
                "model": "served-id",
                "choices": [
                    {"message": {"content": "OK"}, "finish_reason": "stop"}
                ],
                "usage": {},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "key"
            key_file.write_text("dummy-token")
            model = {
                "provider": "openai_compatible",
                "model_id": "served-id",
                "base_url": "http://model.example/v1/",
                "key_file": str(key_file),
                "reasoning_history": "empty",
                "max_output_tokens": 1024,
            }

            result = require_endpoint_canary(model)

        self.assertEqual(result["catalog"], "passed")
        self.assertEqual(result["inference"], "passed")
        self.assertEqual(http_json.call_args_list[0].args[0], "http://model.example/v1/models")
        self.assertEqual(
            http_json.call_args_list[1].args[0],
            "http://model.example/v1/chat/completions",
        )
        self.assertEqual(http_json.call_args_list[1].args[3]["max_tokens"], 32)

        http_json.reset_mock()
        http_json.side_effect = [
            {"data": [{"id": "served-id"}]},
            {
                "model": "unexpected-alias",
                "choices": [{"message": {"content": "OK"}}],
                "usage": {},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "key"
            key_file.write_text("dummy-token")
            with self.assertRaisesRegex(RuntimeError, "expected exact ID"):
                require_endpoint_canary({**model, "key_file": str(key_file)})


if __name__ == "__main__":
    unittest.main()
