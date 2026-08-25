from __future__ import annotations

from typing import Any

from .core import http_json, read_key


def build_payload(provider: str, model: dict[str, Any], prompt: str) -> dict[str, Any]:
    model_id = model["model_id"]
    cap = model["max_output_tokens"]
    if provider == "openai":
        return {
            "model": model_id,
            "input": prompt,
            "reasoning": {"effort": "high"},
            "max_output_tokens": cap,
            "store": False,
        }
    if provider == "anthropic":
        return {
            "model": model_id,
            "max_tokens": cap,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "messages": [{"role": "user", "content": prompt}],
        }
    if provider == "gemini":
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": cap,
                "thinkingConfig": {"thinkingLevel": "HIGH"},
            },
        }
    if provider == "openrouter":
        return {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cap,
            "reasoning": {"enabled": True, "exclude": True},
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "provider": {
                "only": [model["endpoint"]],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
    raise ValueError(f"unsupported provider: {provider}")


def effective_settings(provider: str, model: dict[str, Any]) -> dict[str, Any]:
    payload = build_payload(provider, model, "<PROMPT>")
    payload.pop("input", None)
    payload.pop("messages", None)
    payload.pop("contents", None)
    return {
        "requested_reasoning": "high",
        "effective_payload": payload,
        "tools": False,
        "browsing": False,
        "code_execution": False,
    }


def _openai_text(response: dict[str, Any]) -> str:
    chunks = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "\n".join(chunks).strip()


def _anthropic_text(response: dict[str, Any]) -> str:
    return "\n".join(
        block.get("text", "") for block in response.get("content", []) if block.get("type") == "text"
    ).strip()


def _gemini_text(response: dict[str, Any]) -> str:
    chunks = []
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part and not part.get("thought", False):
                chunks.append(part["text"])
    return "\n".join(chunks).strip()


def _openrouter_text(response: dict[str, Any]) -> str:
    choices = response.get("choices", [])
    return str(choices[0].get("message", {}).get("content", "")).strip() if choices else ""


def call_model(model: dict[str, Any], prompt: str) -> dict[str, Any]:
    provider = model["provider"]
    key = read_key(model["key_file"])
    payload = build_payload(provider, model, prompt)
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        headers["Authorization"] = f"Bearer {key}"
        response = http_json("https://api.openai.com/v1/responses", "POST", headers, payload)
        usage = response.get("usage", {})
        details = usage.get("output_tokens_details", {}) or {}
        return {
            "text": _openai_text(response),
            "resolved_model": response.get("model"),
            "provider": "OpenAI",
            "finish_reason": (response.get("incomplete_details") or {}).get("reason") or response.get("status"),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "reasoning_tokens": details.get("reasoning_tokens", 0),
            },
            "response_id": response.get("id"),
            "raw": response,
        }
    if provider == "anthropic":
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        response = http_json("https://api.anthropic.com/v1/messages", "POST", headers, payload)
        usage = response.get("usage", {})
        return {
            "text": _anthropic_text(response),
            "resolved_model": response.get("model"),
            "provider": "Anthropic",
            "finish_reason": response.get("stop_reason"),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "reasoning_tokens": usage.get("thinking_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
            },
            "response_id": response.get("id"),
            "raw": response,
        }
    if provider == "gemini":
        headers["x-goog-api-key"] = key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model['model_id']}:generateContent"
        response = http_json(url, "POST", headers, payload)
        usage = response.get("usageMetadata", {})
        candidates = response.get("candidates", [])
        return {
            "text": _gemini_text(response),
            "resolved_model": response.get("modelVersion") or model["model_id"],
            "provider": "Google",
            "finish_reason": candidates[0].get("finishReason") if candidates else None,
            "usage": {
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
                "visible_output_tokens": usage.get("candidatesTokenCount", 0),
                "reasoning_tokens": usage.get("thoughtsTokenCount", 0),
            },
            "response_id": response.get("responseId"),
            "raw": response,
        }
    if provider == "openrouter":
        headers["Authorization"] = f"Bearer {key}"
        response = http_json("https://openrouter.ai/api/v1/chat/completions", "POST", headers, payload)
        usage = response.get("usage", {})
        details = usage.get("completion_tokens_details", {}) or {}
        choices = response.get("choices", [])
        return {
            "text": _openrouter_text(response),
            "resolved_model": response.get("model"),
            "provider": response.get("provider") or model.get("endpoint"),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": details.get("reasoning_tokens", 0),
            },
            "response_id": response.get("id"),
            "raw": response,
        }
    raise ValueError(f"unsupported provider: {provider}")


def catalog_check(model: dict[str, Any]) -> dict[str, Any]:
    provider = model["provider"]
    key = read_key(model["key_file"])
    if provider == "openai":
        data = http_json("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"})
        ids = {item.get("id") for item in data.get("data", [])}
        return {"available": model["model_id"] in ids, "resolved": model["model_id"]}
    if provider == "anthropic":
        data = http_json(
            "https://api.anthropic.com/v1/models?limit=1000",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        ids = {item.get("id") for item in data.get("data", [])}
        return {"available": model["model_id"] in ids, "resolved": model["model_id"]}
    if provider == "gemini":
        data = http_json(
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
            headers={"x-goog-api-key": key},
        )
        ids = {str(item.get("name", "")).removeprefix("models/") for item in data.get("models", [])}
        return {"available": model["model_id"] in ids, "resolved": model["model_id"]}
    if provider == "openrouter":
        data = http_json(f"https://openrouter.ai/api/v1/models/{model['model_id']}/endpoints")
        endpoints = data.get("data", {}).get("endpoints", [])
        healthy = [item for item in endpoints if item.get("status") == 0]
        selected = [item for item in healthy if str(item.get("tag", "")).lower() == model["endpoint"].lower()]
        return {
            "available": bool(selected),
            "resolved": model["model_id"],
            "endpoint": model["endpoint"],
            "endpoint_metadata": selected[0] if selected else None,
        }
    raise ValueError(f"unsupported provider: {provider}")
