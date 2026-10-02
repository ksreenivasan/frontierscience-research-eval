from __future__ import annotations

import contextlib
import json
from typing import Any

from .core import http_json, http_sse, read_key

# continuous_usage_stats (vLLM) puts running token counts on every chunk, so a
# response cut off by the timeout still records its usage.
STREAM_OPTIONS = {"include_usage": True, "continuous_usage_stats": True}


def _endpoint_base_url(model: dict[str, Any]) -> str:
    base_url = model.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("openai_compatible requires an explicit base_url")
    if model.get("reasoning_history") not in {"none", "preserve", "empty"}:
        raise ValueError(
            "openai_compatible reasoning_history must be one of: none, preserve, empty"
        )
    return base_url.rstrip("/")


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
    if provider == "openai_compatible":
        _endpoint_base_url(model)
        payload: dict[str, Any] = dict(model.get("request_parameters") or {})
        payload.update(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": cap,
            }
        )
        return payload
    raise ValueError(f"unsupported provider: {provider}")


def effective_settings(provider: str, model: dict[str, Any]) -> dict[str, Any]:
    payload = build_payload(provider, model, "<PROMPT>")
    payload.pop("input", None)
    payload.pop("messages", None)
    payload.pop("contents", None)
    settings = {
        "requested_reasoning": (
            model.get("reasoning_effort")
            if provider == "openai_compatible"
            else "high"
        ),
        "effective_payload": payload,
        "tools": False,
        "browsing": False,
        "code_execution": False,
    }
    if provider == "openai_compatible":
        settings.update(
            {
                "base_url": _endpoint_base_url(model),
                "key_file": model["key_file"],
                "reasoning_history": model["reasoning_history"],
            }
        )
    return settings


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


def _stream_chat_completion(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: int
) -> dict[str, Any]:
    """Stream a chat completion and assemble it in the non-streaming response shape.

    Text received before a timeout is kept. `incomplete_reason` is "timeout",
    "context_limit" or "max_tokens" when output stopped early, otherwise None.
    """
    parts: dict[str, list[str]] = {"content": [], "reasoning_content": [], "reasoning": []}
    response: dict[str, Any] = {"object": "chat.completion"}
    finish_reason = None
    timed_out = False
    try:
        with contextlib.closing(http_sse(url, headers, payload, timeout)) as events:
            for event in events:
                if "error" in event or event.get("object") == "error":
                    raise RuntimeError(f"stream error from provider: {json.dumps(event)[:2000]}")
                for field in ("id", "model", "usage"):
                    if event.get(field):
                        response[field] = event[field]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    for field, chunks in parts.items():
                        if delta.get(field):
                            chunks.append(delta[field])
                    finish_reason = choice.get("finish_reason") or finish_reason
    except TimeoutError:
        timed_out = True
    if finish_reason is None:
        if not timed_out:
            raise RuntimeError("stream ended without a finish_reason")
        if not any("".join(chunks).strip() for chunks in parts.values()):
            raise TimeoutError(f"stream timed out after {timeout} s with no output")
        reason = "timeout"
    elif finish_reason == "length":
        cap = payload.get("max_tokens")
        used = (response.get("usage") or {}).get("completion_tokens")
        reason = "context_limit" if cap is None or (used is not None and used < cap) else "max_tokens"
    else:
        reason = None
    message = {"role": "assistant", **{field: "".join(chunks) or None for field, chunks in parts.items()}}
    response["choices"] = [{"index": 0, "message": message, "finish_reason": finish_reason}]
    response["incomplete_reason"] = reason
    return response


def call_model(model: dict[str, Any], prompt: str, streaming: bool = False) -> dict[str, Any]:
    provider = model["provider"]
    if streaming and provider != "openai_compatible":
        raise ValueError(f"streaming supports only openai_compatible models, not {provider}")
    key = read_key(model["key_file"])
    payload = build_payload(provider, model, prompt)
    timeout = int(model.get("request_timeout_seconds", 1800))
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        headers["Authorization"] = f"Bearer {key}"
        response = http_json(
            "https://api.openai.com/v1/responses",
            "POST",
            headers,
            payload,
            timeout=timeout,
        )
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
        response = http_json(
            "https://api.anthropic.com/v1/messages",
            "POST",
            headers,
            payload,
            timeout=timeout,
        )
        usage = response.get("usage", {})
        output_details = usage.get("output_tokens_details", {}) or {}
        return {
            "text": _anthropic_text(response),
            "resolved_model": response.get("model"),
            "provider": "Anthropic",
            "finish_reason": response.get("stop_reason"),
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "reasoning_tokens": output_details.get("thinking_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
            },
            "response_id": response.get("id"),
            "raw": response,
        }
    if provider == "gemini":
        headers["x-goog-api-key"] = key
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model['model_id']}:generateContent"
        response = http_json(url, "POST", headers, payload, timeout=timeout)
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
    if provider in {"openrouter", "openai_compatible"}:
        headers["Authorization"] = f"Bearer {key}"
        url = (
            "https://openrouter.ai/api/v1/chat/completions"
            if provider == "openrouter"
            else f"{_endpoint_base_url(model)}/chat/completions"
        )
        if streaming:
            stream_payload = {**payload, "stream": True, "stream_options": STREAM_OPTIONS}
            response = _stream_chat_completion(url, headers, stream_payload, timeout)
        else:
            response = http_json(url, "POST", headers, payload, timeout=timeout)
        usage = response.get("usage", {})
        details = usage.get("completion_tokens_details", {}) or {}
        choices = response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        text = str(
            message.get("content")
            or message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        ).strip()
        result = {
            "text": text,
            "resolved_model": response.get("model"),
            "provider": (
                response.get("provider") or model.get("endpoint")
                if provider == "openrouter"
                else model.get("provider_name", "OpenAI-compatible")
            ),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": details.get("reasoning_tokens", 0),
            },
            "response_id": response.get("id"),
            "raw": response,
        }
        if streaming:
            result["incomplete_reason"] = response["incomplete_reason"]
        return result
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
    if provider == "openai_compatible":
        data = http_json(
            f"{_endpoint_base_url(model)}/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        ids = {item.get("id") for item in data.get("data", [])}
        return {
            "available": model["model_id"] in ids,
            "resolved": model["model_id"] if model["model_id"] in ids else None,
        }
    raise ValueError(f"unsupported provider: {provider}")


def require_endpoint_canary(model: dict[str, Any]) -> dict[str, Any] | None:
    if model["provider"] != "openai_compatible":
        return None
    catalog = catalog_check(model)
    if not catalog["available"]:
        raise RuntimeError(
            f"served model {model['model_id']!r} is absent from the endpoint model catalog"
        )
    probe = call_model(
        {**model, "max_output_tokens": int(model.get("canary_max_tokens", 32))},
        "Reply with OK.",
    )
    if not probe["text"]:
        raise RuntimeError("endpoint inference canary returned no text")
    if probe.get("resolved_model") != model["model_id"]:
        raise RuntimeError(
            f"endpoint inference returned model {probe.get('resolved_model')!r}, "
            f"expected exact ID {model['model_id']!r}"
        )
    return {
        "model_id": model["model_id"],
        "catalog": "passed",
        "inference": "passed",
        "resolved_model": probe.get("resolved_model"),
    }
