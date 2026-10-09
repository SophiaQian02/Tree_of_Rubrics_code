from __future__ import annotations

import copy
from functools import lru_cache
import os
import random
import re
import sys
import time
from typing import Any, Callable

DEFAULT_API_REQUEST_USER = ""


def api_request_user() -> str:
    """Return the trace user written into every OpenAI request body."""
    return os.getenv("API_REQUEST_USER", DEFAULT_API_REQUEST_USER).strip()


def api_request_user_kwargs() -> dict[str, str]:
    """Build kwargs for direct SDK calls while allowing an explicit opt-out."""
    value = api_request_user()
    return {"user": value} if value else {}


@lru_cache(maxsize=64)
def _cached_openai_client(
    base_url: str,
    api_key: str,
    api_user: str,
    api_user_header: str,
    timeout: float,
    max_retries: int,
) -> Any:
    from openai import OpenAI

    default_headers = {api_user_header: api_user} if api_user and api_user_header else None
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        timeout=timeout,
        max_retries=max_retries,
    )


def create_openai_client(base_url: str, api_key: str) -> Any:
    """Reuse OpenAI-compatible clients so long runs do not leak socket FDs."""
    api_user = os.getenv("API_USER", "").strip()
    api_user_header = os.getenv("API_USER_HEADER", "X-User-ID").strip()
    return _cached_openai_client(
        base_url=str(base_url).rstrip("/"),
        api_key=str(api_key),
        api_user=api_user,
        api_user_header=api_user_header,
        timeout=float(os.getenv("ARTS_OPENAI_TIMEOUT", "300")),
        max_retries=int(os.getenv("ARTS_OPENAI_CLIENT_MAX_RETRIES", "2")),
    )


def clip_text_middle(text: str, max_chars: int, marker: str = "\n...[middle truncated]...\n") -> str:
    """Keep both ends of a long field so instructions and conclusions survive."""
    value = str(text)
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if max_chars <= len(marker) + 16:
        return value[:max_chars]
    remaining = max_chars - len(marker)
    head = remaining // 2
    tail = remaining - head
    return value[:head] + marker + value[-tail:]


def _context_limit_from_error(error: Exception) -> tuple[int, int] | None:
    message = " ".join(str(error).split())
    patterns = (
        r"maximum context length is\s*(\d+)\s*tokens?.*?request has\s*(\d+)\s*input tokens?",
        r"maximum context length is\s*(\d+)\s*tokens?.*?resulted in\s*(\d+)\s*tokens?",
        r"context length.*?maximum\s*(?:is|of)?\s*(\d+).*?(?:input|prompt).*?(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _truncate_messages_for_context(
    messages: list[dict[str, Any]],
    *,
    context_limit: int,
    input_tokens: int,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    if input_tokens <= 0 or context_limit <= 0:
        return None
    safety = int(os.getenv("ARTS_CONTEXT_SAFETY_TOKENS", "256"))
    allowed_input = context_limit - max(0, max_tokens) - safety
    if allowed_input < 256 or input_tokens <= allowed_input:
        return None
    ratio = max(0.10, min(0.95, allowed_input / input_tokens * 0.95))
    result = copy.deepcopy(messages)
    candidates = [
        (index, str(item.get("content", "")))
        for index, item in enumerate(result)
        if isinstance(item, dict) and isinstance(item.get("content"), str)
    ]
    if not candidates:
        return None
    old_chars = sum(len(content) for _index, content in candidates)
    target_chars = max(512, int(old_chars * ratio))
    longest_index, longest_content = max(candidates, key=lambda item: len(item[1]))
    fixed_chars = sum(len(content) for index, content in candidates if index != longest_index)
    longest_target = max(256, target_chars - max(0, fixed_chars))
    if longest_target >= len(longest_content):
        longest_target = max(256, int(len(longest_content) * ratio))
    result[longest_index]["content"] = clip_text_middle(longest_content, longest_target)
    new_chars = sum(len(str(item.get("content", ""))) for item in result if isinstance(item, dict))
    if new_chars >= old_chars:
        return None
    return result, {
        "context_limit": context_limit,
        "input_tokens": input_tokens,
        "allowed_input_tokens": allowed_input,
        "old_chars": old_chars,
        "new_chars": new_chars,
    }


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_retryable_error(error: Exception) -> bool:
    code = _status_code(error)
    if code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    message = " ".join(str(error).lower().split())
    return any(
        token in message
        for token in (
            "timeout",
            "timed out",
            "connection reset",
            "connection error",
            "temporarily unavailable",
            "internal server error",
            "httpcode': 502",
            '"httpcode": 502',
            "code: 502",
        )
    )


def chat_completion_with_recovery(
    client: Any,
    *,
    retry_log: Callable[[dict[str, Any]], None] | None = None,
    request_retries: int | None = None,
    **kwargs: Any,
) -> Any:
    """Call chat completions with context recovery and transient-error backoff."""
    attempts = max(1, request_retries or int(os.getenv("ARTS_REQUEST_MAX_RETRIES", "5")))
    base_sleep = float(os.getenv("ARTS_REQUEST_RETRY_BASE_SECONDS", "2"))
    max_sleep = float(os.getenv("ARTS_REQUEST_RETRY_MAX_SECONDS", "30"))
    request = dict(kwargs)
    request_user = api_request_user()
    if request_user and not request.get("user"):
        request["user"] = request_user
    # Some OpenAI-compatible relays require an explicit non-streaming flag.
    request.setdefault("stream", False)
    request["messages"] = copy.deepcopy(kwargs.get("messages", []))
    for attempt in range(1, attempts + 1):
        try:
            return client.chat.completions.create(**request)
        except Exception as error:
            limits = _context_limit_from_error(error)
            if limits is not None and attempt < attempts:
                truncated = _truncate_messages_for_context(
                    request["messages"],
                    context_limit=limits[0],
                    input_tokens=limits[1],
                    max_tokens=int(request.get("max_tokens") or request.get("max_completion_tokens") or 0),
                )
                if truncated is not None:
                    request["messages"], diagnostic = truncated
                    event = {
                        "step": "context_input_truncated_retry",
                        "attempt": attempt,
                        "model": str(request.get("model", "")),
                        "error_excerpt": " ".join(str(error).split())[:500],
                        **diagnostic,
                    }
                    if retry_log is not None:
                        retry_log(event)
                    else:
                        print(f"[ARTS RETRY] {event}", file=sys.stderr, flush=True)
                    continue
            if attempt >= attempts or not _is_retryable_error(error):
                raise
            delay = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            delay *= 0.9 + random.random() * 0.2
            event = {
                "step": "transient_request_retry",
                "attempt": attempt,
                "model": str(request.get("model", "")),
                "delay_seconds": round(delay, 3),
                "status_code": _status_code(error),
                "error_excerpt": " ".join(str(error).split())[:500],
            }
            if retry_log is not None:
                retry_log(event)
            else:
                print(f"[ARTS RETRY] {event}", file=sys.stderr, flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")
