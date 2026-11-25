"""Shared Gemini client helpers, text/JSON extraction, and 503 overload recovery."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")

PRIMARY_GEMINI_MODEL = "gemini-3.5-flash-lite"
OVERLOAD_FALLBACK_GEMINI_MODELS = ("gemini-3.1-flash-lite",)
DEFAULT_PRIMARY_503_DELAYS = (0.6, 1.2, 2.0)
DEFAULT_FALLBACK_503_DELAYS = (0.6, 1.0)


def extract_message_text(response: Any) -> str:
    """Extract text cleanly whether LangChain returns an AIMessage, str, or content blocks."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content)


def extract_json_payload(raw_text: str) -> str:
    """Strip optional markdown code fences or locate outer JSON object."""
    text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def build_gemini_chat_kwargs(
    model_name: str,
    api_key: str,
    timeout: float,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Build ChatGoogleGenerativeAI kwargs tuned for fast JSON responses."""
    kwargs: dict[str, Any] = {
        "model": model_name,
        "google_api_key": api_key,
        "timeout": timeout,
        "max_retries": 0,
        "response_mime_type": "application/json",
    }
    if "2.5" in model_name:
        kwargs["thinking_budget"] = 0
        kwargs["temperature"] = temperature
    elif model_name.startswith("gemini-3"):
        level = "minimal" if "lite" in model_name else "low"
        kwargs["thinking_config"] = {"thinking_level": level}
    elif model_name.startswith("gemini-4"):
        kwargs["thinking_config"] = {"thinking_level": "low"}
    else:
        kwargs["temperature"] = temperature
    return kwargs


def is_overload_error(exc: Exception) -> bool:
    """Return True only for temporary server-side 503 overload, never 429 quota or auth errors."""
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    raw_msg = str(exc)
    lower_msg = raw_msg.lower()

    if status in {400, 401, 403, 404, 429}:
        return False
    non_overload_tokens = (
        "429",
        "resource_exhausted",
        "401",
        "403",
        "404",
        "api_key_invalid",
        "api key not valid",
        "not_found",
    )
    if any(tok in lower_msg for tok in non_overload_tokens):
        return False

    return (
        status == 503
        or "503" in raw_msg
        or "UNAVAILABLE" in raw_msg
        or "overloaded" in lower_msg
        or "high demand" in lower_msg
    )


def _retry_on_503(
    attempt_fn: Callable[[], T],
    model_name: str,
    retry_delays: tuple[float, ...],
    deadline: float,
    sleep_fn: Callable[[float], Any],
    logger: logging.Logger,
    context_label: str,
) -> T:
    """Invoke a single model with bounded exponential backoff specifically on HTTP 503."""
    total_attempts = 1 + len(retry_delays)
    for attempt_idx in range(total_attempts):
        try:
            return attempt_fn()
        except Exception as exc:
            if not is_overload_error(exc) or attempt_idx >= len(retry_delays):
                raise
            delay = retry_delays[attempt_idx]
            if time.monotonic() + delay >= deadline:
                raise
            logger.warning(
                "%s %s hit HTTP 503 (%d/%d); retrying in %.1fs.",
                context_label,
                model_name,
                attempt_idx + 1,
                total_attempts,
                delay,
            )
            sleep_fn(delay)
    raise AssertionError("Unreachable retry loop exit")


def invoke_gemini_with_503_recovery(
    invoke_fn: Callable[[str | None], T],
    *,
    is_gemini: bool,
    primary_model: str | None,
    allow_model_fallback: bool,
    sleep_fn: Callable[[float], Any],
    logger: logging.Logger,
    context_label: str,
    primary_retry_delays: tuple[float, ...] = DEFAULT_PRIMARY_503_DELAYS,
    fallback_models: tuple[str, ...] = OVERLOAD_FALLBACK_GEMINI_MODELS,
    fallback_retry_delays: tuple[float, ...] = DEFAULT_FALLBACK_503_DELAYS,
    max_total_seconds: float = 32.0,
) -> tuple[T, str | None]:
    """Run invoke_fn with 503 backoff and optional overload-only Flash-Lite fallback.

    Non-503 failures (such as 429 quota exhaustion or auth errors) are raised
    immediately without sleeping or rotating models.
    """
    if not is_gemini:
        return invoke_fn(None), primary_model

    deadline = time.monotonic() + max_total_seconds
    try:
        result = _retry_on_503(
            lambda: invoke_fn(None),
            model_name=primary_model or PRIMARY_GEMINI_MODEL,
            retry_delays=primary_retry_delays,
            deadline=deadline,
            sleep_fn=sleep_fn,
            logger=logger,
            context_label=context_label,
        )
        return result, primary_model
    except Exception as primary_exc:
        if not (allow_model_fallback and is_overload_error(primary_exc)):
            raise

        for fb_model in fallback_models:
            if fb_model == primary_model or time.monotonic() >= deadline:
                continue
            logger.warning(
                "%s %s remained overloaded (503); trying overload fallback %s.",
                context_label,
                primary_model,
                fb_model,
            )
            try:
                result = _retry_on_503(
                    lambda model=fb_model: invoke_fn(model),
                    model_name=fb_model,
                    retry_delays=fallback_retry_delays,
                    deadline=deadline,
                    sleep_fn=sleep_fn,
                    logger=logger,
                    context_label=context_label,
                )
                return result, fb_model
            except Exception as fb_exc:
                # Never replace a primary 503 with a secondary model's 429 quota error.
                if not is_overload_error(fb_exc):
                    raise primary_exc from fb_exc
                primary_exc = fb_exc

        raise primary_exc
