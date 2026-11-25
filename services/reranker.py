"""LangChain-based crop shortlist reranker with strict Pydantic validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from schemas import (
    LLMRerankOutput,
    MLCandidate,
    RankedCropResult,
    SoilClimateInput,
    WeatherSummary,
    validate_rerank_against_shortlist,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "google": "gemini-2.5-flash",
    "openai_compatible": "gpt-4o-mini",
}

SYSTEM_PROMPT = """You are an agronomy decision assistant helping evaluate a machine-learning shortlist of 3 crops for a farmer's selected location.
Your task is to evaluate whether the location's current weather, current date/sowing season, and near-term forecast (up to 15 days) mean the 3 ML candidate crops should be reordered right now.

Strict rules:
1. Rank ONLY and ALL of the 3 provided candidate crops (`final_rank` 1, 2, and 3). Never invent a new crop, omit a candidate, or repeat a crop.
2. Actively compare all 3 candidates against the selected location, current date/month (regional sowing window), current weather (temperature, humidity, conditions), and the dated forecast values (dates, min/max °C, rainfall mm, total forecast rainfall, days covered).
3. If the ML #1 crop is poorly suited to the location's current weather, near-term rainfall/temperature, or current planting season while another shortlisted candidate with meaningful soil suitability fits current conditions better, REORDER the shortlist and promote that better-suited candidate to `final_rank` 1.
4. Do NOT promote a candidate with ~0% ML soil suitability (such as a temperate fruit crop in hot plains) just to change the order. If ML #1 stays #1 because the other shortlist options are also unsuitable or have near-0% soil fit, clearly warn the farmer if ML #1 is off-season or mismatched with the location's current dry/wet weather (for example, rice in dry November weather with 0.0 mm rain requires assured irrigation or may not be sown now).
5. Ground every weather statement in exact supplied figures. Treat the forecast strictly as near-term planting and early-establishment context; do NOT make claims about final yield or weather beyond the forecast window.
6. Do NOT confuse the farmer's manually entered Model Rainfall input with the forecast rainfall; keep them distinct.
7. Return valid JSON matching the requested schema only."""

USER_PROMPT = """Selected Location & Current Date:
- Location: {location_name}
- Current Date: {current_date}

Farmer's manually entered soil & climate values (used as-is by the ML model):
- Nitrogen (N): {nitrogen} kg/ha
- Phosphorus (P): {phosphorus} kg/ha
- Potassium (K): {potassium} kg/ha
- Temperature: {temperature} °C
- Humidity: {humidity} %
- Soil pH: {ph}
- Model Rainfall input (separate from forecast): {rainfall} mm

Top 3 ML Random Forest candidates (immutable ML ranks and scores):
{candidates_block}

Location current weather & near-term forecast summary:
{weather_block}

{format_instructions}"""


class RerankerUnavailableError(Exception):
    """Raised when LLM provider credentials or configuration are missing."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str | None
    model: str | None
    api_key: str | None
    base_url: str | None
    timeout: float = 20.0


@dataclass
class RerankOutcome:
    final_ranking: list[RankedCropResult]
    order_changed: bool
    why_this_crop: str
    order_change_explanation: str
    ai_status: str
    ai_message: str | None
    provider: str | None = None
    model: str | None = None


def load_llm_settings() -> LLMSettings:
    """Resolve configurable LLM provider, model, and server-side API key from env."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    explicit_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None

    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "45.0"))
    except ValueError:
        timeout = 45.0

    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"

    if not provider:
        return LLMSettings(
            provider=None,
            model=None,
            api_key=None,
            base_url=base_url,
            timeout=timeout,
        )

    provider_keys = {
        "openai": os.getenv("OPENAI_API_KEY", "").strip(),
        "openai_compatible": os.getenv("OPENAI_API_KEY", "").strip(),
        "groq": os.getenv("GROQ_API_KEY", "").strip(),
        "gemini": (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
        ),
        "google": (
            os.getenv("GOOGLE_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
        ),
    }

    api_key = explicit_key or provider_keys.get(provider, "") or None
    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS.get(provider)

    return LLMSettings(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


def build_chat_model(settings: LLMSettings) -> BaseChatModel:
    """Instantiate the configured LangChain chat model using server-side credentials."""
    if not settings.provider or not settings.api_key:
        raise RerankerUnavailableError(
            "Server-side LLM provider or API key is not configured."
        )

    provider = settings.provider
    model_name = settings.model or DEFAULT_MODELS.get(provider, "")

    if provider in {"openai", "openai_compatible"}:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": settings.api_key,
            "temperature": 0,
            "timeout": settings.timeout,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        return ChatOpenAI(**kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name,
            api_key=settings.api_key,
            temperature=0,
            timeout=settings.timeout,
        )

    if provider in {"gemini", "google"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_kwargs: dict[str, Any] = {
            "model": model_name,
            "google_api_key": settings.api_key,
            "timeout": settings.timeout,
            "max_retries": 2,
            "response_mime_type": "application/json",
        }
        if "2.5" in model_name:
            # Disable hidden thinking tokens so JSON reranking completes in ~1-2s.
            gemini_kwargs["thinking_budget"] = 0
            gemini_kwargs["temperature"] = 0
        elif not model_name.startswith("gemini-3"):
            gemini_kwargs["temperature"] = 0
        return ChatGoogleGenerativeAI(**gemini_kwargs)

    raise RerankerUnavailableError(f"Unsupported LLM provider: {provider}")


def _format_candidates_block(ml_top_3: list[MLCandidate]) -> str:
    lines = []
    for candidate in ml_top_3:
        lines.append(
            f"- ML Rank #{candidate.ml_rank}: {candidate.crop} "
            f"(ML model score: {candidate.ml_score_pct:.2f}% / {candidate.ml_score:.4f})"
        )
    return "\n".join(lines)


def _format_weather_block(weather: WeatherSummary) -> str:
    lines = [
        f"Location: {weather.location.display_name} "
        f"({weather.location.latitude:.4f}, {weather.location.longitude:.4f})"
    ]
    if weather.current:
        humidity_text = (
            f"{weather.current.humidity_pct}%"
            if weather.current.humidity_pct is not None
            else "N/A"
        )
        lines.append(
            f"Current weather: {weather.current.temperature_c}°C, "
            f"humidity {humidity_text}, precipitation {weather.current.precipitation_mm} mm, "
            f"{weather.current.condition}"
        )

    lines.append(f"Forecast summary: {weather.forecast_note}")
    lines.append(
        f"Coverage: {weather.days_covered} days (requested {weather.days_requested}), "
        f"total forecast rainfall = {weather.total_rainfall_mm:.2f} mm, "
        f"average min/max = {weather.avg_temp_min_c}°C / {weather.avg_temp_max_c}°C"
    )
    lines.append("Dated daily forecast:")
    for day in weather.daily:
        lines.append(
            f"  - {day.date}: {day.temp_min_c}°C to {day.temp_max_c}°C, "
            f"rainfall {day.rainfall_mm:.2f} mm ({day.condition})"
        )
    return "\n".join(lines)


def _extract_message_text(response: Any) -> str:
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


def _extract_json_payload(raw_text: str) -> str:
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


def _safe_llm_error_summary(exc: Exception) -> str:
    """Summarize an LLM provider failure safely without exposing secrets."""
    exc_name = type(exc).__name__
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    raw_msg = str(exc)
    if "429" in raw_msg or "RESOURCE_EXHAUSTED" in raw_msg or status_code == 429:
        return "AI reranking quota or rate limit exceeded (HTTP 429). Showing the original ML top 3."
    if "401" in raw_msg or "403" in raw_msg or "API_KEY_INVALID" in raw_msg or "API key not valid" in raw_msg:
        return "AI reranking failed due to invalid or unauthorized server API key. Showing the original ML top 3."
    if "404" in raw_msg or "NOT_FOUND" in raw_msg or status_code == 404:
        return "AI reranking model was not found for the configured provider (HTTP 404). Showing the original ML top 3."
    if "timeout" in raw_msg.lower() or "timed out" in raw_msg.lower():
        return "AI reranking request timed out. Showing the original ML top 3."
    code_part = f" (status {status_code})" if status_code else ""
    return f"AI reranking service failed ({exc_name}{code_part}). Showing the original ML top 3."


def build_fallback_ranking(
    ml_top_3: list[MLCandidate],
    ai_status: str,
    ai_message: str,
    provider: str | None = None,
    model: str | None = None,
) -> RerankOutcome:
    """Return the original ML top 3 untouched when AI or weather is unavailable."""
    fallback_rows = [
        RankedCropResult(
            crop=item.crop,
            ml_rank=item.ml_rank,
            ml_score=item.ml_score,
            ml_score_pct=item.ml_score_pct,
            final_rank=item.ml_rank,
            reason=(
                f"Ranked #{item.ml_rank} by the Random Forest model with an ML model score "
                f"of {item.ml_score_pct:.2f}%."
            ),
        )
        for item in ml_top_3
    ]
    top_crop = fallback_rows[0]
    why_text = (
        f"{top_crop.crop.title()} holds ML rank #1 with an ML model score of "
        f"{top_crop.ml_score_pct:.2f}% based on your entered soil and climate parameters."
    )
    return RerankOutcome(
        final_ranking=fallback_rows,
        order_changed=False,
        why_this_crop=why_text,
        order_change_explanation=ai_message,
        ai_status=ai_status,
        ai_message=ai_message,
        provider=provider,
        model=model,
    )


def rerank_with_langchain(
    features: SoilClimateInput,
    ml_top_3: list[MLCandidate],
    weather: WeatherSummary,
    chat_model: BaseChatModel | None = None,
    settings: LLMSettings | None = None,
) -> RerankOutcome:
    """Run LangChain prompt + Pydantic validation to evaluate near-term weather reranking."""
    resolved_settings = settings or load_llm_settings()
    used_model_name = resolved_settings.model
    custom_chat_model = chat_model is not None

    if chat_model is None:
        try:
            chat_model = build_chat_model(resolved_settings)
        except RerankerUnavailableError:
            return build_fallback_ranking(
                ml_top_3=ml_top_3,
                ai_status="unavailable",
                ai_message=(
                    "AI reranking is unavailable (no server-side LLM API key configured). "
                    "Showing the original ML top 3."
                ),
                provider=resolved_settings.provider,
                model=resolved_settings.model,
            )

    parser = PydanticOutputParser(pydantic_object=LLMRerankOutput)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    current_date = (
        weather.start_date
        or (weather.current.observed_at[:10] if weather.current and weather.current.observed_at else None)
        or "unknown"
    )
    prompt_inputs = {
        "location_name": weather.location.display_name,
        "current_date": current_date,
        "nitrogen": features.Nitrogen,
        "phosphorus": features.Phosphorus,
        "potassium": features.Potassium,
        "temperature": features.Temperature,
        "humidity": features.Humidity,
        "ph": features.pH,
        "rainfall": features.Rainfall,
        "candidates_block": _format_candidates_block(ml_top_3),
        "weather_block": _format_weather_block(weather),
        "format_instructions": parser.get_format_instructions(),
    }

    try:
        raw_msg = (prompt | chat_model).invoke(prompt_inputs)
        raw_output = _extract_message_text(raw_msg)
    except Exception as exc:
        transient_tokens = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")
        if (
            not custom_chat_model
            and resolved_settings.provider in {"gemini", "google"}
            and used_model_name == "gemini-2.5-flash"
            and any(tok in str(exc) for tok in transient_tokens)
        ):
            last_fb_exc: Exception = exc
            raw_output = ""
            for fb_name in ("gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"):
                logger.warning("%s hit 429/503; trying fallback %s.", used_model_name, fb_name)
                try:
                    fallback_settings = LLMSettings(
                        provider=resolved_settings.provider,
                        model=fb_name,
                        api_key=resolved_settings.api_key,
                        base_url=resolved_settings.base_url,
                        timeout=resolved_settings.timeout,
                    )
                    fallback_model = build_chat_model(fallback_settings)
                    raw_msg = (prompt | fallback_model).invoke(prompt_inputs)
                    raw_output = _extract_message_text(raw_msg)
                    used_model_name = fb_name
                    break
                except Exception as fb_exc:
                    last_fb_exc = fb_exc
                    used_model_name = fb_name
                    if not any(tok in str(fb_exc) for tok in transient_tokens):
                        break
            if not raw_output:
                logger.warning("LangChain reranker fallback failed: %s", last_fb_exc)
                return build_fallback_ranking(
                    ml_top_3=ml_top_3,
                    ai_status="unavailable",
                    ai_message=_safe_llm_error_summary(last_fb_exc),
                    provider=resolved_settings.provider,
                    model=used_model_name,
                )
        else:
            logger.warning("LangChain reranker call failed: %s", exc)
            return build_fallback_ranking(
                ml_top_3=ml_top_3,
                ai_status="unavailable",
                ai_message=_safe_llm_error_summary(exc),
                provider=resolved_settings.provider,
                model=used_model_name,
            )

    try:
        json_text = _extract_json_payload(raw_output)
        parsed = LLMRerankOutput.model_validate_json(json_text)
        expected_crops = [candidate.crop for candidate in ml_top_3]
        validate_rerank_against_shortlist(parsed, expected_crops)
    except (ValidationError, ValueError, json.JSONDecodeError):
        return build_fallback_ranking(
            ml_top_3=ml_top_3,
            ai_status="invalid_response",
            ai_message=(
                "AI reranking returned an invalid response (incomplete, duplicate, or "
                "non-shortlist crops) and was rejected. Showing the original ML top 3."
            ),
            provider=resolved_settings.provider,
            model=resolved_settings.model,
        )

    ml_lookup = {candidate.crop.lower(): candidate for candidate in ml_top_3}
    sorted_items = sorted(parsed.ranked_crops, key=lambda item: item.final_rank)

    final_ranking: list[RankedCropResult] = []
    for item in sorted_items:
        ml_item = ml_lookup[item.crop]
        final_ranking.append(
            RankedCropResult(
                crop=ml_item.crop,
                ml_rank=ml_item.ml_rank,
                ml_score=ml_item.ml_score,
                ml_score_pct=ml_item.ml_score_pct,
                final_rank=item.final_rank,
                reason=item.reason,
            )
        )

    actual_order_changed = [row.crop for row in final_ranking] != [
        c.crop for c in ml_top_3
    ]
    top_pick = final_ranking[0]

    return RerankOutcome(
        final_ranking=final_ranking,
        order_changed=actual_order_changed,
        why_this_crop=top_pick.reason,
        order_change_explanation=parsed.order_change_explanation,
        ai_status="ok",
        ai_message=None,
        provider=resolved_settings.provider,
        model=used_model_name,
    )
