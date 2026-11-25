"""LangChain-based crop shortlist reranker with strict Pydantic validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
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

SYSTEM_PROMPT = """You are an agronomy decision assistant helping evaluate a machine-learning shortlist of 3 crops.
Your task is to check whether the near-term weather forecast (up to 15 days) gives a strong, grounded reason to reorder those exact 3 ML candidate crops.

Strict rules:
1. Rank ONLY and ALL of the 3 provided candidate crops (`final_rank` 1, 2, and 3). Never invent a new crop, omit a candidate, or repeat a crop.
2. Ground every weather statement in the exact supplied dated forecast values (dates, temperatures in °C, rainfall in mm, total forecast rainfall, and number of days covered).
3. Treat the forecast strictly as near-term planting and early-establishment context. Do NOT make claims about final crop yield or weather beyond the forecast window.
4. Do NOT confuse the user's manually entered model Rainfall input with the 15-day forecast rainfall; keep them distinct.
5. If the forecast gives insufficient reason to alter the ML order, retain the original ML ranking (`1 -> 2 -> 3`) and explain clearly why the ML order was kept.
6. Return valid JSON matching the requested schema only."""

USER_PROMPT = """Farmer's manually entered soil & climate values (used as-is by the ML model):
- Nitrogen (N): {nitrogen} kg/ha
- Phosphorus (P): {phosphorus} kg/ha
- Potassium (K): {potassium} kg/ha
- Temperature: {temperature} °C
- Humidity: {humidity} %
- Soil pH: {ph}
- Model Rainfall input (separate from forecast): {rainfall} mm

Top 3 ML Random Forest candidates (immutable ML ranks and scores):
{candidates_block}

Selected location & near-term weather summary:
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
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "20.0"))
    except ValueError:
        timeout = 20.0

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

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.api_key,
            temperature=0,
            timeout=settings.timeout,
        )

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

    chain = prompt | chat_model | StrOutputParser()

    try:
        raw_output = chain.invoke(
            {
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
        )
    except Exception as exc:
        logger.warning("LangChain reranker call failed: %s", exc)
        return build_fallback_ranking(
            ml_top_3=ml_top_3,
            ai_status="unavailable",
            ai_message=(
                "AI reranking service failed or timed out. Showing the original ML top 3."
            ),
            provider=resolved_settings.provider,
            model=resolved_settings.model,
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
        model=resolved_settings.model,
    )
