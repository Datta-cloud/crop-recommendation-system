"""End-to-end crop recommendation orchestration combining ML, weather, and AI reranking."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from schemas import (
    LocationOption,
    RecommendResponse,
    SelectedLocation,
    SoilClimateInput,
)
from services.ml_service import predict_top_k
import services.reranker as reranker_service
import services.weather as weather_service


def generate_recommendation(
    model: Any,
    crop_names: list[str],
    features: SoilClimateInput,
    selected_location: SelectedLocation | None = None,
    location_query: str | None = None,
    use_ai: bool = True,
    chat_model: BaseChatModel | None = None,
    llm_settings: reranker_service.LLMSettings | None = None,
) -> RecommendResponse:
    """Compute ML top 3, optionally fetch 15-day weather, and run AI reranking safely."""
    ml_top_3 = predict_top_k(model, crop_names, features, k=3)
    input_dict = features.to_display_dict()

    resolved_location = selected_location
    ambiguous_locations: list[LocationOption] = []
    weather_summary = None
    weather_status = "skipped"
    weather_message: str | None = None

    if resolved_location is None and location_query and location_query.strip():
        try:
            matches = weather_service.search_locations(location_query.strip())
            if len(matches) == 1:
                resolved_location = SelectedLocation(**matches[0].model_dump())
            elif len(matches) > 1:
                ambiguous_locations = matches
                weather_status = "ambiguous_location"
                weather_message = (
                    f"Multiple locations matched '{location_query.strip()}'. "
                    "Select the correct district/state and country to include weather."
                )
            else:
                weather_status = "not_found"
                weather_message = (
                    f"No matching location found for '{location_query.strip()}'. "
                    "Showing the original ML top 3."
                )
        except weather_service.WeatherTimeoutError:
            weather_status = "timeout"
            weather_message = (
                "Location lookup timed out. Showing the original ML top 3."
            )
        except weather_service.WeatherUnavailableError:
            weather_status = "unavailable"
            weather_message = (
                "Location lookup service is currently unavailable. Showing the original ML top 3."
            )

    if resolved_location is not None:
        try:
            weather_summary = weather_service.fetch_weather_forecast(resolved_location)
            weather_status = "ok"
            weather_message = weather_summary.forecast_note
        except weather_service.WeatherTimeoutError:
            weather_status = "timeout"
            weather_message = (
                "Weather forecast request timed out. Showing the original ML top 3."
            )
        except weather_service.WeatherUnavailableError as exc:
            weather_status = "unavailable"
            weather_message = f"{exc} Showing the original ML top 3."
    elif weather_status == "skipped":
        weather_message = (
            "No location selected; weather forecast was not fetched. Showing the original ML top 3."
        )

    if weather_summary is not None and use_ai:
        rerank_result = reranker_service.rerank_with_langchain(
            features=features,
            ml_top_3=ml_top_3,
            weather=weather_summary,
            chat_model=chat_model,
            settings=llm_settings,
        )
    elif weather_summary is None:
        ai_fallback_msg = (
            "AI reranking was skipped because 15-day location weather is unavailable."
            if weather_status in {"timeout", "unavailable", "not_found", "ambiguous_location"}
            else "AI reranking was skipped because no location was selected."
        )
        rerank_result = reranker_service.build_fallback_ranking(
            ml_top_3=ml_top_3,
            ai_status="skipped",
            ai_message=ai_fallback_msg,
        )
    else:
        rerank_result = reranker_service.build_fallback_ranking(
            ml_top_3=ml_top_3,
            ai_status="skipped",
            ai_message="AI reranking disabled by request. Showing the original ML top 3.",
        )

    return RecommendResponse(
        input_features=input_dict,
        ml_top_3=ml_top_3,
        recommended_crop=rerank_result.final_ranking[0].crop,
        final_ranking=rerank_result.final_ranking,
        order_changed=rerank_result.order_changed,
        why_this_crop=rerank_result.why_this_crop,
        order_change_explanation=rerank_result.order_change_explanation,
        location=resolved_location,
        ambiguous_locations=ambiguous_locations,
        weather=weather_summary,
        weather_status=weather_status,
        weather_message=weather_message,
        ai_status=rerank_result.ai_status,
        ai_message=rerank_result.ai_message,
        reranker_provider=rerank_result.provider,
        reranker_model=rerank_result.model,
    )
