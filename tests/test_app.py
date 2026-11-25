"""Tests for FastAPI endpoints, ML top-3 probabilities, weather, and LangChain reranking."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import app as app_module
from schemas import (
    CurrentWeather,
    DailyForecast,
    LocationOption,
    SelectedLocation,
    SoilClimateInput,
    WeatherSummary,
)
from services.reranker import LLMSettings, rerank_with_langchain
from services.weather import (
    WeatherTimeoutError,
    fetch_weather_forecast,
    search_locations,
)

BASE_DIR = Path(__file__).resolve().parent.parent

RICE_FORM = {
    "Nitrogen": "90",
    "Phosphorus": "42",
    "Potassium": "43",
    "Temperature": "20.88",
    "Humidity": "82.00",
    "pH": "6.50",
    "Rainfall": "202.94",
}


class RecordingFakeChatModel(BaseChatModel):
    """Deterministic LangChain chat model capturing formatted prompts."""

    response_payload: Any
    recorded_prompts: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "recording-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        joined = "\n".join(str(m.content) for m in messages)
        self.recorded_prompts.append(joined)
        if callable(self.response_payload):
            content = self.response_payload(joined)
        elif isinstance(self.response_payload, dict):
            content = json.dumps(self.response_payload)
        else:
            content = self.response_payload
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def _make_weather_summary(
    location: SelectedLocation,
    daily_rainfalls: list[float],
    start_day: int = 1,
) -> WeatherSummary:
    daily: list[DailyForecast] = []
    for idx, rain in enumerate(daily_rainfalls):
        day_num = start_day + idx
        daily.append(
            DailyForecast(
                date=f"2026-12-{day_num:02d}",
                temp_min_c=19.0 + (idx % 3),
                temp_max_c=31.0 + (idx % 4),
                rainfall_mm=rain,
                rain_probability_pct=20 if rain > 0 else 5,
                weather_code=61 if rain > 0 else 0,
                condition="Slight rain" if rain > 0 else "Clear sky",
            )
        )

    days_covered = len(daily)
    total_rain = round(sum(d.rainfall_mm for d in daily), 2)
    return WeatherSummary(
        location=location,
        current=CurrentWeather(
            observed_at="2026-12-01T10:00",
            temperature_c=29.5,
            humidity_pct=48.0,
            precipitation_mm=0.0,
            weather_code=0,
            condition="Clear sky",
        ),
        daily=daily,
        days_requested=15,
        days_covered=days_covered,
        is_partial_forecast=days_covered < 15,
        total_rainfall_mm=total_rain,
        avg_temp_min_c=round(sum(d.temp_min_c for d in daily) / days_covered, 1),
        avg_temp_max_c=round(sum(d.temp_max_c for d in daily) / days_covered, 1),
        start_date=daily[0].date,
        end_date=daily[-1].date,
        forecast_note=(
            f"Partial forecast: provider returned {days_covered} of 15 requested days "
            f"({daily[0].date} to {daily[-1].date}); total forecast rainfall is {total_rain:.1f} mm."
            if days_covered < 15
            else f"15-day forecast ({daily[0].date} to {daily[-1].date}); "
            f"total expected rainfall is {total_rain:.1f} mm."
        ),
    )


def test_rice_form_and_top_3_match_predict_proba(monkeypatch):
    """Original form with README rice example returns exact predict_proba() top 3."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with open(BASE_DIR / "crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "crop_names.pkl", "rb") as f:
        crop_names = pickle.load(f)

    feature_vec = np.array([[90.0, 42.0, 43.0, 20.88, 82.0, 6.5, 202.94]])
    probs = model.predict_proba(feature_vec)[0]
    expected_pairs = sorted(
        [(crop_names[int(cid)], float(prob), int(cid)) for cid, prob in zip(model.classes_, probs)],
        key=lambda item: (-item[1], item[2]),
    )[:3]

    with TestClient(app_module.app) as client:
        home_resp = client.get("/")
        assert home_resp.status_code == 200

        form_resp = client.post("/predict", data=RICE_FORM)
        assert form_resp.status_code == 200
        html = form_resp.text
        assert "RICE" in html
        assert "Why this crop?" in html
        assert "ML model score" in html
        for crop_name, _, _ in expected_pairs:
            assert crop_name in html

        api_resp = client.post(
            "/api/recommend",
            json={
                "Nitrogen": 90,
                "Phosphorus": 42,
                "Potassium": 43,
                "Temperature": 20.88,
                "Humidity": 82.0,
                "pH": 6.5,
                "Rainfall": 202.94,
            },
        )
        assert api_resp.status_code == 200
        data = api_resp.json()
        assert data["recommended_crop"] == expected_pairs[0][0]
        assert [c["crop"] for c in data["ml_top_3"]] == [p[0] for p in expected_pairs]
        for actual, (_, exp_prob, _) in zip(data["ml_top_3"], expected_pairs):
            assert actual["ml_score"] == pytest.approx(round(exp_prob, 6), abs=1e-6)


def test_selected_location_weather_and_input_separation(monkeypatch):
    """Selected location fetches 15-day weather without overwriting manual climate inputs."""
    pune = SelectedLocation(
        name="Pune",
        district="Pune",
        state="Maharashtra",
        country="India",
        latitude=18.51957,
        longitude=73.85535,
    )
    wx = _make_weather_summary(pune, [0.0, 2.0, 5.5] + [0.0] * 12)

    monkeypatch.setattr("services.weather.fetch_weather_forecast", lambda loc: wx)
    monkeypatch.setattr(
        "services.reranker.load_llm_settings",
        lambda: LLMSettings(provider="openai", model="gpt-4o-mini", api_key=None, base_url=None),
    )

    with TestClient(app_module.app) as client:
        resp = client.post(
            "/api/recommend",
            json={
                "Nitrogen": 90,
                "Phosphorus": 42,
                "Potassium": 43,
                "Temperature": 20.88,
                "Humidity": 82.0,
                "pH": 6.5,
                "Rainfall": 202.94,
                "location": pune.model_dump(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["weather_status"] == "ok"
        assert body["weather"]["days_covered"] == 15
        assert body["weather"]["total_rainfall_mm"] == pytest.approx(7.5)
        assert body["weather"]["location"]["display_name"] == pune.display_name
        # Manual climate values must never be overwritten by forecast/current weather.
        assert body["input_features"]["Temperature"] == 20.88
        assert body["input_features"]["Humidity"] == 82.0
        assert body["input_features"]["Rainfall"] == 202.94
        # AI unavailable fallback still preserves original ML top 3.
        assert body["ai_status"] == "unavailable"
        assert [r["crop"] for r in body["final_ranking"]] == [c["crop"] for c in body["ml_top_3"]]

        # Also verify HTML /predict renders selected location, weather summary, and comparison columns.
        html_form = {
            **RICE_FORM,
            "location_query": "Pune",
            "location_name": pune.name,
            "location_district": pune.district or "",
            "location_state": pune.state or "",
            "location_country": pune.country or "",
            "location_display_name": pune.display_name,
            "latitude": str(pune.latitude),
            "longitude": str(pune.longitude),
        }
        html_resp = client.post("/predict", data=html_form)
        assert html_resp.status_code == 200
        page = html_resp.text
        assert "Why this crop?" in page
        assert "ML rank" in page
        assert "ML model score" in page
        assert "Final rank" in page
        assert "15 of 15 days" in page
        assert "7.5 mm" in page
        assert "202.9 mm" in page
        assert "2026-12-01" in page
        # Recommended crop card appears before the comparison table and weather section.
        assert page.index("Top 3 Shortlist Comparison") > page.index("Why this crop?")
        assert page.index("Why this crop?") > page.index("RICE")


def test_partial_forecast_days_reported_honestly():
    """Shorter forecast responses report actual days covered instead of claiming 15 days."""
    pune = SelectedLocation(
        name="Pune",
        district="Pune",
        state="Maharashtra",
        country="India",
        latitude=18.52,
        longitude=73.86,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-11-25T15:00",
                    "temperature_2m": 30.0,
                    "relative_humidity_2m": 40,
                    "precipitation": 0.0,
                    "weather_code": 0,
                },
                "daily": {
                    "time": [f"2026-11-{25 + i:02d}" for i in range(6)],
                    "temperature_2m_min": [18.0] * 6,
                    "temperature_2m_max": [31.0] * 6,
                    "precipitation_sum": [0.0, 1.5, 0.0, 2.5, 0.0, 0.0],
                    "precipitation_probability_max": [5, 30, 5, 40, 10, 5],
                    "weather_code": [0, 61, 0, 61, 1, 0],
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    summary = fetch_weather_forecast(pune, forecast_days=15, client=client)
    assert summary.days_requested == 15
    assert summary.days_covered == 6
    assert summary.is_partial_forecast is True
    assert summary.total_rainfall_mm == pytest.approx(4.0)
    assert "6 of 15 requested days" in summary.forecast_note


def test_ambiguous_location_names_surface_disambiguation(monkeypatch):
    """Ambiguous location search exposes district/state and country without guessing."""
    candidates = [
        LocationOption(
            name="Aurangabad",
            district="Chhatrapati Sambhajinagar",
            state="Maharashtra",
            country="India",
            latitude=19.8762,
            longitude=75.3433,
        ),
        LocationOption(
            name="Aurangabad",
            district="Aurangabad",
            state="Bihar",
            country="India",
            latitude=24.7521,
            longitude=84.3742,
        ),
    ]
    monkeypatch.setattr("services.weather.search_locations", lambda q: candidates)
    monkeypatch.setattr("app.search_locations", lambda q: candidates)

    with TestClient(app_module.app) as client:
        loc_resp = client.get("/api/locations", params={"q": "Aurangabad"})
        assert loc_resp.status_code == 200
        loc_json = loc_resp.json()
        assert loc_json["ambiguous"] is True
        assert loc_json["count"] == 2
        assert loc_json["results"][0]["state"] == "Maharashtra"
        assert loc_json["results"][1]["state"] == "Bihar"

        rec_resp = client.post(
            "/api/recommend",
            json={
                "Nitrogen": 90,
                "Phosphorus": 42,
                "Potassium": 43,
                "Temperature": 20.88,
                "Humidity": 82.0,
                "pH": 6.5,
                "Rainfall": 202.94,
                "location_query": "Aurangabad",
            },
        )
        assert rec_resp.status_code == 200
        rec_json = rec_resp.json()
        assert rec_json["weather_status"] == "ambiguous_location"
        assert len(rec_json["ambiguous_locations"]) == 2
        assert rec_json["location"] is None
        assert rec_json["recommended_crop"] == rec_json["ml_top_3"][0]["crop"]


def test_invalid_inputs_return_field_errors_and_preserve_form_values():
    """Invalid inputs fail Pydantic validation cleanly and preserve entered values."""
    with TestClient(app_module.app) as client:
        bad_form = {
            "Nitrogen": "-10",
            "Phosphorus": "42",
            "Potassium": "43",
            "Temperature": "25",
            "Humidity": "140",
            "pH": "16.5",
            "Rainfall": "not-a-number",
            "location_query": "Pune",
        }
        form_resp = client.post("/predict", data=bad_form)
        assert form_resp.status_code == 422
        html = form_resp.text
        assert "Please fix the highlighted fields" in html
        assert 'value="-10"' in html
        assert 'value="140"' in html
        assert 'value="16.5"' in html
        assert 'value="not-a-number"' in html
        assert 'value="Pune"' in html
        assert "greater than or equal to 0" in html
        assert "less than or equal to 100" in html
        assert "less than or equal to 14" in html

        api_resp = client.post(
            "/api/recommend",
            json={
                "Nitrogen": -5,
                "Phosphorus": 42,
                "Potassium": 43,
                "Temperature": 25,
                "Humidity": 120,
                "pH": 15,
                "Rainfall": 200,
            },
        )
        assert api_resp.status_code == 422


def test_weather_timeout_falls_back_to_ml_top_3(monkeypatch):
    """Weather timeout preserves ML top 3 and reports which step was unavailable."""
    pune = SelectedLocation(
        name="Pune",
        district="Pune",
        state="Maharashtra",
        country="India",
        latitude=18.52,
        longitude=73.86,
    )

    def raise_timeout(_location):
        raise WeatherTimeoutError("Weather forecast request timed out.")

    monkeypatch.setattr("services.weather.fetch_weather_forecast", raise_timeout)

    with TestClient(app_module.app) as client:
        resp = client.post(
            "/api/recommend",
            json={
                "Nitrogen": 90,
                "Phosphorus": 42,
                "Potassium": 43,
                "Temperature": 20.88,
                "Humidity": 82.0,
                "pH": 6.5,
                "Rainfall": 202.94,
                "location": pune.model_dump(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["weather_status"] == "timeout"
        assert "timed out" in body["weather_message"].lower()
        assert body["ai_status"] == "skipped"
        assert [r["crop"] for r in body["final_ranking"]] == [c["crop"] for c in body["ml_top_3"]]


def test_search_locations_timeout_raises_typed_error():
    """Geocoding client timeout raises WeatherTimeoutError."""

    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = httpx.Client(transport=httpx.MockTransport(timeout_handler))
    with pytest.raises(WeatherTimeoutError):
        search_locations("Pune", client=client)


def test_visual_crossing_15_day_forecast_and_fallback(monkeypatch):
    """When WEATHER_API_KEY is set, Visual Crossing 15-day data is parsed and falls back on error."""
    pune = SelectedLocation(
        name="Pune",
        district="Pune",
        state="Maharashtra",
        country="India",
        latitude=18.52,
        longitude=73.86,
    )
    monkeypatch.setenv("WEATHER_API_KEY", "test-vc-key")

    def vc_and_fallback_handler(request: httpx.Request) -> httpx.Response:
        if "visualcrossing.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "currentConditions": {
                        "datetime": "16:00:00",
                        "temp": 31.2,
                        "humidity": 38.0,
                        "precip": 0.0,
                        "conditions": "Clear",
                    },
                    "days": [
                        {
                            "datetime": f"2026-12-{i + 1:02d}",
                            "tempmin": 18.0,
                            "tempmax": 32.0,
                            "precip": 1.0 if i == 2 else 0.0,
                            "precipprob": 40 if i == 2 else 0,
                            "conditions": "Rain" if i == 2 else "Clear",
                        }
                        for i in range(15)
                    ],
                },
            )
        raise AssertionError(f"Unexpected fallback call: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(vc_and_fallback_handler))
    summary = fetch_weather_forecast(pune, forecast_days=15, client=client)
    assert summary.days_covered == 15
    assert summary.is_partial_forecast is False
    assert summary.total_rainfall_mm == pytest.approx(1.0)
    assert summary.current is not None
    assert summary.current.temperature_c == 31.2

    def vc_error_then_open_meteo_handler(request: httpx.Request) -> httpx.Response:
        if "visualcrossing.com" in str(request.url):
            return httpx.Response(429, json={"error": "rate limit"})
        return httpx.Response(
            200,
            json={
                "current": {"time": "2026-12-01T10:00", "temperature_2m": 29.0, "relative_humidity_2m": 45, "precipitation": 0.0, "weather_code": 0},
                "daily": {
                    "time": [f"2026-12-{i + 1:02d}" for i in range(15)],
                    "temperature_2m_min": [19.0] * 15,
                    "temperature_2m_max": [31.0] * 15,
                    "precipitation_sum": [0.0] * 15,
                    "precipitation_probability_max": [0] * 15,
                    "weather_code": [0] * 15,
                },
            },
        )

    fallback_client = httpx.Client(transport=httpx.MockTransport(vc_error_then_open_meteo_handler))
    fallback_summary = fetch_weather_forecast(pune, forecast_days=15, client=fallback_client)
    assert fallback_summary.days_covered == 15
    assert fallback_summary.total_rainfall_mm == 0.0

    def vc_401_then_weatherapi_handler(request: httpx.Request) -> httpx.Response:
        if "visualcrossing.com" in str(request.url):
            return httpx.Response(401, json={"error": "invalid key for visual crossing"})
        if "weatherapi.com" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "current": {
                        "last_updated": "2026-12-01 12:00",
                        "temp_c": 30.5,
                        "humidity": 42,
                        "precip_mm": 0.0,
                        "condition": {"text": "Sunny"},
                    },
                    "forecast": {
                        "forecastday": [
                            {
                                "date": f"2026-12-0{i + 1}",
                                "day": {
                                    "mintemp_c": 18.0,
                                    "maxtemp_c": 31.0,
                                    "totalprecip_mm": 2.0 if i == 1 else 0.0,
                                    "daily_chance_of_rain": 35 if i == 1 else 5,
                                    "condition": {"text": "Patchy rain" if i == 1 else "Sunny"},
                                },
                            }
                            for i in range(3)
                        ]
                    },
                },
            )
        raise AssertionError(f"Unexpected call: {request.url}")

    wa_client = httpx.Client(transport=httpx.MockTransport(vc_401_then_weatherapi_handler))
    wa_summary = fetch_weather_forecast(pune, forecast_days=15, client=wa_client)
    assert wa_summary.days_covered == 3
    assert wa_summary.is_partial_forecast is True
    assert wa_summary.total_rainfall_mm == pytest.approx(2.0)


def test_llm_response_with_crop_outside_shortlist_is_rejected():
    """LLM responses inventing non-shortlist, duplicate, or incomplete crops are rejected."""
    with open(BASE_DIR / "crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "crop_names.pkl", "rb") as f:
        crop_names = pickle.load(f)

    from services.ml_service import predict_top_k

    soil = SoilClimateInput(
        Nitrogen=90,
        Phosphorus=42,
        Potassium=43,
        Temperature=20.88,
        Humidity=82.0,
        pH=6.5,
        Rainfall=202.94,
    )
    ml_top_3 = predict_top_k(model, crop_names, soil, k=3)
    pune = SelectedLocation(
        name="Pune",
        district="Pune",
        state="Maharashtra",
        country="India",
        latitude=18.52,
        longitude=73.86,
    )
    wx = _make_weather_summary(pune, [0.0] * 15)

    invented_crop_llm = RecordingFakeChatModel(
        response_payload={
            "ranked_crops": [
                {"crop": "wheat", "final_rank": 1, "reason": "Wheat was invented outside shortlist."},
                {"crop": ml_top_3[0].crop, "final_rank": 2, "reason": "Strong ML confidence."},
                {"crop": ml_top_3[1].crop, "final_rank": 3, "reason": "Secondary option."},
            ],
            "order_changed": True,
            "order_change_explanation": "Tried introducing wheat.",
        }
    )

    outcome = rerank_with_langchain(soil, ml_top_3, wx, chat_model=invented_crop_llm)
    assert outcome.ai_status == "invalid_response"
    assert outcome.order_changed is False
    assert [r.crop for r in outcome.final_ranking] == [c.crop for c in ml_top_3]
    assert [r.ml_score for r in outcome.final_ranking] == [c.ml_score for c in ml_top_3]

    duplicate_crop_llm = RecordingFakeChatModel(
        response_payload={
            "ranked_crops": [
                {"crop": ml_top_3[0].crop, "final_rank": 1, "reason": "Primary candidate."},
                {"crop": ml_top_3[0].crop, "final_rank": 2, "reason": "Duplicate candidate."},
                {"crop": ml_top_3[1].crop, "final_rank": 3, "reason": "Third candidate."},
            ],
            "order_changed": False,
            "order_change_explanation": "Duplicate test.",
        }
    )
    dup_outcome = rerank_with_langchain(soil, ml_top_3, wx, chat_model=duplicate_crop_llm)
    assert dup_outcome.ai_status == "invalid_response"

    incomplete_crop_llm = RecordingFakeChatModel(
        response_payload={
            "ranked_crops": [
                {"crop": ml_top_3[0].crop, "final_rank": 1, "reason": "Primary candidate."},
                {"crop": ml_top_3[1].crop, "final_rank": 2, "reason": "Second candidate only."},
            ],
            "order_changed": False,
            "order_change_explanation": "Incomplete list test.",
        }
    )
    inc_outcome = rerank_with_langchain(soil, ml_top_3, wx, chat_model=incomplete_crop_llm)
    assert inc_outcome.ai_status == "invalid_response"
    assert [r.crop for r in inc_outcome.final_ranking] == [c.crop for c in ml_top_3]


def test_dry_forecast_explanation_uses_supplied_rainfall_figures():
    """Dry 15-day forecast supplies exact rainfall figures and grounds explanation without forcing a winner."""
    with open(BASE_DIR / "crop_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "crop_names.pkl", "rb") as f:
        crop_names = pickle.load(f)

    from services.ml_service import predict_top_k

    # Ambiguous/close input where multiple crops sit in the shortlist
    soil = SoilClimateInput(
        Nitrogen=75,
        Phosphorus=48,
        Potassium=40,
        Temperature=24.0,
        Humidity=72.0,
        pH=6.6,
        Rainfall=135.0,
    )
    ml_top_3 = predict_top_k(model, crop_names, soil, k=3)
    location = SelectedLocation(
        name="Solapur",
        district="Solapur",
        state="Maharashtra",
        country="India",
        latitude=17.6599,
        longitude=75.9064,
    )
    dry_daily = [0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]
    dry_wx = _make_weather_summary(location, dry_daily)

    def build_grounded_response(prompt_text: str) -> str:
        # Verify the LangChain prompt passed exact rainfall metrics before generating output.
        assert "1.40 mm" in prompt_text
        assert "2026-12-05: 20.0°C to 31.0°C, rainfall 0.60 mm" in prompt_text
        assert "2026-12-11: 20.0°C to 33.0°C, rainfall 0.80 mm" in prompt_text
        assert "Model Rainfall input (separate from forecast): 135.0 mm" in prompt_text

        ranked = [
            {
                "crop": candidate.crop,
                "final_rank": candidate.ml_rank,
                "reason": (
                    f"{candidate.crop} retains ML rank #{candidate.ml_rank} (score {candidate.ml_score_pct:.2f}%); "
                    f"the 15-day forecast supplies only {dry_wx.total_rainfall_mm:.2f} mm total rainfall "
                    f"(0.60 mm on 2026-12-05 and 0.80 mm on 2026-12-11), so near-term irrigation planning is advised."
                ),
            }
            for candidate in ml_top_3
        ]
        return json.dumps(
            {
                "ranked_crops": ranked,
                "order_changed": False,
                "order_change_explanation": (
                    f"Retained the original ML shortlist order because the {dry_wx.days_covered}-day "
                    f"forecast totals only {dry_wx.total_rainfall_mm:.2f} mm (0.60 mm on 2026-12-05 and "
                    f"0.80 mm on 2026-12-11), providing near-term dry-planting context without overriding soil suitability."
                ),
            }
        )

    fake_llm = RecordingFakeChatModel(response_payload=build_grounded_response)
    outcome = rerank_with_langchain(soil, ml_top_3, dry_wx, chat_model=fake_llm)

    assert outcome.ai_status == "ok"
    assert len(fake_llm.recorded_prompts) == 1
    assert outcome.final_ranking[0].crop in {c.crop for c in ml_top_3}
    assert "1.40 mm" in outcome.why_this_crop
    assert "1.40 mm" in outcome.order_change_explanation
    assert "2026-12-05" in outcome.order_change_explanation
    # ML scores remain immutable after reranking.
    ml_scores_by_crop = {c.crop: c.ml_score for c in ml_top_3}
    for row in outcome.final_ranking:
        assert row.ml_score == ml_scores_by_crop[row.crop]
