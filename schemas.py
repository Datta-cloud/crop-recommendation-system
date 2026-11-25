"""Pydantic schemas for requests, ML predictions, weather data, and AI reranking."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

FEATURE_LABELS = {
    "Nitrogen": "Nitrogen (N)",
    "Phosphorus": "Phosphorus (P)",
    "Potassium": "Potassium (K)",
    "Temperature": "Temperature (°C)",
    "Humidity": "Humidity (%)",
    "pH": "Soil pH",
    "Rainfall": "Rainfall (mm)",
}


def format_location_display(
    name: str,
    district: str | None = None,
    state: str | None = None,
    country: str | None = None,
) -> str:
    """Build a readable location label with district/state and country."""
    admin_parts: list[str] = []
    for part in (district, state, country):
        cleaned = (part or "").strip()
        if cleaned and cleaned not in admin_parts:
            admin_parts.append(cleaned)

    if not admin_parts:
        return name.strip()

    # Keep district/state/country visible while avoiding redundant leading repeats.
    if admin_parts[0].lower() == name.strip().lower() and len(admin_parts) > 1:
        admin_parts = admin_parts[1:]

    return f"{name.strip()} — {', '.join(admin_parts)}"


class SoilClimateInput(BaseModel):
    """Validated soil nutrient and climate inputs matching training feature order."""

    model_config = ConfigDict(populate_by_name=True)

    Nitrogen: float = Field(
        ...,
        validation_alias=AliasChoices("Nitrogen", "N", "nitrogen"),
        ge=0,
        le=200,
        description="Nitrogen content ratio in soil (kg/ha)",
    )
    Phosphorus: float = Field(
        ...,
        validation_alias=AliasChoices("Phosphorus", "P", "phosphorus"),
        ge=0,
        le=200,
        description="Phosphorus content ratio in soil (kg/ha)",
    )
    Potassium: float = Field(
        ...,
        validation_alias=AliasChoices("Potassium", "K", "potassium"),
        ge=0,
        le=205,
        description="Potassium content ratio in soil (kg/ha)",
    )
    Temperature: float = Field(
        ...,
        validation_alias=AliasChoices("Temperature", "temperature"),
        ge=-10,
        le=60,
        description="Ambient temperature (°C)",
    )
    Humidity: float = Field(
        ...,
        validation_alias=AliasChoices("Humidity", "humidity"),
        ge=0,
        le=100,
        description="Relative humidity (%)",
    )
    pH: float = Field(
        ...,
        validation_alias=AliasChoices("pH", "ph"),
        ge=0,
        le=14,
        description="Soil pH level (0–14)",
    )
    Rainfall: float = Field(
        ...,
        validation_alias=AliasChoices("Rainfall", "rainfall"),
        ge=0,
        le=500,
        description="Model rainfall input (mm), kept separate from forecast rainfall",
    )

    @field_validator(
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "Temperature",
        "Humidity",
        "pH",
        "Rainfall",
        mode="after",
    )
    @classmethod
    def ensure_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Value must be a finite number.")
        return value

    def to_feature_array(self) -> list[float]:
        """Return features in exact training order: N, P, K, temperature, humidity, ph, rainfall."""
        return [
            float(self.Nitrogen),
            float(self.Phosphorus),
            float(self.Potassium),
            float(self.Temperature),
            float(self.Humidity),
            float(self.pH),
            float(self.Rainfall),
        ]

    def to_display_dict(self) -> dict[str, float]:
        return {
            "Nitrogen": float(self.Nitrogen),
            "Phosphorus": float(self.Phosphorus),
            "Potassium": float(self.Potassium),
            "Temperature": float(self.Temperature),
            "Humidity": float(self.Humidity),
            "pH": float(self.pH),
            "Rainfall": float(self.Rainfall),
        }


class LocationOption(BaseModel):
    """Geocoded location option returned by location search."""

    name: str
    district: str | None = None
    state: str | None = None
    country: str | None = None
    country_code: str | None = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timezone: str | None = None
    display_name: str = ""

    @model_validator(mode="after")
    def populate_display_name(self) -> "LocationOption":
        if not self.display_name.strip():
            self.display_name = format_location_display(
                self.name, self.district, self.state, self.country
            )
        return self


class SelectedLocation(LocationOption):
    """Explicit location selected by the user for weather lookup."""


class RecommendRequest(SoilClimateInput):
    """JSON request payload for POST /api/recommend."""

    location: SelectedLocation | None = None
    location_query: str | None = None
    use_ai: bool = True


class MLCandidate(BaseModel):
    """Original top-k crop prediction produced by Random Forest predict_proba()."""

    crop: str
    ml_rank: int = Field(..., ge=1, le=3)
    ml_score: float = Field(..., ge=0.0, le=1.0)
    ml_score_pct: float = Field(..., ge=0.0, le=100.0)


class RankedCropResult(BaseModel):
    """Shortlisted crop carrying immutable ML rank/score alongside final rank and reason."""

    crop: str
    ml_rank: int = Field(..., ge=1, le=3)
    ml_score: float = Field(..., ge=0.0, le=1.0)
    ml_score_pct: float = Field(..., ge=0.0, le=100.0)
    final_rank: int = Field(..., ge=1, le=3)
    reason: str


class CurrentWeather(BaseModel):
    """Current observed weather at the selected coordinates."""

    observed_at: str | None = None
    temperature_c: float
    humidity_pct: float | None = None
    precipitation_mm: float = 0.0
    weather_code: int | None = None
    condition: str = "Unknown"


class DailyForecast(BaseModel):
    """Single day forecast entry."""

    date: str
    temp_min_c: float
    temp_max_c: float
    rainfall_mm: float = Field(..., ge=0.0)
    rain_probability_pct: int | None = None
    weather_code: int | None = None
    condition: str = "Unknown"


class WeatherSummary(BaseModel):
    """Current weather and multi-day forecast summary (targeting 15 days)."""

    location: SelectedLocation
    current: CurrentWeather | None = None
    daily: list[DailyForecast] = Field(default_factory=list)
    days_requested: int = 15
    days_covered: int = 0
    is_partial_forecast: bool = False
    total_rainfall_mm: float = 0.0
    avg_temp_min_c: float | None = None
    avg_temp_max_c: float | None = None
    start_date: str | None = None
    end_date: str | None = None
    forecast_note: str = ""


class LLMRerankItem(BaseModel):
    """Single ranked crop item produced by the LLM."""

    crop: str = Field(..., min_length=1)
    final_rank: int = Field(..., ge=1, le=3)
    reason: str = Field(..., min_length=5)

    @field_validator("crop")
    @classmethod
    def normalize_crop(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("Crop name cannot be empty.")
        return cleaned

    @field_validator("reason")
    @classmethod
    def strip_reason(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 5:
            raise ValueError("Reason must be a concise non-empty explanation.")
        return cleaned


class LLMRerankOutput(BaseModel):
    """Structured LangChain reranker output validated strictly by Pydantic."""

    ranked_crops: list[LLMRerankItem]
    order_changed: bool
    order_change_explanation: str = Field(..., min_length=5)

    @field_validator("order_change_explanation")
    @classmethod
    def strip_explanation(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 5:
            raise ValueError("Order change explanation cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> "LLMRerankOutput":
        if len(self.ranked_crops) != 3:
            raise ValueError(
                f"Expected exactly 3 ranked crops, received {len(self.ranked_crops)}."
            )

        ranks = [item.final_rank for item in self.ranked_crops]
        if set(ranks) != {1, 2, 3}:
            raise ValueError(
                f"final_rank values must be exactly {{1, 2, 3}}, received {ranks}."
            )

        crops = [item.crop for item in self.ranked_crops]
        if len(set(crops)) != 3:
            raise ValueError(
                f"Duplicate crops are not allowed in reranked output: {crops}."
            )

        return self


def validate_rerank_against_shortlist(
    output: LLMRerankOutput, expected_crops: list[str]
) -> LLMRerankOutput:
    """Verify the LLM ranked exactly the 3 ML candidates with no invented or omitted crops."""
    expected_normalized = [crop.strip().lower() for crop in expected_crops]
    expected_set = set(expected_normalized)
    actual_set = {item.crop for item in output.ranked_crops}

    if actual_set != expected_set:
        invented = sorted(actual_set - expected_set)
        missing = sorted(expected_set - actual_set)
        raise ValueError(
            f"LLM output does not match ML shortlist. Invented={invented}, Missing={missing}."
        )

    return output


class RecommendResponse(BaseModel):
    """Unified response payload returned by POST /api/recommend and rendered in the UI."""

    input_features: dict[str, float]
    ml_top_3: list[MLCandidate]
    recommended_crop: str
    final_ranking: list[RankedCropResult]
    order_changed: bool = False
    why_this_crop: str
    order_change_explanation: str
    location: SelectedLocation | None = None
    ambiguous_locations: list[LocationOption] = Field(default_factory=list)
    weather: WeatherSummary | None = None
    weather_status: Literal[
        "ok", "skipped", "ambiguous_location", "not_found", "timeout", "unavailable"
    ]
    weather_message: str | None = None
    ai_status: Literal["ok", "skipped", "unavailable", "invalid_response"]
    ai_message: str | None = None
    reranker_provider: str | None = None
    reranker_model: str | None = None


class ChatRequest(BaseModel):
    """JSON request payload for POST /api/chat ('Sheti Mitra')."""

    question: str = Field(..., min_length=2, max_length=500)
    language: Literal["en", "mr"] = "en"

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("Please enter a valid farming question.")
        return cleaned


class ChatSource(BaseModel):
    """Official note source returned alongside Sheti Mitra's answer."""

    id: str
    title: str
    topic: str
    source_name: str
    url: str
    score: float


class ChatResponse(BaseModel):
    """Structured JSON response for POST /api/chat."""

    question: str
    language: Literal["en", "mr"]
    answer: str
    covered: bool
    sources: list[ChatSource] = Field(default_factory=list)
    status: Literal["ok", "not_covered", "unavailable"]
    message: str | None = None
