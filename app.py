"""FastAPI application for ML-powered Crop Recommendation with 15-day weather context."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from schemas import (
    FEATURE_LABELS,
    LocationOption,
    RecommendRequest,
    RecommendResponse,
    SelectedLocation,
    SoilClimateInput,
    format_location_display,
)
from services.ml_service import load_ml_artifacts
from services.recommendation import generate_recommendation
from services.weather import (
    WeatherTimeoutError,
    WeatherUnavailableError,
    search_locations,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.pkl"
LABELS_PATH = BASE_DIR / "crop_names.pkl"
TEMPLATES_DIR = BASE_DIR / "templates"

SOIL_FIELD_NAMES = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "pH",
    "Rainfall",
]

LOCATION_FORM_FIELDS = [
    "location_query",
    "location_name",
    "location_district",
    "location_state",
    "location_country",
    "location_display_name",
    "latitude",
    "longitude",
]

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def ensure_artifacts_loaded(application: FastAPI) -> tuple[Any, list[str]]:
    """Load ML artifacts once and cache them on app.state."""
    if not hasattr(application.state, "model") or not hasattr(
        application.state, "crop_names"
    ):
        model, crop_names = load_ml_artifacts(MODEL_PATH, LABELS_PATH)
        application.state.model = model
        application.state.crop_names = crop_names
    return application.state.model, application.state.crop_names


@asynccontextmanager
async def lifespan(application: FastAPI):
    ensure_artifacts_loaded(application)
    yield


app = FastAPI(
    title="Crop Recommendation System",
    description=(
        "Random Forest crop recommendation with 15-day Open-Meteo weather context "
        "and LangChain shortlist evaluation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def _format_pydantic_field_errors(exc: ValidationError) -> dict[str, str]:
    """Convert Pydantic validation errors into readable per-field messages."""
    errors: dict[str, str] = {}
    for err in exc.errors():
        loc = err.get("loc") or ()
        raw_field = str(loc[0]) if loc else "form"
        # Normalize alias names back to canonical HTML field names.
        alias_map = {
            "N": "Nitrogen",
            "nitrogen": "Nitrogen",
            "P": "Phosphorus",
            "phosphorus": "Phosphorus",
            "K": "Potassium",
            "potassium": "Potassium",
            "temperature": "Temperature",
            "humidity": "Humidity",
            "ph": "pH",
            "rainfall": "Rainfall",
        }
        field = alias_map.get(raw_field, raw_field)
        err_type = err.get("type", "")
        label = FEATURE_LABELS.get(field, field)

        if err_type in {"missing", "float_parsing"}:
            msg = f"Enter a valid number for {label}."
        else:
            msg = err.get("msg", f"Invalid value for {label}.")
            if msg.startswith("Value error, "):
                msg = msg[len("Value error, ") :]

        if field not in errors:
            errors[field] = msg
    return errors


def _build_selected_location_from_form(
    form_data: dict[str, str],
) -> tuple[SelectedLocation | None, str | None]:
    """Construct SelectedLocation when coordinates are present in form submission."""
    lat_raw = form_data.get("latitude", "").strip()
    lon_raw = form_data.get("longitude", "").strip()
    if not lat_raw and not lon_raw:
        return None, None

    if not lat_raw or not lon_raw:
        return None, "Both latitude and longitude are required when selecting a location."

    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except ValueError:
        return None, "Selected location coordinates must be valid numbers."

    name = form_data.get("location_name", "").strip() or form_data.get(
        "location_query", ""
    ).strip() or "Selected Location"
    district = form_data.get("location_district", "").strip() or None
    state = form_data.get("location_state", "").strip() or None
    country = form_data.get("location_country", "").strip() or None
    display_name = form_data.get("location_display_name", "").strip() or format_location_display(
        name, district, state, country
    )

    try:
        location = SelectedLocation(
            name=name,
            district=district,
            state=state,
            country=country,
            latitude=lat,
            longitude=lon,
            display_name=display_name,
        )
        return location, None
    except ValidationError:
        return None, "Selected location coordinates are out of range."


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint for deployment probes."""
    _, crop_names = ensure_artifacts_loaded(app)
    return {
        "status": "ok",
        "model_loaded": True,
        "crop_count": len(crop_names),
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Render the main crop recommendation form."""
    ensure_artifacts_loaded(app)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "form_data": {},
            "field_errors": {},
            "recommendation": None,
        },
    )


@app.get("/api/locations")
def api_search_locations(q: str = Query(..., min_length=2, max_length=120)) -> JSONResponse:
    """Search villages, towns, or cities with district/state and country metadata."""
    try:
        matches: list[LocationOption] = search_locations(q)
    except WeatherTimeoutError as exc:
        return JSONResponse(status_code=504, content={"detail": str(exc), "results": []})
    except WeatherUnavailableError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc), "results": []})

    return JSONResponse(
        status_code=200,
        content={
            "query": q.strip(),
            "count": len(matches),
            "ambiguous": len(matches) > 1,
            "results": [item.model_dump() for item in matches],
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request) -> HTMLResponse:
    """Validate HTML form inputs, compute top-3 ML crops, fetch weather, and rerank."""
    model, crop_names = ensure_artifacts_loaded(app)
    raw_form = await request.form()

    form_data: dict[str, str] = {}
    for key in SOIL_FIELD_NAMES + LOCATION_FORM_FIELDS:
        val = raw_form.get(key)
        form_data[key] = str(val).strip() if val is not None else ""

    soil_payload: dict[str, Any] = {}
    for field in SOIL_FIELD_NAMES:
        val = form_data[field]
        if val != "":
            soil_payload[field] = val

    field_errors: dict[str, str] = {}
    soil_input: SoilClimateInput | None = None
    try:
        soil_input = SoilClimateInput.model_validate(soil_payload)
    except ValidationError as exc:
        field_errors.update(_format_pydantic_field_errors(exc))

    selected_location, location_error = _build_selected_location_from_form(form_data)
    if location_error:
        field_errors["location"] = location_error

    if field_errors or soil_input is None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "form_data": form_data,
                "field_errors": field_errors,
                "recommendation": None,
            },
            status_code=422,
        )

    recommendation = generate_recommendation(
        model=model,
        crop_names=crop_names,
        features=soil_input,
        selected_location=selected_location,
        location_query=form_data.get("location_query") or None,
        use_ai=True,
    )

    if recommendation.location is not None:
        form_data["location_name"] = recommendation.location.name
        form_data["location_district"] = recommendation.location.district or ""
        form_data["location_state"] = recommendation.location.state or ""
        form_data["location_country"] = recommendation.location.country or ""
        form_data["location_display_name"] = recommendation.location.display_name
        form_data["latitude"] = str(recommendation.location.latitude)
        form_data["longitude"] = str(recommendation.location.longitude)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "form_data": form_data,
            "field_errors": {},
            "recommendation": recommendation,
        },
    )


@app.post("/api/recommend", response_model=RecommendResponse)
def api_recommend(payload: RecommendRequest) -> RecommendResponse:
    """Return structured JSON recommendations with ML scores, weather summary, and AI ranking."""
    model, crop_names = ensure_artifacts_loaded(app)
    return generate_recommendation(
        model=model,
        crop_names=crop_names,
        features=payload,
        selected_location=payload.location,
        location_query=payload.location_query,
        use_ai=payload.use_ai,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("APP_RELOAD", "false").lower() == "true"
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload_enabled)
