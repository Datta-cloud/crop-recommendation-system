"""Weather geocoding and 15-day forecast client (Open-Meteo + optional Visual Crossing key)."""

from __future__ import annotations

import logging
import os

import httpx

from schemas import (
    CurrentWeather,
    DailyForecast,
    LocationOption,
    SelectedLocation,
    WeatherSummary,
    format_location_display,
)

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
VISUAL_CROSSING_URL = (
    "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
)
WEATHERAPI_URL = "https://api.weatherapi.com/v1/forecast.json"
MET_NO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"

DEFAULT_HEADERS = {
    "User-Agent": "CropRecommendationSystem/2.0 (https://github.com/Datta-cloud/crop-recommendation-system)"
}

WMO_CONDITIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Freezing light drizzle",
    57: "Freezing dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Freezing light rain",
    67: "Freezing heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherServiceError(Exception):
    """Base error raised for weather lookup failures."""


class WeatherTimeoutError(WeatherServiceError):
    """Raised when geocoding or forecast requests time out."""


class WeatherUnavailableError(WeatherServiceError):
    """Raised when the upstream weather service fails or returns invalid data."""


def describe_weather_code(code: int | None) -> str:
    if code is None:
        return "Unknown"
    return WMO_CONDITIONS.get(int(code), f"Weather code {code}")


def _get_json_with_retry(
    url: str,
    params: dict[str, object],
    timeout: float,
    client: httpx.Client | None = None,
    retries: int = 1,
) -> dict:
    """Execute GET JSON request with User-Agent header and light transient retry."""
    last_exc: Exception | None = None
    attempts = 1 + (retries if client is None else 0)

    for attempt in range(attempts):
        try:
            if client is not None:
                response = client.get(
                    url, params=params, headers=DEFAULT_HEADERS, timeout=timeout
                )
            else:
                with httpx.Client(
                    timeout=timeout, headers=DEFAULT_HEADERS, follow_redirects=True
                ) as http_client:
                    response = http_client.get(url, params=params)

            if response.status_code in {429, 502, 503, 504} and attempt < attempts - 1:
                logger.warning(
                    "Transient HTTP %s from %s (attempt %d/%d)",
                    response.status_code,
                    url,
                    attempt + 1,
                    attempts,
                )
                continue

            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                raise WeatherTimeoutError("Weather request timed out.") from exc
        except (httpx.HTTPError, ValueError) as exc:
            last_exc = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning("Weather API error on %s (status=%s): %s", url, status_code, exc)
            if attempt >= attempts - 1:
                if status_code == 429:
                    detail = "Weather forecast service rate-limited the host (HTTP 429)."
                elif status_code in {401, 403}:
                    detail = f"Weather forecast API rejected credentials (HTTP {status_code})."
                elif status_code is not None:
                    detail = f"Weather forecast service returned HTTP {status_code}."
                else:
                    detail = f"Weather forecast service is unavailable ({type(exc).__name__})."
                raise WeatherUnavailableError(detail) from exc

    raise WeatherUnavailableError("Weather forecast service is currently unavailable.") from last_exc


def search_locations(
    query: str,
    count: int = 8,
    timeout: float = 8.0,
    client: httpx.Client | None = None,
) -> list[LocationOption]:
    """Search villages, towns, or cities with district/state and country details."""
    cleaned = query.strip()
    if len(cleaned) < 2:
        return []

    params = {
        "name": cleaned,
        "count": count,
        "language": "en",
        "format": "json",
    }

    try:
        payload = _get_json_with_retry(
            GEOCODING_URL, params=params, timeout=timeout, client=client
        )
    except WeatherTimeoutError as exc:
        raise WeatherTimeoutError("Location search timed out.") from exc
    except WeatherUnavailableError as exc:
        raise WeatherUnavailableError("Unable to reach location geocoding service.") from exc

    raw_results = payload.get("results") or []
    options: list[LocationOption] = []
    for item in raw_results:
        name = (item.get("name") or "").strip()
        lat = item.get("latitude")
        lon = item.get("longitude")
        if not name or lat is None or lon is None:
            continue

        district = item.get("admin2") or item.get("admin3")
        state = item.get("admin1")
        country = item.get("country")

        options.append(
            LocationOption(
                name=name,
                district=district,
                state=state,
                country=country,
                country_code=item.get("country_code"),
                latitude=float(lat),
                longitude=float(lon),
                timezone=item.get("timezone"),
                display_name=format_location_display(name, district, state, country),
            )
        )

    return options


def _build_weather_summary(
    location: SelectedLocation,
    current_weather: CurrentWeather | None,
    daily_items: list[DailyForecast],
    forecast_days: int,
) -> WeatherSummary:
    days_covered = len(daily_items)
    if days_covered == 0:
        raise WeatherUnavailableError(
            "Weather service returned no valid daily forecast data."
        )

    total_rainfall = round(sum(item.rainfall_mm for item in daily_items), 2)
    avg_min = round(sum(item.temp_min_c for item in daily_items) / days_covered, 1)
    avg_max = round(sum(item.temp_max_c for item in daily_items) / days_covered, 1)
    start_date = daily_items[0].date
    end_date = daily_items[-1].date
    is_partial = days_covered < forecast_days

    if is_partial:
        forecast_note = (
            f"Partial forecast: provider returned {days_covered} of {forecast_days} requested days "
            f"({start_date} to {end_date}); total forecast rainfall is {total_rainfall:.1f} mm."
        )
    else:
        forecast_note = (
            f"{days_covered}-day forecast ({start_date} to {end_date}); "
            f"total expected rainfall is {total_rainfall:.1f} mm."
        )

    return WeatherSummary(
        location=location,
        current=current_weather,
        daily=daily_items,
        days_requested=forecast_days,
        days_covered=days_covered,
        is_partial_forecast=is_partial,
        total_rainfall_mm=total_rainfall,
        avg_temp_min_c=avg_min,
        avg_temp_max_c=avg_max,
        start_date=start_date,
        end_date=end_date,
        forecast_note=forecast_note,
    )


def _fetch_open_meteo_forecast(
    location: SelectedLocation,
    forecast_days: int,
    timeout: float,
    client: httpx.Client | None = None,
) -> WeatherSummary:
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
        "daily": (
            "temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max,weather_code"
        ),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    payload = _get_json_with_retry(
        FORECAST_URL, params=params, timeout=timeout, client=client
    )

    current_raw = payload.get("current") or {}
    current_weather: CurrentWeather | None = None
    if current_raw and current_raw.get("temperature_2m") is not None:
        wcode = current_raw.get("weather_code")
        humidity_val = current_raw.get("relative_humidity_2m")
        precip_val = current_raw.get("precipitation")
        current_weather = CurrentWeather(
            observed_at=current_raw.get("time"),
            temperature_c=round(float(current_raw["temperature_2m"]), 1),
            humidity_pct=round(float(humidity_val), 1)
            if humidity_val is not None
            else None,
            precipitation_mm=round(float(precip_val or 0.0), 2),
            weather_code=int(wcode) if wcode is not None else None,
            condition=describe_weather_code(wcode),
        )

    daily_raw = payload.get("daily") or {}
    dates = daily_raw.get("time") or []
    t_mins = daily_raw.get("temperature_2m_min") or []
    t_maxs = daily_raw.get("temperature_2m_max") or []
    rains = daily_raw.get("precipitation_sum") or []
    rain_probs = daily_raw.get("precipitation_probability_max") or []
    codes = daily_raw.get("weather_code") or []

    row_count = min(len(dates), len(t_mins), len(t_maxs), forecast_days)
    daily_items: list[DailyForecast] = []
    for idx in range(row_count):
        date_str = dates[idx]
        t_min = t_mins[idx]
        t_max = t_maxs[idx]
        if date_str is None or t_min is None or t_max is None:
            continue

        rain_val = rains[idx] if idx < len(rains) and rains[idx] is not None else 0.0
        prob_val = rain_probs[idx] if idx < len(rain_probs) else None
        code_val = codes[idx] if idx < len(codes) else None

        daily_items.append(
            DailyForecast(
                date=str(date_str),
                temp_min_c=round(float(t_min), 1),
                temp_max_c=round(float(t_max), 1),
                rainfall_mm=round(max(0.0, float(rain_val)), 2),
                rain_probability_pct=int(prob_val) if prob_val is not None else None,
                weather_code=int(code_val) if code_val is not None else None,
                condition=describe_weather_code(code_val),
            )
        )

    return _build_weather_summary(location, current_weather, daily_items, forecast_days)


def _fetch_visual_crossing_forecast(
    location: SelectedLocation,
    api_key: str,
    forecast_days: int,
    timeout: float,
    client: httpx.Client | None = None,
) -> WeatherSummary:
    url = f"{VISUAL_CROSSING_URL}/{location.latitude},{location.longitude}/next14days"
    params = {
        "unitGroup": "metric",
        "include": "days,current",
        "key": api_key,
        "contentType": "json",
    }

    payload = _get_json_with_retry(url, params=params, timeout=timeout, client=client)

    current_raw = payload.get("currentConditions") or {}
    current_weather: CurrentWeather | None = None
    if current_raw and current_raw.get("temp") is not None:
        humidity_val = current_raw.get("humidity")
        precip_val = current_raw.get("precip")
        current_weather = CurrentWeather(
            observed_at=current_raw.get("datetime"),
            temperature_c=round(float(current_raw["temp"]), 1),
            humidity_pct=round(float(humidity_val), 1)
            if humidity_val is not None
            else None,
            precipitation_mm=round(max(0.0, float(precip_val or 0.0)), 2),
            weather_code=None,
            condition=str(current_raw.get("conditions") or "Current"),
        )

    daily_items: list[DailyForecast] = []
    for day in (payload.get("days") or [])[:forecast_days]:
        date_str = day.get("datetime")
        t_min = day.get("tempmin")
        t_max = day.get("tempmax")
        if date_str is None or t_min is None or t_max is None:
            continue

        rain_val = day.get("precip")
        prob_val = day.get("precipprob")
        daily_items.append(
            DailyForecast(
                date=str(date_str),
                temp_min_c=round(float(t_min), 1),
                temp_max_c=round(float(t_max), 1),
                rainfall_mm=round(max(0.0, float(rain_val or 0.0)), 2),
                rain_probability_pct=int(round(float(prob_val)))
                if prob_val is not None
                else None,
                weather_code=None,
                condition=str(day.get("conditions") or "Forecast"),
            )
        )

    return _build_weather_summary(location, current_weather, daily_items, forecast_days)


def _fetch_weatherapi_forecast(
    location: SelectedLocation,
    api_key: str,
    forecast_days: int,
    timeout: float,
    client: httpx.Client | None = None,
) -> WeatherSummary:
    params = {
        "key": api_key,
        "q": f"{location.latitude},{location.longitude}",
        "days": min(forecast_days, 14),
        "aqi": "no",
        "alerts": "no",
    }

    payload = _get_json_with_retry(
        WEATHERAPI_URL, params=params, timeout=timeout, client=client
    )

    current_raw = payload.get("current") or {}
    current_weather: CurrentWeather | None = None
    if current_raw and current_raw.get("temp_c") is not None:
        cond_text = (current_raw.get("condition") or {}).get("text") or "Current"
        humidity_val = current_raw.get("humidity")
        precip_val = current_raw.get("precip_mm")
        current_weather = CurrentWeather(
            observed_at=current_raw.get("last_updated"),
            temperature_c=round(float(current_raw["temp_c"]), 1),
            humidity_pct=round(float(humidity_val), 1)
            if humidity_val is not None
            else None,
            precipitation_mm=round(max(0.0, float(precip_val or 0.0)), 2),
            weather_code=None,
            condition=str(cond_text),
        )

    forecast_days_raw = (payload.get("forecast") or {}).get("forecastday") or []
    daily_items: list[DailyForecast] = []
    for entry in forecast_days_raw[:forecast_days]:
        date_str = entry.get("date")
        day = entry.get("day") or {}
        t_min = day.get("mintemp_c")
        t_max = day.get("maxtemp_c")
        if date_str is None or t_min is None or t_max is None:
            continue

        rain_val = day.get("totalprecip_mm")
        prob_val = day.get("daily_chance_of_rain")
        cond_text = (day.get("condition") or {}).get("text") or "Forecast"
        daily_items.append(
            DailyForecast(
                date=str(date_str),
                temp_min_c=round(float(t_min), 1),
                temp_max_c=round(float(t_max), 1),
                rainfall_mm=round(max(0.0, float(rain_val or 0.0)), 2),
                rain_probability_pct=int(round(float(prob_val)))
                if prob_val is not None
                else None,
                weather_code=None,
                condition=str(cond_text),
            )
        )

    return _build_weather_summary(location, current_weather, daily_items, forecast_days)


def _fetch_met_no_forecast(
    location: SelectedLocation,
    forecast_days: int,
    timeout: float,
    client: httpx.Client | None = None,
) -> WeatherSummary:
    """Fetch up to ~10 days of global forecast from MET Norway (no API key required)."""
    params = {
        "lat": round(location.latitude, 4),
        "lon": round(location.longitude, 4),
    }

    payload = _get_json_with_retry(MET_NO_URL, params=params, timeout=timeout, client=client)
    timeseries = (payload.get("properties") or {}).get("timeseries") or []
    if not timeseries:
        raise WeatherUnavailableError("MET Norway returned empty timeseries.")

    first_entry = timeseries[0]
    first_data = first_entry.get("data") or {}
    first_instant = first_data.get("instant", {}).get("details") or {}
    first_next = first_data.get("next_1_hours") or first_data.get("next_6_hours") or {}
    first_symbol = str((first_next.get("summary") or {}).get("symbol_code") or "Current")
    current_weather: CurrentWeather | None = None
    if first_instant.get("air_temperature") is not None:
        current_weather = CurrentWeather(
            observed_at=first_entry.get("time"),
            temperature_c=round(float(first_instant["air_temperature"]), 1),
            humidity_pct=round(float(first_instant["relative_humidity"]), 1)
            if first_instant.get("relative_humidity") is not None
            else None,
            precipitation_mm=round(
                max(
                    0.0,
                    float((first_next.get("details") or {}).get("precipitation_amount") or 0.0),
                ),
                2,
            ),
            weather_code=None,
            condition=first_symbol.replace("_", " ").capitalize(),
        )

    grouped: dict[str, dict[str, object]] = {}
    for entry in timeseries:
        timestamp = str(entry.get("time") or "")
        date_str = timestamp[:10]
        if len(date_str) != 10:
            continue

        data = entry.get("data") or {}
        details = (data.get("instant") or {}).get("details") or {}
        temp = details.get("air_temperature")
        if temp is None:
            continue

        # Prefer 1h blocks on hourly steps to avoid double-counting overlapping 6h blocks.
        period = data.get("next_1_hours") or data.get("next_6_hours") or {}
        precip = float((period.get("details") or {}).get("precipitation_amount") or 0.0)
        symbol = str((period.get("summary") or {}).get("symbol_code") or "")

        bucket = grouped.setdefault(
            date_str,
            {"temps": [], "rain": 0.0, "conditions": []},
        )
        bucket["temps"].append(float(temp))  # type: ignore[union-attr]
        bucket["rain"] = float(bucket["rain"]) + max(0.0, precip)  # type: ignore[arg-type]
        if symbol:
            bucket["conditions"].append(symbol.replace("_", " ").capitalize())  # type: ignore[union-attr]

    daily_items: list[DailyForecast] = []
    for date_str in sorted(grouped.keys())[:forecast_days]:
        bucket = grouped[date_str]
        temps: list[float] = bucket["temps"]  # type: ignore[assignment]
        conds: list[str] = bucket["conditions"]  # type: ignore[assignment]
        daily_items.append(
            DailyForecast(
                date=date_str,
                temp_min_c=round(min(temps), 1),
                temp_max_c=round(max(temps), 1),
                rainfall_mm=round(float(bucket["rain"]), 2),  # type: ignore[arg-type]
                rain_probability_pct=None,
                weather_code=None,
                condition=conds[len(conds) // 2] if conds else "Forecast",
            )
        )

    return _build_weather_summary(location, current_weather, daily_items, forecast_days)


def fetch_weather_forecast(
    location: SelectedLocation,
    forecast_days: int = 15,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> WeatherSummary:
    """Fetch current weather and up to 15 days of forecast for coordinates.

    Uses Visual Crossing (or WeatherAPI.com if a WeatherAPI key is supplied) when
    WEATHER_API_KEY is configured, and falls back to Open-Meteo when unset or on error.
    """
    weather_api_key = (
        os.getenv("WEATHER_API_KEY", "").strip()
        or os.getenv("VISUAL_CROSSING_API_KEY", "").strip()
    )
    provider_pref = os.getenv("WEATHER_PROVIDER", "").strip().lower()

    if weather_api_key:
        if provider_pref == "weatherapi":
            try:
                return _fetch_weatherapi_forecast(
                    location=location,
                    api_key=weather_api_key,
                    forecast_days=forecast_days,
                    timeout=timeout,
                    client=client,
                )
            except WeatherServiceError as exc:
                logger.warning("WeatherAPI failed, falling back to Open-Meteo: %s", exc)
        else:
            try:
                return _fetch_visual_crossing_forecast(
                    location=location,
                    api_key=weather_api_key,
                    forecast_days=forecast_days,
                    timeout=timeout,
                    client=client,
                )
            except WeatherServiceError as exc:
                logger.warning("Visual Crossing failed, trying WeatherAPI/Open-Meteo: %s", exc)
                if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
                    try:
                        return _fetch_weatherapi_forecast(
                            location=location,
                            api_key=weather_api_key,
                            forecast_days=forecast_days,
                            timeout=timeout,
                            client=client,
                        )
                    except WeatherServiceError as wa_exc:
                        logger.warning("WeatherAPI fallback also failed: %s", wa_exc)

    try:
        return _fetch_open_meteo_forecast(
            location=location,
            forecast_days=forecast_days,
            timeout=timeout,
            client=client,
        )
    except WeatherTimeoutError as exc:
        raise WeatherTimeoutError("Weather forecast request timed out.") from exc
    except WeatherUnavailableError as exc:
        logger.warning("Open-Meteo unavailable (%s); falling back to MET Norway.", exc)
        return _fetch_met_no_forecast(
            location=location,
            forecast_days=forecast_days,
            timeout=timeout,
            client=client,
        )
