"""Open-Meteo geocoding and 15-day weather forecast client."""

from __future__ import annotations

import httpx

from schemas import (
    CurrentWeather,
    DailyForecast,
    LocationOption,
    SelectedLocation,
    WeatherSummary,
    format_location_display,
)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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
        if client is not None:
            response = client.get(GEOCODING_URL, params=params, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as http_client:
                response = http_client.get(GEOCODING_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise WeatherTimeoutError("Location search timed out.") from exc
    except (httpx.HTTPError, ValueError) as exc:
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


def fetch_weather_forecast(
    location: SelectedLocation,
    forecast_days: int = 15,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
) -> WeatherSummary:
    """Fetch current weather and up to 15 days of daily forecast for coordinates."""
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

    try:
        if client is not None:
            response = client.get(FORECAST_URL, params=params, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as http_client:
                response = http_client.get(FORECAST_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    except httpx.TimeoutException as exc:
        raise WeatherTimeoutError("Weather forecast request timed out.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise WeatherUnavailableError("Weather forecast service is unavailable.") from exc

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

    days_covered = len(daily_items)
    if days_covered == 0:
        raise WeatherUnavailableError("Weather service returned no valid daily forecast data.")

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
