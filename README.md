# Crop Recommendation System

A machine learning and weather-aware web application built with **FastAPI**, **scikit-learn**, **Open-Meteo**, and **LangChain** that recommends crops from soil nutrients and climate inputs, fetches a location's 15-day weather forecast, and evaluates whether near-term weather should adjust the ML shortlist order.

---

## Key Capabilities

1. **Random Forest Top-3 Shortlist (`scikit-learn`)**
   - Loads `crop_model.pkl` and `crop_names.pkl` once at FastAPI startup.
   - Passes features in exact training order (`N, P, K, temperature, humidity, ph, rainfall`) to `predict_proba()`, matches columns to `model.classes_`, and decodes the top 3 crops with immutable **ML rank** and **ML model score**.

2. **Location Search & 15-Day Forecast (`Open-Meteo`)**
   - Search villages, towns, or cities (`GET /api/locations?q=...`) displaying district/state and country so ambiguous names can be disambiguated cleanly.
   - Fetches current weather and up to 15 days of dated daily min/max temperature and expected rainfall using the selected coordinates (no API key required).
   - Summarizes total forecast rainfall and actual days covered honestly (including partial responses under 15 days).
   - Keeps the ML model's `Rainfall` input strictly separate from forecast rainfall and never overwrites manual climate entries.

3. **LangChain AI Reranking + Strict Pydantic Validation**
   - Sends the 3 ML candidates, immutable ML scores, entered soil/climate values, and dated weather summary to a configurable server-side LLM (`openai`, `groq`, `gemini`, or `openai_compatible`).
   - Validates LLM output strictly with Pydantic, rejecting invented crops, duplicate crops, or incomplete rankings.
   - Falls back cleanly to the original ML top 3 whenever location weather or AI reranking is unavailable, timed out, or invalid.

4. **FastAPI Form UI & JSON API**
   - `GET /` and `POST /predict` preserve the classic 7-field form with Pydantic field-level errors, preserved input values, and a mobile-readable side-by-side recommendation + weather view.
   - `POST /api/recommend` returns structured JSON for API clients.

---

## Project Structure

```text
Crop-Recommendation-System-Using-ML/
├── app.py                     # FastAPI application, routes, and startup artifact loading
├── schemas.py                 # Pydantic models for validation, weather, and reranking
├── services/
│   ├── ml_service.py          # Model loading and predict_proba() top-3 decoding
│   ├── weather.py             # Open-Meteo geocoding & 15-day forecast client
│   ├── reranker.py            # Configurable LangChain reranker & strict validation
│   └── recommendation.py      # End-to-end recommendation orchestration
├── templates/
│   └── index.html             # Responsive Jinja2 UI with location picker and comparison table
├── tests/
│   └── test_app.py            # Pytest suite covering ML, weather, validation, and AI edge cases
├── train_model.py             # Random Forest training and artifact export script
├── Crop_recommendation.csv    # Labeled soil & climate dataset (2,200 rows, 22 crops)
├── crop_model.pkl             # Trained Random Forest artifact
├── crop_names.pkl             # Sorted crop class mapping artifact
├── Procfile                   # Production startup command for PaaS deployment
├── .env.example               # Example environment configuration
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Setup & Run Commands

### 1. Create and Activate a Virtual Environment

```bash
python -m venv venv
# Windows (PowerShell)
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional for AI Reranking)

Copy `.env.example` to `.env` and add your server-side LLM provider key if you want LangChain weather reranking enabled:

```bash
cp .env.example .env
```

Supported `LLM_PROVIDER` values:
- `openai` (`LLM_MODEL=gpt-4o-mini`, `OPENAI_API_KEY` or `LLM_API_KEY`)
- `groq` (`LLM_MODEL=llama-3.3-70b-versatile`, `GROQ_API_KEY` or `LLM_API_KEY`)
- `gemini` (`LLM_MODEL=gemini-2.5-flash`, `GEMINI_API_KEY` or `LLM_API_KEY`)
- `openai_compatible` (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`)

Without an LLM API key, the application still serves ML top-3 predictions and live 15-day Open-Meteo weather forecasts, falling back gracefully to the ML ranking.

### 4. Retrain the ML Model (Optional)

Pre-trained artifacts (`crop_model.pkl` and `crop_names.pkl`) are already included (`99.32%` test accuracy on a 20% split of `Crop_recommendation.csv`). To retrain:

```bash
python train_model.py
```

### 5. Start the FastAPI Server

Development (auto-reload):
```bash
uvicorn app:app --reload
```

Production (`Procfile`):
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` in your browser (interactive API docs at `http://127.0.0.1:8000/docs`).

### 6. Run Tests

```bash
pytest
```

---

## API Endpoints

- `GET /` — Web UI form and results view
- `POST /predict` — Form endpoint accepting `Nitrogen`, `Phosphorus`, `Potassium`, `Temperature`, `Humidity`, `pH`, `Rainfall`, plus optional location fields
- `GET /api/locations?q=<place>` — Search locations by village, town, or city with district/state and country metadata
- `POST /api/recommend` — Structured JSON endpoint accepting soil/climate features and optional `location` or `location_query`
- `GET /health` — Health check reporting loaded model status and crop class count

---

## Sample Test Case (README Rice Example)

| N | P | K | Temperature (°C) | Humidity (%) | pH | Rainfall (mm) | Top ML Prediction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 90 | 42 | 43 | 20.88 | 82.00 | 6.50 | 202.94 | `rice` |

Example JSON request to `POST /api/recommend`:

```json
{
  "Nitrogen": 90,
  "Phosphorus": 42,
  "Potassium": 43,
  "Temperature": 20.88,
  "Humidity": 82.0,
  "pH": 6.5,
  "Rainfall": 202.94,
  "location_query": "Pune"
}
```
