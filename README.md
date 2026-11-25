# Crop Recommendation System

A machine learning web application built with **Python**, **scikit-learn**, and **Flask** that recommends the optimal crop to cultivate based on soil nutrient composition and local climate conditions.

---

## Overview

Selecting the right crop for a given soil and climate profile directly impacts agricultural yield and resource efficiency. This project trains a multi-class **Random Forest** classifier on 2,200 labeled agricultural samples across 22 crop varieties and serves real-time predictions through a Flask web interface.

### Input Features (7 Parameters)

| Feature | Description | Unit | Dataset Range |
| :--- | :--- | :--- | :--- |
| `N` | Nitrogen content ratio in soil | kg/ha | `0 – 140` |
| `P` | Phosphorus content ratio in soil | kg/ha | `5 – 145` |
| `K` | Potassium content ratio in soil | kg/ha | `5 – 205` |
| `temperature` | Ambient temperature | °C | `8.83 – 43.68` |
| `humidity` | Relative humidity | % | `14.26 – 99.98` |
| `ph` | Soil pH value | 0–14 | `3.50 – 9.94` |
| `rainfall` | Annual/seasonal rainfall | mm | `20.21 – 298.56` |

---

## Dataset

- **Source file:** `Crop_recommendation.csv`
- **Samples:** 2,200 rows, 8 columns (7 numerical features + 1 target `label`), zero missing values
- **Classes:** 22 balanced crop categories (100 samples per class):
  `apple`, `banana`, `blackgram`, `chickpea`, `coconut`, `coffee`, `cotton`, `grapes`, `jute`, `kidneybeans`, `lentil`, `maize`, `mango`, `mothbeans`, `mungbean`, `muskmelon`, `orange`, `papaya`, `pigeonpeas`, `pomegranate`, `rice`, `watermelon`

---

## Model & Evaluation

- **Algorithm:** `RandomForestClassifier` (`n_estimators=100`, `random_state=42`)
- **Train / Test Split:** 80% training (`1,760` samples) / 20% testing (`440` samples) with `random_state=42`
- **Test Accuracy:** **99.32%**

### Feature Importance (Trained Model)

| Rank | Feature | Importance |
| :---: | :--- | :---: |
| 1 | `rainfall` | `0.227` |
| 2 | `humidity` | `0.211` |
| 3 | `K` (Potassium) | `0.181` |
| 4 | `P` (Phosphorus) | `0.144` |
| 5 | `N` (Nitrogen) | `0.109` |
| 6 | `temperature` | `0.076` |
| 7 | `ph` | `0.052` |

---

## Project Structure

```text
Crop-Recommendation-System-Using-ML/
├── app.py                     # Flask application and /predict inference route
├── train_model.py             # Data loading, model training, evaluation, and serialization
├── Crop_recommendation.csv    # Labeled soil & climate dataset (2,200 samples)
├── crop_model.pkl             # Serialized Random Forest model
├── crop_names.pkl             # Serialized sorted crop class list
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules for virtualenvs and caches
├── templates/
│   └── index.html             # Responsive web UI for input and prediction display
└── README.md
```

---

## Setup & Usage

### 1. Clone and Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

Pre-trained model artifacts (`crop_model.pkl` and `crop_names.pkl`) are included. To retrain from `Crop_recommendation.csv` and print evaluation metrics:

```bash
python train_model.py
```

### 4. Run the Web Application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser, enter the 7 soil/climate values, and submit to view the recommended crop.

*(Set `FLASK_DEBUG=true` in your environment if you want Flask auto-reload during development.)*

---

## Sample Test Inputs

| N | P | K | Temperature (°C) | Humidity (%) | pH | Rainfall (mm) | Predicted Crop |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 90 | 42 | 43 | 20.88 | 82.00 | 6.50 | 202.94 | `rice` |
| 18 | 30 | 29 | 26.76 | 92.86 | 6.42 | 224.59 | `coconut` |
| 89 | 47 | 38 | 25.52 | 72.25 | 6.00 | 151.89 | `jute` |
| 24 | 130 | 195 | 30.00 | 81.54 | 6.11 | 67.13 | `grapes` |

---

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning & Data:** scikit-learn, NumPy, pandas
- **Frontend:** HTML5, CSS3 (Jinja2 templates)
