"""Flask web application for serving Crop Recommendation ML predictions."""

import os
from pathlib import Path
import pickle

from flask import Flask, render_template, request
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.pkl"
LABELS_PATH = BASE_DIR / "crop_names.pkl"

FEATURE_FIELDS = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "pH",
    "Rainfall",
]

app = Flask(__name__)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(LABELS_PATH, "rb") as file:
    crop_names = pickle.load(file)


@app.route("/")
def home():
    """Render the main crop recommendation form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Parse soil/climate form inputs and render the predicted crop."""
    form_data = {field: request.form.get(field, "").strip() for field in FEATURE_FIELDS}

    try:
        features = [float(form_data[field]) for field in FEATURE_FIELDS]
    except ValueError:
        return (
            render_template(
                "index.html",
                error="Please provide valid numeric values for all seven fields.",
                form_data=form_data,
            ),
            400,
        )

    input_data = np.array([features], dtype=float)
    pred_idx = int(model.predict(input_data)[0])
    predicted_crop = crop_names[pred_idx]

    return render_template(
        "index.html",
        result=True,
        crop=predicted_crop,
        form_data=form_data,
    )


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)
