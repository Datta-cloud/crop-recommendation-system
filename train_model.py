"""Training and evaluation pipeline for the Crop Recommendation Random Forest model."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "crop_model.pkl"
LABELS_PATH = BASE_DIR / "crop_names.pkl"

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Unique crop classes: {df['label'].nunique()}")

    crop_list = sorted(df["label"].unique())
    crop_to_num = {crop: idx for idx, crop in enumerate(crop_list)}
    df["crop_num"] = df["label"].map(crop_to_num)

    # Convert to NumPy arrays so inference inputs match training format cleanly.
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["crop_num"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print("\nFeature Importances:")
    for feat, imp in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda item: -item[1]
    ):
        print(f"  {feat:12s} {imp:.4f}")

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    with open(LABELS_PATH, "wb") as file:
        pickle.dump(crop_list, file)

    print(f"\nSaved model artifact: {MODEL_PATH.name}")
    print(f"Saved label artifact: {LABELS_PATH.name}")

    sample = np.array([[90.0, 42.0, 43.0, 20.8, 82.0, 6.5, 202.9]])
    pred_crop = crop_list[model.predict(sample)[0]]
    print(f"Verification sample prediction: {pred_crop}")


if __name__ == "__main__":
    main()
