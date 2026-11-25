"""Machine learning inference utilities reusing the trained Random Forest artifact."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np

from schemas import MLCandidate, SoilClimateInput


def load_ml_artifacts(model_path: Path, labels_path: Path) -> tuple[Any, list[str]]:
    """Load the serialized scikit-learn model and sorted crop label mapping."""
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(labels_path, "rb") as labels_file:
        crop_names = list(pickle.load(labels_file))

    return model, crop_names


def predict_top_k(
    model: Any,
    crop_names: list[str],
    features: SoilClimateInput,
    k: int = 3,
) -> list[MLCandidate]:
    """Compute top-k crop probabilities matched explicitly via model.classes_."""
    feature_row = np.array([features.to_feature_array()], dtype=float)
    probabilities = model.predict_proba(feature_row)[0]

    scored_crops: list[tuple[str, float, int]] = []
    for class_id, prob in zip(model.classes_, probabilities):
        idx = int(class_id)
        crop_name = crop_names[idx]
        scored_crops.append((crop_name, float(prob), idx))

    scored_crops.sort(key=lambda item: (-item[1], item[2]))

    candidates: list[MLCandidate] = []
    for rank, (crop_name, prob, _) in enumerate(scored_crops[:k], start=1):
        candidates.append(
            MLCandidate(
                crop=crop_name,
                ml_rank=rank,
                ml_score=round(prob, 6),
                ml_score_pct=round(prob * 100.0, 2),
            )
        )

    return candidates
