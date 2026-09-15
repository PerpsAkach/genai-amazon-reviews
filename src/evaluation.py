from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REQUIRED_COLUMNS = ("source_rating", "sentiment_stars")
STAR_LABELS = [1, 2, 3, 4, 5]


def evaluate_sentiment_predictions(df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate 1–5 star predictions against the source Amazon rating."""

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing evaluation columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Cannot evaluate an empty prediction table")

    y_true = pd.to_numeric(df["source_rating"], errors="raise").round().astype(int)
    y_pred = pd.to_numeric(df["sentiment_stars"], errors="raise").astype(int)

    if not y_true.between(1, 5).all() or not y_pred.between(1, 5).all():
        raise ValueError("Evaluation ratings must lie in the 1–5 star range")

    error = (y_true - y_pred).abs()
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    return {
        "n": int(len(df)),
        "exact_accuracy": float((y_true == y_pred).mean()),
        "within_one_star_accuracy": float((error <= 1).mean()),
        "mean_absolute_error": float(error.mean()),
        "quadratic_weighted_kappa": None if np.isnan(qwk) else float(qwk),
        "confusion_matrix_labels": STAR_LABELS,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=STAR_LABELS,
        ).tolist(),
    }
