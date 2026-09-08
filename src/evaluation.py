import pandas as pd


def evaluate_sentiment_predictions(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["source_rating"].astype(float).round().astype(int)
    y_pred = df["sentiment_stars"].astype(int)
    error = (y_true - y_pred).abs()

    return {
        "n": float(len(df)),
        "exact_accuracy": float((y_true == y_pred).mean()),
        "within_one_star_accuracy": float((error <= 1).mean()),
        "mean_absolute_error": float(error.mean()),
    }
