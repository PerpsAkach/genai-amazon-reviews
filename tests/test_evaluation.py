import pandas as pd
import pytest

from src.evaluation import evaluate_sentiment_predictions


def test_evaluation_returns_ordinal_metrics_and_confusion_matrix():
    df = pd.DataFrame(
        {
            "source_rating": [1, 2, 4, 5],
            "sentiment_stars": [1, 3, 4, 4],
        }
    )

    metrics = evaluate_sentiment_predictions(df)

    assert metrics["n"] == 4
    assert metrics["exact_accuracy"] == pytest.approx(0.5)
    assert metrics["within_one_star_accuracy"] == pytest.approx(1.0)
    assert metrics["mean_absolute_error"] == pytest.approx(0.5)
    assert metrics["quadratic_weighted_kappa"] is not None
    assert metrics["confusion_matrix_labels"] == [1, 2, 3, 4, 5]
    assert len(metrics["confusion_matrix"]) == 5
    assert all(len(row) == 5 for row in metrics["confusion_matrix"])


def test_evaluation_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        evaluate_sentiment_predictions(
            pd.DataFrame(columns=["source_rating", "sentiment_stars"])
        )


def test_evaluation_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing evaluation columns"):
        evaluate_sentiment_predictions(pd.DataFrame({"source_rating": [5]}))


def test_evaluation_rejects_out_of_range_predictions():
    with pytest.raises(ValueError, match="1–5"):
        evaluate_sentiment_predictions(
            pd.DataFrame({"source_rating": [5], "sentiment_stars": [6]})
        )
