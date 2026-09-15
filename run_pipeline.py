from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import load_video_game_reviews
from src.evaluation import evaluate_sentiment_predictions
from src.genai_pipeline import ReviewGenAIPipeline


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def write_results(df: pd.DataFrame, output: Path) -> None:
    """Write prediction records as CSV, JSON, or JSON Lines."""

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()

    if suffix == ".csv":
        df.to_csv(output, index=False)
    elif suffix == ".json":
        output.write_text(
            json.dumps(df.to_dict(orient="records"), indent=2, default=str),
            encoding="utf-8",
        )
    elif suffix in {".jsonl", ".ndjson"}:
        df.to_json(output, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError("Output extension must be .csv, .json, .jsonl, or .ndjson")


def write_metrics(metrics: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process Amazon Video Game reviews with FLAN-T5 and a 1–5 star sentiment model."
    )
    parser.add_argument("--limit", type=positive_int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reviews.csv"),
        help="Prediction output (.csv, .json, .jsonl, or .ndjson).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON path for evaluation metrics. Defaults beside --output.",
    )
    args = parser.parse_args()

    reviews = load_video_game_reviews(limit=args.limit)
    processor = ReviewGenAIPipeline()
    rows = [processor.process_review(dict(review)) for review in reviews]

    df = pd.DataFrame(rows)
    metrics = evaluate_sentiment_predictions(df)

    write_results(df, args.output)
    metrics_output = args.metrics_output or args.output.with_suffix(".metrics.json")
    write_metrics(metrics, metrics_output)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {len(df)} predictions to {args.output}")
    print(f"Wrote evaluation metrics to {metrics_output}")


if __name__ == "__main__":
    main()
