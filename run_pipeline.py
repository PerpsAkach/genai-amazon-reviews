from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_video_game_reviews
from src.evaluation import evaluate_sentiment_predictions
from src.genai_pipeline import ReviewGenAIPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/reviews.csv"))
    args = parser.parse_args()

    reviews = load_video_game_reviews(limit=args.limit)
    processor = ReviewGenAIPipeline()
    rows = [processor.process_review(dict(review)) for review in reviews]

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(evaluate_sentiment_predictions(df))


if __name__ == "__main__":
    main()
