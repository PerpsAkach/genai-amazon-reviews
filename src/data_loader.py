from datasets import load_dataset

from .config import PipelineConfig


def load_video_game_reviews(limit: int | None = None):
    cfg = PipelineConfig()
    dataset = load_dataset(cfg.dataset_name, cfg.review_config, split=cfg.split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def validate_review(record: dict) -> None:
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        raise ValueError("Review text must be non-empty.")
    rating = float(record.get("rating"))
    if not 1 <= rating <= 5:
        raise ValueError("Rating must lie in [1, 5].")
