from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .config import PipelineConfig

DatasetLoader = Callable[..., Any]


def load_video_game_reviews(
    limit: int | None = None,
    config: PipelineConfig | None = None,
    *,
    dataset_loader: DatasetLoader | None = None,
):
    """Load the configured Amazon Video Games review split.

    ``limit`` is applied after the dataset is loaded and must be a positive
    integer when supplied. ``dataset_loader`` can be injected for deterministic
    tests; the Hugging Face dependency is loaded lazily for real runtime use.
    """

    cfg = config or PipelineConfig()
    cfg.validate()

    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer when provided")

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    dataset = dataset_loader(cfg.dataset_name, cfg.review_config, split=cfg.split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def validate_review(record: dict[str, Any]) -> None:
    """Validate the fields required by the implemented pipeline."""

    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Review text must be a non-empty string")

    try:
        rating = float(record.get("rating"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Rating must be numeric") from exc

    if not math.isfinite(rating) or not 1 <= rating <= 5:
        raise ValueError("Rating must lie in [1, 5]")
