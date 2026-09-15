from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the reconstructed review pipeline."""

    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    review_config: str = "raw_review_Video_Games"
    metadata_config: str = "raw_meta_Video_Games"
    split: str = "full"
    generator_model: str = "google/flan-t5-base"
    sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    max_input_tokens: int = 512
    max_new_tokens: int = 100
    sentiment_character_limit: int = 4000

    def validate(self) -> None:
        required_strings = {
            "dataset_name": self.dataset_name,
            "review_config": self.review_config,
            "split": self.split,
            "generator_model": self.generator_model,
            "sentiment_model": self.sentiment_model,
        }
        for name, value in required_strings.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.sentiment_character_limit <= 0:
            raise ValueError("sentiment_character_limit must be positive")


DEFAULT_CONFIG = PipelineConfig()
DEFAULT_CONFIG.validate()
