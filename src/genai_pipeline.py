from __future__ import annotations

import re
from typing import Any

from .config import PipelineConfig
from .data_loader import validate_review

PROMPTS = {
    "clean": "Clean and correct this product review: {text}",
    "summary": "Summarize this review in one sentence: {text}",
    "features": "List the main product features or issues in this review: {text}",
}

SOURCE_FIELDS = (
    "asin",
    "parent_asin",
    "title",
    "timestamp",
    "verified_purchase",
    "helpful_vote",
)


def parse_star_label(label: str) -> int:
    """Extract a 1–5 star class from a classifier label."""

    match = re.search(r"(?<!\d)([1-5])(?!\d)", str(label))
    if not match:
        raise ValueError(f"Could not parse star label: {label!r}")
    return int(match.group(1))


class ReviewGenAIPipeline:
    """Prompt-based review transformation plus ordinal sentiment inference.

    Model objects can be injected for deterministic unit testing. When they are
    not supplied, Hugging Face components are loaded lazily on first use.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        tokenizer: Any | None = None,
        generator: Any | None = None,
        sentiment: Any | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.config.validate()
        self.tokenizer = tokenizer
        self.generator = generator
        self.sentiment = sentiment

    def _ensure_models(self) -> None:
        if (
            self.tokenizer is not None
            and self.generator is not None
            and self.sentiment is not None
        ):
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from transformers import pipeline as hf_pipeline

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.generator_model)
        if self.generator is None:
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.generator_model
            )
        if self.sentiment is None:
            self.sentiment = hf_pipeline(
                "sentiment-analysis",
                model=self.config.sentiment_model,
            )

    def _generate(self, prompt: str) -> str:
        self._ensure_models()
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        output = self.generator.generate(
            **encoded,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def process_review(self, review: dict[str, Any]) -> dict[str, Any]:
        validate_review(review)
        self._ensure_models()

        text = review["text"].strip()
        sentiment_result = self.sentiment(
            text[: self.config.sentiment_character_limit]
        )[0]

        result: dict[str, Any] = {
            "original": text,
            "clean": self._generate(PROMPTS["clean"].format(text=text)),
            "summary": self._generate(PROMPTS["summary"].format(text=text)),
            "features": self._generate(PROMPTS["features"].format(text=text)),
            "source_rating": float(review["rating"]),
            "sentiment_label": str(sentiment_result["label"]),
            "sentiment_stars": parse_star_label(sentiment_result["label"]),
            "sentiment_confidence": float(sentiment_result["score"]),
        }

        for field in SOURCE_FIELDS:
            if field in review and review[field] is not None:
                result[f"source_{field}"] = review[field]

        return result
