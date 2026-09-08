from __future__ import annotations

import re
from typing import Any

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline

from .config import PipelineConfig
from .data_loader import validate_review

PROMPTS = {
    "clean": "Clean and correct this product review: {text}",
    "summary": "Summarize this review in one sentence: {text}",
    "features": "List the main product features or issues in this review: {text}",
}


def parse_star_label(label: str) -> int:
    match = re.search(r"\b([1-5])\b", str(label))
    if not match:
        raise ValueError(f"Could not parse star label: {label!r}")
    return int(match.group(1))


class ReviewGenAIPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.generator_model)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(self.config.generator_model)
        self.sentiment = hf_pipeline("sentiment-analysis", model=self.config.sentiment_model)

    def _generate(self, prompt: str) -> str:
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
        text = review["text"].strip()
        sentiment_result = self.sentiment(text[:4000])[0]
        return {
            "original": text,
            "clean": self._generate(PROMPTS["clean"].format(text=text)),
            "summary": self._generate(PROMPTS["summary"].format(text=text)),
            "features": self._generate(PROMPTS["features"].format(text=text)),
            "source_rating": float(review["rating"]),
            "sentiment_label": sentiment_result["label"],
            "sentiment_stars": parse_star_label(sentiment_result["label"]),
            "sentiment_confidence": float(sentiment_result["score"]),
        }
