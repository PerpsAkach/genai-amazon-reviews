import math

import pytest

from src.config import PipelineConfig
from src.data_loader import validate_review
from src.genai_pipeline import ReviewGenAIPipeline, parse_star_label


class FakeTokenizer:
    def __call__(self, prompt, **kwargs):
        assert prompt
        assert kwargs["truncation"] is True
        return {"input_ids": [[1, 2, 3]]}

    def decode(self, output, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " generated output "


class FakeGenerator:
    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        return [[10, 11]]


class FakeSentiment:
    def __call__(self, text):
        assert text
        return [{"label": "4 stars", "score": 0.91}]


def test_parse_star_label_supports_expected_labels():
    assert parse_star_label("5 stars") == 5
    assert parse_star_label("1 star") == 1
    assert parse_star_label("class: 3") == 3


def test_parse_star_label_rejects_unparseable_label():
    with pytest.raises(ValueError):
        parse_star_label("LABEL_POSITIVE")
    with pytest.raises(ValueError):
        parse_star_label("10 stars")


def test_validate_review_accepts_valid_record():
    validate_review({"text": "Great game.", "rating": 5})


@pytest.mark.parametrize(
    "record",
    [
        {"text": "", "rating": 5},
        {"text": "   ", "rating": 5},
        {"text": None, "rating": 5},
        {"text": "Bad.", "rating": 8},
        {"text": "Bad.", "rating": 0},
        {"text": "Bad.", "rating": None},
        {"text": "Bad.", "rating": "not-a-number"},
        {"text": "Bad.", "rating": math.nan},
    ],
)
def test_validate_review_rejects_invalid_record(record):
    with pytest.raises(ValueError):
        validate_review(record)


def test_pipeline_uses_injected_models_without_network_access():
    processor = ReviewGenAIPipeline(
        PipelineConfig(sentiment_character_limit=12),
        tokenizer=FakeTokenizer(),
        generator=FakeGenerator(),
        sentiment=FakeSentiment(),
    )

    result = processor.process_review(
        {
            "text": "A long but useful review.",
            "rating": 4,
            "asin": "DEMO-ASIN",
            "verified_purchase": True,
        }
    )

    assert result["clean"] == "generated output"
    assert result["summary"] == "generated output"
    assert result["features"] == "generated output"
    assert result["sentiment_stars"] == 4
    assert result["sentiment_confidence"] == pytest.approx(0.91)
    assert result["source_rating"] == 4.0
    assert result["source_asin"] == "DEMO-ASIN"
    assert result["source_verified_purchase"] is True
