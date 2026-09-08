import pytest

from src.data_loader import validate_review
from src.genai_pipeline import parse_star_label


def test_parse_star_label():
    assert parse_star_label("5 stars") == 5
    assert parse_star_label("1 star") == 1


def test_validate_review_accepts_valid_record():
    validate_review({"text": "Great game.", "rating": 5})


def test_validate_review_rejects_out_of_range_rating():
    with pytest.raises(ValueError):
        validate_review({"text": "Bad.", "rating": 8})
