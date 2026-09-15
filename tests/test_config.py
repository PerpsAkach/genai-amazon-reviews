import pytest

from src.config import PipelineConfig


def test_default_config_is_valid():
    PipelineConfig().validate()


@pytest.mark.parametrize(
    "config",
    [
        PipelineConfig(dataset_name=""),
        PipelineConfig(review_config=""),
        PipelineConfig(split=""),
        PipelineConfig(generator_model=""),
        PipelineConfig(sentiment_model=""),
        PipelineConfig(max_input_tokens=0),
        PipelineConfig(max_new_tokens=0),
        PipelineConfig(sentiment_max_tokens=0),
    ],
)
def test_invalid_configurations_are_rejected(config):
    with pytest.raises(ValueError):
        config.validate()
