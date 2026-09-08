from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    review_config: str = "raw_review_Video_Games"
    metadata_config: str = "raw_meta_Video_Games"
    split: str = "full"
    generator_model: str = "google/flan-t5-base"
    sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    max_input_tokens: int = 512
    max_new_tokens: int = 100
