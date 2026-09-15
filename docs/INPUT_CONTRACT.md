# Input Contract

The implemented public runner reads records from the Hugging Face dataset configured by `PipelineConfig`.

## Required fields

Each processed review must contain:

| Field | Requirement |
|---|---|
| `text` | Non-empty string after trimming |
| `rating` | Numeric, finite value in the inclusive range 1–5 |

Invalid required values raise `ValueError`; the current runner is fail-fast rather than silently dropping malformed reviews.

## Optional source fields

When available, the pipeline may preserve these non-user fields for traceability:

- `asin`
- `parent_asin`
- `title`
- `timestamp`
- `verified_purchase`
- `helpful_vote`

Other upstream fields are ignored by the current structured output.

## Dataset configuration

Default configuration:

```text
dataset: McAuley-Lab/Amazon-Reviews-2023
review config: raw_review_Video_Games
split: full
generator: google/flan-t5-base
sentiment: nlptown/bert-base-multilingual-uncased-sentiment
```

The recovered project context also includes `raw_meta_Video_Games`, but the current public runner does not join the metadata dataset.

## Sampling

`--limit` must be a positive integer. When the requested limit exceeds the available dataset length, the loader returns the available rows without indexing beyond the dataset.

## Token handling

Generator inputs are tokenized with truncation up to `max_input_tokens`. Sentiment inputs are passed to the Hugging Face pipeline with tokenizer-aware truncation up to `sentiment_max_tokens`. This avoids relying on character-count heuristics for transformer input limits.

## External dependency boundary

The dataset is not committed to this repository. A real run therefore depends on the configured upstream Hugging Face dataset being available and on the runtime having network access when the dataset/model artifacts are not already cached.
