# Provenance

This repository separates recovered project choices from the current portfolio reconstruction so that historical and implementation claims remain defensible.

## RECOVERED

Supported by prior project artifacts or preserved project context:

- Amazon Reviews 2023 as the source dataset family;
- the `raw_review_Video_Games` review configuration;
- the `raw_meta_Video_Games` metadata configuration as part of the original project context;
- `google/flan-t5-base` for prompt-conditioned text transformation;
- `nlptown/bert-base-multilingual-uncased-sentiment` for 1–5 star sentiment classification;
- the cleaning, one-sentence summarization, and feature/issue extraction prompt pattern.

Recovered context does **not** establish that the current public source tree is literal historical source code.

## RECONSTRUCTED

Current code rebuilt from the supported project design where original source bytes were unavailable:

- modular configuration, loading, generation, sentiment, evaluation, and CLI structure;
- review validation;
- structured row-level outputs;
- CSV/JSON/JSONL serialization;
- evaluation against the source star rating.

## ENHANCED

New portfolio-quality engineering added after reconstruction:

- validated runtime configuration;
- positive-limit validation for dataset sampling;
- lazy model loading and dependency injection for deterministic offline unit tests;
- token-aware truncation for both FLAN-T5 inputs and sentiment inference;
- preservation of non-user source fields such as ASIN, title, timestamp, verified-purchase status, and helpful-vote count when present;
- exact accuracy, within-one-star accuracy, mean absolute error, quadratic weighted kappa, and a fixed-label confusion matrix;
- metrics JSON output;
- bounded dependency ranges;
- Python 3.11 / 3.12 / 3.13 CI;
- source compilation, Ruff linting, CLI-surface validation, and dependency auditing;
- expanded tests for configuration, input validation, evaluation, serialization, dataset limits, and injected-model execution.

## UNVERIFIED / NOT CLAIMED

This repository does not present the following as verified facts:

- the current code as byte-for-byte historical project source;
- the previously remembered approximately 87% text/GenAI metric, because the original result artifact has not been recovered;
- a production deployment or hosted inference service;
- domain-specific fine-tuning of FLAN-T5 or the sentiment model on this Amazon Video Games subset;
- human-reference evaluation of generated summaries or extracted features/issues;
- guaranteed factuality of generated text;
- deterministic reproducibility of remote Hugging Face model/dataset contents without pinned upstream revisions.

## Data boundary

No Amazon review dataset is committed to this repository. Runtime use depends on the public upstream dataset and model downloads. The public code intentionally avoids persisting user identifiers from source review records in its structured output.
