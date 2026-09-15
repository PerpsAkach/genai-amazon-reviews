# Output Contract

The public runner emits one structured record per processed review plus a separate metrics JSON document.

## Prediction fields

Core fields produced for every valid processed review:

| Field | Meaning |
|---|---|
| `original` | Trimmed source review text |
| `clean` | FLAN-T5 cleaning/correction output |
| `summary` | FLAN-T5 one-sentence summary output |
| `features` | FLAN-T5 feature/issue extraction output |
| `source_rating` | Source Amazon star rating as a float |
| `sentiment_label` | Raw sentiment-model label |
| `sentiment_stars` | Parsed 1–5 integer star class |
| `sentiment_confidence` | Sentiment-model score returned by the Hugging Face pipeline |

Selected source fields are copied only when present:

- `source_asin`
- `source_parent_asin`
- `source_title`
- `source_timestamp`
- `source_verified_purchase`
- `source_helpful_vote`

The implementation intentionally does not add source user identifiers to the structured output.

## Supported result formats

The `--output` extension selects serialization:

- `.csv` → comma-separated table;
- `.json` → indented JSON array of records;
- `.jsonl` or `.ndjson` → one JSON record per line.

Unknown output extensions are rejected rather than silently reinterpreted.

## Metrics document

The metrics JSON contains:

- `n`
- `exact_accuracy`
- `within_one_star_accuracy`
- `mean_absolute_error`
- `quadratic_weighted_kappa`
- `confusion_matrix_labels`
- `confusion_matrix`

The confusion matrix always uses label order `[1, 2, 3, 4, 5]` so matrices from different runs remain structurally comparable.

`quadratic_weighted_kappa` may be `null` when the statistic is mathematically undefined for the observed labels.

## Interpretation boundary

The source Amazon rating is used as the comparison target for sentiment predictions. It is not a human-authored reference for the generated cleaning, summary, or feature/issue outputs. Those generated-text tasks therefore require a separate evaluation design if reference-quality claims are needed.
