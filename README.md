# GenAI Amazon Reviews

Generative AI and NLP pipeline built around Amazon Reviews 2023 for review cleaning, one-sentence summarization, product feature/issue extraction, and 1–5 star sentiment analysis.

## Stack

- Python
- Hugging Face Datasets / Transformers
- FLAN-T5
- BERT sentiment classification
- Pandas / scikit-learn

## Pipeline

```text
Amazon Reviews 2023
        |
        v
Review validation
        |
        +--> FLAN-T5: cleaning
        +--> FLAN-T5: summarization
        +--> FLAN-T5: feature/issue extraction
        +--> BERT: 1–5 star sentiment
        |
        v
Structured output + evaluation
```

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py --limit 25 --output outputs/reviews.csv
```

Historical model/dataset choices are separated from reconstructed and enhanced implementation details in `PROVENANCE.md`.
