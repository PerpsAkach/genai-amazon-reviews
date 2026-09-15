import argparse
import json

import pandas as pd
import pytest

from run_pipeline import positive_int, write_metrics, write_results


def test_positive_int_accepts_positive_value():
    assert positive_int("3") == 3


def test_positive_int_rejects_zero():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")


def test_write_results_supports_csv_json_and_jsonl(tmp_path):
    df = pd.DataFrame([{"source_rating": 5, "sentiment_stars": 4}])

    csv_path = tmp_path / "results.csv"
    json_path = tmp_path / "results.json"
    jsonl_path = tmp_path / "results.jsonl"

    write_results(df, csv_path)
    write_results(df, json_path)
    write_results(df, jsonl_path)

    assert "source_rating" in csv_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["sentiment_stars"] == 4
    assert json.loads(jsonl_path.read_text(encoding="utf-8").strip())["source_rating"] == 5


def test_write_results_rejects_unknown_extension(tmp_path):
    df = pd.DataFrame([{"x": 1}])
    with pytest.raises(ValueError, match="Output extension"):
        write_results(df, tmp_path / "results.txt")


def test_write_metrics_writes_json(tmp_path):
    path = tmp_path / "metrics.json"
    write_metrics({"n": 2, "exact_accuracy": 0.5}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["n"] == 2
