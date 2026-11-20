import json

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from review_classifier.train import load_dataset, evaluate


def test_load_dataset(tmp_path):
    dataset = tmp_path / "samples.jsonl"

    rows = [
        {"text": "Excellent product", "label": 1},
        {"text": "Terrible product", "label": 0},
    ]

    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    texts, labels = load_dataset(dataset)

    assert texts == ["Excellent product", "Terrible product"]
    assert labels == [1, 0]


def test_load_empty_dataset(tmp_path):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset vazio"):
        load_dataset(dataset)


def test_model_training_and_evaluation():
    texts = [
        "excellent quality",
        "really good product",
        "amazing experience",
        "very happy customer",
        "terrible quality",
        "really bad product",
        "awful experience",
        "very unhappy customer",
    ]

    labels = [1, 1, 1, 1, 0, 0, 0, 0]

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    model.fit(texts, labels)

    predictions = model.predict(texts)

    assert len(predictions) == len(texts)
    assert set(predictions).issubset({0, 1})

    metrics = evaluate(model, texts, labels)

    assert set(metrics) == {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    }

    assert all(0 <= value <= 1 for value in metrics.values())