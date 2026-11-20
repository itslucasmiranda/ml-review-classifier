import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from review_classifier.train import load_dataset


def main():
    model_path = Path("models/tfidf_logistic_regression.joblib")
    test_path = Path("data/processed/uci-v1/test.jsonl")
    output_path = Path("reports/final_evaluation.json")

    model = joblib.load(model_path)

    texts, labels = load_dataset(test_path)
    predictions = model.predict(texts)

    results = {
        "model": "tfidf_logistic_regression",
        "dataset": "uci-v1",
        "test_samples": len(texts),
        "accuracy": accuracy_score(labels, predictions),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()