
import argparse
import json
from pathlib import Path
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


def load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    texts = [row["text"] for row in rows]
    labels = [int(row["label"]) for row in rows]

    if not texts:
        raise ValueError(f"Dataset vazio: {path}")

    return texts, labels


def evaluate(model, texts, labels):
    predictions = model.predict(texts)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/uci-v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/baseline_metrics.json"),
    )
    args = parser.parse_args()

    x_train, y_train = load_dataset(args.data_dir / "train.jsonl")
    x_val, y_val = load_dataset(args.data_dir / "validation.jsonl")

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)

    classifier = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
            )),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42,
            )),
        ]
    )
    classifier.fit(x_train, y_train)

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "tfidf_logistic_regression.joblib"
    joblib.dump(classifier, model_path)

    print(f"Modelo salvo em: {model_path}")

    results = {
        "dummy_classifier": evaluate(dummy, x_val, y_val),
        "tfidf_logistic_regression": evaluate(
            classifier, x_val, y_val
        ),
        "train_samples": len(x_train),
        "validation_samples": len(x_val),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
