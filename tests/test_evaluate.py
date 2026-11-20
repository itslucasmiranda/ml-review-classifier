
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from review_classifier.train import evaluate


def test_confusion_matrix():
    actual = [0, 0, 1, 1]
    predicted = [0, 1, 0, 1]

    result = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    assert result.tolist() == [
        [1, 1],
        [1, 1],
    ]


def test_model_serialization(tmp_path):
    model = DummyClassifier(strategy="most_frequent")

    texts = [
        "Excellent product",
        "Terrible product",
        "Amazing service",
    ]
    labels = [1, 0, 1]

    model.fit(texts, labels)

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)

    restored = joblib.load(path)

    assert restored.predict(texts).tolist() == (
        model.predict(texts).tolist()
    )


def test_evaluation_metrics():
    model = DummyClassifier(strategy="most_frequent")

    texts = ["good", "bad", "excellent", "terrible"]
    labels = [1, 0, 1, 0]

    model.fit(texts, labels)

    results = evaluate(model, texts, labels)

    assert set(results) == {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    }

    assert all(
        0 <= value <= 1
        for value in results.values()
    )
