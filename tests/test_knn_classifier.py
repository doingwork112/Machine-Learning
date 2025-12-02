"""Unit tests for the custom KNN classifier."""
import numpy as np
import pytest

from ml_package import KNNClassifier


@pytest.fixture
def simple_dataset():
    """Toy binary classification dataset."""
    X_train = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.1, 0.2], [0.9, 0.8]])
    y_test = np.array([0, 1])
    return X_train, y_train, X_test, y_test


def test_knn_classifier_predicts_correct_labels(simple_dataset):
    """Model should classify points based on the majority label of nearest neighbors."""
    X_train, y_train, X_test, y_test = simple_dataset
    model = KNNClassifier(k=3).fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert predictions.tolist() == y_test.tolist()


def test_knn_classifier_score_returns_accuracy(simple_dataset):
    """score should match the manual accuracy for the toy dataset."""
    X_train, y_train, X_test, y_test = simple_dataset
    model = KNNClassifier(k=3).fit(X_train, y_train)
    assert model.score(X_test, y_test) == pytest.approx(1.0)

