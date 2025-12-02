"""Unit tests for the custom Decision Tree classifier."""
import numpy as np
import pytest

from ml_package import DecisionTreeClassifier


@pytest.fixture
def simple_dataset():
    """Toy dataset with one feature and a clear split at x=1.5."""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.5], [2.5]])
    y_test = np.array([0, 1])
    return X_train, y_train, X_test, y_test


def test_decision_tree_predicts_correct_labels(simple_dataset):
    """Tree should perfectly separate the toy dataset."""
    X_train, y_train, X_test, y_test = simple_dataset
    model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert predictions.tolist() == y_test.tolist()


def test_decision_tree_score_returns_accuracy(simple_dataset):
    """Accuracy should be perfect on the simple dataset."""
    X_train, y_train, X_test, y_test = simple_dataset
    model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
    assert model.score(X_test, y_test) == pytest.approx(1.0)

