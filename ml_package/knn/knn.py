"""K-Nearest Neighbors Classifier Implementation"""
import numpy as np
from collections import Counter


class KNNClassifier:
    """Simple K-Nearest Neighbors Classifier"""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        """Predict class labels"""
        X = np.array(X)
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

