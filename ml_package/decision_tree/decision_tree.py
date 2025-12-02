"""Decision Tree Classifier Implementation"""
import numpy as np
from collections import Counter


class Node:
    """Node class for decision tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    """Decision Tree Classifier (CART algorithm)"""
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        gini = 1 - sum(p ** 2 for p in probabilities)
        return gini
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain from a split"""
        parent_gini = self._gini_impurity(y)
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        if n == 0:
            return 0
        child_gini = (n_left / n) * self._gini_impurity(y_left) + \
                     (n_right / n) * self._gini_impurity(y_right)
        gain = parent_gini - child_gini
        return gain
    
    def _best_split(self, X, y):
        """Find the best split for a node"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            for threshold in unique_values:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                y_left = y[left_indices]
                y_right = y[right_indices]
                gain = self._information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        if (depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature] > best_threshold)[0]
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def fit(self, X, y):
        """Build the decision tree"""
        X = np.array(X)
        y = np.array(y)
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        """Predict a single sample by traversing the tree"""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict class labels for samples"""
        X = np.array(X)
        predictions = [self._predict_sample(x, self.root) for x in X]
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

