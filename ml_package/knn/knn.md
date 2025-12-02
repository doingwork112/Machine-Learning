# K-Nearest Neighbors (KNN)

## Algorithm

K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm used for **classification** tasks. The core idea is that similar data points tend to have similar labels.

### Core Idea

KNN makes predictions based on the k closest training examples in the feature space:
- The algorithm finds the k nearest neighbors and assigns the most common class label among them (majority voting).

The distance metric used is typically Euclidean distance: `distance = sqrt(sum((x_i - x_j)^2))`

### Objective

Minimize classification error by leveraging local similarity patterns in the data.

### Key Hyperparameters

- **k** (default=3): Number of neighbors to consider
  - Small k (e.g., k=1): More sensitive to noise, may overfit
  - Large k (e.g., k=20): More stable but may underfit, loses local patterns
  - Optimal k is typically found through cross-validation
- **Distance Metric**: Euclidean by default
  - Other options: Manhattan, Minkowski, cosine distance

### Advantages

- Simple to understand and implement
- No training phase (lazy learning)
- Works well for non-linear problems

### Disadvantages

- Computationally expensive for large datasets (O(n) for each prediction)
- Sensitive to irrelevant features and outliers
- Requires feature scaling for best performance
- Memory intensive (stores all training data)

## Data

### Input Features

- **X**: Array-like of shape `(n_samples, n_features)`
  - Training samples with numeric features
  - Features should be scaled (e.g., using StandardScaler) for best performance
  - All features should be continuous numeric values

### Labels/Targets

- **y**: Array-like of shape `(n_samples,)`
  - Integer class labels (e.g., 0, 1, 2, ...)

### Data Loading and Preprocessing

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ml_package import KNNClassifier

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Preprocessing: Scale features (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Use KNN
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
```

### Preprocessing Requirements

1. **Feature Scaling**: Essential for KNN because it uses distance metrics
   - Use `StandardScaler` to normalize features (mean=0, std=1)
   - Or use `MinMaxScaler` to scale to [0, 1] range

2. **Missing Values**: Handle missing values before training
   - KNN cannot handle missing values directly

3. **Categorical Features**: Convert to numeric if present
   - Use one-hot encoding or label encoding



