# Machine Learning Final Project

## Project Overview

 It combines **Jupyter notebooks** that walk through full ML workflows with a small, reusable **Python machine learning package**.

The core task is **binary classification** on the `Breast Cancer Wisconsin (Diagnostic)` dataset from `sklearn.datasets`, predicting whether a tumor is malignant or benign.

Main goals:

- **Notebooks**: end-to-end examples of data exploration, preprocessing, modeling, and evaluation.
- **Custom package**: clean implementations of classic algorithms (KNN, Decision Tree) with tests and documentation.
- **Best practices**: clear structure, simple APIs, and basic unit testing with `pytest`.

## Dataset

- **Source**: `sklearn.datasets.load_breast_cancer`
- **Task**: Predict tumor type (**malignant** vs **benign**)  
- **Samples**: 569  
- **Features**: 30 numeric features describing cell nuclei  
- **Target encoding**: `0 = malignant`, `1 = benign`

The dataset is loaded directly from scikit-learn, so no manual download is required.

## Repository Structure

```text
Machine-Learning/
├── ml_package/                  # Custom ML package
│   ├── __init__.py
│   ├── knn/
│   │   ├── __init__.py
│   │   ├── knn.py              # KNN classifier implementation
│   │   └── knn.md              # Algorithm documentation
│   └── decision_tree/
│       ├── __init__.py
│       ├── decision_tree.py    # Decision Tree classifier implementation
│       └── decision_tree.md    # Algorithm documentation
├── notebooks/                  # Jupyter notebooks (analysis and demos)
│   ├── 01_data_exploration_and_preprocessing.ipynb
│   ├── 02_classical_ml_algorithms.ipynb
│   ├── 03_custom_package_demo.ipynb
│   ├── 04_neural_networks.ipynb
│   └── 05_clustering_and_dimensionality_reduction.ipynb
├── tests/                      # Unit tests for the custom package
│   ├── test_knn_classifier.py
│   └── test_decision_tree_classifier.py
└── README.md
```

## Installation

In a fresh environment (for example `conda` or `venv`):

```bash
cd Machine-Learning

# install dependencies
pip install numpy scikit-learn matplotlib seaborn pandas pytest jupyter tensorflow

# install the custom package in editable mode
pip install -e .
```

## Using the Custom Package

Example usage of `ml_package` in Python:

```python
from ml_package import KNNClassifier, DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)
print("KNN accuracy:", knn.score(X_test, y_test))

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
print("Decision Tree accuracy:", dt.score(X_test, y_test))
```

## Notebooks

- **01\_data\_exploration\_and\_preprocessing.ipynb**: basic EDA, feature scaling, and train/test split.  
- **02\_classical\_ml\_algorithms.ipynb**: compares Logistic Regression, k-NN, Decision Trees, Random Forest, Gradient Boosting, Naive Bayes, Voting Classifier, Stacking Classifier SVM on the breast cancer task, with short analysis.  
- **03\_custom\_package\_demo.ipynb**: end-to-end demo of the custom `ml_package` (KNN + Decision Tree) on the dataset.  
- **04\_neural\_networks.ipynb**: MLP (sklearn) and Keras models, training curves, and result discussion.  
- **05\_clustering\_and\_dimensionality\_reduction.ipynb**: PCA / t-SNE visualization and clustering (K-Means, hierarchical, DBSCAN) with ARI and silhouette analysis.

## Package Documentation

Each algorithm in `ml_package` has a dedicated markdown file:

- `ml_package/knn/knn.md` – **K-Nearest Neighbors**: core idea, hyperparameters, data assumptions, and example code.  
- `ml_package/decision_tree/decision_tree.md` – **Decision Tree**: CART splitting, stopping criteria, and example usage.  

These documents explain the **purpose**, **functionality**, and **usage** of the custom implementations.

## Running Tests

From the project root:

```bash
pytest tests/ -v
```

This runs the unit tests for:

- **KNNClassifier** (`tests/test_knn_classifier.py`)  
- **DecisionTreeClassifier** (`tests/test_decision_tree_classifier.py`)

## Results

- **Classical models**: Logistic Regression and SVM reach accuracy around **0.98** on the breast cancer test set.  
- **Custom package**:  
  - `KNNClassifier (k=3)` achieves accuracy ≈ **0.98**, with very high precision/recall.  
  - `DecisionTreeClassifier (max_depth=3)` achieves accuracy ≈ **0.94**, trading some accuracy for interpretability.  
- **Neural networks**: Feed-forward networks in `04_neural_networks.ipynb` reach similar or slightly higher performance, confirming that the dataset is relatively easy once properly preprocessed.  

More detailed analysis for each method is included inside the corresponding notebooks.

## Note on commit identity:
My earliest commits were made before I configured Git identity on this Mac, so the system’s default hostname/email appeared.
I updated my Git configuration from this point.