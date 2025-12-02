# Machine-Learning
## 1. Project Overview

This repository is my final project for the course on Machine Learning and Data Science.

The goal is to:

- Use **Jupyter notebooks** to show complete ML workflows.
- Build a small **Python machine learning package** with reusable models (two methods).
- Follow good practices in software development, testing, and documentation.

The project focuses on **binary classification** using the
`Breast Cancer Wisconsin (Diagnostic)` dataset from `sklearn.datasets`.

## 2. Dataset

- Source: `sklearn.datasets.load_breast_cancer`
- Task: Predict whether a tumor is **malignant** or **benign**.
- Features: 30 numeric features describing properties of cell nuclei.
- Target: Binary label (`0 = malignant`, `1 = benign`).

The dataset is loaded directly from scikit-learn, so no extra download is needed.

## 3. Package Documentation

Each algorithm in `ml_package` ships with its own README that explains the **purpose**, **core functionality**, and **usage examples**:

- `ml_package/knn/knn.md`: K-Nearest Neighbors classifier overview, hyperparameters, and sample code.
- `ml_package/decision_tree/decision_tree.md`: Decision Tree classifier overview, splitting strategy, and sample code.

