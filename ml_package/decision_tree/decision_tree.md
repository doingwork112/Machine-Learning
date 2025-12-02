# Decision Tree Classifier

## Algorithm

Decision Tree is a tree-based algorithm that makes decisions by recursively splitting the data based on feature values. This implementation uses the CART (Classification and Regression Trees) algorithm.

### Core Idea

The algorithm builds a tree structure where:
- **Internal nodes**: Represent feature tests (e.g., "feature_1 <= threshold")
- **Branches**: Represent outcomes of the test
- **Leaf nodes**: Represent class labels (predictions)

The tree is built by recursively:
1. Finding the best feature and threshold to split the data
2. Splitting data into left and right subsets
3. Repeating until stopping criteria are met
4. Assigning majority class to leaf nodes

### Objective

Minimize Gini impurity (or maximize information gain) at each split to create pure leaf nodes that accurately classify instances.

**Gini Impurity**: Measures how often a randomly chosen element would be incorrectly labeled.
- Formula: `Gini = 1 - Σ(p_i)^2` where p_i is the proportion of class i
- Gini = 0: Perfect purity (all samples same class)
- Gini = 0.5: Maximum impurity (equal class distribution)

**Information Gain**: Measures the reduction in impurity after a split.
- Formula: `Gain = Gini(parent) - [weighted average of Gini(children)]`
- Higher gain = better split

### Key Hyperparameters

- **max_depth** (default=10): Maximum depth of the tree
  - Controls model complexity and prevents overfitting
  - Smaller depth: simpler model, may underfit
  - Larger depth: more complex model, may overfit
  - Typical range: 3-20

- **min_samples_split** (default=2): Minimum number of samples required to split a node
  - Prevents splitting nodes with too few samples
  - Larger values: simpler trees, less overfitting
  - Smaller values: more complex trees, may overfit

### Advantages

- Easy to interpret and visualize
- Handles both numeric and categorical features
- No feature scaling required
- Can capture non-linear relationships
- Feature importance can be extracted

### Disadvantages

- Prone to overfitting (especially deep trees)
- Sensitive to small changes in data
- May create biased trees if classes are imbalanced
- Can be unstable (small data changes → different trees)

## Data

### Input Features

- **X**: Array-like of shape `(n_samples, n_features)`
  - Training samples with numeric features
  - Features can be continuous or discrete
  - **No scaling required** (unlike KNN)
  - Missing values should be handled before training

### Labels/Targets

- **y**: Array-like of shape `(n_samples,)`
  - Integer class labels (e.g., 0, 1, 2, ...)
  - Binary or multi-class classification supported
  - Class labels should be integers starting from 0

### Data Loading and Preprocessing

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from ml_package import DecisionTreeClassifier

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Note: No scaling required for Decision Trees
# But scaling can still be beneficial in some cases

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use Decision Tree
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=2)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
accuracy = dt.score(X_test, y_test)
```

### Preprocessing Requirements

1. **Feature Scaling**: **Not required** for Decision Trees
   - Tree splits are based on feature thresholds, not distances
   - However, scaling may still help in some ensemble methods

2. **Missing Values**: Handle missing values before training
   - Decision Trees cannot handle missing values directly
   - Options: imputation, removal, or use algorithms that handle missing values

3. **Categorical Features**: Can handle numeric features directly
   - For categorical features, use label encoding or one-hot encoding
   - Current implementation works best with numeric features

4. **Class Imbalance**: Consider class weights if classes are imbalanced
   - Current implementation uses majority voting at leaves


