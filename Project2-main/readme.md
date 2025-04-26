This repository contains 2implementation of gradient-boosted classification tree ensemble, built entirely from first principles. It follows the deviance-based boosting framework (negative log-likelihood) as outlined in Sections 10.9–10.10 of Hastie, Tibshirani & Friedman’s Elements of Statistical Learning. You’ll find full support for binary and multiclass targets, subsampling, early stopping, custom losses, and comprehensive evaluation routines.

---

## Overview

Our `GradientBoostingClassifier` constructs an additive model of shallow regression trees by repeatedly fitting each tree to the negative gradient (residual) of the log-loss.  At each iteration:

1. **Computing scores** for each class (initialized to log-odds for binary, zero for multiclass)  
2. **Evaluating probabilities** via sigmoid (binary) or softmax (multiclass)  
3. **Forming residuals** = (one-hot or {0,1}) – predicted probabilities  
4. **Fitting** one small decision tree per class on those residuals (with optional subsampling and parallel split search)  
5. **Updating** raw class scores by adding a shrunken tree output  
6. **Logging** training and validation loss; stopping early if it stalls  

In addition, the model offers:

- **`predict_proba`** and `predict`  
- **`evaluate`**: accuracy, precision, recall, F1, confusion matrix  
- **Error-handling** for bad inputs and calling methods before fitting  

---

## Project Structure

```
gradientboosting/
│
├── model/
│   └── gradientboosting.py        # Core GBM implementation
│
├── tests/
│   ├── __init__.py
│   └── test_gradientboost.py      # Pytest suite exercising all features
│
├── test_data/                     # Small CSVs for unit tests
│   ├── binary_linear.csv
│   ├── binary_xor.csv
│   ├── multiclass_clusters.csv
│   ├── binary_imbalanced.csv
│   └── multiclass_highdim.csv
│
├── visualizations/
│   └── visualizations.ipynb       # Notebook plotting loss curves & comparisons
│
├── requirements.txt               # pip dependencies
└── README.md                      # This documentation
```

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/dhanvanth342/Math-Behind-Machine-Learning.git
   cd Project2-main
   ```

2. **Create & activate a virtual environment**  
   Using **venv**:
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```
   Or **conda**:
   ```bash
   conda create -n gbm python=3.10
   conda activate gbm
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Basic Usage

```python
from model.gradientboosting import GradientBoostingClassifier

# Initializing the model
model = GradientBoostingClassifier(
    n_estimators=100,           # number of boosting rounds
    learning_rate=0.05,         # shrinkage factor
    max_depth=3,                # depth of each tree
    min_samples_split=2,        # min samples per split
    subsample=0.7,              # row subsampling fraction
    early_stopping_rounds=10,   # stop if val loss stalls
    validation_fraction=0.1     # fraction held out for early stopping
)

# Fitting on numpy arrays X_train, y_train
model.fit(X_train, y_train)

# Predicting probabilities and classes
probs = model.predict_proba(X_test)
preds = model.predict(X_test)

# Evaluating performance
metrics = model.evaluate(X_test, y_test)
print(metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score'])
```

---

## Running the Test Suite

We use **pytest** to verify correctness:

```bash
# From project root
pytest -v -s tests/test_gradientboost.py
```

<details>
<summary>📸 Preview of test run output</summary>

```
collected 10 items                                                                                                                                              

test_gradientboost.py::test_learning_simple_boundary test_learning_simple_boundary passed
PASSED
test_gradientboost.py::test_probability_bounds test_probability_bounds passed
PASSED
test_gradientboost.py::test_learning_nonlinear test_learning_nonlinear passed
PASSED
test_gradientboost.py::test_multiclass_basic test_multiclass_basic passed
PASSED
test_gradientboost.py::test_early_stopping_triggers test_early_stopping_triggers passed
PASSED
test_gradientboost.py::test_evaluate_outputs test_evaluate_outputs passed
PASSED
test_gradientboost.py::test_custom_loss_support test_custom_loss_support passed
PASSED
test_gradientboost.py::test_init_validation test_init_validation passed
PASSED
test_gradientboost.py::test_feature_dimension_mismatch test_feature_dimension_mismatch passed
PASSED
test_gradientboost.py::test_edge_cases test_edge_cases passed
PASSED
```

 
</details>

---

## Model Details

**Q: What does this model do and when should I use it?**  
Our `GradientBoostingClassifier` constructs an additive model of shallow regression trees by repeatedly fitting each tree to the negative gradient (residual) of the log-loss.  At each iteration:

1. **Computing scores** for each class (initialized to log-odds for binary, zero for multiclass)  
2. **Evaluating probabilities** via sigmoid (binary) or softmax (multiclass)  
3. **Forming residuals** = (one-hot or {0,1}) – predicted probabilities  
4. **Fitting** one small decision tree per class on those residuals (with optional subsampling and parallel split search)  
5. **Updating** raw class scores by adding a shrunken tree output  
6. **Logging** training and validation loss; stopping early if it stalls  

In addition, the model offers:

- **`predict_proba`** and `predict`  
- **`evaluate`**: accuracy, precision, recall, F1, confusion matrix  
- **Error-handling** for bad inputs and calling methods before fitting  

You can use this model for classification purpose, as it can compete with state of the art algorithms, and less computationally cost compare to neural networks. We have tested our model on digit classification from 1 to 10 and receieved 95% accuracy. 

**Q: How did you test your model to determine if it is working reasonably correctly?

We generated five small synthetic datasets under `test_data/`, each designed to probe a different aspect of the classifier:

1. **Linear separability (binary)**  
   – A 3-feature, 25-row set of two well-separated Gaussian clusters, verifying that the model can learn a straightforward decision boundary with high accuracy.

2. **XOR pattern (binary, non-linear)**  
   – A 4-feature, 30-row dataset embedding the classic XOR in two dimensions (plus noise), ensuring the boosting loop can capture non-linear structure.

3. **Clustered multiclass**  
   – A 5-feature, 40-row dataset with three overlapping Gaussian clusters, validating correct softmax handling, multi-class residuals, and confusion-matrix metrics.

4. **Imbalanced binary**  
   – A 6-feature, 50-row set with an 80/20 class split, testing early-stopping and subsampling under skewed class priors.

5. **High-dimensional rare classes (multiclass)**  
   – A 10-feature, 50-row dataset with five classes (one very small), stressing numeric stability and capacity to learn from scarce labels.

Our pytest suite then exercises:

- Basic and **non-linear decision boundaries**  
- **Probability validity** (0 ≤ p ≤ 1)  
- **Multiclass recall** and confusion-matrix construction  
- **Early stopping** under noisy or imbalanced data  
- **Custom loss-function** support  
- **Hyperparameter** and **dimension** checks  
- **Edge cases**: single sample, constant target, NaNs, predict-before-fit  

Passing these tests ensures both the **core boosting logic** and the **robustness checks** are functioning as intended..

**Q: What parameters have you exposed to users of your implementation in order to tune performance?**

- `n_estimators`  
  Number of boosting rounds (i.e. how many trees are added).  More trees can improve fit but increase runtime and risk overfitting.

- `learning_rate`  
  Shrinkage factor applied to each tree’s predictions.  Smaller values slow down learning and typically require more trees, but can yield better generalization.

- `max_depth`  
  Maximum depth of each regression tree.  Controls how complex each weak learner can be—deeper trees can capture more interactions but may overfit.

- `min_samples_split`  
  Minimum number of samples required to split an internal node.  Larger values make trees more conservative and help prevent overfitting on small datasets.

- `subsample`  
  Fraction of the training set randomly sampled for fitting each tree.  Values <1.0 introduce stochasticity (bagging), which can reduce variance and improve robustness.

- `early_stopping_rounds`  
  Number of consecutive iterations without improvement on a hold-out set before stopping training.  Works together with `validation_fraction`.

- `validation_fraction`  
  Proportion of the training data held out to monitor validation loss for early stopping.


**Q: Any inputs the model struggles with?**  
Small datasets with extremely low sample counts or highly imbalanced rare-class scenarios can be unstable.  In our visualization notebook you’ll see that after ≈10 iterations the validation and training curves diverge—indicating overfitting. Given more time, We could add:

- **Lower early-stop thresholds** (e.g. 5 rounds)  
- **Adaptive learning-rate schedules** (decay η as training progresses)  

to mitigate this and smooth out the loss plateau.  

---

## Additional Information

Open **`visualizations/visualizations.ipynb`** to:

- Plot training vs. validation loss curves  
- Compare performance on both binary (breast cancer) and multiclass (digits) datasets against `sklearn`’s GBM and `RandomForestClassifier`  
- Observe that on the multiclass digits task:  
  ```
  Random Forest > Custom GBM > Single Decision Tree
  ```
  confirming that our implementation trains reasonably well on real-world data.


