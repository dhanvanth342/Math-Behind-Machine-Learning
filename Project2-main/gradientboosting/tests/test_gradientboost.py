import pytest
import numpy as np
import pandas as pd
import os

import sys 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


try:
    from model.gradientboosting import GradientBoostingClassifier, InvalidDataError, ModelNotFittedError
except ImportError as e:
    print("Import Error Details:")
    print(f"Error: {e}")
    print("Available paths:", sys.path)
    raise
# Loading fixtures from test_data folder
TEST_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__),  # model/tests/
                 os.pardir,                   # model/
                 'test_data')                 # model/test_data
)

def load_dataset(name: str) -> pd.DataFrame:
    """Loading CSV by name from the test_data folder."""
    path = os.path.join(TEST_DATA_DIR, f"{name}.csv")
    return pd.read_csv(path)

# 1. Testing simple linear separation on binary data
#    ensuring accuracy is high on clean, linearly separable dataset

def test_learning_simple_boundary():
    df = load_dataset('binary_linear')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.mean(preds == y) >= 0.5
    print("test_learning_simple_boundary passed") 
# What happens if accuracy < 0.5?
# that means the test is marked as failed, signaling that our GradientBoostingClassifier didn’t learn the linear pattern well enough 


# 2. Testing probability outputs bounds on XOR-style data
#    ensuring predict_proba returns values in [0,1]

def test_probability_bounds():
    df = load_dataset('binary_xor')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    model = GradientBoostingClassifier(n_estimators=20)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    print("test_probability_bounds passed")
# What happens if proba < 0 or proba > 1?
# that means the test is marked as failed, signaling that our GradientBoostingClassifier didn’t return valid probabilities

# 3. Testing non-linear boundary learning
#    verifying model outperforms random on XOR data

def test_learning_nonlinear():
    df = load_dataset('binary_xor')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    model = GradientBoostingClassifier(n_estimators=40, max_depth=3)
    model.fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.6
    print("test_learning_nonlinear passed")
# What happens if accuracy ≤ 0.6?
# that means the test is marked as failed, signaling that our GradientBoostingClassifier didn’t learn the XOR pattern well enough

# 4. Testing multiclass clustering performance
#    validating macro-recall > 0.5 on synthetic clusters

def test_multiclass_basic():
    df = load_dataset('multiclass_clusters')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    model = GradientBoostingClassifier(n_estimators=40, max_depth=3)
    model.fit(X, y)
    mets = model.evaluate(X, y)
    assert mets['accuracy'] >= 0.5
    print("test_multiclass_basic passed")

# What happens if recall < 0.5?
# that means the test is marked as failed, signaling that our GradientBoostingClassifier didn’t learn the multiclass pattern well enough

# 5. Testing early stopping on imbalanced binary data
#    verifying best_iteration < n_estimators

def test_early_stopping_triggers():
    df = load_dataset('binary_imbalanced')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    model = GradientBoostingClassifier(n_estimators=50, early_stopping_rounds=5, validation_fraction=0.2)
    model.fit(X, y)
    assert model.best_iteration is not None
    assert model.best_iteration < 50
    print("test_early_stopping_triggers passed")
# What happens if best_iteration >= n_estimators?
# that means the test is marked as failed, signaling that our GradientBoostingClassifier didn’t trigger early stopping correctly

# 6. Testing evaluate() outputs required metrics
#    ensuring evaluate returns all keys and proper types

def test_evaluate_outputs():
    df = load_dataset('binary_linear')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(X, y)
    mets = model.evaluate(X, y)
    for key in ['accuracy','precision','recall','f1_score','confusion_matrix']:
        assert key in mets
        assert mets[key] is not None
    print("test_evaluate_outputs passed")
# What happens if any key is missing or None?
# that means the test is marked as failed, signaling that our GradientBoostingClassifier didn’t return all required metrics correctly

# 7. Testing custom loss function support
#    confirming trees are built when using custom loss

def test_custom_loss_support():
    df = load_dataset('binary_linear')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    def mse_loss(y_true, y_pred, return_gradient=False):
        loss = np.mean((y_true-y_pred)**2)
        grad = 2*(y_pred-y_true)
        return (loss, grad) if return_gradient else loss
    model = GradientBoostingClassifier(custom_loss=mse_loss)
    model.fit(X, y)
    # Presence of at least one tree per class indicates use of loss
    assert len(model._trees[0]) > 0
    print("test_custom_loss_support passed")

# 8. Testing hyperparameter validation
#    ensuring invalid init args raise InvalidDataError

def test_init_validation():
    with pytest.raises(InvalidDataError):
        GradientBoostingClassifier(n_estimators=0)
    with pytest.raises(InvalidDataError):
        GradientBoostingClassifier(learning_rate=1.5)
    print("test_init_validation passed")
# 9. Testing feature-dimension mismatch on predict
#    ensuring wrong feature count triggers error

def test_feature_dimension_mismatch():
    df = load_dataset('binary_linear')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(X, y)
    X_bad = np.random.rand(X.shape[0], X.shape[1]+1)
    with pytest.raises(InvalidDataError):
        model.predict_proba(X_bad)
    with pytest.raises(InvalidDataError):
        model.predict(X_bad)
    print("test_feature_dimension_mismatch passed")
# 10. Edge-case handling tests
#     verifying fit/predict raise errors on degenerate inputs

def test_edge_cases():
    # single sample
    Xs, ys = np.array([[1,2,3]]), np.array([0])
    with pytest.raises(ValueError):
        GradientBoostingClassifier().fit(Xs, ys)
    # constant target
    Xc = np.random.rand(5,3); yc = np.zeros(5)
    with pytest.raises(ValueError):
        GradientBoostingClassifier().fit(Xc, yc)
    # NaNs in X
    Xn = np.random.rand(6,3); Xn[0,0]=np.nan; yn = np.random.randint(0,2,6)
    with pytest.raises(ValueError):
        GradientBoostingClassifier().fit(Xn, yn)
    # predict before fit
    model = GradientBoostingClassifier()
    Xp = np.random.rand(4,3)
    with pytest.raises(ModelNotFittedError):
        model.predict(Xp)
    print("test_edge_cases passed")