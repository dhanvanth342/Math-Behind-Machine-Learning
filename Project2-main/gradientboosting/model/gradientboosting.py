# model/gradientboosting.py

import numpy as np
from typing import List, Optional, Dict, Callable
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import os


# Defining custom exceptions for model state and input validation
class ModelNotFittedError(Exception):
    """Raising when methods are called before fitting."""
    pass


class InvalidDataError(ValueError):
    """Raising when provided inputs are invalid."""
    pass


@dataclass
class TreeNode:
    """Representing a node in the regression tree."""
    is_leaf: bool = False
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[float] = None
    gain: float = 0.0


class GradientBoostingClassifier:
    """Building gradient-boosted trees for classification tasks."""
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        n_jobs: int = -1,
        early_stopping_rounds: Optional[int] = 5,
        validation_fraction: float = 0.1,
        custom_loss: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    ):
        # Validating hyperparameters [will be useful for testing]
        if not isinstance(n_estimators, int) or n_estimators < 1:
            raise InvalidDataError("n_estimators must be integer ≥ 1.")
        if not (0 < learning_rate <= 1):
            raise InvalidDataError("learning_rate must be in (0, 1].")
        if not isinstance(max_depth, int) or max_depth < 1:
            raise InvalidDataError("max_depth must be integer ≥ 1.")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise InvalidDataError("min_samples_split must be integer ≥ 2.")
        if not (0 < subsample <= 1):
            raise InvalidDataError("subsample must be in (0, 1].")
        if early_stopping_rounds is not None and (not isinstance(early_stopping_rounds, int) or early_stopping_rounds < 1):
            raise InvalidDataError("early_stopping_rounds must be integer ≥ 1 or None.")
        if not (0 <= validation_fraction < 1):
            raise InvalidDataError("validation_fraction must be in [0, 1).")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.custom_loss = custom_loss
        
        # Preparing internal state
        self._trees: List[List[TreeNode]] = []
        self._init_score: float = 0.0
        self.training_loss_history: List[float] = []
        self.validation_loss_history: List[float] = []
        #self.feature_importance_: Dict[int, float] = defaultdict(float)
        self.best_iteration: Optional[int] = None
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.is_fitted_: bool = False
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray):
        """Checking that X and y have compatible shapes."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise InvalidDataError("X must be 2-D array.")
        if y_arr.ndim != 1:
            raise InvalidDataError("y must be 1-D array.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise InvalidDataError("X and y must have same number of samples.")
        return X_arr, y_arr

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Applying sigmoid to raw scores."""
        return np.clip(1 / (1 + np.exp(-x)), 1e-15, 1 - 1e-15)

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        """Applying softmax to raw class scores."""
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, features: List[int]) -> Optional[tuple]:
        """Searching for the best split across given features."""
        best_gain = float("-inf")
        best_split = None
        n = len(y)
        if n < self.min_samples_split:
            return None

        parent_var = np.var(y)
        for f in features:
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2
            for thr in thresholds:
                left = y[X[:, f] <= thr]
                right = y[X[:, f] > thr]
                if len(left) < self.min_samples_split or len(right) < self.min_samples_split:
                    continue
                gain = parent_var - (np.var(left)*len(left) + np.var(right)*len(right)) / n
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature": f,
                        "threshold": thr,
                        "left_idx": np.where(X[:, f] <= thr)[0],
                        "right_idx": np.where(X[:, f] > thr)[0]
                    }
        if best_split is None:
            return None
        return best_gain, best_split

    def _build_tree(self, X: np.ndarray, y: np.ndarray) -> TreeNode:
        """Building a regression tree on residuals in parallel."""
        if len(y) < self.min_samples_split:
            leaf = TreeNode(is_leaf=True, value=np.mean(y))
            return leaf

        # Splitting features into chunks for parallel search
        n_jobs = os.cpu_count() if self.n_jobs < 1 else min(self.n_jobs, os.cpu_count())
        feats = list(range(X.shape[1]))
        chunk_size = max(1, len(feats) // n_jobs)
        chunks = [feats[i:i+chunk_size] for i in range(0, len(feats), chunk_size)]

        # Running best-split search in parallel
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = executor.map(lambda fs: self._find_best_split(X, y, fs), chunks)

        # Selecting the overall best split
        best_gain = float("-inf")
        split_info = None
        for res in results:
            if res and res[0] > best_gain:
                best_gain, split_info = res

        if split_info is None:
            return TreeNode(is_leaf=True, value=np.mean(y))

        # Creating internal node and recursing
        node = TreeNode(
            feature_idx=split_info["feature"],
            threshold=split_info["threshold"],
            gain=best_gain
        )
        left_idx, right_idx = split_info["left_idx"], split_info["right_idx"]
        node.left = self._build_tree(X[left_idx], y[left_idx])
        node.right = self._build_tree(X[right_idx], y[right_idx])
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        """Fitting the gradient boosting model to the training data."""
        # Validating and storing training data
        X_arr, y_arr = self._validate_inputs(X, y)
        if np.isnan(X_arr).any() or np.isnan(y_arr).any():
            raise ValueError("Input contains NaN values.")
        # Reject too-few samples or only one class
        if len(X_arr) < self.min_samples_split or len(np.unique(y_arr)) < 2:
           raise ValueError("Need at least two samples from two classes to fit.")
        # Ensuring class labels are integer for one-hot indexing
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_arr = y_arr.astype(int)
        self.n_classes_ = len(np.unique(y_arr))
        self.n_features_ = X_arr.shape[1]
        self.X_train_, self.y_train_ = X_arr, y_arr

        # Splitting off validation set if early stopping is enabled
        if self.early_stopping_rounds:
            n_val = max(1, int(len(X_arr) * self.validation_fraction))
            X_train, X_val = X_arr[:-n_val], X_arr[-n_val:]
            y_train, y_val = y_arr[:-n_val], y_arr[-n_val:]
        else:
            X_train, y_train = X_arr, y_arr
            X_val = y_val = None

        # Initializing raw predictions
        if self.n_classes_ == 2:
            p0 = np.clip(np.mean(y_train), 1e-15, 1-1e-15)
            init_score = np.log(p0 / (1-p0))
        else:
            init_score = 0.0
        self._init_score = init_score
        raw_train = np.full((len(X_train), self.n_classes_), init_score)
        raw_val = np.full((len(X_val), self.n_classes_), init_score) if X_val is not None else None

        # One-hot encoding of y
        Y_train = np.eye(self.n_classes_)[y_train]
        Y_val = np.eye(self.n_classes_)[y_val] if X_val is not None else None

        # Preparing storage
        self._trees = [[] for _ in range(self.n_classes_)]
        self.training_loss_history = []
        self.validation_loss_history = []
        best_val_loss = float("inf")
        rounds_no_improve = 0

        # Main boosting loop
        for m in range(self.n_estimators):
            # Computing probabilities and residuals
            prob_train = self._softmax(raw_train)
            grad = Y_train - prob_train

            # Fitting one tree per class
            for k in range(self.n_classes_):
                # Subsampling rows
                if self.subsample < 1:
                    idx = np.random.choice(len(X_train), int(self.subsample * len(X_train)), replace=False)
                    Xs, ys = X_train[idx], grad[idx, k]
                else:
                    Xs, ys = X_train, grad[:, k]
                # Building tree on residuals
                tree = self._build_tree(Xs, ys)
                self._trees[k].append(tree)
                # Updating raw scores
                preds = np.array([self._predict_tree(xi, tree) for xi in X_train])
                raw_train[:, k] += self.learning_rate * preds
                if X_val is not None:
                    preds_val = np.array([self._predict_tree(xi, tree) for xi in X_val])
                    raw_val[:, k] += self.learning_rate * preds_val

            # Logging training loss
            prob_train = self._softmax(raw_train)
            loss_train = -np.mean(np.sum(Y_train * np.log(prob_train + 1e-15), axis=1))
            self.training_loss_history.append(loss_train)

            # Logging validation loss and checking early stopping
            if X_val is not None:
                prob_val = self._softmax(raw_val)
                loss_val = -np.mean(np.sum(Y_val * np.log(prob_val + 1e-15), axis=1))
                self.validation_loss_history.append(loss_val)
                if loss_val + 1e-4 < best_val_loss:
                    best_val_loss, rounds_no_improve = loss_val, 0
                else:
                    rounds_no_improve += 1
                    if rounds_no_improve >= self.early_stopping_rounds:
                        break

        # Finalizing fit
        self.best_iteration = m - rounds_no_improve if X_val is not None else None
        self.is_fitted_ = True
        return self

    def _predict_tree(self, x: np.ndarray, node: TreeNode) -> float:
        """Traversing a single sample through a regression tree."""
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise ModelNotFittedError("Model has not been fitted yet.")
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise InvalidDataError("X must be a 2-D array for prediction.")
        if X_arr.shape[1] != self.n_features_:
            raise InvalidDataError(f"Each sample in X must have {self.n_features_} features.")
        raw = np.full((X_arr.shape[0], self.n_classes_), self._init_score)
        for k in range(self.n_classes_):
            for tree in self._trees[k]:
                preds = np.array([self._predict_tree(xi, tree) for xi in X_arr])
                raw[:, k] += self.learning_rate * preds
        return self._softmax(raw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicting class labels for input data."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, object]:
        """Calculating accuracy, precision, recall, f1-score, and confusion matrix."""
        if not self.is_fitted_:
            raise ModelNotFittedError("Model has not been fitted yet.")
        X_arr, y_arr = self._validate_inputs(X, y)
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_arr = y_arr.astype(int)
        proba = self.predict_proba(X_arr)
        preds = self.predict(X_arr)
        # Building confusion matrix
        cm = np.zeros((self.n_classes_, self.n_classes_), dtype=int)
        for t, p in zip(y_arr, preds):
            cm[t, p] += 1
        # Computing per-class metrics
        precisions, recalls, f1s = [], [], []
        for k in range(self.n_classes_):
            tp = cm[k, k]
            fp = cm[:, k].sum() - tp
            fn = cm[k, :].sum() - tp
            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0
            f = 2 * p * r / (p + r) if p + r > 0 else 0.0
            precisions.append(p); recalls.append(r); f1s.append(f)
        # Aggregating metrics
        metrics = {
            "confusion_matrix": cm,
            "accuracy": np.mean(preds == y_arr),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1_score": np.mean(f1s)
        }
        return metrics

    def plot_learning_curves(self):
        """Plotting training and validation loss over boosting rounds."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.training_loss_history, label="Training Loss")
        if self.validation_loss_history:
            plt.plot(self.validation_loss_history, label="Validation Loss")
        plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    
