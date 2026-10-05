"""Candidate classifiers and the saved-model bundle.

Every candidate is a sklearn Pipeline that takes the *full* feature matrix
(all blocks) and starts with a ColumnSelector picking its feature set. This way
(feature set x model) combinations, and soft-voting ensembles of the best
combinations, all share one cached feature matrix.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_NAMES = ("logreg", "rf", "svm", "xgb")


class ColumnSelector(TransformerMixin, BaseEstimator):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X if self.columns is None else X[:, self.columns]


def make_model(name: str, n_features: int, seed: int = 0, n_jobs: int = -1):
    """Return (estimator, small hyper-parameter grid) for a model name."""
    if name == "logreg":
        est = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=3000))])
        grid = {"clf__C": [0.01, 0.1, 1, 10]}
    elif name == "rf":
        est = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1, max_features="sqrt", n_jobs=n_jobs, random_state=seed
        )
        grid = {"n_estimators": [200, 500], "max_depth": [None, 20], "min_samples_leaf": [1, 3]}
    elif name == "svm":
        steps = [("scale", StandardScaler())]
        # RBF-SVM is slow on very high-dimensional HOG; PCA keeps it tractable.
        if n_features > 300:
            steps.append(("pca", PCA(n_components=0.95, random_state=seed)))
        steps.append(("clf", SVC(kernel="rbf", C=10, gamma="scale")))
        est = Pipeline(steps)
        grid = {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01]}
    elif name == "xgb":
        est = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=seed,
        )
        grid = {"n_estimators": [200, 500], "max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1]}
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return est, grid


def make_candidate(model_name: str, columns: np.ndarray, seed: int = 0, n_jobs: int = -1):
    est, grid = make_model(model_name, len(columns), seed, n_jobs)
    pipe = Pipeline([("select", ColumnSelector(columns)), ("model", est)])
    grid = {f"model__{k}": v for k, v in grid.items()}
    return pipe, grid


def positive_score(est, X) -> np.ndarray:
    """P(occupied) if the model has probabilities, else a sigmoid of the margin."""
    try:
        return est.predict_proba(X)[:, 1]
    except AttributeError:
        return expit(est.decision_function(X))


class SoftVoteEnsemble(ClassifierMixin, BaseEstimator):
    """Average positive-class scores of several fitted-from-scratch candidates."""

    def __init__(self, estimators=None):
        self.estimators = estimators  # list of (name, unfitted estimator)

    def fit(self, X, y):
        self.fitted_ = [(n, clone(e).fit(X, y)) for n, e in self.estimators]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.mean([positive_score(e, X) for _, e in self.fitted_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class OccupancyClassifier:
    """What gets saved to disk: feature config + fitted pipeline + threshold."""

    def __init__(self, feature_config, estimator, name: str, threshold: float = 0.5, metrics: dict | None = None):
        self.feature_config = feature_config
        self.estimator = estimator
        self.name = name
        self.threshold = threshold
        self.metrics = metrics or {}

    def _extractor(self):
        from .features import FeatureExtractor

        return FeatureExtractor(self.feature_config)

    def predict_proba(self, patches) -> np.ndarray:
        X = self._extractor().transform(patches, n_jobs=1)
        return positive_score(self.estimator, X)

    def predict(self, patches) -> np.ndarray:
        return (self.predict_proba(patches) >= self.threshold).astype(int)

    def save(self, path):
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "OccupancyClassifier":
        import joblib

        return joblib.load(path)
