"""Compare (feature set x model) combinations and select the best one(s).

Protocol
--------
1. Hold out a test split (by group if groups are given, e.g. whole cameras/days).
2. On the training split, run k-fold CV for every combination
   (StratifiedGroupKFold when groups exist, otherwise StratifiedKFold).
3. Rank by mean CV F1 for the "occupied" class, tie-break by ROC-AUC and then
   by inference latency (it will run on every spot of every frame).
4. Optionally grid-search hyper-parameters for the top-N combinations.
5. Optionally build a soft-voting ensemble of the top-k and CV it the same way.
6. The winner (chosen on CV only, never on the test set) is refit on the whole
   training split and evaluated once on the held-out test split.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from .features import FEATURE_SETS, FeatureExtractor
from .models import MODEL_NAMES, SoftVoteEnsemble, make_candidate, positive_score

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def split_holdout(y, groups=None, test_size=0.2, seed=0):
    idx = np.arange(len(y))
    if groups is not None and len(np.unique(groups)) >= 2:
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, y, groups))
        if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
            return tr, te
    return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)


def make_cv(y, groups=None, n_splits=5, seed=0):
    if groups is not None and len(np.unique(groups)) >= n_splits:
        return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed), groups
    return StratifiedKFold(n_splits, shuffle=True, random_state=seed), None


def latency_ms(est, X, n=200, repeats=3) -> float:
    """Median per-patch prediction time (classifier only, features excluded)."""
    Xs = X[: min(n, len(X))]
    est.predict(Xs)  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        est.predict(Xs)
        times.append(time.perf_counter() - t)
    return 1000 * float(np.median(times)) / len(Xs)


def evaluate(est, X, y) -> dict:
    s = positive_score(est, X)
    p = (s >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "roc_auc": roc_auc_score(y, s) if len(np.unique(y)) == 2 else float("nan"),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    }


@dataclass
class SelectionResult:
    leaderboard: pd.DataFrame
    best_name: str
    best_estimator: object  # fitted on full training split
    test_metrics: dict
    test_metrics_all: pd.DataFrame  # test metrics of every top candidate (for reporting only)


def run_selection(
    X: np.ndarray,
    y: np.ndarray,
    extractor: FeatureExtractor,
    groups: np.ndarray | None = None,
    feature_sets=None,
    models=MODEL_NAMES,
    n_splits: int = 5,
    tune_top: int = 0,
    ensemble_top: int = 3,
    min_ensemble_gain: float = 0.002,
    test_size: float = 0.2,
    seed: int = 0,
    n_jobs: int = -1,
    log=print,
) -> SelectionResult:
    feature_sets = list(feature_sets or FEATURE_SETS)
    tr, te = split_holdout(y, groups, test_size, seed)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    gtr = None if groups is None else groups[tr]
    cv, cv_groups = make_cv(ytr, gtr, n_splits, seed)
    log(f"train={len(tr)} test={len(te)}  cv={type(cv).__name__}({n_splits})  "
        f"occupied-rate train={ytr.mean():.2f} test={yte.mean():.2f}")

    candidates: dict[str, tuple] = {}
    rows = []
    for fs in feature_sets:
        cols = extractor.columns_for(fs)
        for m in models:
            name = f"{fs} | {m}"
            est, grid = make_candidate(m, cols, seed, n_jobs=1)  # parallelism is over CV folds
            candidates[name] = (est, grid)
            res = cross_validate(est, Xtr, ytr, groups=cv_groups, cv=cv, scoring=SCORING, n_jobs=n_jobs)
            row = {"candidate": name, "features": fs, "model": m, "n_features": len(cols)}
            for k in SCORING:
                row[f"cv_{k}"] = res[f"test_{k}"].mean()
                row[f"cv_{k}_std"] = res[f"test_{k}"].std()
            row["fit_s"] = res["fit_time"].mean()
            rows.append(row)
            log(f"  {name:<28} F1={row['cv_f1']:.4f}±{row['cv_f1_std']:.4f}  AUC={row['cv_roc_auc']:.4f}")

    board = pd.DataFrame(rows)

    # Latency for ranking ties: fit once on a small subset is enough for timing.
    def _rank(df):
        return df.sort_values(["cv_f1", "cv_roc_auc", "latency_ms"], ascending=[False, False, True]).reset_index(drop=True)

    board["latency_ms"] = np.nan
    top_names = board.sort_values("cv_f1", ascending=False).head(max(tune_top, ensemble_top, 5))["candidate"]
    for name in top_names:
        est = clone(candidates[name][0]).fit(Xtr, ytr)
        board.loc[board.candidate == name, "latency_ms"] = latency_ms(est, Xte)
    board = _rank(board)

    # Optional hyper-parameter tuning of the top-N.
    if tune_top:
        log(f"Tuning top {tune_top} candidates ...")
        for name in board.head(tune_top)["candidate"].tolist():
            est, grid = candidates[name]
            gs = GridSearchCV(est, grid, scoring="f1", cv=cv, n_jobs=n_jobs, refit=False)
            gs.fit(Xtr, ytr, groups=cv_groups)
            current = {k: est.get_params()[k] for k in gs.best_params_}
            if gs.best_params_ == current:
                log(f"  {name:<36} defaults already best")
                continue
            tuned = clone(est).set_params(**gs.best_params_)
            tname = f"{name} (tuned)"
            candidates[tname] = (tuned, {})
            res = cross_validate(tuned, Xtr, ytr, groups=cv_groups, cv=cv, scoring=SCORING, n_jobs=n_jobs)
            row = board[board.candidate == name].iloc[0].to_dict()
            row.update({"candidate": tname, "params": str(gs.best_params_)})
            for k in SCORING:
                row[f"cv_{k}"] = res[f"test_{k}"].mean()
                row[f"cv_{k}_std"] = res[f"test_{k}"].std()
            row["latency_ms"] = latency_ms(clone(tuned).fit(Xtr, ytr), Xte)
            board = pd.concat([board, pd.DataFrame([row])], ignore_index=True)
            log(f"  {tname:<36} F1={row['cv_f1']:.4f}  params={gs.best_params_}")
        board = _rank(board)

    # Optional soft-voting ensemble: best candidate of each of the top-k model
    # families (diversity matters more than raw rank for voting).
    if ensemble_top and ensemble_top > 1:
        members = board.drop_duplicates("model").head(ensemble_top)["candidate"].tolist()
        ens = SoftVoteEnsemble([(n, candidates[n][0]) for n in members])
        ename = "ensemble[" + " + ".join(members) + "]"
        candidates[ename] = (ens, {})
        res = cross_validate(ens, Xtr, ytr, groups=cv_groups, cv=cv, scoring=SCORING, n_jobs=n_jobs)
        row = {"candidate": ename, "features": "mixed", "model": "soft-vote", "n_features": X.shape[1]}
        for k in SCORING:
            row[f"cv_{k}"] = res[f"test_{k}"].mean()
            row[f"cv_{k}_std"] = res[f"test_{k}"].std()
        row["fit_s"] = res["fit_time"].mean()
        row["latency_ms"] = latency_ms(clone(ens).fit(Xtr, ytr), Xte)
        board = _rank(pd.concat([board, pd.DataFrame([row])], ignore_index=True))
        log(f"  {ename} F1={row['cv_f1']:.4f}")

    # Final: refit winner on all training data, evaluate once on the test split.
    # The (slower, more complex) ensemble must beat the best single model by
    # ``min_ensemble_gain`` CV-F1 to be chosen.
    best_name = board.iloc[0]["candidate"]
    if best_name.startswith("ensemble["):
        single = board[~board.candidate.str.startswith("ensemble[")].iloc[0]
        if board.iloc[0]["cv_f1"] - single["cv_f1"] < min_ensemble_gain:
            best_name = single["candidate"]
    best = clone(candidates[best_name][0]).fit(Xtr, ytr)
    test_metrics = evaluate(best, Xte, yte)

    # For the report: test metrics of the top-5 as well (not used for selection).
    test_rows = []
    for name in board.head(5)["candidate"]:
        est = best if name == best_name else clone(candidates[name][0]).fit(Xtr, ytr)
        m = evaluate(est, Xte, yte)
        m.pop("confusion_matrix")
        test_rows.append({"candidate": name, **{f"test_{k}": v for k, v in m.items()}})

    return SelectionResult(board, best_name, best, test_metrics, pd.DataFrame(test_rows))
