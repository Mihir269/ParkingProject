"""Extract HOG / LBP / GLCM / RGB / HSV features, compare LogReg, RF, SVM and XGBoost
on every feature set, select the best and save it.

Examples
--------
# folder layout (our own crops, PKLotSegmented, ACPDS patches):
python scripts/train_models.py --data data/acpds_patches --group-level 1

# CNRPark-EXT (label list) with camera as group:
python scripts/train_models.py --labels CNR-EXT/LABELS/all.txt --images CNR-EXT/PATCHES \
    --group-level 2 --max-per-class 5000 --tune-top 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.data import load_folder, load_label_list  # noqa: E402
from parking.features import FEATURE_SETS, FeatureConfig, FeatureExtractor  # noqa: E402
from parking.models import MODEL_NAMES, OccupancyClassifier  # noqa: E402
from parking.selection import run_selection  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--data", help="root folder with empty/ and occupied/ subfolders (any depth)")
src.add_argument("--labels", help="CNRPark-style label list: '<rel_path> <0|1>' per line")
ap.add_argument("--images", help="image root for --labels")
ap.add_argument("--group-level", type=int, default=None,
                help="path component (relative) used as CV group, e.g. camera / lot / day")
ap.add_argument("--max-per-class", type=int, default=None)
ap.add_argument("--patch-size", type=int, nargs=2, default=(64, 64), metavar=("W", "H"))
ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
ap.add_argument("--cv", type=int, default=5)
ap.add_argument("--tune-top", type=int, default=0, help="grid-search the top-N combinations")
ap.add_argument("--ensemble-top", type=int, default=3, help="soft-vote the top-k (0/1 disables)")
ap.add_argument("--test-size", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-jobs", type=int, default=-1)
ap.add_argument("--out", default="outputs/run")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

ds = load_folder(a.data, a.group_level) if a.data else load_label_list(a.labels, a.images, a.group_level)
ds = ds.subsample(a.max_per_class, a.seed)
print(f"{len(ds)} patches  (empty={int((ds.labels == 0).sum())}, occupied={int((ds.labels == 1).sum())})"
      + (f", {len(np.unique(ds.groups))} groups" if ds.groups is not None else ""))

cfg = FeatureConfig(patch_size=tuple(a.patch_size))
ext = FeatureExtractor(cfg)
cache = out / "features.npz"
if cache.exists() and np.load(cache)["X"].shape[0] == len(ds):
    X = np.load(cache)["X"]
    print("loaded cached features", X.shape)
else:
    t = time.time()
    X = ext.transform(ds.paths, n_jobs=a.n_jobs)
    np.savez_compressed(cache, X=X, y=ds.labels)
    print(f"extracted features {X.shape} in {time.time() - t:.1f}s  "
          + ", ".join(f"{k}={s.stop - s.start}" for k, s in ext.block_slices.items()))

res = run_selection(
    X, ds.labels, ext, ds.groups,
    feature_sets=a.feature_sets, models=a.models, n_splits=a.cv,
    tune_top=a.tune_top, ensemble_top=a.ensemble_top,
    test_size=a.test_size, seed=a.seed, n_jobs=a.n_jobs,
)

res.leaderboard.to_csv(out / "leaderboard.csv", index=False)
res.test_metrics_all.to_csv(out / "test_top5.csv", index=False)
OccupancyClassifier(cfg, res.best_estimator, res.best_name, metrics=res.test_metrics).save(out / "best_model.joblib")
(out / "summary.json").write_text(json.dumps(
    {"best": res.best_name, "test": res.test_metrics, "n": len(ds), "args": vars(a)}, indent=2, default=str))

# Report: leaderboard heatmap (feature set x model) + confusion matrix
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = res.leaderboard[~res.leaderboard.candidate.str.contains(r"tuned|ensemble")]
    piv = base.pivot(index="features", columns="model", values="cv_f1").reindex(a.feature_sets)
    fig, ax = plt.subplots(1, 2, figsize=(13, 0.45 * len(piv) + 2), gridspec_kw={"width_ratios": [3, 2]})
    im = ax[0].imshow(piv.values, cmap="viridis", aspect="auto")
    ax[0].set_xticks(range(piv.shape[1]), piv.columns)
    ax[0].set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax[0].text(j, i, f"{v:.3f}", ha="center", va="center", color="w" if v < np.nanmean(piv.values) else "k")
    ax[0].set_title("Mean CV F1 (occupied) by feature set and model")
    fig.colorbar(im, ax=ax[0])
    cm = np.array(res.test_metrics["confusion_matrix"])
    ax[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, cm[i, j], ha="center", va="center")
    ax[1].set_xticks([0, 1], ["empty", "occupied"])
    ax[1].set_yticks([0, 1], ["empty", "occupied"])
    ax[1].set_xlabel("predicted")
    ax[1].set_ylabel("true")
    ax[1].set_title(f"Test confusion matrix\n{res.best_name[:60]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "report.png", dpi=130)
except Exception as e:  # plotting is optional
    print("plot skipped:", e)

cols = ["candidate", "cv_f1", "cv_f1_std", "cv_roc_auc", "cv_accuracy", "latency_ms"]
print("\nTop 10 (by CV F1):")
print(res.leaderboard[cols].head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nSelected: {res.best_name}")
print("Held-out test:", {k: round(v, 4) for k, v in res.test_metrics.items() if k != "confusion_matrix"},
      "CM:", res.test_metrics["confusion_matrix"])
print(f"Saved to {out}/")
