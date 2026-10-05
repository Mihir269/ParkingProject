"""Train the sliding-window car detector.

Patches (cars vs background) are sampled from annotated frames, described with
HOG / LBP / GLCM / RGB / HSV, and every (feature set x LogReg / RF / SVM / XGBoost)
combination is compared with grouped CV; the best becomes stage 2 of the cascade.
Then hard-negative mining, and detection AP on held-out frames.

python scripts/train_detector.py --scenes data/scenes/train_* --out outputs/detector
python scripts/train_detector.py --yolo PKLot.v1/train/images --car-classes 1 --empty-classes 0 \
       --test-yolo PKLot.v1/valid/images --out outputs/detector_pklot
python scripts/train_detector.py --pklot PKLot/PKLot --out outputs/detector_pklot
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from parking.annotations import load_pklot_xml, load_scene_json, load_yolo, sample_patches, suggest_window_sizes  # noqa: E402
from parking.detector import CarDetector, DetectorConfig, Stage1HOG, average_precision, mine_hard_negatives  # noqa: E402
from parking.features import FEATURE_SETS, FeatureConfig, FeatureExtractor  # noqa: E402
from parking.models import MODEL_NAMES  # noqa: E402
from parking.selection import run_selection  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--scenes", nargs="*", default=[], help="scene folders containing annotations.json")
ap.add_argument("--yolo", help="YOLO images dir (labels in ../labels)")
ap.add_argument("--test-yolo", help="separate YOLO images dir for evaluation")
ap.add_argument("--car-classes", type=int, nargs="+", default=[0])
ap.add_argument("--empty-classes", type=int, nargs="*", default=[])
ap.add_argument("--no-random-negatives", action="store_true",
                help="dataset only labels cars inside spaces (e.g. PKLot): sample negatives only from empty spaces")
ap.add_argument("--pklot", help="PKLot root with .jpg + .xml")
ap.add_argument("--max-frames", type=int, default=None, help="cap on training frames used for patches")
ap.add_argument("--neg-per-frame", type=int, default=30)
ap.add_argument("--windows", type=int, nargs="+", help="window sizes w1 h1 w2 h2 ... (default: from annotations)")
ap.add_argument("--n-windows", type=int, default=5)
ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
ap.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
ap.add_argument("--cv", type=int, default=5)
ap.add_argument("--tune-top", type=int, default=0)
ap.add_argument("--ensemble-top", type=int, default=3)
ap.add_argument("--hard-neg-frames", type=int, default=30, help="frames used for hard-negative mining (0 = off)")
ap.add_argument("--ap-top", type=int, default=3, help="also report detection AP of the top-k candidates")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-jobs", type=int, default=-1)
ap.add_argument("--out", default="outputs/detector")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)

# ---------------------------------------------------------------- data
frames, test = [], None
for sc in a.scenes:
    frames += load_scene_json(Path(sc) / "annotations.json")
if a.yolo:
    frames += load_yolo(a.yolo, car_classes=a.car_classes, empty_classes=a.empty_classes,
                        random_negatives=not a.no_random_negatives)
if a.test_yolo:
    test = load_yolo(a.test_yolo, car_classes=a.car_classes, empty_classes=a.empty_classes,
                     random_negatives=not a.no_random_negatives)
if a.pklot:
    frames += load_pklot_xml(a.pklot)
if not frames:
    sys.exit("no annotated frames given")
if test is None:  # hold out whole groups (scene-days / parking lots)
    groups = np.array([f.group for f in frames])
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=a.seed).split(frames, groups=groups))
    frames, test = [frames[i] for i in tr], [frames[i] for i in te]
train = frames
if a.max_frames and len(train) > a.max_frames:
    train = [train[i] for i in sorted(rng.choice(len(train), a.max_frames, replace=False))]
print(f"frames: train={len(train)} test={len(test)}  cars: train={sum(len(f.cars) for f in train)} "
      f"test={sum(len(f.cars) for f in test)}")

windows = ([tuple(a.windows[i:i + 2]) for i in range(0, len(a.windows), 2)] if a.windows
           else suggest_window_sizes(train, a.n_windows, a.seed))
print("window sizes (w,h):", windows)

# ---------------------------------------------------------------- patch classifier selection
t = time.time()
if len({f.group for f in train}) < a.cv + 2:
    # Too few scenes/days for grouped CV: group by frame instead, so the shifted and
    # mirrored copies of one car never sit on both sides of a split.
    print("few groups -> grouping CV by frame")
    for f in train:
        f.group = f"{f.group}/{Path(f.path).name}"
P, y, g = sample_patches(train, windows, a.neg_per_frame, seed=a.seed)
cfg = FeatureConfig()
ext = FeatureExtractor(cfg)
X = ext.transform(P, n_jobs=a.n_jobs)
print(f"{len(P)} patches (car={int(y.sum())}, background={int((y == 0).sum())}), features {X.shape} "
      f"in {time.time() - t:.0f}s")
res = run_selection(X, y, ext, g, feature_sets=a.feature_sets, models=a.models, n_splits=a.cv,
                    tune_top=a.tune_top, ensemble_top=a.ensemble_top, seed=a.seed, n_jobs=a.n_jobs)
res.leaderboard.to_csv(out / "patch_leaderboard.csv", index=False)
print(f"stage 2 = {res.best_name}")

# ---------------------------------------------------------------- cascade + hard negatives
stage1 = Stage1HOG(cfg).fit(P, y)
det = CarDetector(cfg, DetectorConfig(windows, threshold=a.threshold), stage1,
                  clone(res.candidates[res.best_name]).fit(X, y), res.best_name)
if a.hard_neg_frames:
    pool = [train[i] for i in rng.choice(len(train), min(a.hard_neg_frames, len(train)), replace=False)]
    hn = mine_hard_negatives(det, pool)
    if hn:
        X = np.vstack([X, ext.transform(hn, n_jobs=a.n_jobs)])
        y = np.r_[y, np.zeros(len(hn), int)]
        P = P + hn
        det.stage1 = Stage1HOG(cfg).fit(P, y)
        det.stage2 = clone(res.candidates[res.best_name]).fit(X, y)
    print(f"hard negatives mined: {len(hn)}")

# ---------------------------------------------------------------- evaluation
t = time.time()
det.metrics = average_precision(det, test)
sec_per_frame = (time.time() - t) / max(len(test), 1)
print(f"held-out detection: AP@0.5={det.metrics['ap']:.4f}  precision={det.metrics['precision']:.4f}  "
      f"recall={det.metrics['recall']:.4f}  ({sec_per_frame:.2f} s/frame)")
det.save(out / "car_detector.joblib")

ap_rows = []
names = [n for n in res.leaderboard.candidate if not n.startswith("ensemble")][: a.ap_top]
for name in names:  # for information only: selection was done on patch CV
    d2 = CarDetector(cfg, det.config, det.stage1, clone(res.candidates[name]).fit(X, y), name)
    m = average_precision(d2, test)
    ap_rows.append({"candidate": name, **m})
    print(f"  {name:<28} AP={m['ap']:.4f} P={m['precision']:.3f} R={m['recall']:.3f}")
(out / "summary.json").write_text(json.dumps({
    "stage2": res.best_name, "patch_test": res.test_metrics, "detection": det.metrics,
    "sec_per_frame": sec_per_frame, "windows": windows, "top_candidates_ap": ap_rows,
    "n_patches": int(len(y)), "args": vars(a)}, indent=2, default=str))

# ---------------------------------------------------------------- figure
try:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = res.leaderboard[~res.leaderboard.candidate.str.contains(r"tuned|ensemble")]
    piv = base.pivot(index="features", columns="model", values="cv_f1").reindex(a.feature_sets)
    show = [test[i] for i in np.linspace(0, len(test) - 1, 2).astype(int)]
    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot2grid((2, 3), (0, 0), rowspan=2)
    ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="w" if v < np.nanmean(piv.values) else "k")
    ax.set_title("Car vs background: mean CV F1")
    for k, f in enumerate(show):
        img = f.image()
        boxes, p = det.detect(img)
        vis = img.copy()
        for b in f.cars.astype(int):
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (255, 255, 0), 1)
        for b, s in zip(boxes.astype(int), p):
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (255, 0, 0), 2)
            cv2.putText(vis, f"{s:.2f}", (b[0], b[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        ax = plt.subplot2grid((2, 3), (k, 1), colspan=2)
        ax.imshow(vis)
        ax.set_axis_off()
        ax.set_title(f"held-out frame {f.timestamp or ''}: red = detections, yellow = ground truth", fontsize=9)
    fig.suptitle(f"stage 2: {res.best_name}   |   held-out AP@0.5 = {det.metrics['ap']:.3f}")
    fig.tight_layout()
    fig.savefig(out / "report.png", dpi=110)
except Exception as e:
    print("plot skipped:", e)
print("saved", out)
