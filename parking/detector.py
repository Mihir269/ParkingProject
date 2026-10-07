"""Sliding-window car detector built from the hand-crafted features + selected model.

Two-stage cascade (classic Dalal-Triggs style, made fast enough for Python):

Stage 1  - linear classifier on HOG only, evaluated densely. For each window size
           (w, h) the frame is resized so that window becomes ``patch_size``; HOG is
           computed ONCE for that resized frame and every window position is scored
           by sliding the weight tensor over the block grid (no per-window feature
           extraction). Threshold set for ~99% recall on training cars: it only
           has to throw away the obvious background.
Stage 2  - the best (features x model) combination from ``selection.run_selection``
           (HOG / LBP / GLCM / colour; LogReg / RF / SVM / XGBoost) re-scores the
           surviving windows.
Then non-maximum suppression with box voting (kept box = score-weighted mean of
the windows overlapping it), which keeps the box steady from frame to frame.

Window sizes come from the training annotations (``suggest_window_sizes``), so the
detector adapts to whatever car sizes the camera sees.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .annotations import FrameAnnotation, iou, nms_vote
from .features import FeatureConfig, FeatureExtractor, hog_features
from .models import positive_score


@dataclass
class DetectorConfig:
    window_sizes: list[tuple[int, int]]
    step_cells: int = 1  # stride in HOG cells (1 cell = 1/8 of the window at 64px)
    stage1_recall: float = 0.99
    stage1_max_windows: int = 400  # per frame, highest stage-1 scores
    threshold: float = 0.5  # stage-2 P(car)
    nms_iou: float = 0.3
    extra: dict = field(default_factory=dict)


class Stage1HOG:
    """Linear HOG scorer whose weights can be slid over a full-frame HOG block grid."""

    def __init__(self, cfg: FeatureConfig, C: float = 0.1):
        self.cfg, self.C = cfg, C

    def _gray(self, patch):
        rgb = cv2.resize(patch, self.cfg.patch_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def fit(self, patches, y, recall=0.99):
        X = np.vstack([hog_features(self._gray(p), self.cfg) for p in patches])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
        # fold the scaler into the weights: w.(x-mu)/sd + b = (w/sd).x + (b - w.mu/sd)
        w = clf.coef_.ravel() / sc.scale_
        self.b = float(clf.intercept_[0] - w @ sc.mean_)
        ppc, cpb = self.cfg.hog_pixels_per_cell, self.cfg.hog_cells_per_block
        pw, ph = self.cfg.patch_size
        self.nbx, self.nby = pw // ppc - cpb + 1, ph // ppc - cpb + 1
        self.W = w.reshape(self.nby, self.nbx, -1)
        s = X @ w + self.b
        self.threshold = float(np.quantile(s[y == 1], 1 - recall))
        return self

    def score_map(self, frame_rgb, win):
        """Scores of every window of size ``win`` (w,h). Returns (boxes, scores)."""
        cfg, ppc = self.cfg, self.cfg.hog_pixels_per_cell
        pw, ph = cfg.patch_size
        sx, sy = pw / win[0], ph / win[1]
        H, W = frame_rgb.shape[:2]
        rw, rh = int(round(W * sx)), int(round(H * sy))
        if rw < pw or rh < ph:
            return np.zeros((0, 4)), np.zeros(0)
        gray = cv2.cvtColor(cv2.resize(frame_rgb, (rw, rh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        from skimage.feature import hog

        B = hog(gray, orientations=cfg.hog_orientations, pixels_per_cell=(ppc, ppc),
                cells_per_block=(cfg.hog_cells_per_block,) * 2, block_norm="L2-Hys", feature_vector=False)
        B = B.reshape(B.shape[0], B.shape[1], -1)  # (blocks_y, blocks_x, 36)
        ny, nx = B.shape[0] - self.nby + 1, B.shape[1] - self.nbx + 1
        if ny <= 0 or nx <= 0:
            return np.zeros((0, 4)), np.zeros(0)
        S = np.full((ny, nx), self.b)
        for i in range(self.nby):
            for j in range(self.nbx):
                S += B[i:i + ny, j:j + nx] @ self.W[i, j]
        yy, xx = np.mgrid[0:ny, 0:nx]
        x1, y1 = xx.ravel() * ppc / sx, yy.ravel() * ppc / sy
        boxes = np.column_stack([x1, y1, x1 + win[0], y1 + win[1]])
        return boxes, S.ravel()


class CarDetector:
    def __init__(self, feature_config: FeatureConfig, config: DetectorConfig, stage1: Stage1HOG,
                 stage2, stage2_name: str = "", metrics: dict | None = None):
        self.feature_config, self.config = feature_config, config
        self.stage1, self.stage2, self.stage2_name = stage1, stage2, stage2_name
        self.metrics = metrics or {}

    def proposals(self, frame):
        boxes, scores = [], []
        for win in self.config.window_sizes:
            b, s = self.stage1.score_map(frame, win)
            if self.config.step_cells > 1:  # subsample positions
                keep = np.arange(len(s)) % self.config.step_cells == 0
                b, s = b[keep], s[keep]
            m = s >= self.stage1.threshold
            boxes.append(b[m])
            scores.append(s[m])
        boxes, scores = np.vstack(boxes), np.concatenate(scores)
        order = np.argsort(-scores)[: self.config.stage1_max_windows]
        return boxes[order], scores[order]

    def detect(self, frame_rgb: np.ndarray, return_all: bool = False):
        boxes, _ = self.proposals(frame_rgb)
        if not len(boxes):
            return np.zeros((0, 4)), np.zeros(0)
        H, W = frame_rgb.shape[:2]
        patches = []
        for x1, y1, x2, y2 in boxes.astype(int):
            patches.append(frame_rgb[max(0, y1):min(H, y2), max(0, x1):min(W, x2)])
        X = FeatureExtractor(self.feature_config).transform(patches, n_jobs=1)
        p = positive_score(self.stage2, X)
        if return_all:
            return boxes, p
        m = p >= self.config.threshold
        return nms_vote(boxes[m], p[m], self.config.nms_iou)

    def save(self, path):
        import joblib

        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "CarDetector":
        import joblib

        return joblib.load(path)


def mine_hard_negatives(det: CarDetector, frames: list[FrameAnnotation], max_per_frame: int = 20):
    """Confident detections that are not cars -> extra negative patches.

    Only boxes known to be car-free count: anywhere if ``random_negatives``,
    otherwise only inside annotated empty spaces."""
    out = []
    for f in frames:
        img = f.image()
        boxes, p = det.detect(img, return_all=True)
        m = p >= det.config.threshold * 0.6
        boxes, p = boxes[m], p[m]
        known = np.vstack([f.cars, f.ignore])
        if len(known) and len(boxes):
            far = iou(boxes, known).max(1) < 0.3
            boxes, p = boxes[far], p[far]
        if not f.random_negatives:
            if not len(f.negatives):
                continue
            ok = iou(boxes, f.negatives).max(1) > 0.5 if len(boxes) else np.zeros(0, bool)
            boxes, p = boxes[ok], p[ok]
        for b in boxes[np.argsort(-p)][:max_per_frame].astype(int):
            out.append(img[max(0, b[1]):b[3], max(0, b[0]):b[2]])
    return out


def average_precision(det: CarDetector, frames: list[FrameAnnotation], iou_thr: float = 0.5) -> dict:
    """VOC-style AP@IoU plus precision/recall at the detector's threshold.

    For datasets without ``random_negatives`` (space labels only), detections outside the
    annotated spaces are ignored instead of counted as false positives."""
    recs, n_gt = [], 0
    tp_thr = fp_thr = 0
    for f in frames:
        boxes, p = det.detect(f.image())
        if not f.random_negatives and len(boxes):
            known = np.vstack([f.cars, f.negatives]) if len(f.negatives) else f.cars
            if len(known):
                inside = iou(boxes, known).max(1) > 0.3
                boxes, p = boxes[inside], p[inside]
            else:
                boxes, p = boxes[:0], p[:0]
        n_gt += len(f.cars)
        used = np.zeros(len(f.cars), bool)
        for i in np.argsort(-p):
            hit = False
            if len(f.cars):
                o = iou(boxes[i], f.cars)[0]
                o[used] = 0
                j = int(o.argmax())
                if o[j] >= iou_thr:
                    used[j] = hit = True
            if not hit and len(f.ignore) and iou(boxes[i], f.ignore).max() >= iou_thr:
                continue  # "difficult" car: neither a hit nor a false positive (VOC protocol)
            recs.append((p[i], hit))
            tp_thr += hit
            fp_thr += not hit
    if not recs:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "n_gt": n_gt}
    recs.sort(key=lambda r: -r[0])
    hits = np.array([r[1] for r in recs], float)
    tp, fp = np.cumsum(hits), np.cumsum(1 - hits)
    rec, prec = tp / max(n_gt, 1), tp / (tp + fp)
    # all-point interpolated AP
    mrec, mpre = np.r_[0, rec, 1], np.r_[1, prec, 0]
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"ap": ap, "precision": tp_thr / max(tp_thr + fp_thr, 1), "recall": tp_thr / max(n_gt, 1), "n_gt": n_gt}

