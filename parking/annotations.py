"""Frame-level car annotations and training-patch sampling for the car detector.

Supported sources:

* our scene format: ``annotations.json`` written by ``scenes.save_scene``
  (or by hand: ``{"frames": [{"image": ..., "timestamp": ..., "cars": [[x1,y1,x2,y2], ...]}]}``)
* YOLO format (e.g. the Roboflow export of PKLot, or anything labelled in
  Roboflow / CVAT / Label Studio): ``images/*.jpg`` + ``labels/*.txt`` with
  ``class cx cy w h`` normalised. Choose which class ids mean "car".
* PKLot original XML: every parking space with ``occupied="0|1"`` and its contour.

PKLot and other parking-space datasets only label cars *inside spaces*. Cars
driving or parked in the lane are unlabelled, so random background crops could
contain unlabelled cars. For such sources ``random_negatives`` is False and only
empty spaces are used as negatives.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import IMG_EXTS, read_rgb


@dataclass
class FrameAnnotation:
    path: str
    cars: np.ndarray  # (N, 4) x1 y1 x2 y2
    negatives: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # known car-free boxes
    random_negatives: bool = True  # is everything outside `cars` guaranteed car-free?
    timestamp: datetime | None = None
    group: str = "0"  # camera / sequence / day, for leakage-free splits
    ignore: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # "difficult" cars: neither pos nor neg

    def image(self) -> np.ndarray:
        return read_rgb(self.path)


# ---------------------------------------------------------------- box utils
def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) boxes -> (N, M)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.3) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        o = iou(boxes[i], boxes[order[1:]])[0]
        order = order[1:][o < thr]
    return np.array(keep, int)


def nms_vote(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.3, vote_iou: float = 0.5):
    """NMS, then replace each kept box by the score-weighted mean of the boxes
    overlapping it (box voting). Sliding windows on a coarse grid otherwise make
    the box jump between neighbouring positions from frame to frame."""
    keep = nms(boxes, scores, thr)
    if not len(keep):
        return boxes[:0], scores[:0]
    o = iou(boxes[keep], boxes)
    out = []
    for k, row in zip(keep, o):
        m = row >= vote_iou
        w = scores[m]
        out.append((boxes[m] * w[:, None]).sum(0) / w.sum())
    return np.array(out), scores[keep]


# ---------------------------------------------------------------- loaders
def load_scene_json(path: str | Path) -> list[FrameAnnotation]:
    path = Path(path)
    data = json.loads(path.read_text())
    out = []
    for f in data["frames"]:
        ts = datetime.fromisoformat(f["timestamp"]) if f.get("timestamp") else None
        out.append(FrameAnnotation(
            str(path.parent / f["image"]), np.asarray(f["cars"], float).reshape(-1, 4),
            timestamp=ts, group=f.get("group", f"{path.parent.name}:{ts.date()}" if ts else path.parent.name)))
    return out


def load_yolo(images_dir, labels_dir=None, car_classes=(0,), empty_classes=(),
              random_negatives: bool = True, group_from_name=None) -> list[FrameAnnotation]:
    """``car_classes``: class ids that are cars (e.g. space-occupied).
    ``empty_classes``: class ids that are guaranteed car-free (e.g. space-empty)."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir else images_dir.parent / "labels"
    out = []
    for img in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        lab = labels_dir / (img.stem + ".txt")
        rows = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
        import cv2

        h, w = cv2.imread(str(img), cv2.IMREAD_REDUCED_GRAYSCALE_2).shape[:2]
        h, w = h * 2, w * 2

        def to_xyxy(r):
            cx, cy, bw, bh = r[:, 1] * w, r[:, 2] * h, r[:, 3] * w, r[:, 4] * h
            return np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])

        cars = to_xyxy(rows[np.isin(rows[:, 0], car_classes)])
        neg = to_xyxy(rows[np.isin(rows[:, 0], empty_classes)])
        group = group_from_name(img.name) if group_from_name else "0"
        out.append(FrameAnnotation(str(img), cars, neg, random_negatives, group=group))
    return out


def load_pklot_xml(root) -> list[FrameAnnotation]:
    """PKLot/<lot>/<weather>/<date>/<image>.jpg + .xml. Group = parking lot."""
    root = Path(root)
    out = []
    for xml in sorted(root.rglob("*.xml")):
        img = xml.with_suffix(".jpg")
        if not img.exists():
            continue
        cars, empty = [], []
        for sp in ET.parse(xml).getroot().iter("space"):
            pts = np.array([[float(p.get("x")), float(p.get("y"))] for p in sp.iter("point")])
            if len(pts) < 3 or sp.get("occupied") is None:
                continue
            box = [*pts.min(0), *pts.max(0)]
            (cars if sp.get("occupied") == "1" else empty).append(box)
        out.append(FrameAnnotation(str(img), np.array(cars).reshape(-1, 4), np.array(empty).reshape(-1, 4),
                                   random_negatives=False, group=xml.relative_to(root).parts[0]))
    return out


# ---------------------------------------------------------------- patch sampling
def suggest_window_sizes(frames: list[FrameAnnotation], k: int = 5, seed: int = 0) -> list[tuple[int, int]]:
    """Typical car box sizes in this camera -> sliding-window sizes (k-means on log w, log h)."""
    from sklearn.cluster import KMeans

    wh = np.vstack([np.column_stack([f.cars[:, 2] - f.cars[:, 0], f.cars[:, 3] - f.cars[:, 1]])
                    for f in frames if len(f.cars)])
    k = min(k, len(np.unique(wh.round(), axis=0)))
    km = KMeans(k, n_init=10, random_state=seed).fit(np.log(wh))
    sizes = sorted({(int(round(w)), int(round(h))) for w, h in np.exp(km.cluster_centers_)})
    return sizes


def _crop(img, box):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img[y1:y2, x1:x2]


def _jitter(box, rng, amount):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = rng.uniform(-amount, amount, 2) * (w, h)
    s = 1 + rng.uniform(-amount, amount)
    cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
    return np.array([cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2])


def sample_patches(frames: list[FrameAnnotation], window_sizes, neg_per_frame: int = 30,
                   pos_jitter: int = 2, seed: int = 0, context_negatives: int = 0):
    """Car / not-car training patches.

    positives : every car box + ``pos_jitter`` slightly shifted/scaled copies + mirror
    negatives : known empty boxes, ``context_negatives`` shifted/oversized windows per
                car, random windows with IoU < 0.3 to every car
                (only if ``random_negatives``), and *partial* cars (IoU 0.1-0.3), which
                teach the classifier to fire only on a well-centred car
    Returns (patches, labels, groups).
    """
    rng = np.random.default_rng(seed)
    P, y, g = [], [], []

    def add(img, box, label, grp, flip=False):
        c = _crop(img, box)
        if c is not None:
            P.append(c[:, ::-1].copy() if flip else c)
            y.append(label)
            g.append(grp)

    for f in frames:
        img = f.image()
        H, W = img.shape[:2]
        for box in f.cars:
            add(img, box, 1, f.group)
            add(img, box, 1, f.group, flip=True)
            for _ in range(pos_jitter):
                add(img, _jitter(box, rng, 0.06), 1, f.group)
        for box in f.negatives:
            add(img, box, 0, f.group)
        # "Badly placed" windows around labelled cars: shifted half off the car, or
        # scaled up over several cars. Safe negatives even when other cars in the image
        # are unlabelled, and they teach the classifier to fire only on a centred car.
        for box in f.cars:
            for _ in range(context_negatives):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if rng.random() < 0.6:
                    ang = rng.uniform(0, 2 * np.pi)
                    d = rng.uniform(0.5, 0.8)
                    cand = box + np.array([np.cos(ang) * w, np.sin(ang) * h] * 2) * d
                else:
                    s = rng.uniform(1.8, 2.6)
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    cand = np.array([cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2])
                if cand[0] < 0 or cand[1] < 0 or cand[2] > W or cand[3] > H:
                    continue
                if iou(cand, f.cars).max() < 0.3:
                    add(img, cand, 0, f.group)
        n_rand = neg_per_frame if f.random_negatives else 0
        tries = 0
        while n_rand > 0 and tries < 50 * neg_per_frame:
            tries += 1
            w, h = window_sizes[rng.integers(len(window_sizes))]
            s = rng.uniform(0.8, 1.25)
            w, h = w * s, h * s
            if w >= W or h >= H:
                continue
            x, yy = rng.uniform(0, W - w), rng.uniform(0, H - h)
            box = np.array([x, yy, x + w, yy + h])
            if len(f.ignore) and iou(box, f.ignore).max() >= 0.1:
                continue
            o = iou(box, f.cars).max() if len(f.cars) else 0.0
            partial = 0.1 < o < 0.3 and rng.random() < 0.5
            if o < 0.1 or partial:
                add(img, box, 0, f.group)
                n_rand -= 1
    return P, np.array(y), np.array(g)


# ---------------------------------------------------------------- CNRPark-EXT
CNR_WEATHER = {"S": "SUNNY", "O": "OVERCAST", "R": "RAINY"}


def cnrpark_ext_frames(root, cameras=None, full_size=(2592, 1944), img_size=(1000, 750)) -> dict:
    """Per-camera frame annotations for CNR-EXT (data/cnrpark after download).

    Returns {camera: {"bays": {slot_id: box}, "frames": [FrameAnnotation, ...]}}.
    Labels are per parking space (square box around the space, from cameraN.csv),
    so "cars" are boxes of occupied spaces and "negatives" boxes of free spaces.
    Cars outside the monitored spaces are unlabelled -> random_negatives=False.
    """
    import pandas as pd

    root = Path(root)
    meta = pd.read_csv(root / "CNRPark+EXT.csv", dtype={"camera": str})
    meta = meta[~meta.camera.isin(["A", "B"])].copy()
    meta["cam"] = meta.camera.astype(int)
    sx, sy = img_size[0] / full_size[0], img_size[1] / full_size[1]
    out = {}
    for cam in sorted(meta.cam.unique()):
        if cameras and cam not in cameras:
            continue
        boxes = pd.read_csv(root / f"camera{cam}.csv")
        bays = {int(r.SlotId): (r.X * sx, r.Y * sy, (r.X + r.W) * sx, (r.Y + r.H) * sy) for r in boxes.itertuples()}
        frames = []
        for dt, g in meta[meta.cam == cam].groupby("datetime"):
            ts = datetime.strptime(dt, "%Y-%m-%d_%H.%M")
            weather = CNR_WEATHER[g.weather.iloc[0]]
            img = root / "FULL_IMAGE_1000x750" / weather / f"{ts:%Y-%m-%d}" / f"camera{cam}" / f"{ts:%Y-%m-%d_%H%M}.jpg"
            if not img.exists():
                continue
            g = g[g.slot_id.isin(bays)]
            cars = np.array([bays[s] for s in g.slot_id[g.occupancy == 1]]).reshape(-1, 4)
            free = np.array([bays[s] for s in g.slot_id[g.occupancy == 0]]).reshape(-1, 4)
            frames.append(FrameAnnotation(str(img), cars, free, random_negatives=False, timestamp=ts,
                                          group=f"camera{cam}:{ts.date()}"))  # camera-day
        out[cam] = {"bays": bays, "frames": sorted(frames, key=lambda f: f.timestamp)}
    return out


# ---------------------------------------------------------------- COCO (NDISPark and others)
def load_coco(json_path, images_dir=None, category: str = "car", group_fn=None) -> list[FrameAnnotation]:
    """COCO detection annotations -> frames. Every car is boxed in datasets like
    NDISPark, so random background windows are safe negatives.

    ``group_fn(file_name)`` -> camera id for leakage-free splits; NDISPark file names
    are ``<camera>_<unix time>.jpg``, which is the default."""
    json_path = Path(json_path)
    images_dir = Path(images_dir) if images_dir else json_path.parent / "imgs"
    data = json.loads(json_path.read_text())
    cat_ids = {c["id"] for c in data.get("categories", []) if c["name"] == category} or None
    boxes: dict[int, list] = {}
    for a in data["annotations"]:
        if cat_ids is None or a["category_id"] in cat_ids:
            x, y, w, h = a["bbox"]
            boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    group_fn = group_fn or (lambda n: Path(n).stem.split("_")[0])
    out = []
    for im in data["images"]:
        name = im["file_name"]
        ts = None
        stem = Path(name).stem.split("_")
        if len(stem) > 1 and stem[-1].isdigit() and len(stem[-1]) == 10:  # unix time
            ts = datetime.fromtimestamp(int(stem[-1]))
        out.append(FrameAnnotation(str(images_dir / name), np.array(boxes.get(im["id"], []), float).reshape(-1, 4),
                                   random_negatives=True, timestamp=ts, group=group_fn(name)))
    return out


# ---------------------------------------------------------------- ACPDS (action camera, ~10 m high)
def load_acpds(root, split: str = "all") -> list[dict]:
    """Action-Camera Parking Dataset (Marek 2021): ``images/`` + ``annotations.json``
    with per-image ``rois_list`` (4 corners per space, normalised to [0,1]) and
    ``occupancy_list``, split into train / valid / test by parking lot.

    Returns [{"path", "split", "quads" (N,4,2 pixels), "occupied" (N,)}]."""
    import cv2

    root = Path(root)
    if not (root / "annotations.json").exists():  # archive may add a top-level folder
        found = sorted(root.rglob("annotations.json"))
        if not found:
            raise FileNotFoundError(f"no annotations.json under {root}")
        root = found[0].parent
    ann = json.loads((root / "annotations.json").read_text())
    splits = ["train", "valid", "test"] if split == "all" else [split]
    out = []
    for sp in splits:
        a = ann[sp]
        for name, rois, occ in zip(a["file_names"], a["rois_list"], a["occupancy_list"]):
            path = root / "images" / name
            img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
            h, w = (img.shape[0] * 4, img.shape[1] * 4) if img is not None else (1, 1)
            q = np.asarray(rois, float).reshape(-1, 4, 2) * [w - 1, h - 1]
            out.append({"path": str(path), "split": sp, "quads": q, "occupied": np.asarray(occ, int)})
    return out


def acpds_frames(root, split: str = "all") -> list[FrameAnnotation]:
    """ACPDS as detector annotations: occupied spaces = cars, free spaces = negatives.
    Cars outside annotated spaces are unlabelled -> random_negatives=False.
    Each image is its own viewpoint, so group = image."""
    out = []
    for r in load_acpds(root, split):
        boxes = np.column_stack([r["quads"].min(1), r["quads"].max(1)]) if len(r["quads"]) else np.zeros((0, 4))
        occ = r["occupied"].astype(bool)
        out.append(FrameAnnotation(r["path"], boxes[occ], boxes[~occ], random_negatives=False,
                                   group=f"{r['split']}:{Path(r['path']).stem}"))
    return out


# ---------------------------------------------------------------- PASCAL VOC
def load_voc(root, split: str = "trainval", cls: str = "car", n_background: int | None = None,
             seed: int = 0) -> list[FrameAnnotation]:
    """PASCAL VOC (e.g. VOC2007 via the Ultralytics GitHub mirror): every car in every
    image is boxed, so random background windows are safe negatives. Cars marked
    ``difficult`` (tiny / heavily occluded) are ``ignore`` boxes, as in the VOC protocol.

    Returns all images of ``split`` containing a ``cls`` object, plus ``n_background``
    randomly chosen images without one (None = all of them)."""
    root = Path(root)
    if not (root / "Annotations").exists():
        found = sorted(root.rglob("VOC2007")) or sorted(p.parent for p in root.rglob("Annotations"))
        root = found[0]
    ids = (root / "ImageSets" / "Main" / f"{split}.txt").read_text().split()
    with_cls, without = [], []
    for i in ids:
        r = ET.parse(root / "Annotations" / f"{i}.xml").getroot()
        cars, hard = [], []
        for o in r.iter("object"):
            if o.findtext("name") != cls:
                continue
            b = o.find("bndbox")
            box = [float(b.findtext(k)) - (1 if k.startswith(("xmin", "ymin")) else 0)
                   for k in ("xmin", "ymin", "xmax", "ymax")]
            (hard if o.findtext("difficult") == "1" else cars).append(box)
        f = FrameAnnotation(str(root / "JPEGImages" / f"{i}.jpg"), np.array(cars).reshape(-1, 4),
                            random_negatives=True, group=i, ignore=np.array(hard).reshape(-1, 4))
        (with_cls if len(cars) or len(hard) else without).append(f)
    if n_background is not None and len(without) > n_background:
        rng = np.random.default_rng(seed)
        without = [without[i] for i in sorted(rng.choice(len(without), n_background, replace=False))]
    return with_cls + without
