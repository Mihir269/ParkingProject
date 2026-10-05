"""Dataset loading for spot-level (patch) occupancy classification.

Two layouts are supported:

1. Folder layout (works with PKLot "PKLotSegmented" and our own crops)::

       root/.../<anything>/{empty|occupied}/*.jpg

   The label comes from the nearest ancestor directory named like
   ``empty/free/vacant/0`` or ``occupied/busy/full/1``.

2. Label-list layout (CNRPark / CNRPark-EXT ``LABELS/*.txt``)::

       SUNNY/2015-11-12/camera1/S_2015-11-12_07.09_C01_184.jpg 0

Labels: 0 = empty, 1 = occupied.

``groups`` (camera / parking lot / day) are returned so that cross-validation
can keep all patches of one group on the same side of a split. Random splits of
patches from the same camera leak near-duplicate images and give over-optimistic
scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
EMPTY_NAMES = {"empty", "free", "vacant", "0"}
OCCUPIED_NAMES = {"occupied", "busy", "full", "1"}
CLASS_NAMES = ("empty", "occupied")


@dataclass
class PatchDataset:
    paths: list[str]
    labels: np.ndarray
    groups: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.paths)

    def subsample(self, max_per_class: int | None, seed: int = 0) -> "PatchDataset":
        if not max_per_class:
            return self
        rng = np.random.default_rng(seed)
        keep = []
        for c in np.unique(self.labels):
            idx = np.flatnonzero(self.labels == c)
            if len(idx) > max_per_class:
                idx = rng.choice(idx, max_per_class, replace=False)
            keep.extend(idx.tolist())
        keep = np.sort(keep)
        return PatchDataset(
            [self.paths[i] for i in keep],
            self.labels[keep],
            None if self.groups is None else self.groups[keep],
        )


def read_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _label_from_parts(parts: tuple[str, ...]) -> int | None:
    for part in reversed(parts):
        p = part.lower()
        if p in EMPTY_NAMES:
            return 0
        if p in OCCUPIED_NAMES:
            return 1
    return None


def load_folder(root: str | Path, group_level: int | None = None) -> PatchDataset:
    """Recursively load ``root``; ``group_level`` = index of the path component
    (relative to root) to use as group, e.g. 0 -> parking lot / camera folder."""
    root = Path(root)
    paths, labels, groups = [], [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(root).parts
        y = _label_from_parts(rel[:-1])
        if y is None:
            continue
        paths.append(str(p))
        labels.append(y)
        if group_level is not None:
            groups.append(rel[group_level] if group_level < len(rel) - 1 else "root")
    if not paths:
        raise ValueError(f"No labelled images found under {root}")
    return PatchDataset(paths, np.array(labels), np.array(groups) if group_level is not None else None)


def load_label_list(list_file: str | Path, image_root: str | Path, group_level: int | None = None) -> PatchDataset:
    """CNRPark-style ``<relative_path> <label>`` file."""
    image_root = Path(image_root)
    paths, labels, groups = [], [], []
    for line in Path(list_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rel, lab = line.rsplit(maxsplit=1)
        paths.append(str(image_root / rel))
        labels.append(int(lab))
        if group_level is not None:
            groups.append(Path(rel).parts[group_level])
    return PatchDataset(paths, np.array(labels), np.array(groups) if group_level is not None else None)
