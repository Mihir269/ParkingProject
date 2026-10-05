"""Frame-level occupancy detection: crop every spot, classify, smooth over time."""
from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np

from .models import OccupancyClassifier
from .spots import Spot, crop_all, scale_spots


def classify_frame(frame_rgb: np.ndarray, spots: list[Spot], clf: OccupancyClassifier,
                   annotated_size=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (occupied 0/1 array, P(occupied) array), one entry per spot."""
    h, w = frame_rgb.shape[:2]
    spots = scale_spots(spots, annotated_size, (w, h))
    patches = crop_all(frame_rgb, spots, clf.feature_config.patch_size)
    p = clf.predict_proba(patches)
    return (p >= clf.threshold).astype(int), p


class TemporalSmoother:
    """Majority vote over the last ``window`` frames per spot, so a person walking
    past or a flicker doesn't toggle a spot's state."""

    def __init__(self, n_spots: int, window: int = 5):
        self.buf = [deque(maxlen=window) for _ in range(n_spots)]

    def update(self, occupied) -> np.ndarray:
        for b, o in zip(self.buf, occupied):
            b.append(int(o))
        return np.array([int(sum(b) * 2 > len(b)) for b in self.buf])


def log_rows(ts: datetime, spots: list[Spot], occupied, scores) -> list[dict]:
    return [{"timestamp": ts, "spot_id": s.id, "occupied": int(o), "score": float(p)}
            for s, o, p in zip(spots, occupied, scores)]
