"""Parking-spot geometry: definition, storage, cropping and visualisation.

Camera assumption (per discussion with Prof. Kulkarni): a fixed camera mounted
~10 ft high, looking down the lane rather than from the side. Spots then appear
as quadrilaterals (perspective-distorted rectangles). Each spot is stored as
4 image points (clockwise from the top-left as seen in the image) and is warped
to an upright rectangle before feature extraction, so near and far spots look
alike to the classifier.

spots.json::

    {"image_size": [W, H],
     "spots": [{"id": "A1", "owner": "flat-101", "polygon": [[x,y],[x,y],[x,y],[x,y]]}, ...]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Spot:
    id: str
    polygon: np.ndarray  # (4, 2) float32
    owner: str | None = None  # resident / flat the spot is allotted to
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"id": self.id, "polygon": self.polygon.round(1).tolist()}
        if self.owner:
            d["owner"] = self.owner
        if self.meta:
            d["meta"] = self.meta
        return d


def load_spots(path: str | Path) -> tuple[list[Spot], tuple[int, int] | None]:
    data = json.loads(Path(path).read_text())
    spots = [
        Spot(str(s["id"]), np.asarray(s["polygon"], np.float32), s.get("owner"), s.get("meta", {}))
        for s in data["spots"]
    ]
    size = tuple(data["image_size"]) if data.get("image_size") else None
    return spots, size


def save_spots(path: str | Path, spots: list[Spot], image_size: tuple[int, int] | None = None) -> None:
    data = {"image_size": list(image_size) if image_size else None, "spots": [s.to_dict() for s in spots]}
    Path(path).write_text(json.dumps(data, indent=2))


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def scale_spots(spots: list[Spot], from_size, to_size) -> list[Spot]:
    """Rescale polygons if the frame resolution differs from the annotated one."""
    if not from_size or tuple(from_size) == tuple(to_size):
        return spots
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    return [Spot(s.id, s.polygon * np.array([sx, sy], np.float32), s.owner, s.meta) for s in spots]


def crop_spot(frame: np.ndarray, spot: Spot, out_size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Perspective-warp the spot quadrilateral into an upright ``out_size`` patch."""
    w, h = out_size
    src = order_quad(spot.polygon)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_all(frame: np.ndarray, spots: list[Spot], out_size=(64, 64)) -> list[np.ndarray]:
    return [crop_spot(frame, s, out_size) for s in spots]


def draw_spots(
    frame: np.ndarray,
    spots: list[Spot],
    occupied: list[bool] | None = None,
    scores: list[float] | None = None,
) -> np.ndarray:
    """Overlay spots on an RGB frame: green = free, red = occupied, yellow = unknown."""
    out = frame.copy()
    for i, s in enumerate(spots):
        color = (255, 255, 0) if occupied is None else ((255, 0, 0) if occupied[i] else (0, 200, 0))
        pts = s.polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 2)
        label = s.id if scores is None else f"{s.id} {scores[i]:.2f}"
        cx, cy = s.polygon.mean(0).astype(int)
        cv2.putText(out, label, (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
