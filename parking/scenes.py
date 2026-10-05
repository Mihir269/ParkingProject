"""Synthetic camera *sequences* of a residential parking area, for developing the
car detector and spot discovery without real footage.

A fixed camera sees two rows of bays on either side of a driving lane. Residents
park and leave on their own schedules (office-goers, work-from-home, shift workers,
one car that never moves, unused bays), and other cars drive along the lane.
Ground truth is kept for every frame: car boxes, bay boxes, which bay each car is in.

Like ``synthetic.py``, this only checks that the pipeline works. It is not
evidence of real-world accuracy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class SceneFrame:
    timestamp: datetime
    image: np.ndarray | None  # RGB; None once written to disk
    car_boxes: list[Box]
    parked_in: list[str | None]  # bay id of each car (None = moving)
    path: str | None = None


@dataclass
class Scene:
    bays: dict[str, Box]
    frames: list[SceneFrame] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)  # bay -> resident type


def draw_car(img: np.ndarray, box: Box, color, rng) -> None:
    """Top/oblique view of a car: body, shadow, front and rear glass, roof line."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    w, h = x2 - x1, y2 - y1
    color = tuple(float(c) for c in color)
    dark = (35.0, 40.0, 50.0)
    shadow = tuple(float(v) * 0.45 for v in img[min(y2 + 2, img.shape[0] - 1), min(x2 + 2, img.shape[1] - 1)])
    cv2.rectangle(img, (x1 + 3, y1 + 3), (x2 + 3, y2 + 3), shadow, -1)
    r = max(2, min(w, h) // 6)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    if h >= w:  # nose up or down: glass across the car
        g = max(2, int(h * rng.uniform(0.12, 0.18)))
        cv2.rectangle(img, (x1 + 3, y1 + g), (x2 - 3, y1 + 2 * g), dark, -1)
        cv2.rectangle(img, (x1 + 4, y2 - 2 * g), (x2 - 4, y2 - g), dark, -1)
        cv2.line(img, ((x1 + x2) // 2, y1 + 2 * g), ((x1 + x2) // 2, y2 - 2 * g), tuple(c * 0.8 for c in color), 1)
    else:  # driving sideways along the lane
        g = max(2, int(w * rng.uniform(0.12, 0.18)))
        cv2.rectangle(img, (x1 + g, y1 + 3), (x1 + 2 * g, y2 - 3), dark, -1)
        cv2.rectangle(img, (x2 - 2 * g, y1 + 4), (x2 - g, y2 - 4), dark, -1)
        cv2.line(img, (x1 + 2 * g, (y1 + y2) // 2), (x2 - 2 * g, (y1 + y2) // 2), tuple(c * 0.8 for c in color), 1)


def _background(rng, W, H, bays: dict[str, Box], lane: Box) -> np.ndarray:
    base = rng.normal(105, 9, (H, W)).astype(np.float32)
    base = cv2.GaussianBlur(base, (5, 5), 0) + rng.normal(0, 5, (H, W))
    img = np.repeat(base[..., None], 3, -1) + np.array([2, 0, -3], np.float32)
    # lane: slightly darker, worn centre line
    lx1, ly1, lx2, ly2 = (int(v) for v in lane)
    img[ly1:ly2, lx1:lx2] *= 0.88
    for x in range(lx1, lx2, 40):
        cv2.line(img, (x, (ly1 + ly2) // 2), (x + 20, (ly1 + ly2) // 2), (200.0, 190.0, 120.0), 2)
    for x1, y1, x2, y2 in bays.values():  # bay side lines
        cv2.line(img, (int(x1), int(y1)), (int(x1), int(y2)), (225.0, 225.0, 225.0), 2)
        cv2.line(img, (int(x2), int(y1)), (int(x2), int(y2)), (225.0, 225.0, 225.0), 2)
    # some static clutter: oil stains, a planter, a pillar
    for _ in range(12):
        c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.ellipse(img, c, (int(rng.integers(3, 14)), int(rng.integers(3, 10))), 0, 0, 360, (70.0, 70.0, 70.0), -1)
    cv2.rectangle(img, (W - 34, 10), (W - 10, H - 10), (90.0, 130.0, 70.0), -1)  # planter strip
    return img


def _away_intervals(kind: str, weekend: bool, rng) -> list[tuple[float, float]]:
    """Hours of the day when the resident's car is NOT in the bay."""
    if kind == "unused":
        return [(0, 24)]
    if kind == "static":
        return []
    out = []
    if kind == "office" and not weekend and rng.random() > 0.1:
        out.append((rng.normal(9, 0.4), rng.normal(18.5, 0.6)))
    elif kind == "shift" and not weekend:
        out.append((rng.normal(14, 0.3), rng.normal(23, 0.3)))
    if rng.random() < (0.5 if weekend else 0.2):
        a = rng.uniform(10, 19)
        out.append((a, a + rng.uniform(1, 3)))
    return out


def make_scene(
    days: int = 2,
    step_min: int = 10,
    start: str = "2026-09-28",
    n_per_row: int = 7,
    bay: tuple[int, int] = (72, 124),
    p_moving: float = 0.4,
    seed: int = 0,
    render: bool = True,
) -> Scene:
    rng = np.random.default_rng(seed)
    bw, bh = bay
    W, H = 30 + n_per_row * bw + 50, 2 * bh + 140
    bays: dict[str, Box] = {}
    for r, (row, y) in enumerate((("A", 20), ("B", H - 20 - bh))):
        for c in range(n_per_row):
            x = 30 + c * bw
            bays[f"{row}{c + 1}"] = (x, y, x + bw, y + bh)
    lane = (0, 20 + bh + 10, W - 40, H - 30 - bh)
    bg = _background(rng, W, H, bays, lane)

    kinds_pool = ["office", "office", "office", "wfh", "shift", "static", "unused"]
    kinds = {b: kinds_pool[i % len(kinds_pool)] for i, b in enumerate(rng.permutation(list(bays)))}
    colors = {b: rng.uniform(20, 235, 3) if rng.random() > 0.3 else np.full(3, rng.uniform(70, 210)) for b in bays}

    scene = Scene(bays, kinds=kinds)
    t0 = datetime.fromisoformat(start)
    for d in range(days):
        day = t0 + timedelta(days=d)
        away = {b: _away_intervals(kinds[b], day.weekday() >= 5, rng) for b in bays}
        # each time a car comes back it parks slightly differently
        offsets = {b: [] for b in bays}
        for m in range(0, 24 * 60, step_min):
            hr = m / 60
            boxes, parked = [], []
            for b, (x1, y1, x2, y2) in bays.items():
                if any(a <= hr < e for a, e in away[b]):
                    continue
                stay = sum(e <= hr for _, e in away[b])  # index of current stay
                while len(offsets[b]) <= stay:
                    offsets[b].append((rng.uniform(0.12, 0.2), rng.uniform(0.04, 0.09), rng.integers(-5, 6), rng.integers(-5, 6)))
                mx, my, dx, dy = offsets[b][stay]
                boxes.append((x1 + bw * mx + dx, y1 + bh * my + dy, x2 - bw * mx + dx, y2 - bh * my + dy))
                parked.append(b)
            if rng.random() < p_moving:  # a car driving along the lane
                cw, ch = rng.uniform(100, 120), rng.uniform(48, 58)
                cx = rng.uniform(0, lane[2] - cw)
                cy = (lane[1] + lane[3]) / 2 - ch / 2 + rng.uniform(-6, 6)
                boxes.append((cx, cy, cx + cw, cy + ch))
                parked.append(None)
            ts = day + timedelta(minutes=m)
            img = _render(bg, boxes, parked, colors, hr, rng) if render else None
            scene.frames.append(SceneFrame(ts, img, boxes, parked))
    return scene


def _render(bg, boxes, parked, colors, hour, rng) -> np.ndarray:
    img = bg.copy()
    for box, b in zip(boxes, parked):
        draw_car(img, box, colors[b] if b else rng.uniform(20, 235, 3), rng)
    # daylight curve: darker early morning / night (street lights, not pitch black)
    light = 0.55 + 0.6 * max(0.0, np.sin(np.pi * (hour - 6) / 14)) if 6 <= hour <= 20 else 0.5
    img = img * light * rng.uniform(0.92, 1.08)
    if 7 <= hour <= 17 and rng.random() < 0.5:  # building shadow sweeping across
        H, W = img.shape[:2]
        x = int(W * (hour - 7) / 10)
        pts = np.array([[x, 0], [x + 120, 0], [x + 60, H], [x - 60, H]], np.int32)
        mask = np.zeros((H, W), np.float32)
        cv2.fillPoly(mask, [pts], 1.0)
        img *= 1 - 0.35 * mask[..., None]
    img += rng.normal(0, 4 if light > 0.7 else 9, img.shape)
    return img.clip(0, 255).astype(np.uint8)


def save_scene(scene: Scene, out_dir: str | Path) -> Path:
    """Write frames as PNG + annotations.json (format read by annotations.load_scene_json)."""
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    recs = []
    for f in scene.frames:
        name = f"frames/{f.timestamp:%Y%m%d_%H%M}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR))
        f.path = str(out / name)
        recs.append({"image": name, "timestamp": f.timestamp.isoformat(),
                     "cars": [list(map(float, b)) for b in f.car_boxes], "parked_in": f.parked_in})
    (out / "annotations.json").write_text(json.dumps(
        {"bays": {k: list(v) for k, v in scene.bays.items()}, "bay_kinds": scene.kinds, "frames": recs}, indent=1))
    return out
