"""Synthetic data, ONLY for smoke tests and demos of the pipeline.

Real results must come from real images (PKLot, CNRPark-EXT, or crops from the
society camera). Synthetic patches are deliberately varied (lighting, shadows,
grey cars, oil stains) so the comparison is not trivially perfect, but numbers
on them say nothing about real-world accuracy.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .spots import Spot


def _asphalt(rng, h, w):
    base = rng.uniform(70, 140)
    img = rng.normal(base, rng.uniform(4, 14), (h, w)).clip(0, 255)
    img = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
    tint = rng.uniform(-6, 6, 3)
    rgb = np.stack([img + tint[0], img + tint[1], img + tint[2]], -1)
    # painted bay lines on the sides
    lw = int(rng.integers(2, 5))
    line = rng.uniform(190, 245)
    rgb[:, :lw] = line
    rgb[:, w - lw:] = line
    # random stains
    for _ in range(rng.integers(0, 3)):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.ellipse(rgb, c, (int(rng.integers(3, 12)), int(rng.integers(3, 10))), 0, 0, 360,
                    float(base * 0.6), -1)
    return rgb


def _car(rng, rgb):
    h, w = rgb.shape[:2]
    gray_car = rng.random() < 0.35  # grey/silver cars resemble asphalt -> hard cases
    color = np.full(3, rng.uniform(80, 200)) if gray_car else rng.uniform(20, 235, 3)
    mx, my = int(w * rng.uniform(0.1, 0.2)), int(h * rng.uniform(0.04, 0.12))
    dx, dy = int(rng.integers(-4, 5)), int(rng.integers(-4, 5))
    x0, y0, x1, y1 = mx + dx, my + dy, w - mx + dx, h - my + dy
    # shadow under the car
    cv2.rectangle(rgb, (x0 + 3, y0 + 3), (x1 + 3, y1 + 3), tuple(float(v) for v in rgb[h // 2, w // 2] * 0.5), -1)
    cv2.rectangle(rgb, (x0, y0), (x1, y1), tuple(float(c) for c in color), -1)
    # windshield + rear window + roof highlight
    ww = int((y1 - y0) * rng.uniform(0.15, 0.25))
    cv2.rectangle(rgb, (x0 + 4, y0 + ww), (x1 - 4, y0 + 2 * ww), (35.0, 40.0, 50.0), -1)
    cv2.rectangle(rgb, (x0 + 5, y1 - 2 * ww), (x1 - 5, y1 - ww), (35.0, 40.0, 50.0), -1)
    cv2.line(rgb, (x0 + 2, (y0 + y1) // 2), (x1 - 2, (y0 + y1) // 2), tuple(float(c) * 0.8 for c in color), 1)
    return rgb


def _intrusion(rng, rgb):
    """Part of a neighbouring car poking into an empty bay (common at 10 ft height)."""
    h, w = rgb.shape[:2]
    side_w = int(w * rng.uniform(0.1, 0.3))
    color = tuple(float(c) for c in rng.uniform(20, 235, 3))
    x0, x1 = (0, side_w) if rng.random() < 0.5 else (w - side_w, w)
    cv2.rectangle(rgb, (x0, int(h * 0.1)), (x1, int(h * 0.9)), color, -1)
    return rgb


def synthetic_patch(rng, occupied: bool, size=(64, 64)) -> np.ndarray:
    w, h = size
    rgb = _asphalt(rng, h, w)
    if occupied:
        rgb = _car(rng, rgb)
        if rng.random() < 0.25:  # low-contrast car: blend toward the asphalt colour
            rgb = 0.45 * rgb + 0.55 * _asphalt(rng, h, w)
    elif rng.random() < 0.3:
        rgb = _intrusion(rng, rgb)
    # cast shadow from building / tree (also on empty spots)
    if rng.random() < 0.4:
        mask = np.zeros((h, w), np.float32)
        pts = rng.integers(0, max(w, h), (4, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
        rgb = rgb * (1 - 0.45 * mask[..., None])
    rgb = rgb * rng.uniform(0.55, 1.35)  # global illumination (morning / noon / evening)
    rgb = rgb + rng.normal(0, rng.uniform(3, 18), rgb.shape)  # sensor noise, night grain
    out = rgb.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.4:  # far spots: low effective resolution
        f = rng.uniform(0.2, 0.45)
        small = cv2.resize(out, (max(4, int(w * f)), max(4, int(h * f))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def make_patch_dataset(out_dir: str | Path, n_per_class=300, n_cameras=4, seed=0) -> Path:
    """Write out_dir/camN/{empty,occupied}/*.png so load_folder(out_dir, group_level=0) works."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    for i in range(n_per_class * 2):
        occ = i % 2 == 1
        cam = f"cam{rng.integers(n_cameras)}"
        d = out / cam / ("occupied" if occ else "empty")
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{i:05d}.png"), cv2.cvtColor(synthetic_patch(rng, occ), cv2.COLOR_RGB2BGR))
    return out


def make_frame(occupancy: list[bool], seed=0, n_cols=6, bay=(70, 120)):
    """A fake ~elevated-front-view frame with 2 rows of bays, plus their spot polygons."""
    rng = np.random.default_rng(seed)
    bw, bh = bay
    rows = int(np.ceil(len(occupancy) / n_cols))
    W, H = n_cols * bw + 40, rows * (bh + 40) + 40
    frame = np.full((H, W, 3), 110, np.float32) + rng.normal(0, 6, (H, W, 3))
    spots = []
    for k, occ in enumerate(occupancy):
        r, c = divmod(k, n_cols)
        x, y = 20 + c * bw, 20 + r * (bh + 40)
        frame[y:y + bh, x:x + bw] = synthetic_patch(rng, occ, (bw, bh))
        spots.append(Spot(f"{'AB'[r % 2]}{c + 1}", np.array(
            [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], np.float32), owner=f"flat-{101 + k}"))
    return frame.clip(0, 255).astype(np.uint8), spots


def make_occupancy_log(n_spots=8, days=28, step_min=15, start="2026-09-01", seed=0) -> pd.DataFrame:
    """Simulated residents: office-goers leave ~9:00 return ~18:30, some WFH, some
    shift workers. Weekends mostly at home. Returns columns timestamp, spot_id, occupied."""
    rng = np.random.default_rng(seed)
    kinds = rng.choice(["office", "office", "office", "wfh", "shift"], n_spots)
    t0 = datetime.fromisoformat(start)
    recs = []
    for d in range(days):
        day = t0 + timedelta(days=d)
        weekend = day.weekday() >= 5
        for s in range(n_spots):
            kind = kinds[s]
            away = []
            if kind == "office" and not weekend and rng.random() > 0.08:
                away.append((rng.normal(9.0, 0.4), rng.normal(18.5, 0.6)))
            elif kind == "shift" and not weekend:
                away.append((rng.normal(14.0, 0.3), rng.normal(23.0, 0.3)))
            if rng.random() < (0.5 if weekend else 0.15):  # errands
                a = rng.uniform(10, 19)
                away.append((a, a + rng.uniform(0.5, 3)))
            for m in range(0, 24 * 60, step_min):
                hr = m / 60
                occ = not any(a <= hr < b for a, b in away)
                recs.append((day + timedelta(minutes=m), f"S{s + 1}", int(occ)))
    return pd.DataFrame(recs, columns=["timestamp", "spot_id", "occupied"])
