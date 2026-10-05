"""Discover parking spots from car detections over time.

Idea: a parking spot is a place where a car stays parked AND that is later seen
empty (the car drove away) or was empty before a car arrived and stayed.

Pipeline (offline, over a sequence of frames, e.g. one frame per 1-30 min):

1. Run the car detector on every frame; keep detections + a small grey thumbnail.
2. Cluster detections across time into *sites* by box overlap (IoU), then merge
   sites whose median boxes overlap (the same car parks a little differently).
3. For every site, rebuild a presence timeline: present in a frame if any
   detection there overlaps the site box.
4. *Stays* = runs of presence lasting >= ``min_stay_minutes`` (a car driving down
   the lane is never a stay). Short absences inside a stay are bridged, but the
   car must really be detected in most frames of it (``min_stay_density``).
5. *Transitions* = a stay followed (departure) or preceded (arrival) by an absence of
   >= ``min_absence_minutes``. Each one is verified by appearance: the site's
   pixels while occupied vs while empty must differ (``min_change``). This rejects
   detector misses where the car is actually still there.
6. Status per site:
       confirmed  - at least one verified departure/arrival   -> parking spot
       candidate  - car parked the whole time, never seen empty -> probably a spot
       dropped    - never a stay (passing cars, false positives)

Output: Spots (for spots.json) and the occupancy log for the scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .annotations import iou
from .spots import Spot


@dataclass
class DiscoveryConfig:
    match_iou: float = 0.5  # detection <-> site
    merge_iou: float = 0.2  # site <-> site overlap considered for merging ...
    max_cooccurrence: float = 0.1  # ... unless both are detected in the same frame this often
    min_stay_minutes: float = 45
    min_absence_minutes: float = 30
    min_change: float = 0.6  # 1 - correlation of occupied vs empty appearance
    min_stay_density: float = 0.75  # share of a stay's frames with a real detection
    pad: float = 0.08  # grow the car box a little to get the spot box
    thumb_width: int = 480


@dataclass
class Site:
    box: np.ndarray
    members: list = field(default_factory=list)  # (frame_idx, box)
    presence: np.ndarray | None = None
    stays: list = field(default_factory=list)  # (start_idx, end_idx) inclusive
    departures: list = field(default_factory=list)  # (stay_end_idx, change)
    arrivals: list = field(default_factory=list)  # (stay_start_idx, change)
    status: str = "dropped"


class SpotDiscovery:
    def __init__(self, config: DiscoveryConfig | None = None):
        self.cfg = config or DiscoveryConfig()
        self.timestamps: list[datetime] = []
        self.detections: list[np.ndarray] = []
        self.thumbs: list[np.ndarray] = []
        self.scale = 1.0
        self.frame_size = None
        self.sites: list[Site] = []

    # ------------------------------------------------------------ collection
    def add_frame(self, ts: datetime, frame_rgb: np.ndarray, boxes: np.ndarray) -> None:
        H, W = frame_rgb.shape[:2]
        self.frame_size = (W, H)
        self.scale = min(1.0, self.cfg.thumb_width / W)
        thumb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if self.scale < 1:
            thumb = cv2.resize(thumb, (int(W * self.scale), int(H * self.scale)), interpolation=cv2.INTER_AREA)
        self.timestamps.append(ts)
        self.detections.append(np.asarray(boxes, float).reshape(-1, 4))
        self.thumbs.append(thumb)

    # ------------------------------------------------------------ analysis
    def _cluster(self) -> list[Site]:
        sites: list[Site] = []
        for t in np.argsort(self.timestamps, kind="stable"):
            dets = self.detections[t]
            if not len(dets):
                continue
            if sites:
                o = iou(dets, np.array([s.box for s in sites]))
                r, c = linear_sum_assignment(-o)
                matched = {i: j for i, j in zip(r, c) if o[i, j] >= self.cfg.match_iou}
            else:
                matched = {}
            for i, d in enumerate(dets):
                if i in matched:
                    s = sites[matched[i]]
                    s.members.append((t, d))
                    s.box = np.median([b for _, b in s.members], axis=0)
                else:
                    sites.append(Site(d.copy(), [(t, d)]))
        # Merge sites describing the same place (detector boxes jitter, the car parks
        # a bit differently each time). Two overlapping sites seen in the SAME frames
        # are two different cars side by side and are kept apart.
        def cooc(a, b):
            fa, fb = {t for t, _ in a.members}, {t for t, _ in b.members}
            return len(fa & fb) / max(min(len(fa), len(fb)), 1)

        changed = True
        while changed and len(sites) > 1:
            changed = False
            boxes = np.array([s.box for s in sites])
            o = iou(boxes, boxes)
            np.fill_diagonal(o, 0)
            for i, j in zip(*np.unravel_index(np.argsort(-o, axis=None), o.shape)):
                if o[i, j] < self.cfg.merge_iou:
                    break
                if i < j and cooc(sites[i], sites[j]) <= self.cfg.max_cooccurrence:
                    a, b = sites[i], sites.pop(j)
                    a.members += b.members
                    a.box = np.median([m for _, m in a.members], axis=0)
                    changed = True
                    break
        return sites

    def _appearance(self, idx: list[int], box) -> np.ndarray | None:
        x1, y1, x2, y2 = (np.asarray(box) * self.scale).round().astype(int)
        vecs = []
        for t in idx:
            th = self.thumbs[t]
            c = th[max(0, y1):y2, max(0, x1):x2]
            if c.size < 16:
                continue
            v = cv2.resize(c, (16, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            vecs.append((v - v.mean()) / (v.std() + 1e-6))  # cancel lighting changes
        return np.mean(vecs, 0) if vecs else None

    def _change(self, occ_idx, empty_idx, box) -> float:
        a, b = self._appearance(occ_idx, box), self._appearance(empty_idx, box)
        if a is None or b is None:
            return 0.0
        return float(1 - np.corrcoef(a, b)[0, 1])

    def analyse(self) -> list[Site]:
        cfg = self.cfg
        order = np.argsort(self.timestamps, kind="stable")
        ts = pd.to_datetime(pd.Series(self.timestamps)).iloc[order].reset_index(drop=True)
        dets = [self.detections[i] for i in order]
        thumbs = [self.thumbs[i] for i in order]
        self.timestamps, self.detections, self.thumbs = list(ts), dets, thumbs
        step = ts.diff().median().total_seconds() / 60 if len(ts) > 1 else 1.0
        dur = lambda a, b: (ts[b] - ts[a]).total_seconds() / 60 + step  # inclusive run length

        self.sites = self._cluster()
        for s in self.sites:
            member_frames = {t for t, _ in s.members}
            s.presence = np.array([t in member_frames or (bool(len(d)) and iou(s.box, d).max() >= cfg.match_iou)
                                   for t, d in enumerate(dets)])
            raw = s.presence.copy()
            runs = _runs(s.presence)
            # bridge short absences inside a stay (missed detections)
            for v, a, b in runs:
                if not v and 0 < a and b < len(ts) - 1 and dur(a, b) < cfg.min_absence_minutes:
                    s.presence[a:b + 1] = True
            runs = _runs(s.presence)
            # a parked car is detected in (almost) every frame of its stay; scattered
            # sightings of different passing cars bridged together are not a stay
            s.stays = [(a, b) for v, a, b in runs if v and dur(a, b) >= cfg.min_stay_minutes
                       and raw[a:b + 1].mean() >= cfg.min_stay_density]
            if not s.stays:
                s.status = "dropped"
                continue
            for k, (v, a, b) in enumerate(runs):
                if not v or (a, b) not in s.stays:
                    continue
                if k + 1 < len(runs):  # what follows the stay
                    _, a2, b2 = runs[k + 1]
                    if dur(a2, b2) >= cfg.min_absence_minutes:
                        s.departures.append((b, self._change(range(a, b + 1), range(a2, b2 + 1), s.box)))
                if k > 0:  # what precedes it
                    _, a0, b0 = runs[k - 1]
                    if dur(a0, b0) >= cfg.min_absence_minutes:
                        s.arrivals.append((a, self._change(range(a, b + 1), range(a0, b0 + 1), s.box)))
            verified = [c for _, c in s.departures + s.arrivals if c >= cfg.min_change]
            s.status = "confirmed" if verified else "candidate"
        return self.sites

    # ------------------------------------------------------------ outputs
    def spots(self, include_candidates: bool = True) -> list[Spot]:
        keep = [s for s in self.sites if s.status == "confirmed" or (include_candidates and s.status == "candidate")]
        # name spots in reading order (rows top to bottom, then left to right)
        keep.sort(key=lambda s: (round(s.box[1] / max(s.box[3] - s.box[1], 1)), s.box[0]))
        out = []
        W, H = self.frame_size
        for k, s in enumerate(keep, 1):
            x1, y1, x2, y2 = s.box
            px, py = (x2 - x1) * self.cfg.pad, (y2 - y1) * self.cfg.pad
            x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            meta = {
                "status": s.status,
                "n_stays": len(s.stays),
                "n_departures": sum(c >= self.cfg.min_change for _, c in s.departures),
                "n_arrivals": sum(c >= self.cfg.min_change for _, c in s.arrivals),
                "occupancy_rate": round(float(s.presence.mean()), 3),
                "car_box": [round(float(v), 1) for v in s.box],
            }
            out.append(Spot(f"S{k}", poly, meta=meta))
            s.spot_id = f"S{k}"
        return out

    def occupancy_log(self) -> pd.DataFrame:
        rows = []
        for s in self.sites:
            sid = getattr(s, "spot_id", None)
            if sid is None or s.status == "dropped":
                continue
            rows += [(t, sid, int(p)) for t, p in zip(self.timestamps, s.presence)]
        return pd.DataFrame(rows, columns=["timestamp", "spot_id", "occupied"])

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "box": np.round(s.box).astype(int).tolist(), "status": s.status, "detections": len(s.members),
            "stays": len(s.stays), "departures": len(s.departures), "arrivals": len(s.arrivals),
            "max_change": round(max([c for _, c in s.departures + s.arrivals], default=0.0), 3),
        } for s in self.sites])


def _runs(x: np.ndarray) -> list[tuple[bool, int, int]]:
    """Run-length encoding: [(value, start, end_inclusive), ...]."""
    out, start = [], 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[start]:
            out.append((bool(x[start]), start, i - 1))
            start = i
    return out


def evaluate_discovery(spots: list[Spot], bays: dict[str, tuple], kinds: dict[str, str] | None = None,
                       min_iou: float = 0.3) -> tuple[dict, pd.DataFrame]:
    """Match discovered spots to ground-truth bays (Hungarian on IoU)."""
    names = list(bays)
    gt = np.array([bays[n] for n in names], float)
    pred = np.array([[*s.polygon.min(0), *s.polygon.max(0)] for s in spots], float).reshape(-1, 4)
    o = iou(pred, gt) if len(pred) else np.zeros((0, len(gt)))
    r, c = linear_sum_assignment(-o) if len(pred) else ([], [])
    match = {names[j]: spots[i] for i, j in zip(r, c) if o[i, j] >= min_iou}
    rows = []
    for n in names:
        s = match.get(n)
        rows.append({"bay": n, "kind": (kinds or {}).get(n, ""), "found": s is not None,
                     "spot_id": s.id if s else None, "status": s.meta.get("status") if s else None,
                     "iou": round(float(o[[i for i, j in zip(r, c) if names[j] == n][0], names.index(n)]), 3) if s else None})
    df = pd.DataFrame(rows)
    confirmed = [s for s in spots if s.meta.get("status") == "confirmed"]
    matched_ids = {s.id for s in match.values()}
    metrics = {
        "n_bays": len(names),
        "n_discovered": len(spots),
        "n_confirmed": len(confirmed),
        "precision": len(matched_ids) / max(len(spots), 1),
        "precision_confirmed": sum(s.id in matched_ids for s in confirmed) / max(len(confirmed), 1),
        "recall": len(match) / max(len(names), 1),
    }
    return metrics, df
