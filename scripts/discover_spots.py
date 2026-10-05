"""Discover parking spots automatically: detect cars in every frame, find places
where cars park and later leave (or arrive), and write spots.json + occupancy log.

python scripts/discover_spots.py --detector outputs/detector/car_detector.joblib \
       --frames data/scenes/discovery/frames --out outputs/discovery \
       --ground-truth data/scenes/discovery/annotations.json      # optional evaluation

Frame timestamps come from the file name (``--ts-format``, default
``%Y%m%d_%H%M``), falling back to file modification time. A video works too
(``--video``, one frame every ``--every`` seconds).
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.data import IMG_EXTS, read_rgb  # noqa: E402
from parking.detector import CarDetector  # noqa: E402
from parking.discovery import DiscoveryConfig, SpotDiscovery, evaluate_discovery  # noqa: E402
from parking.spots import save_spots  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--detector", required=True)
src = ap.add_mutually_exclusive_group(required=True)
src.add_argument("--frames", help="folder of frames")
src.add_argument("--video", help="video file / RTSP URL")
ap.add_argument("--every", type=float, default=300, help="seconds between sampled video frames")
ap.add_argument("--start", default=None, help="video start time (ISO), default now")
ap.add_argument("--ts-format", default="%Y%m%d_%H%M")
ap.add_argument("--min-stay", type=float, default=45, help="minutes a car must stay to count as parked")
ap.add_argument("--min-absence", type=float, default=30, help="minutes a place must be empty to count as left")
ap.add_argument("--min-change", type=float, default=0.6)
ap.add_argument("--no-candidates", action="store_true", help="only write confirmed spots")
ap.add_argument("--ground-truth", help="annotations.json with 'bays' to evaluate against")
ap.add_argument("--out", default="outputs/discovery")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

det = CarDetector.load(a.detector)
disc = SpotDiscovery(DiscoveryConfig(min_stay_minutes=a.min_stay, min_absence_minutes=a.min_absence,
                                     min_change=a.min_change))


def frames():
    if a.frames:
        for f in sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in IMG_EXTS):
            try:
                ts = datetime.strptime(f.stem, a.ts_format)
            except ValueError:
                ts = datetime.fromtimestamp(f.stat().st_mtime)
            yield ts, read_rgb(f)
    else:
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(fps * a.every))
        t0 = datetime.fromisoformat(a.start) if a.start else datetime.now()
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield t0 + timedelta(seconds=i / fps), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1


t = time.time()
last = None
for n, (ts, img) in enumerate(frames(), 1):
    boxes, _ = det.detect(img)
    disc.add_frame(ts, img, boxes)
    last = img
    if n % 25 == 0:
        print(f"  {n} frames ({(time.time() - t) / n:.2f} s/frame)  {ts}  cars={len(boxes)}")
print(f"detected cars in {n} frames in {time.time() - t:.0f}s")

disc.analyse()
spots = disc.spots(include_candidates=not a.no_candidates)
H, W = last.shape[:2]
save_spots(out / "spots.json", spots, (W, H))
disc.occupancy_log().to_csv(out / "occupancy_log.csv", index=False)
disc.summary().to_csv(out / "sites.csv", index=False)
print(f"\n{sum(s.meta['status'] == 'confirmed' for s in spots)} confirmed + "
      f"{sum(s.meta['status'] == 'candidate' for s in spots)} candidate spots "
      f"({sum(s.status == 'dropped' for s in disc.sites)} sites dropped: passing cars / false detections)")
for s in spots:
    print(f"  {s.id:<4} {s.meta['status']:<9} stays={s.meta['n_stays']} left={s.meta['n_departures']} "
          f"arrived={s.meta['n_arrivals']} occupied={s.meta['occupancy_rate']:.0%}")

# overlay: confirmed = green, candidate = orange, on an emptiest-looking frame
vis = last.copy()
for s in spots:
    c = (0, 200, 0) if s.meta["status"] == "confirmed" else (255, 150, 0)
    cv2.polylines(vis, [s.polygon.astype(np.int32).reshape(-1, 1, 2)], True, c, 2)
    x, y = s.polygon[0].astype(int)
    cv2.putText(vis, s.id, (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
cv2.imwrite(str(out / "discovered_spots.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

if a.ground_truth:
    gt = json.loads(Path(a.ground_truth).read_text())
    metrics, table = evaluate_discovery(spots, {k: tuple(v) for k, v in gt["bays"].items()}, gt.get("bay_kinds"))
    table.to_csv(out / "evaluation.csv", index=False)
    (out / "evaluation.json").write_text(json.dumps(metrics, indent=2))
    print("\nvs ground truth:", {k: round(v, 3) for k, v in metrics.items()})
    print(table.to_string(index=False))
print("saved", out)
