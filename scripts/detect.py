"""Run the trained classifier on a frame, a folder of frames, or a video/RTSP stream.

python scripts/detect.py --model outputs/cnrext_side_cams/best_model.joblib --spots configs/society_spots.json \
    --source rtsp://<camera> --every 60 --smooth 5 --out outputs/detect
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.data import IMG_EXTS, read_rgb  # noqa: E402
from parking.detect import TemporalSmoother, classify_frame, log_rows  # noqa: E402
from parking.models import OccupancyClassifier  # noqa: E402
from parking.spots import draw_spots, load_spots  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--spots", required=True)
ap.add_argument("--source", required=True, help="image, folder of images, video file or rtsp:// URL")
ap.add_argument("--every", type=float, default=60, help="seconds between processed video frames")
ap.add_argument("--smooth", type=int, default=1, help="majority-vote window (frames)")
ap.add_argument("--out", default="outputs/detect")
a = ap.parse_args()

clf = OccupancyClassifier.load(a.model)
spots, ann_size = load_spots(a.spots)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
smoother = TemporalSmoother(len(spots), a.smooth)
rows = []


def process(frame, ts, name):
    occ, p = classify_frame(frame, spots, clf, ann_size)
    occ = smoother.update(occ)
    rows.extend(log_rows(ts, spots, occ, p))
    vis = draw_spots(frame, spots, occ.astype(bool).tolist(), p.tolist())
    cv2.imwrite(str(out / f"{name}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"{ts}  free={int((occ == 0).sum())}/{len(spots)}  " + " ".join(
        f"{s.id}:{'X' if o else '.'}" for s, o in zip(spots, occ)))


src = Path(a.source)
if src.is_dir() or (src.suffix.lower() in IMG_EXTS):
    files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    for f in files:
        process(read_rgb(f), datetime.fromtimestamp(f.stat().st_mtime), f.stem)
else:
    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step, i = max(1, int(fps * a.every)), 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), datetime.now(), f"frame_{i:07d}")
        i += 1

log = out / "occupancy_log.csv"
pd.DataFrame(rows).to_csv(log, mode="a", header=not log.exists(), index=False)
print(f"annotated frames + {log}")
