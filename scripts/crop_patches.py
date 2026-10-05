"""Build a training set from your own camera: crop every spot of every frame.

Crops go to <out>/<frame-stem>/unlabeled/; move them into empty/ or occupied/
(or pre-label with an existing model via --model and just fix the mistakes).
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.data import IMG_EXTS, read_rgb  # noqa: E402
from parking.models import OccupancyClassifier  # noqa: E402
from parking.spots import crop_spot, load_spots, scale_spots  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True)
ap.add_argument("--spots", required=True)
ap.add_argument("--out", default="data/own_patches")
ap.add_argument("--size", type=int, nargs=2, default=(64, 64))
ap.add_argument("--model", help="optional model to pre-label crops")
a = ap.parse_args()

spots, ann = load_spots(a.spots)
clf = OccupancyClassifier.load(a.model) if a.model else None
for f in sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in IMG_EXTS):
    frame = read_rgb(f)
    ss = scale_spots(spots, ann, (frame.shape[1], frame.shape[0]))
    crops = [crop_spot(frame, s, tuple(a.size)) for s in ss]
    labels = clf.predict(crops) if clf else [None] * len(crops)
    for s, c, y in zip(ss, crops, labels):
        d = Path(a.out) / f.stem / ({0: "empty", 1: "occupied"}.get(y, "unlabeled"))
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{s.id}.png"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
print("done ->", a.out)
