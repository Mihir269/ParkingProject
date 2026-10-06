"""Evaluate a saved car detector on PASCAL VOC (official test split), optionally
overriding its sliding-window sizes or threshold.

python scripts/eval_detector.py --detector outputs/detector_voc0712/car_detector.joblib --voc data/voc \
       --windows 36 22 53 35 68 86 94 46 161 75 168 132 336 153 407 264
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.annotations import load_voc  # noqa: E402
from parking.detector import CarDetector, average_precision  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--detector", required=True)
ap.add_argument("--voc", required=True)
ap.add_argument("--voc-background", type=int, default=300)
ap.add_argument("--windows", type=int, nargs="+")
ap.add_argument("--threshold", type=float)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out")
a = ap.parse_args()

det = CarDetector.load(a.detector)
if a.windows:
    det.config.window_sizes = [tuple(a.windows[i:i + 2]) for i in range(0, len(a.windows), 2)]
if a.threshold is not None:
    det.config.threshold = a.threshold
test = load_voc(a.voc, "test", n_background=a.voc_background, seed=a.seed + 1)  # same images as training script
t = time.time()
m = average_precision(det, test)
m["sec_per_frame"] = (time.time() - t) / len(test)
m["windows"] = det.config.window_sizes
print(json.dumps(m, default=str))
if a.out:
    Path(a.out).write_text(json.dumps(m, indent=2, default=str))
