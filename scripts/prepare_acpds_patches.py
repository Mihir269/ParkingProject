"""Cut every annotated parking space out of ACPDS images (perspective-corrected) into
<out>/<split>/<image>/{empty,occupied}/<k>.png for train_models.py.

python scripts/prepare_acpds_patches.py --root data/acpds --out data/acpds_patches
python scripts/train_models.py --data data/acpds_patches --group-level 1 --out outputs/acpds_occupancy
(--group-level 1 = image; ACPDS has one image per viewpoint, so a CV fold never sees the test views)
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.annotations import load_acpds  # noqa: E402
from parking.data import read_rgb  # noqa: E402
from parking.spots import Spot, crop_spot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/acpds")
ap.add_argument("--out", default="data/acpds_patches")
ap.add_argument("--size", type=int, nargs=2, default=(96, 96))
a = ap.parse_args()

n = 0
for r in load_acpds(a.root):
    img = read_rgb(r["path"])
    for k, (q, occ) in enumerate(zip(r["quads"], r["occupied"])):
        d = Path(a.out) / r["split"] / Path(r["path"]).stem / ("occupied" if occ else "empty")
        d.mkdir(parents=True, exist_ok=True)
        patch = crop_spot(img, Spot(str(k), q.astype("float32")), tuple(a.size))
        cv2.imwrite(str(d / f"{k:03d}.png"), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        n += 1
print(f"wrote {n} patches to {a.out}")
