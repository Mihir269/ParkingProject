"""Generate synthetic demo data (patches, a frame + spots.json, an occupancy log)."""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.spots import save_spots  # noqa: E402
from parking.synthetic import make_frame, make_occupancy_log, make_patch_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--n-per-class", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

out = Path(a.out)
make_patch_dataset(out / "patches", a.n_per_class, seed=a.seed)
occ = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
frame, spots = make_frame(occ, seed=a.seed + 1)
cv2.imwrite(str(out / "frame.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
save_spots(out / "spots.json", spots, (frame.shape[1], frame.shape[0]))
make_occupancy_log(seed=a.seed).to_csv(out / "occupancy_log.csv", index=False)
print(f"wrote {out}/patches, frame.png (+spots.json, truth={occ}), occupancy_log.csv")
