"""Mark parking spots on a reference camera frame (needs opencv-python with GUI).

Click the 4 corners of a spot (any order), then:
  n / Enter : save spot and start next     u : undo last point / last spot
  s         : save spots.json and quit     q / Esc : quit without saving

python scripts/annotate_spots.py --image ref.jpg --out spots.json
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.spots import Spot, draw_spots, load_spots, order_quad, save_spots  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--out", default="spots.json")
ap.add_argument("--prefix", default="S", help="spot id prefix")
a = ap.parse_args()

img = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
spots = load_spots(a.out)[0] if Path(a.out).exists() else []
pts: list[tuple[int, int]] = []


def on_mouse(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((x, y))


cv2.namedWindow("annotate", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("annotate", on_mouse)
while True:
    vis = draw_spots(img, spots)
    for p in pts:
        cv2.circle(vis, p, 4, (255, 0, 255), -1)
    if len(pts) > 1:
        cv2.polylines(vis, [np.array(pts, np.int32)], len(pts) == 4, (255, 0, 255), 1)
    cv2.imshow("annotate", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    k = cv2.waitKey(20) & 0xFF
    if k in (ord("n"), 13) and len(pts) == 4:
        spots.append(Spot(f"{a.prefix}{len(spots) + 1}", order_quad(np.array(pts, np.float32))))
        pts.clear()
    elif k == ord("u"):
        if pts:
            pts.pop()
        elif spots:
            spots.pop()
    elif k == ord("s"):
        save_spots(a.out, spots, (img.shape[1], img.shape[0]))
        print(f"saved {len(spots)} spots -> {a.out}")
        break
    elif k in (ord("q"), 27):
        break
cv2.destroyAllWindows()
