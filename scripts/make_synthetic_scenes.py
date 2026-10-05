"""Generate synthetic camera sequences (frames + annotations.json) for the car detector
and spot discovery. Each scene has its own background, residents and car colours."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.scenes import make_scene, save_scene  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/scenes")
ap.add_argument("--train-scenes", type=int, default=3)
ap.add_argument("--days", type=int, default=2)
ap.add_argument("--train-step", type=int, default=30, help="minutes between training frames")
ap.add_argument("--discovery-step", type=int, default=10, help="minutes between frames of the discovery scene")
a = ap.parse_args()

for i in range(a.train_scenes):
    p = save_scene(make_scene(days=a.days, step_min=a.train_step, seed=100 + i, start="2026-09-28"),
                   Path(a.out) / f"train_{i}")
    print("wrote", p)
# unseen scene (new background/residents), sampled densely, for discovery
p = save_scene(make_scene(days=a.days, step_min=a.discovery_step, seed=7, start="2026-10-05"), Path(a.out) / "discovery")
print("wrote", p)
