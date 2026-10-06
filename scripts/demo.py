"""End-to-end demo on REAL footage: CNRPark-EXT camera 3 (a side-angled CCTV camera
the occupancy model never saw during training).

1. Occupancy: classify every parking space in every frame (23 days) and compare
   with the human labels.
2. Time-lapse: one day as an animated GIF (green = free, red = occupied,
   yellow outline = model disagrees with the label).
3. Availability: learn from the first days when each space is usually free.
4. Guest allotment on the held-out last days: book guests into spots predicted to
   stay free, then check against the real labels whether the spot really stayed
   free. Compared with the naive rule "take any spot that is free right now".
5. Car detector examples (optional): VOC 2007 test photos + a CNRPark frame.

python scripts/demo.py --occupancy-model outputs/cnrext_side_cams/best_model.joblib \
       --detector outputs/detector_voc/car_detector.joblib --out docs/demo
"""
import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.annotations import cnrpark_ext_frames, load_voc  # noqa: E402
from parking.models import OccupancyClassifier  # noqa: E402
from parking.scheduler import AvailabilityModel, GuestAllocator, OccupancyHistory  # noqa: E402
from parking.spots import Spot, crop_all  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cnrpark", default="data/cnrpark")
ap.add_argument("--camera", type=int, default=3)
ap.add_argument("--occupancy-model", required=True)
ap.add_argument("--detector", help="car detector for the detection examples")
ap.add_argument("--voc", default="data/voc")
ap.add_argument("--test-days", type=int, default=7)
ap.add_argument("--guest-hours", type=float, default=2)
ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9],
                help="confidence thresholds compared for guest allotment")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="docs/demo")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)
GREEN, RED, YELLOW = (40, 200, 70), (230, 50, 50), (255, 220, 0)

# ------------------------------------------------------------------ 1. occupancy on every frame
cam = cnrpark_ext_frames(a.cnrpark, [a.camera])[a.camera]
frames, bays = cam["frames"], cam["bays"]
spots = [Spot(str(s), np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)) for s, (x1, y1, x2, y2) in bays.items()]
clf = OccupancyClassifier.load(a.occupancy_model)
rows = []
for f in frames:
    img = f.image()
    p = clf.predict_proba(crop_all(img, spots, clf.feature_config.patch_size))
    for s, prob in zip(bays, p):
        rows.append({"timestamp": f.timestamp, "spot_id": str(s), "p_occupied": float(prob),
                     "pred": int(prob >= clf.threshold), "truth": f.slots.get(s, -1)})
log = pd.DataFrame(rows)
lab = log[log.truth >= 0]
tp = int(((lab.pred == 1) & (lab.truth == 1)).sum())
occ_metrics = {
    "frames": len(frames), "spaces": len(bays), "days": int(log.timestamp.dt.date.nunique()),
    "labelled_decisions": len(lab), "accuracy": float((lab.pred == lab.truth).mean()),
    "f1_occupied": float(2 * tp / max(lab.pred.sum() + lab.truth.sum(), 1)),
}
print("occupancy:", {k: round(v, 4) if isinstance(v, float) else v for k, v in occ_metrics.items()})


# ------------------------------------------------------------------ 2. time-lapse GIF of one day
def overlay(img, f, preds):
    vis = img.copy()
    for (s, (x1, y1, x2, y2)), pr in zip(bays.items(), preds):
        c = RED if pr else GREEN
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        tint = vis[y1:y2, x1:x2].astype(np.float32)
        vis[y1:y2, x1:x2] = (0.65 * tint + 0.35 * np.array(c)).astype(np.uint8)
        wrong = f.slots.get(s, -1) >= 0 and f.slots[s] != pr
        cv2.rectangle(vis, (x1, y1), (x2, y2), YELLOW if wrong else c, 4 if wrong else 2)
    free = int(len(preds) - sum(preds))
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(vis, f"camera {a.camera}   {f.timestamp:%a %d %b %Y  %H:%M}   free {free}/{len(preds)}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


by_day = log.groupby(log.timestamp.dt.date).timestamp.nunique()
day = by_day.idxmax()
day_frames = [f for f in frames if f.timestamp.date() == day]
pil = []
from PIL import Image  # noqa: E402

for f in day_frames:
    preds = log[log.timestamp == f.timestamp].pred.tolist()
    vis = overlay(f.image(), f, preds)
    pil.append(Image.fromarray(cv2.resize(vis, (600, 450), interpolation=cv2.INTER_AREA)))
pil[0].save(out / "timelapse.gif", save_all=True, append_images=pil[1:], duration=700, loop=0, optimize=True)
mid = day_frames[len(day_frames) // 2]
cv2.imwrite(str(out / "occupancy_frame.jpg"),
            cv2.cvtColor(overlay(mid.image(), mid, log[log.timestamp == mid.timestamp].pred.tolist()), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"time-lapse: {len(day_frames)} frames of {day}")

# ------------------------------------------------------------------ 3. availability from the first days
dates = sorted(log.timestamp.dt.date.unique())
train_days, test_days = dates[:-a.test_days], dates[-a.test_days:]
hist_log = log[log.timestamp.dt.date.isin(train_days)][["timestamp", "spot_id", "pred"]].rename(columns={"pred": "occupied"})
hist = OccupancyHistory(hist_log, slot_minutes=30)
model = AvailabilityModel(hist, min_days=3)
prof = hist.profile()
wk = prof[(prof.day_type == "weekday") & prof.slot.between(14, 35)]  # 07:00-17:30
heat = wk.pivot(index="spot_id", columns="slot", values="p_free")
heat = heat.loc[sorted(heat.index, key=int)]

# ------------------------------------------------------------------ 4. guest allotment on held-out days
truth_by_ts = {ts: g.set_index("spot_id") for ts, g in log.groupby("timestamp")}
all_ts = pd.DatetimeIndex(sorted(truth_by_ts))


def stayed_free(spot, start, end):
    """Ground truth: was ``spot`` labelled free in every frame within [start, end)?"""
    ts = [t for t in all_ts if start <= t < end]
    vals = [truth_by_ts[t].at[spot, "truth"] for t in ts]
    vals = [v for v in vals if v >= 0]
    return None if not vals else all(v == 0 for v in vals)


def nearest_frame(t):
    return all_ts[np.argmin(np.abs((all_ts - pd.Timestamp(t)).total_seconds()))]


results = []
for d in test_days:
    allocators = {t: GuestAllocator(model, min_confidence=t, buffer_minutes=0) for t in a.thresholds}
    naive_taken = set()
    for h in range(7, 16):
        start = pd.Timestamp(d) + timedelta(hours=h)
        end = start + timedelta(hours=a.guest_hours)
        if end > pd.Timestamp(d) + timedelta(hours=18):
            continue
        now = truth_by_ts[nearest_frame(start)]
        live = now.pred.to_dict()  # what the camera sees when the guest asks
        row = {"day": str(d), "start": f"{h:02d}:00"}
        for t, alloc in allocators.items():
            b = alloc.request(f"guest@{h}", start.to_pydatetime(), a.guest_hours, live_status=live)
            row[f"spot@{t}"] = b.spot_id if b else None
            row[f"ok@{t}"] = stayed_free(b.spot_id, start, end) if b else None
        free_now = [s for s, v in live.items() if v == 0 and s not in naive_taken]
        naive_spot = rng.choice(free_now) if free_now else None
        if naive_spot:
            naive_taken.add(naive_spot)
        row["naive_spot"], row["naive_ok"] = naive_spot, stayed_free(naive_spot, start, end) if naive_spot else None
        results.append(row)
res = pd.DataFrame(results)
res.to_csv(out / "guest_allotment.csv", index=False)


def rate(col):
    v = res[col].dropna()
    return (float(v.mean()) if len(v) else float("nan")), int(len(v))


sweep = []
for t in a.thresholds:
    r, n = rate(f"ok@{t}")
    sweep.append({"confidence": t, "allotted": n, "stayed_free": r})
naive_rate, n_naive = rate("naive_ok")
guest_metrics = {"train_days": len(train_days), "test_days": len(test_days), "requests": len(res),
                 "guest_hours": a.guest_hours, "sweep": sweep,
                 "naive": {"allotted": n_naive, "stayed_free": naive_rate}}
print("guests:", json.dumps(guest_metrics, default=str))

# ------------------------------------------------------------------ figures: availability heatmap + guest bar
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
im = ax[0].imshow(heat.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax[0].set_yticks(range(len(heat)), [f"spot {s}" for s in heat.index], fontsize=7)
xt = list(range(0, heat.shape[1], 2))
ax[0].set_xticks(xt, [f"{(heat.columns[i] * 30) // 60:02d}:{(heat.columns[i] * 30) % 60:02d}" for i in xt], fontsize=8)
ax[0].set_title(f"Learned from {len(train_days)} days: probability each spot is FREE (weekdays)")
fig.colorbar(im, ax=ax[0], fraction=0.03)
labels = [f"model\n\u2265{x['confidence']:.0%}" for x in sweep] + ["naive:\nfree now"]
vals = [x["stayed_free"] for x in sweep] + [naive_rate]
ns = [x["allotted"] for x in sweep] + [n_naive]
bars = ax[1].bar(labels, np.nan_to_num(vals), color=["#2e7d32"] * len(sweep) + ["#9e9e9e"])
for b, v, n in zip(bars, vals, ns):
    ax[1].text(b.get_x() + b.get_width() / 2, (0 if np.isnan(v) else v) + 0.01,
               ("-" if np.isnan(v) else f"{v:.0%}") + f"\n{n} booked", ha="center", fontsize=8)
ax[1].set_ylim(0, 1.15)
ax[1].set_ylabel("booked spot really stayed free")
ax[1].set_title(f"{len(res)} guest requests ({a.guest_hours:g} h) on {len(test_days)} unseen days", fontsize=10)
fig.tight_layout()
fig.savefig(out / "availability_and_guests.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------ 5. car detector examples
det_metrics = None
if a.detector:
    from parking.detector import CarDetector

    det = CarDetector.load(a.detector)
    det_metrics = det.metrics
    voc = [f for f in load_voc(a.voc, "test", n_background=0) if 2 <= len(f.cars) <= 6]
    picks = [voc[i] for i in rng.choice(len(voc), 3, replace=False)]
    tiles = []
    for f in picks:
        img = f.image()
        boxes, p = det.detect(img)
        keep = p >= 0.75  # demo threshold: fewer false alarms
        vis = img.copy()
        for b in f.cars.astype(int):
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (255, 255, 0), 1)
        for b, s in zip(boxes[keep].astype(int), p[keep]):
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), RED, 2)
            cv2.putText(vis, f"car {s:.2f}", (b[0], max(12, b[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)
        h = 300
        tiles.append(cv2.resize(vis, (int(vis.shape[1] * h / vis.shape[0]), h)))
    strip = np.hstack(tiles)
    cv2.imwrite(str(out / "detector_examples.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

summary = {"occupancy": occ_metrics, "guests": guest_metrics, "detector": det_metrics,
           "camera": a.camera, "occupancy_model": clf.name, "timelapse_day": str(day)}
(out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("saved demo to", out)
