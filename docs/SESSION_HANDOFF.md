# Session handoff: Dynamic Parking Allotment

Last updated: 2026-10-07 · Branch: `claude/dreamy-heisenberg-atzi7s` (repo `Mihir269/ParkingProject`, no `main` branch yet, no PR open)

## 1. The project in one paragraph

Mihir Gune (IIT Bombay, with Prof. Vinay Kulkarni). In housing societies, allotted parking
spots stand empty for hours while guests have nowhere to park. A fixed CCTV camera
(1) finds the parking spots, (2) classifies each spot free/occupied, (3) learns when each
spot is usually free, (4) lends free spots to guests for as long as they will likely
stay free. Constraint from the user: **classical CV only**: HOG, LBP/GLCM, RGB/HSV
histograms with Logistic Regression, Random Forest, SVM, XGBoost, picking the best.
**Train and report on public datasets only**: the user explicitly does not want
synthetic data used for training or results.

## 2. Decisions made (and why)

| Decision | Reason |
|---|---|
| Camera: elevated (~10 m) **side view**, looking diagonally across the rows (like mall CCTV) | User's choice. Prof. Kulkarni's email suggested ~10 ft "above, not at the sides"; user should confirm with him. Code supports both. |
| Occupancy model trained on **CNRPark-EXT**, side-angled cameras 1, 2, 3, 9 | Real CCTV, matches the camera placement; grouped CV by camera = "new society" test |
| Car detector trained on **PASCAL VOC 2007** | Needed boxes around every car. CNRPark labels are squares over spaces, so a detector trained on them got AP 0.02 (abandoned). |
| VOC 2012 added? **No** | 07+12 gave AP 0.29 vs 0.30 for 07 only and was 2× slower; a learning curve showed only ~+0.01 F1 per doubling of data |
| Spot discovery from parked cars | User's idea: a place where a car stays ≥45 min and later leaves (with a real pixel change) becomes a confirmed spot |
| Synthetic generators kept only in `tests/helpers/` | Used only to check the code runs; never for reported numbers |

## 3. Results (all on real public data)

| What | Result |
|---|---|
| Occupancy, all 9 CNRPark cameras, unseen cameras | F1 0.978 (vote of SVM+XGB+LogReg, all features) |
| Occupancy, side-angled cams 1/2/9 → unseen cam 3 | F1 0.955, **texture+colour (LBP, GLCM, RGB, HSV) + Random Forest**; HOG alone only 0.889 |
| Car detector, VOC 2007 | patch CV F1 0.903 (**all 5 features + SVM**); detection **AP@0.5 = 0.30** (scored with 300 of 4,231 car-free test images → slightly optimistic) |
| Demo, camera 3, 23 days, 451 frames, 21 spots | occupancy accuracy 95.2 % (9,471 decisions) |
| Guest allotment, 7 unseen days, 63 requests of 2 h | model booked 15, **all 15 stayed free**; naive "any spot free now" booked 49, only 35 % stayed free |

Features per 64×64 patch: HOG 1764, LBP 28, GLCM 36, RGB 48, HSV 50 = 1926 numbers.
Detector stage 1 = HOG + linear LogReg (fast pre-filter); stage 2 = best model above.

## 4. Repository map

```
parking/  features.py  models.py  selection.py  data.py  annotations.py  spots.py
          detect.py  detector.py  discovery.py  scheduler.py
scripts/  download_datasets.py  train_models.py  train_detector.py  demo.py
          annotate_spots.py  detect.py  discover_spots.py  crop_patches.py
configs/  cnrpark_camera3_spots.json        docs/demo/  docs/results/   tests/
```

## 5. How to reproduce (and how long it takes on 4 CPU cores)

```bash
pip install -r requirements.txt
python scripts/download_datasets.py cnrpark --patches      # ~1.6 GB
python scripts/download_datasets.py voc                    # ~0.9 GB
cat data/cnrpark/LABELS/camera{1,2,3,9}.txt > data/cnrpark/LABELS/side_cams.txt
python scripts/train_models.py --labels data/cnrpark/LABELS/side_cams.txt --images data/cnrpark/PATCHES \
       --group-level 2 --max-per-class 3000 --cv 3 --test-size 0.25 --out outputs/occupancy   # ~15 min
python scripts/train_detector.py --voc data/voc --out outputs/detector                    # ~1.5 h (eval ~50 min)
python scripts/demo.py --occupancy-model outputs/occupancy/best_model.joblib \
       --detector outputs/detector/car_detector.joblib --out docs/demo                     # ~10 min
python -m pytest                                                                           # 14 tests, ~30 s
```

Trained models are **not in git** (`*.joblib` ignored; 9 MB occupancy, 31.5 MB detector).
Run on a camera: `annotate_spots.py` once, then
`detect.py --model ... --spots ... --source rtsp://<camera> --every 60 --smooth 5`.

## 6. Environment gotchas (cloud session)

* Network allow-list: **only GitHub** (incl. release downloads) and package registries work.
  Kaggle, Hugging Face, Zenodo, Cloudflare R2, UFPR (PKLot), cocodataset.org are blocked.
  VOC comes from the Ultralytics GitHub mirror, CNRPark from fabiocarrara/deep-parking releases.
* Background tasks are killed after 2 h. Start long runs detached:
  `setsid nohup python ... > log 2>&1 < /dev/null & disown`, then poll the log.
* Never `pkill -f <pattern>` with a pattern that also appears in your own command line:
  it kills your own shell. Use the task-stop tool or `pgrep` + `kill <pid>`.
* SVM detector evaluation is slow (3–7 s per photo); the 07+12 SVM was ~6.6 s/photo.

## 7. Open items / next steps

1. **Ask the user**: open a PR / make this branch `main` so GitHub shows the README.
2. **Ask the user**: publish the two trained models as a GitHub release download.
3. Confirm camera placement with Prof. Kulkarni.
4. Footage from a real housing-society camera: the demo lot is an office (busy 9–17),
   societies have the opposite pattern, which is where guest allotment helps most.
5. Run `discover_spots.py` with the VOC detector on CNRPark camera 3 frames (not done yet
   on real data; discovery was only validated on synthetic scenes earlier).
6. Optional datasets matching the camera better (download them on the user's machine,
   blocked here): ACPDS (~10 m side views), NDISPark (every car boxed, day/night).
   Loaders for these were removed in the clean-up; re-add if used.
7. Guest allotment currently lives inside `demo.py`; a standalone booking command
   for a society is not written yet.
