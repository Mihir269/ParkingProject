# Dynamic Parking Allotment for Residential Societies

Residential societies have fixed, allotted parking, yet many allotted spots sit
empty for long stretches (owners away at the office) while guests have nowhere to
park. This project:

1. **Identifies parking spots** in a fixed camera view.
2. **Classifies each spot as empty or occupied** using classical CV features
   (HOG, LBP/GLCM, RGB/HSV histograms) with Logistic Regression, Random Forest,
   SVM and XGBoost, and **selects the best model**.
3. **Learns when each spot is usually free** (for example, flat 101 leaves at 9:00
   and returns at 18:30 on weekdays) from the occupancy log.
4. **Gives free spots to guests** for a requested number of hours, but only
   when the spot is very likely to stay free for the whole visit.

**Camera assumption** (from Prof. Kulkarni): fixed camera about **10 ft high**,
looking along the lane rather than from the side. Spots then look like
quadrilaterals. Each spot is perspective-warped to an upright patch before
classification, so near and far spots look alike to the model.

---

## Plan

| Phase | What | Status |
|---|---|---|
| **0. Setup** | Repo structure, feature extraction, model comparison, scheduler, tests | ✅ done (this commit) |
| **1. Data** | (a) Public benchmarks: **CNRPark-EXT** (elevated cameras, similar to our setup, 150×150 spot patches, sunny/overcast/rainy) and **PKLot** (top-down, about 700k patches). (b) **Our own society**: mount or borrow a camera at about 10 ft, capture a frame every 1–5 min for 1–2 weeks, mark spots once with `annotate_spots.py`, crop with `crop_patches.py`, then label (pre-label with the benchmark model and fix the mistakes) | ⏳ next |
| **2. Spot detection** | v1: manual polygons per camera (one-time, about 10 min). v2 (stretch): automatic spot finding from line detection, or from where cars park over many days | v1 ✅ |
| **3. Occupancy model** | Features: HOG, LBP, GLCM, RGB and HSV histograms, plus combinations (11 feature sets). Models: LogReg, RF, SVM, XGBoost. Grouped CV (by camera/day, to avoid leakage), optional grid-search of the top N, soft-voting ensemble of the top model families, selection on CV F1, one final held-out test | ✅ pipeline; ⏳ run on real data |
| **4. Robustness** | Cross-dataset tests (train PKLot → test CNRPark, train benchmark → test our society), weather/night breakdown, per-spot error analysis (far spots, pillar occlusion), temporal smoothing across frames | ⏳ |
| **5. Availability model** | Occupancy log → time-slot grid → P(spot stays free for the whole window) from comparable past days (weekday vs. weekend), with a safety buffer for residents who return early | ✅ v1 |
| **6. Guest allotment** | Rank spots by that probability, plus resident "I'm away" declarations, live camera status and existing bookings; book the best spot above a confidence threshold | ✅ v1 |
| **7. Evaluation of allotment** | Replay held-out weeks: how often would a guest have been bumped (resident returned early)? Trade-off between utilisation and conflicts as the threshold changes | ⏳ |
| **8. Demo / report** | Live overlay of free/occupied spots, a simple booking UI or WhatsApp bot, and the write-up | ⏳ |

Possible extensions: compare against a small CNN (MobileNet / mAlexNet from the
CNRPark paper) as an upper bound, model residents' leave/return times directly
(survival model) instead of empirical rates, and notify the resident when a guest
is parked in their spot.

---

## Layout

```
parking/
  features.py    HOG, LBP, GLCM, RGB/HSV histogram features (named blocks, combinable)
  data.py        dataset loaders: folder layout (PKLot / own crops) and CNRPark label lists
  models.py      LogReg / RF / SVM / XGBoost candidates, soft-vote ensemble, saved-model bundle
  selection.py   grouped CV over feature set x model, tuning, ensemble, best-model selection
  spots.py       spot polygons (JSON), perspective crop, overlay drawing
  detect.py      per-frame classification and temporal smoothing
  scheduler.py   occupancy history -> availability model -> guest allocator
  synthetic.py   synthetic patches/frames/logs for tests & demos only
scripts/
  annotate_spots.py   click 4 corners per spot on a reference frame -> spots.json
  crop_patches.py     crop all spots from many frames -> training patches
  train_models.py     feature extraction + model comparison + selection
  detect.py           run on image / folder / video / RTSP, write overlay + occupancy log
  guest_demo.py       learn availability from a log, allot guests
  make_synthetic.py   generate demo data
tests/               pytest suite (synthetic data)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest                         # ~15 s

# End-to-end demo on synthetic data
python scripts/make_synthetic.py
python scripts/train_models.py --data data/synthetic/patches --group-level 0 --tune-top 3 --out outputs/synth
python scripts/detect.py --model outputs/synth/best_model.joblib --spots data/synthetic/spots.json \
       --source data/synthetic/frame.png
python scripts/guest_demo.py --log data/synthetic/occupancy_log.csv \
       --request "2026-09-29 10:00" 4 --request "2026-10-03 11:00" 3
```

### On real data

```bash
# CNRPark-EXT (http://cnrpark.it): group by camera (path component 2)
python scripts/train_models.py --labels CNR-EXT/LABELS/all.txt --images CNR-EXT/PATCHES \
       --group-level 2 --max-per-class 5000 --tune-top 3 --out outputs/cnrext

# PKLot segmented: group by parking lot (UFPR04 / UFPR05 / PUCPR)
python scripts/train_models.py --data PKLot/PKLotSegmented --group-level 0 --max-per-class 5000 --out outputs/pklot

# Our society
python scripts/annotate_spots.py --image ref_frame.jpg --out configs/society_spots.json
python scripts/crop_patches.py --frames captures/ --spots configs/society_spots.json \
       --model outputs/cnrext/best_model.joblib       # pre-labels; fix by moving files
python scripts/train_models.py --data data/own_patches --out outputs/society
python scripts/detect.py --model outputs/society/best_model.joblib --spots configs/society_spots.json \
       --source rtsp://<camera> --every 60 --smooth 5
```

Outputs per training run: `leaderboard.csv` (every combination, CV mean ± std
of accuracy, precision, recall, F1, ROC-AUC, fit time, latency), `test_top5.csv`,
`report.png` (F1 heatmap + confusion matrix), `summary.json`, `best_model.joblib`.

## Model selection details

* **Features** (64×64 patch): HOG 1764-d (9 bins, 8 px cells, 2×2 blocks);
  LBP uniform at (P=8, R=1) and (16, 2), 28-d; GLCM at distances 1/2/4,
  4 angles, 6 properties (mean and range over angles), 36-d; RGB 3×16 bins;
  HSV 18+16+16 bins.
* **Feature sets compared**: each block alone, `texture` (LBP+GLCM), `color`
  (RGB+HSV), `hog+texture`, `hog+color`, `texture+color`, `all`.
* **Models**: LogReg and SVM get standardisation (the SVM also gets PCA at 95%
  variance on large feature sets); RF; XGBoost (hist).
* **No leakage**: patches of the same spot from consecutive frames are near
  duplicates. Use `--group-level` so CV and the test split keep a whole
  camera/lot/day on one side. Random splits give inflated scores.
* **Selection**: by mean CV F1 for "occupied", tie-break on ROC-AUC, then
  latency. The ensemble must beat the best single model by ≥0.002 F1, otherwise
  the simpler model wins. The test set is used once, after selection.

### Synthetic sanity run (pipeline check only, not a real result)

![synthetic report](docs/synthetic_report.png)

On synthetic patches, HOG-based sets with SVM or LogReg score best and colour
alone is weakest. Real-image numbers come in Phase 1.
