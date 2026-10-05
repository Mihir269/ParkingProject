from datetime import datetime

import numpy as np
import pytest

from parking.detect import TemporalSmoother, classify_frame
from parking.features import BLOCKS, FEATURE_SETS, FeatureConfig, FeatureExtractor
from parking.models import OccupancyClassifier
from parking.scheduler import AvailabilityModel, AwayDeclaration, GuestAllocator, OccupancyHistory
from parking.selection import run_selection
from parking.spots import Spot, crop_spot, load_spots, order_quad, save_spots
from parking.synthetic import make_frame, make_occupancy_log, synthetic_patch


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    y = np.array([i % 2 for i in range(160)])
    patches = [synthetic_patch(rng, bool(v)) for v in y]
    groups = np.array([f"cam{i % 5}" for i in range(len(y))])
    return patches, y, groups


def test_feature_blocks_and_sets():
    ext = FeatureExtractor()
    p = synthetic_patch(np.random.default_rng(1), True)
    v = ext.extract_one(p)
    sl = ext.block_slices
    assert list(sl) == list(BLOCKS)
    assert v.shape == (sl["hsv"].stop,) and np.isfinite(v).all()
    # 64x64, 8px cells, 2x2 blocks, 9 bins -> 7*7*2*2*9
    assert sl["hog"].stop - sl["hog"].start == 1764
    # histograms are normalised
    for b in ("rgb", "lbp"):
        assert np.isclose(v[sl[b]].sum(), 3.0 if b == "rgb" else 2.0, atol=1e-4)
    for fs in FEATURE_SETS:
        assert len(ext.columns_for(fs)) > 0


def test_feature_config_roundtrip():
    cfg = FeatureConfig(patch_size=(48, 48))
    assert FeatureConfig.from_dict(cfg.to_dict()) == cfg


def test_selection_and_inference(data, tmp_path):
    patches, y, groups = data
    ext = FeatureExtractor()
    X = ext.transform(patches, n_jobs=1)
    res = run_selection(X, y, ext, groups, feature_sets=["hog", "color"], models=["logreg", "rf", "svm", "xgb"],
                        n_splits=3, ensemble_top=3, n_jobs=1, log=lambda *_: None)
    assert len(res.leaderboard) == 2 * 4 + 1  # + ensemble
    assert res.test_metrics["f1"] > 0.8
    clf = OccupancyClassifier(ext.config, res.best_estimator, res.best_name)
    clf.save(tmp_path / "m.joblib")
    clf2 = OccupancyClassifier.load(tmp_path / "m.joblib")
    assert np.allclose(clf.predict_proba(patches[:10]), clf2.predict_proba(patches[:10]))

    truth = [1, 0, 1, 0, 0, 1]
    frame, spots = make_frame(truth, seed=3)
    occ, p = classify_frame(frame, spots, clf2)
    assert occ.shape == (6,) and ((p >= 0) & (p <= 1)).all()


def test_spots_io_and_crop(tmp_path):
    s = Spot("A1", np.array([[10, 10], [50, 12], [52, 60], [8, 58]], np.float32), owner="flat-1")
    save_spots(tmp_path / "s.json", [s], (100, 80))
    (s2,), size = load_spots(tmp_path / "s.json")
    assert size == (100, 80) and s2.owner == "flat-1"
    frame = np.zeros((80, 100, 3), np.uint8)
    frame[10:60, 8:52] = 200
    assert crop_spot(frame, s2, (32, 32)).mean() > 150
    shuffled = s.polygon[[2, 0, 3, 1]]
    assert np.allclose(order_quad(shuffled), s.polygon)


def test_smoother():
    sm = TemporalSmoother(1, window=3)
    out = [sm.update([o])[0] for o in (1, 1, 0, 1, 1)]
    assert out == [1, 1, 1, 1, 1]  # single-frame flicker ignored


def test_scheduler():
    log = make_occupancy_log(n_spots=6, days=21, seed=2)
    hist = OccupancyHistory(log, 30)
    assert hist.n_slots == 48 and len(hist.spot_ids) == 6
    model = AvailabilityModel(hist)
    wk = model.window_free_prob(datetime(2026, 9, 29, 10), datetime(2026, 9, 29, 14))
    night = model.window_free_prob(datetime(2026, 9, 29, 1), datetime(2026, 9, 29, 5))
    assert wk.p_free.max() > 0.7  # office-goers are away on weekday mornings
    assert night.p_free.max() < 0.2  # everyone is home at night

    alloc = GuestAllocator(model, min_confidence=0.7)
    start = datetime(2026, 9, 29, 10)
    booked = []
    while (b := alloc.request(f"g{len(booked)}", start, 4)) is not None:
        booked.append(b.spot_id)
    assert booked and len(set(booked)) == len(booked)  # no double booking

    # An explicit away declaration makes a night slot available.
    alloc = GuestAllocator(model, declarations=[AwayDeclaration("S1", datetime(2026, 9, 29, 0), datetime(2026, 9, 29, 8))])
    b = alloc.request("night-guest", datetime(2026, 9, 29, 1), 4)
    assert b is not None and b.spot_id == "S1"
    # Live status overrides history.
    c = alloc.candidates(datetime(2026, 9, 29, 1), 4, live_status={"S1": 1})
    assert not c.set_index("spot_id").loc["S1", "eligible"]
