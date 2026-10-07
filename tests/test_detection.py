import numpy as np
import pytest

from parking.annotations import iou, nms, sample_patches, suggest_window_sizes
from parking.detector import CarDetector, DetectorConfig, Stage1HOG, average_precision
from parking.discovery import SpotDiscovery, _runs, evaluate_discovery
from parking.features import FeatureConfig, FeatureExtractor
from parking.models import make_candidate
from tests.helpers.scenes import make_scene, save_scene


def test_iou_and_nms():
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], float)
    o = iou(a, a)
    assert np.allclose(np.diag(o), 1) and np.isclose(o[0, 1], 25 / 175) and o[0, 2] == 0
    keep = nms(np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float), np.array([0.9, 0.8, 0.7]))
    assert list(keep) == [0, 2]


def test_runs():
    assert _runs(np.array([1, 1, 0, 1])) == [(True, 0, 1), (False, 2, 2), (True, 3, 3)]


@pytest.fixture(scope="module")
def scene_frames(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    scene = make_scene(days=1, step_min=120, seed=3)
    save_scene(scene, d)
    from parking.annotations import load_scene_json

    return scene, load_scene_json(d / "annotations.json")


def test_patch_sampling(scene_frames):
    _, frames = scene_frames
    win = suggest_window_sizes(frames, k=3)
    assert 1 <= len(win) <= 3
    P, y, g = sample_patches(frames, win, neg_per_frame=10)
    assert len(P) == len(y) == len(g) and set(np.unique(y)) == {0, 1}
    # negatives never centred on a car: IoU with every car < 0.3 is enforced at sampling time
    assert (y == 0).sum() >= 10 * len(frames) * 0.9


def test_detector_end_to_end(scene_frames):
    _, frames = scene_frames
    train, test = frames[:8], frames[8:]
    win = suggest_window_sizes(train, k=3)
    P, y, _ = sample_patches(train, win, neg_per_frame=25)
    cfg = FeatureConfig()
    ext = FeatureExtractor(cfg)
    X = ext.transform(P, n_jobs=1)
    est, _ = make_candidate("logreg", ext.columns_for("hog+color"))
    det = CarDetector(cfg, DetectorConfig(win), Stage1HOG(cfg).fit(P, y), est.fit(X, y))
    boxes, scores = det.detect(test[0].image())
    assert boxes.shape[1] == 4 and len(boxes) == len(scores)
    m = average_precision(det, test)
    assert m["ap"] > 0.5, m  # synthetic scene: should find most cars


def test_discovery_with_oracle_detections():
    scene = make_scene(days=2, step_min=15, seed=5, start="2026-10-05")
    disc = SpotDiscovery()
    rng = np.random.default_rng(0)
    for f in scene.frames:  # perfect detector that misses 5% of cars
        disc.add_frame(f.timestamp, f.image, np.array([b for b in f.car_boxes if rng.random() > 0.05]))
    disc.analyse()
    spots = disc.spots()
    m, table = evaluate_discovery(spots, scene.bays, scene.kinds)
    assert m["precision"] == 1.0  # no spots in the driving lane
    assert m["precision_confirmed"] == 1.0
    kinds = dict(zip(table.bay, table.kind))
    status = dict(zip(table.bay, table.status))
    found = dict(zip(table.bay, table.found))
    for bay, kind in kinds.items():
        if kind == "unused":
            assert not found[bay]  # never had a car: cannot be discovered
        if kind == "static":
            assert status[bay] == "candidate"  # never left: not confirmed
        if kind == "shift":
            assert status[bay] == "confirmed"
    log = disc.occupancy_log()
    assert set(log.spot_id) == {s.id for s in spots}
