"""Format tests for the real-dataset loaders (tiny files in each dataset's exact
format), plus a check on the real CNRPark+EXT data when it has been downloaded."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parking.annotations import acpds_frames, cnrpark_ext_frames, load_acpds, load_coco


def _img(path, w=200, h=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), np.uint8))


def test_coco_ndispark_format(tmp_path):
    _img(tmp_path / "train/imgs/60_1537470146.jpg")
    ann = {"images": [{"id": 1, "file_name": "60_1537470146.jpg", "width": 200, "height": 100}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 20, 30, 40]}],
           "categories": [{"id": 3, "name": "car"}]}
    (tmp_path / "train/train_coco_annotations.json").write_text(json.dumps(ann))
    (f,) = load_coco(tmp_path / "train/train_coco_annotations.json")
    assert np.allclose(f.cars, [[10, 20, 40, 60]]) and f.group == "60" and f.random_negatives
    assert f.timestamp is not None and Path(f.path).exists()


def test_acpds_format(tmp_path):
    root = tmp_path / "rois_gopro"  # archive's top-level folder is found automatically
    _img(root / "images/GOPR0001.JPG")
    quad = [[0.1, 0.2], [0.3, 0.2], [0.3, 0.6], [0.1, 0.6]]
    split = {"file_names": ["GOPR0001.JPG"], "rois_list": [[quad, quad]], "occupancy_list": [[1, 0]]}
    (root / "annotations.json").write_text(json.dumps({"train": split, "valid": split, "test": split}))
    (r,) = load_acpds(tmp_path, "train")
    assert r["quads"].shape == (2, 4, 2) and np.isclose(r["quads"][0, 1, 0], 0.3 * 199)
    (f,) = acpds_frames(tmp_path, "test")
    assert len(f.cars) == 1 and len(f.negatives) == 1 and not f.random_negatives


CNR = Path("data/cnrpark")


@pytest.mark.skipif(not (CNR / "CNRPark+EXT.csv").exists(), reason="CNRPark+EXT not downloaded")
def test_cnrpark_real():
    d = cnrpark_ext_frames(CNR, [8])[8]
    f = d["frames"][0]
    img = f.image()
    assert img.shape == (750, 1000, 3)
    assert len(f.cars) + len(f.negatives) > 0
    allb = np.vstack([f.cars, f.negatives])
    assert (allb[:, 2] <= 1000 + 1).all() and (allb[:, 3] <= 750 + 1).all()
