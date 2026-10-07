"""Format tests for the dataset loaders (tiny files in the dataset's exact format),
plus checks on the real CNRPark-EXT / VOC data when it has been downloaded."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from parking.annotations import cnrpark_ext_frames, load_voc


def _img(path, w=200, h=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), np.uint8))


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


def test_voc_format(tmp_path):
    root = tmp_path / "VOCdevkit/VOC2007"
    _img(root / "JPEGImages/000001.jpg")
    _img(root / "JPEGImages/000002.jpg")
    (root / "ImageSets/Main").mkdir(parents=True)
    (root / "ImageSets/Main/trainval.txt").write_text("000001\n000002\n")
    obj = "<object><name>{}</name><difficult>{}</difficult><bndbox><xmin>11</xmin><ymin>21</ymin>" \
          "<xmax>50</xmax><ymax>60</ymax></bndbox></object>"
    (root / "Annotations").mkdir()
    (root / "Annotations/000001.xml").write_text(
        f"<annotation>{obj.format('car', 0)}{obj.format('car', 1)}{obj.format('dog', 0)}</annotation>")
    (root / "Annotations/000002.xml").write_text(f"<annotation>{obj.format('person', 0)}</annotation>")
    frames = load_voc(tmp_path, "trainval")
    assert len(frames) == 2
    f = frames[0]
    assert np.allclose(f.cars, [[10, 20, 50, 60]]) and len(f.ignore) == 1 and f.random_negatives
    assert len(frames[1].cars) == 0  # background image
    assert len(load_voc(tmp_path, "trainval", n_background=0)) == 1


VOC = Path("data/voc")


@pytest.mark.skipif(not VOC.exists(), reason="PASCAL VOC not downloaded")
def test_voc_real():
    f = load_voc(VOC, "test", n_background=0)
    # VOC's own car_test.txt lists 721 images with a non-difficult car
    assert sum(len(x.cars) > 0 for x in f) == 721 and sum(len(x.cars) for x in f) == 1201
