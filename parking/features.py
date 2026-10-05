"""Hand-crafted features for parking-spot occupancy classification.

Every patch (one parking spot, cropped and perspective-corrected) is resized to a
fixed size and described by independent feature *blocks*:

    hog   Histogram of Oriented Gradients  -> shape/edges (car outline, windshield)
    lbp   Local Binary Pattern histograms   -> micro-texture (asphalt vs. car body)
    glcm  Grey-Level Co-occurrence stats    -> macro-texture (contrast, homogeneity)
    rgb   per-channel RGB histograms        -> colour content
    hsv   per-channel HSV histograms        -> colour, more robust to illumination

Blocks are extracted once and concatenated, so any combination ("feature set")
can be evaluated by selecting columns, without re-extracting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

BLOCKS = ("hog", "lbp", "glcm", "rgb", "hsv")

# Named combinations compared during model selection.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "hog": ("hog",),
    "lbp": ("lbp",),
    "glcm": ("glcm",),
    "rgb": ("rgb",),
    "hsv": ("hsv",),
    "texture": ("lbp", "glcm"),
    "color": ("rgb", "hsv"),
    "hog+texture": ("hog", "lbp", "glcm"),
    "hog+color": ("hog", "rgb", "hsv"),
    "texture+color": ("lbp", "glcm", "rgb", "hsv"),
    "all": BLOCKS,
}

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")


@dataclass
class FeatureConfig:
    patch_size: tuple[int, int] = (64, 64)  # (width, height) after resizing
    # HOG
    hog_orientations: int = 9
    hog_pixels_per_cell: int = 8
    hog_cells_per_block: int = 2
    # LBP: one uniform-LBP histogram per (points, radius) scale
    lbp_scales: tuple[tuple[int, int], ...] = ((8, 1), (16, 2))
    # GLCM
    glcm_levels: int = 32
    glcm_distances: tuple[int, ...] = (1, 2, 4)
    glcm_angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    # Colour histograms
    rgb_bins: int = 16
    hsv_bins: tuple[int, int, int] = (18, 16, 16)
    blocks: tuple[str, ...] = field(default=BLOCKS)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        d = dict(d)
        d["patch_size"] = tuple(d["patch_size"])
        d["lbp_scales"] = tuple(tuple(s) for s in d["lbp_scales"])
        d["glcm_distances"] = tuple(d["glcm_distances"])
        d["glcm_angles"] = tuple(d["glcm_angles"])
        d["hsv_bins"] = tuple(d["hsv_bins"])
        d["blocks"] = tuple(d["blocks"])
        return cls(**d)


def _norm_hist(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32).ravel()
    s = h.sum()
    return h / s if s > 0 else h


def hog_features(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    return hog(
        gray,
        orientations=cfg.hog_orientations,
        pixels_per_cell=(cfg.hog_pixels_per_cell,) * 2,
        cells_per_block=(cfg.hog_cells_per_block,) * 2,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def lbp_features(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    feats = []
    for p, r in cfg.lbp_scales:
        codes = local_binary_pattern(gray, p, r, method="uniform")
        # uniform LBP yields p + 2 distinct codes
        hist, _ = np.histogram(codes, bins=p + 2, range=(0, p + 2))
        feats.append(_norm_hist(hist))
    return np.concatenate(feats)


def glcm_features(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    q = (gray.astype(np.uint16) * cfg.glcm_levels // 256).astype(np.uint8)
    m = graycomatrix(
        q,
        distances=list(cfg.glcm_distances),
        angles=list(cfg.glcm_angles),
        levels=cfg.glcm_levels,
        symmetric=True,
        normed=True,
    )
    feats = []
    for prop in GLCM_PROPS:
        v = graycoprops(m, prop)  # (n_distances, n_angles)
        # mean over angles -> rotation invariance; range keeps directionality cue
        feats.append(v.mean(axis=1))
        feats.append(v.max(axis=1) - v.min(axis=1))
    return np.nan_to_num(np.concatenate(feats)).astype(np.float32)


def rgb_features(rgb: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    return np.concatenate(
        [_norm_hist(cv2.calcHist([rgb], [c], None, [cfg.rgb_bins], [0, 256])) for c in range(3)]
    )


def hsv_features(rgb: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # OpenCV: H in [0,180), S,V in [0,256)
    ranges = ([0, 180], [0, 256], [0, 256])
    return np.concatenate(
        [_norm_hist(cv2.calcHist([hsv], [c], None, [cfg.hsv_bins[c]], ranges[c])) for c in range(3)]
    )


class FeatureExtractor:
    """Turns RGB uint8 patches into a feature matrix with named column blocks."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self._slices: dict[str, slice] | None = None

    def _prepare(self, patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if patch.ndim == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        rgb = cv2.resize(patch, self.config.patch_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return rgb, gray

    def extract_blocks(self, patch: np.ndarray) -> dict[str, np.ndarray]:
        rgb, gray = self._prepare(patch)
        cfg = self.config
        fns = {
            "hog": lambda: hog_features(gray, cfg),
            "lbp": lambda: lbp_features(gray, cfg),
            "glcm": lambda: glcm_features(gray, cfg),
            "rgb": lambda: rgb_features(rgb, cfg),
            "hsv": lambda: hsv_features(rgb, cfg),
        }
        return {b: fns[b]() for b in cfg.blocks}

    def extract_one(self, patch: np.ndarray) -> np.ndarray:
        return np.concatenate(list(self.extract_blocks(patch).values()))

    @property
    def block_slices(self) -> dict[str, slice]:
        """Column range of each block inside the concatenated feature vector."""
        if self._slices is None:
            dummy = np.zeros((*self.config.patch_size[::-1], 3), np.uint8)
            start, slices = 0, {}
            for name, v in self.extract_blocks(dummy).items():
                slices[name] = slice(start, start + len(v))
                start += len(v)
            self._slices = slices
        return self._slices

    def columns_for(self, feature_set: str | tuple[str, ...]) -> np.ndarray:
        blocks = FEATURE_SETS[feature_set] if isinstance(feature_set, str) else feature_set
        sl = self.block_slices
        return np.concatenate([np.arange(sl[b].start, sl[b].stop) for b in blocks])

    def transform(self, patches, n_jobs: int = -1, batch_size: int = 64) -> np.ndarray:
        """Extract features for an iterable of RGB patches (or a callable loader per item)."""
        patches = list(patches)
        if len(patches) < 2 * batch_size or n_jobs == 1:
            rows = [self.extract_one(_load(p)) for p in patches]
        else:
            rows = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
                delayed(_extract)(self.config, p) for p in patches
            )
        return np.vstack(rows).astype(np.float32)


def _load(p):
    if isinstance(p, np.ndarray):
        return p
    from .data import read_rgb

    return read_rgb(p)


def _extract(cfg: FeatureConfig, p) -> np.ndarray:
    return FeatureExtractor(cfg).extract_one(_load(p))
