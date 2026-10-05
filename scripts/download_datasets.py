"""Download public parking datasets into data/.

    python scripts/download_datasets.py cnrpark            # CNRPark+EXT: metadata + full frames (~1.1 GB)
    python scripts/download_datasets.py cnrpark --patches  # + 150x150 spot patches (~0.5 GB)
    python scripts/download_datasets.py pklot              # PKLot full frames + XML (~4.6 GB)
    python scripts/download_datasets.py side               # ACPDS + NDISPark + CNRPark+EXT (elevated side views)
    python scripts/download_datasets.py all --patches

Downloads resume if interrupted (HTTP Range), archives are extracted and then
deleted (``--keep-archives`` to keep them). Re-running skips finished parts.

Sources
-------
CNRPark+EXT  http://cnrpark.it  (files hosted on GitHub releases of fabiocarrara/deep-parking)
             Amato et al., "Deep learning for decentralized parking lot occupancy
             detection", Expert Systems with Applications 72, 2017.
ACPDS        https://github.com/martin-marek/parking-space-occupancy  (MIT)
             Marek, "Image-Based Parking Space Occupancy Classification: Dataset
             and Baseline", arXiv:2107.12207, 2021.
NDISPark     https://zenodo.org/records/6560823
             Ciampi et al., "Domain Adaptation for Traffic Density Estimation",
             VISAPP 2021.
PKLot        https://web.inf.ufpr.br/vri/databases/parking-lot-database/  (CC BY 4.0)
             Almeida et al., "PKLot - A robust dataset for parking lot
             classification", Expert Systems with Applications 42(11), 2015.
             If the UFPR server is slow or down, a mirror exists on Roboflow
             (https://public.roboflow.com/object-detection/pklot, free account,
             export "YOLO v8") -> use train_detector.py --yolo instead.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

GH = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/"
FILES = {
    "cnrpark": [
        # (url, size in bytes approx, extract?)
        (GH + "CNRPark+EXT.csv", 18_132_695, False),
        (GH + "CNR-EXT_FULL_IMAGE_1000x750.tar", 1_100_000_000, True),
    ],
    "cnrpark-patches": [
        (GH + "CNR-EXT-Patches-150x150.zip", 449_500_000, True),
        (GH + "CNRPark-Patches-150x150.zip", 36_600_000, True),
    ],
    "pklot": [
        ("http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz", 4_600_000_000, True),
    ],
    # Action-Camera Parking Dataset: 293 images from ~10 m high, every image a different view
    "acpds": [
        ("https://pub-e8bbdcbe8f6243b2a9933704a9b1d8bc.r2.dev/parking%2Frois_gopro.zip", 300_000_000, True),
    ],
    # NDISPark: ~250 images, 7 parking-lot cameras, EVERY car boxed (COCO), day + night
    "ndispark": [
        ("https://zenodo.org/records/6560823/files/ndis_park.zip?download=1", 118_200_000, True),
    ],
}
TARGET = {"cnrpark": "cnrpark", "cnrpark-patches": "cnrpark", "pklot": "pklot", "acpds": "acpds",
          "ndispark": "ndispark"}


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, retries: int = 5) -> Path:
    """Resumable download with a progress line."""
    part = dest.with_suffix(dest.suffix + ".part")
    if dest.exists():
        print(f"  already have {dest.name}")
        return dest
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "parking-project/0.1"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                    part.unlink(missing_ok=True)
                total = have + int(r.headers.get("Content-Length", 0) or 0)
                t0, done = time.time(), have
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        speed = (done - have) / max(time.time() - t0, 1e-6)
                        pct = f"{100 * done / total:5.1f}%" if total else ""
                        print(f"\r  {dest.name}: {pct} {_fmt(done)} / {_fmt(total)}  {_fmt(speed)}/s   ",
                              end="", flush=True)
            print()
            part.rename(dest)
            return dest
        except Exception as e:  # network hiccup: back off and resume
            wait = 2 ** attempt
            print(f"\n  error ({e}); retry {attempt}/{retries} in {wait}s")
            time.sleep(wait)
    sys.exit(f"failed to download {url}")


def extract(archive: Path, out: Path) -> None:
    marker = out / f".extracted_{archive.name}"
    if marker.exists():
        return
    print(f"  extracting {archive.name} ...")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data") if sys.version_info >= (3, 12) else t.extractall(out)
    marker.touch()


def fetch(name: str, root: Path, keep: bool) -> None:
    out = root / TARGET[name]
    out.mkdir(parents=True, exist_ok=True)
    need = sum(s for _, s, _ in FILES[name]) * 2.2  # archive + extracted copy
    free = shutil.disk_usage(out).free
    if free < need:
        print(f"  warning: {_fmt(free)} free, ~{_fmt(need)} needed")
    for url, _, do_extract in FILES[name]:
        fname = url.split("?")[0].rsplit("/", 1)[1].replace("%2F", "_")
        f = download(url, out / fname)
        if do_extract:
            extract(f, out)
            if not keep:
                f.unlink()
                f.touch()  # keep an empty placeholder so re-runs skip the download
    print(f"  {name} ready in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["cnrpark", "pklot", "acpds", "ndispark", "side", "all"],
                    help="side = the datasets matching an elevated side-angle camera (acpds, ndispark, cnrpark)")
    ap.add_argument("--patches", action="store_true", help="also CNRPark spot patches (occupancy classifier)")
    ap.add_argument("--root", default="data")
    ap.add_argument("--keep-archives", action="store_true")
    a = ap.parse_args()
    names = {"cnrpark": ["cnrpark"], "pklot": ["pklot"], "acpds": ["acpds"], "ndispark": ["ndispark"],
             "side": ["acpds", "ndispark", "cnrpark"], "all": ["acpds", "ndispark", "cnrpark", "pklot"]}[a.dataset]
    if a.patches and "cnrpark" in names:
        names.append("cnrpark-patches")
    for n in names:
        print(f"[{n}]")
        fetch(n, Path(a.root), a.keep_archives)
    print("\nNext:")
    if "cnrpark" in names:
        print("  python scripts/train_models.py --labels data/cnrpark/LABELS/all.txt --images data/cnrpark/PATCHES "
              "--group-level 2 --max-per-class 3000        # occupancy classifier, grouped by camera")
        print("  python scripts/train_detector.py --cnrpark data/cnrpark --max-frames 200   # car detector")
    if "pklot" in names:
        print("  python scripts/train_detector.py --pklot data/pklot/PKLot/PKLot --max-frames 300")
    if "acpds" in names:
        print("  python scripts/prepare_acpds_patches.py --root data/acpds && "
              "python scripts/train_models.py --data data/acpds_patches --group-level 1")
    if "ndispark" in names:
        print("  python scripts/train_detector.py --coco data/ndispark/train/train_coco_annotations.json "
              "--test-coco data/ndispark/validation/val_coco_annotations.json")


if __name__ == "__main__":
    main()
