"""Download the public datasets used in this project into data/.

    python scripts/download_datasets.py cnrpark --patches   # CNRPark-EXT (~1.6 GB): occupancy model + demo
    python scripts/download_datasets.py voc                 # PASCAL VOC 2007 (~0.9 GB): car detector

Downloads resume if interrupted; archives are extracted and then deleted.

CNRPark-EXT  http://cnrpark.it  (files on GitHub releases of fabiocarrara/deep-parking)
             Amato et al., "Deep learning for decentralized parking lot occupancy
             detection", Expert Systems with Applications 72, 2017.
PASCAL VOC   http://host.robots.ox.ac.uk/pascal/VOC/voc2007/  (mirrored on GitHub by Ultralytics)
             Everingham et al., "The PASCAL Visual Object Classes (VOC) Challenge", IJCV 2010.
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

GH_CNR = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/"
GH_VOC = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
FILES = {  # name -> [(url, approx bytes, extract?)]
    "cnrpark": [
        (GH_CNR + "CNRPark+EXT.csv", 18_132_695, False),
        (GH_CNR + "CNR-EXT_FULL_IMAGE_1000x750.tar", 1_100_000_000, True),
    ],
    "cnrpark-patches": [
        (GH_CNR + "CNR-EXT-Patches-150x150.zip", 449_500_000, True),
    ],
    "voc": [
        (GH_VOC + "VOCtrainval_06-Nov-2007.zip", 445_914_070, True),
        (GH_VOC + "VOCtest_06-Nov-2007.zip", 438_316_827, True),
    ],
}
TARGET = {"cnrpark": "cnrpark", "cnrpark-patches": "cnrpark", "voc": "voc"}


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
    ap.add_argument("dataset", choices=["cnrpark", "voc", "all"])
    ap.add_argument("--patches", action="store_true", help="also CNRPark-EXT spot patches (occupancy model)")
    ap.add_argument("--root", default="data")
    ap.add_argument("--keep-archives", action="store_true")
    a = ap.parse_args()
    names = {"cnrpark": ["cnrpark"], "voc": ["voc"], "all": ["cnrpark", "voc"]}[a.dataset]
    if a.patches and "cnrpark" in names:
        names.append("cnrpark-patches")
    for n in names:
        print(f"[{n}]")
        fetch(n, Path(a.root), a.keep_archives)


if __name__ == "__main__":
    main()
