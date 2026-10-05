"""Learn per-spot availability from an occupancy log and allot spots to guests.

python scripts/guest_demo.py --log data/synthetic/occupancy_log.csv \
    --request "2026-09-29 10:00" 4 --request "2026-09-29 10:00" 4 --request "2026-09-27 11:00" 3
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parking.scheduler import AvailabilityModel, GuestAllocator, OccupancyHistory  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--slot", type=int, default=30, help="slot size in minutes")
ap.add_argument("--min-confidence", type=float, default=0.8)
ap.add_argument("--request", nargs=2, action="append", metavar=("START", "HOURS"), required=True)
ap.add_argument("--profile-out", help="optional CSV of P(free) per spot/day-type/slot")
a = ap.parse_args()

hist = OccupancyHistory(pd.read_csv(a.log, parse_dates=["timestamp"]), a.slot)
alloc = GuestAllocator(AvailabilityModel(hist), min_confidence=a.min_confidence)
if a.profile_out:
    hist.profile().to_csv(a.profile_out, index=False)
for i, (start, hours) in enumerate(a.request, 1):
    start = pd.Timestamp(start).to_pydatetime()
    c = alloc.candidates(start, float(hours))
    print(f"\nGuest {i}: {start:%a %d %b %H:%M} for {hours} h")
    print(c.head(5).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    b = alloc.request(f"guest{i}", start, float(hours))
    print("->", f"allotted {b.spot_id} until {b.end:%H:%M}" if b else "no spot meets the confidence threshold")
