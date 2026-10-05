"""From per-spot occupancy history to guest allotments.

1. ``OccupancyHistory`` turns the raw detector log (timestamp, spot_id, occupied)
   into a regular grid of time slots per day (a slot counts as occupied if the
   spot was occupied at any observation inside it -> conservative).
2. ``AvailabilityModel`` estimates, for any future window, the probability that
   each spot stays free for the *whole* window, empirically: the fraction of
   comparable past days (same weekday type) on which it actually did.
3. ``GuestAllocator`` ranks candidate spots for a guest request, combining that
   learned probability with residents' explicit "I'm away" declarations,
   live status and existing bookings, and books the best one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def day_type(d: pd.Timestamp | datetime, mode: str = "weekday/weekend") -> str:
    if mode == "dow":
        return pd.Timestamp(d).day_name()
    return "weekend" if pd.Timestamp(d).weekday() >= 5 else "weekday"


class OccupancyHistory:
    def __init__(self, log: pd.DataFrame, slot_minutes: int = 30):
        if 1440 % slot_minutes:
            raise ValueError("slot_minutes must divide 1440")
        self.slot_minutes = slot_minutes
        df = log[["timestamp", "spot_id", "occupied"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.normalize()
        df["slot"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) // slot_minutes
        g = df.groupby(["spot_id", "date", "slot"])["occupied"].max()
        # grid[spot] -> DataFrame(index=date, columns=slot) with 1/0/NaN (NaN = no observation)
        self.grid = {
            s: sub.droplevel(0).unstack("slot").reindex(columns=range(self.n_slots))
            for s, sub in g.groupby(level=0)
        }
        self.spot_ids = sorted(self.grid)

    @property
    def n_slots(self) -> int:
        return 1440 // self.slot_minutes

    def profile(self, mode: str = "weekday/weekend") -> pd.DataFrame:
        """P(free) per spot x day-type x slot — handy for heatmaps / the report."""
        rows = []
        for s, g in self.grid.items():
            types = g.index.map(lambda d: day_type(d, mode))
            for t, sub in g.groupby(types):
                pf = 1 - sub.mean(axis=0, skipna=True)
                for slot, p in pf.items():
                    rows.append((s, t, slot, p, sub[slot].notna().sum()))
        return pd.DataFrame(rows, columns=["spot_id", "day_type", "slot", "p_free", "n_days"])


class AvailabilityModel:
    def __init__(self, history: OccupancyHistory, mode: str = "weekday/weekend", min_days: int = 3):
        self.h = history
        self.mode = mode
        self.min_days = min_days

    def _slots(self, start: datetime, end: datetime) -> list[tuple[int, int]]:
        """(day_offset, slot) pairs covered by [start, end); windows may cross midnight."""
        m = self.h.slot_minutes
        t = pd.Timestamp(start).floor(f"{m}min")
        d0 = pd.Timestamp(start).normalize()
        out = []
        while t < pd.Timestamp(end):
            out.append(((t.normalize() - d0).days, (t.hour * 60 + t.minute) // m))
            t += pd.Timedelta(minutes=m)
        return out

    def window_free_prob(self, start: datetime, end: datetime) -> pd.DataFrame:
        """For each spot: P(free during the whole window) and the number of past days used."""
        slots = self._slots(start, end)
        target = day_type(start, self.mode)
        rows = []
        for s, g in self.h.grid.items():
            days = [d for d in g.index if day_type(d, self.mode) == target]
            ok = n = 0
            for d in days:
                vals = []
                for off, slot in slots:
                    dd = d + pd.Timedelta(days=off)
                    vals.append(g.at[dd, slot] if dd in g.index else np.nan)
                vals = np.asarray(vals, float)
                if np.isnan(vals).mean() > 0.25:  # too little evidence that day
                    continue
                n += 1
                ok += int(np.nansum(vals) == 0)
            # Laplace smoothing keeps 0/0 and tiny samples away from 0 or 1.
            p = (ok + 0.5) / (n + 1) if n else np.nan
            rows.append({"spot_id": s, "p_free": p, "n_days": n, "free_days": ok})
        return pd.DataFrame(rows)


@dataclass
class Booking:
    spot_id: str
    guest: str
    start: datetime
    end: datetime


@dataclass
class AwayDeclaration:
    """A resident telling the system their spot is free (e.g. via an app)."""

    spot_id: str
    start: datetime
    end: datetime


@dataclass
class GuestAllocator:
    model: AvailabilityModel
    min_confidence: float = 0.8
    buffer_minutes: int = 30  # guest must leave this long before the resident tends to return
    bookings: list[Booking] = field(default_factory=list)
    declarations: list[AwayDeclaration] = field(default_factory=list)

    def _booked(self, spot, start, end) -> bool:
        return any(b.spot_id == spot and b.start < end and start < b.end for b in self.bookings)

    def _declared(self, spot, start, end) -> bool:
        return any(a.spot_id == spot and a.start <= start and end <= a.end for a in self.declarations)

    def candidates(self, start: datetime, hours: float, live_status: dict[str, int] | None = None) -> pd.DataFrame:
        end = start + timedelta(hours=hours)
        probs = self.model.window_free_prob(start, end + timedelta(minutes=self.buffer_minutes))
        probs["declared_away"] = [self._declared(s, start, end) for s in probs.spot_id]
        probs["booked"] = [self._booked(s, start, end) for s in probs.spot_id]
        probs["occupied_now"] = [bool(live_status.get(s, 0)) if live_status else False for s in probs.spot_id]
        probs["confidence"] = np.where(probs.declared_away, 1.0, probs.p_free)
        probs.loc[probs.n_days < self.model.min_days, "confidence"] = np.where(
            probs.loc[probs.n_days < self.model.min_days, "declared_away"], 1.0, 0.0)
        probs["eligible"] = (~probs.booked) & (~probs.occupied_now) & (probs.confidence >= self.min_confidence)
        return probs.sort_values(["eligible", "confidence", "n_days"], ascending=False).reset_index(drop=True)

    def request(self, guest: str, start: datetime, hours: float, live_status=None) -> Booking | None:
        c = self.candidates(start, hours, live_status)
        c = c[c.eligible]
        if c.empty:
            return None
        b = Booking(c.iloc[0].spot_id, guest, start, start + timedelta(hours=hours))
        self.bookings.append(b)
        return b
