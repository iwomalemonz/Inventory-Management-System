"""Everything about the calendar: ML features, festivals and demand assumptions.

Two jobs live here:
1. `calendar_frame` turns dates into the numeric features the ML models use.
2. The FESTIVALS table and the *_PRIOR dictionaries are editable assumptions
   about how weekdays, paydays, seasons and festivals change demand. The sales
   dataset has no real festival signal, so these are applied on top of the
   models. Edit them freely to match your shop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from . import config

SEASON_NAMES = ["Winter", "Summer", "Monsoon", "Autumn"]


def season_id(month: int) -> int:
    """Month (1-12) -> season index. Tweak for your region."""
    if month in (11, 12, 1, 2):
        return 0   # Winter
    if month in (3, 4, 5, 6):
        return 1   # Summer
    if month in (7, 8, 9):
        return 2   # Monsoon
    return 3       # Autumn (October)


def dom_bucket(day: int) -> int:
    """Day of month -> 0: 1-5 (payday), 1: 6-15, 2: 16-25, 3: 26-31."""
    if day <= 5:
        return 0
    if day <= 15:
        return 1
    if day <= 25:
        return 2
    return 3


def calendar_frame(dates) -> pd.DataFrame:
    """Dates -> calendar features. Used for BOTH training and prediction."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    return pd.DataFrame(
        {
            "dow": idx.dayofweek.values,                     # 0 = Monday
            "dom": idx.day.values,                           # 1-31
            "dom_bucket": [dom_bucket(d) for d in idx.day],
            "month": idx.month.values - 1,                   # 0-11
            "season_id": [season_id(m) for m in idx.month],
            "is_weekend": (idx.dayofweek.values >= 5).astype(int),
        }
    )


# ---------------------------------------------------------------------------
# Festivals (editable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Festival:
    name: str
    day: date
    peak: float                      # demand multiplier on the day itself
    run_up_days: int = 0             # days before with rising demand
    run_up_peak: float = 1.0         # multiplier on the day right before
    category_boost: dict = field(default_factory=dict, hash=False, compare=False)
    # category_boost scales the *extra* demand per category.
    # 1.0 = normal lift, 1.4 = 40% more lift, 0.5 = half the lift.


DIWALI_BOOST = {"Clothing": 1.3, "Electronics": 1.2, "Groceries": 1.2,
                "Furniture": 0.9, "Toys": 0.7}

# Lunar festivals move every year, so they are listed by hand.
# Dates below were checked for 2026; add or fix years as needed.
LUNAR_FESTIVALS = [
    Festival("Dhanteras", date(2026, 11, 6), 1.9, 3, 1.5, DIWALI_BOOST),
    Festival("Choti Diwali", date(2026, 11, 7), 2.0, 0, 1.0, DIWALI_BOOST),
    Festival("Diwali", date(2026, 11, 8), 2.3, 2, 1.9, DIWALI_BOOST),
    Festival("Govardhan Puja", date(2026, 11, 9), 1.5, 0, 1.0, DIWALI_BOOST),
    Festival("Bhai Dooj", date(2026, 11, 10), 1.4, 0, 1.0, DIWALI_BOOST),
    Festival("Holi", date(2027, 3, 22), 1.7, 3, 1.4, {"Clothing": 1.3, "Groceries": 1.3}),
]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + 7 * (n - 1))


@lru_cache(maxsize=None)
def _fixed_festivals(year: int) -> tuple:
    """Festivals that fall on the same date every year."""
    return (
        Festival("New Year's Day", date(year, 1, 1), 1.4, 2, 1.2),
        Festival("Republic Day", date(year, 1, 26), 1.3, 1, 1.1, {"Clothing": 1.2, "Electronics": 1.3}),
        Festival("Valentine's Day", date(year, 2, 14), 1.3, 2, 1.15, {"Toys": 1.2, "Clothing": 1.1}),
        Festival("Independence Day", date(year, 8, 15), 1.3, 1, 1.1, {"Electronics": 1.4, "Clothing": 1.2}),
        Festival("Children's Day", date(year, 11, 14), 1.4, 2, 1.2, {"Toys": 2.0}),
        Festival("Black Friday", _nth_weekday(year, 11, 4, 4), 1.9, 2, 1.4,
                 {"Electronics": 1.6, "Clothing": 1.3, "Groceries": 0.4}),
        Festival("Christmas", date(year, 12, 25), 1.8, 4, 1.5, {"Toys": 1.8, "Groceries": 1.2}),
        Festival("New Year's Eve", date(year, 12, 31), 1.5, 1, 1.2, {"Groceries": 1.5}),
    )


def all_festivals_near(d: date) -> list[Festival]:
    fixed = list(_fixed_festivals(d.year)) + list(_fixed_festivals(d.year + 1))
    return fixed + LUNAR_FESTIVALS


@dataclass(frozen=True)
class EventInfo:
    name: str
    kind: str            # "day" (the festival itself) or "run_up" (days before)
    days_until: int      # 0 on the day itself
    multiplier: float    # overall lift, before category weighting
    category_boost: dict


def active_event(d: date) -> EventInfo | None:
    """The strongest festival effect on date d (or None)."""
    best_day = None
    best_run = None
    for f in all_festivals_near(d):
        delta = (f.day - d).days
        if delta == 0:
            if best_day is None or f.peak > best_day.multiplier:
                best_day = EventInfo(f.name, "day", 0, f.peak, f.category_boost)
        elif 0 < delta <= f.run_up_days:
            frac = (f.run_up_days - delta + 1) / f.run_up_days
            m = 1 + (f.run_up_peak - 1) * frac
            if best_run is None or m > best_run.multiplier:
                best_run = EventInfo(f.name, "run_up", delta, m, f.category_boost)
    return best_day or best_run


def event_multiplier(event: EventInfo | None, category: str) -> float:
    if event is None:
        return 1.0
    return 1.0 + (event.multiplier - 1.0) * event.category_boost.get(category, 1.0)


# ---------------------------------------------------------------------------
# Other demand assumptions (editable)
# ---------------------------------------------------------------------------
DOW_PRIOR = {0: 0.97, 1: 0.92, 2: 0.93, 3: 0.97, 4: 1.05, 5: 1.15, 6: 1.18}


def dom_prior(day: int) -> float:
    """Payday effect: busy at the start and end of the month, quiet mid-month."""
    if day <= 5:
        return 1.12
    if day >= 28:
        return 1.06
    if 10 <= day <= 20:
        return 0.97
    return 1.0


# (category, season) -> multiplier
SEASON_PRIOR = {
    ("Clothing", "Winter"): 1.20,
    ("Groceries", "Winter"): 1.05,
    ("Electronics", "Summer"): 1.10,
    ("Furniture", "Monsoon"): 0.95,
}


def calendar_prior(d: date, category: str, event: EventInfo | None) -> float:
    """Combined multiplier for one product category on one date."""
    if not config.USE_CALENDAR_PRIORS:
        return 1.0
    season = SEASON_NAMES[season_id(d.month)]
    return (
        DOW_PRIOR[d.weekday()]
        * dom_prior(d.day)
        * SEASON_PRIOR.get((category, season), 1.0)
        * event_multiplier(event, category)
    )


def busy_reason(d: date) -> str:
    """Short reason used in the banner when a normal day is still busy."""
    payday = d.day <= 5 or d.day >= 28
    weekend = d.weekday() >= 5
    if payday and weekend:
        return "Payday weekend"
    if payday:
        return "Payday week"
    if weekend:
        return "Weekend rush"
    return "Busy day"


def upcoming_festivals(start: date, days: int) -> list[dict]:
    """Festival days inside [start, start + days)."""
    out = []
    for i in range(days):
        d = start + timedelta(days=i)
        ev = active_event(d)
        if ev and ev.kind == "day":
            out.append({"name": ev.name, "date": d.isoformat(), "label": d.strftime("%d %b").lstrip("0")})
    return out
