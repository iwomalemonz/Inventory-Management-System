"""Day-by-day inventory simulation.

For every day:
  1. deliveries ordered earlier arrive (green "+N product" lines)
  2. customers buy: units ~ Poisson(forecast demand), split into small purchases
     spread across the day (red "-N x product" lines)
  3. if stock runs out, the missed units are recorded as lost sales
  4. at closing time each product is re-checked and a restock may be ordered

Restocking policies:
  smart=True  -> order when stock + on-order would not cover the forecast demand
                 for the next (max lead time + safety days). Uses the ML forecast,
                 so it orders earlier before festivals.
  smart=False -> order when stock drops below 30% of capacity.
"""
from __future__ import annotations

import numpy as np

from . import config


def _split(units: int, rng: np.random.Generator) -> list[int]:
    """Split a day's units into purchase lines (each at least 1 unit)."""
    n = max(1, min(units, round(units / config.AVG_ORDER_SIZE)))
    extra = rng.multinomial(units - n, [1.0 / n] * n)
    return [int(1 + e) for e in extra]


def run(lam: np.ndarray, infos: list[dict], products: list[dict], days: int,
        rng: np.random.Generator, smart: bool = True, record: bool = True) -> dict:
    """Simulate `days` days. `lam` needs a few extra rows after `days` (look-ahead)."""
    P = len(products)
    cap = config.CAPACITY
    lead_lo, lead_hi = config.LEAD_TIME_RANGE
    lookahead = lead_hi + config.SAFETY_DAYS

    stock = np.full(P, cap, dtype=int)
    pending: list[tuple[int, int, int]] = []        # (arrival day, product, qty)
    sold = np.zeros(P, dtype=int)
    lost = np.zeros(P, dtype=int)
    restocked = np.zeros(P, dtype=int)
    had_stockout = np.zeros(P, dtype=bool)
    day_records = []

    for d in range(days):
        events = []

        # 1. deliveries
        waiting = []
        for arrive, p, qty in pending:
            if arrive <= d:
                add = int(min(qty, cap - stock[p]))
                if add > 0:
                    stock[p] += add
                    restocked[p] += add
                    if record:
                        events.append({"t": round(float(rng.uniform(0.03, 0.12)), 3),
                                       "type": "restock", "p": p, "qty": add})
            else:
                waiting.append((arrive, p, qty))
        pending = waiting

        # 2. customers
        day_factor = rng.lognormal(0.0, config.DAY_NOISE_SIGMA)
        lam_day = lam[d] * day_factor * rng.lognormal(0.0, config.PRODUCT_NOISE_SIGMA, size=P)
        units = rng.poisson(lam_day)
        day_sold = day_lost = 0

        for p in range(P):
            u = int(units[p])
            if u == 0:
                continue
            sizes = _split(u, rng)
            times = np.sort(rng.beta(2.2, 2.4, size=len(sizes)) * 0.88 + 0.08)
            lost_here = 0
            first_miss_t = None
            for t, qty in zip(times, sizes):
                got = int(min(qty, stock[p]))
                if got > 0:
                    stock[p] -= got
                    sold[p] += got
                    day_sold += got
                    if record:
                        events.append({"t": round(float(t), 3), "type": "sale", "p": p, "qty": got})
                if got < qty:
                    lost_here += qty - got
                    if first_miss_t is None:
                        first_miss_t = float(t)
            if lost_here:
                lost[p] += lost_here
                day_lost += lost_here
                had_stockout[p] = True
                if record:
                    events.append({"t": round(first_miss_t + 0.001, 3), "type": "stockout",
                                   "p": p, "qty": lost_here})

        # 4. closing-time restock decisions
        on_order = np.zeros(P, dtype=int)
        for _, p, qty in pending:
            on_order[p] += qty
        for p in range(P):
            position = int(stock[p] + on_order[p])
            if smart:
                need = lam[d + 1: d + 1 + lookahead, p].sum() * config.SAFETY_FACTOR
            else:
                need = config.BASIC_REORDER_FRACTION * cap
            if position <= need:
                qty = cap - position
                step = config.ORDER_ROUNDING
                qty = int(round(qty / step) * step)
                if qty >= config.MIN_ORDER:
                    lead = int(rng.integers(lead_lo, lead_hi + 1))
                    pending.append((d + lead, p, qty))
                    on_order[p] += qty

        if record:
            events.sort(key=lambda e: e["t"])
            info = infos[d]
            day_records.append({
                "day": d + 1,
                "iso": info["iso"],
                "weekday": info["weekday"],
                "date_label": info["date_label"],
                "headline": info["headline"],
                "level": info["level"],
                "event": info["event"],
                "events": events,
                "end_stock": [int(x) for x in stock],
                "on_order": [int(x) for x in on_order],
                "sold": int(day_sold),
                "lost": int(day_lost),
            })

    return {
        "days": day_records,
        "sold": sold, "lost": lost, "restocked": restocked,
        "had_stockout": had_stockout,
    }


def stockout_risk(lam: np.ndarray, infos: list[dict], products: list[dict],
                  days: int, smart: bool, runs: int, seed: int) -> list[dict]:
    """Run the simulation many times; report how often each product sells out."""
    P = len(products)
    hits = np.zeros(P)
    lost = np.zeros(P)
    for r in range(runs):
        res = run(lam, infos, products, days, np.random.default_rng(seed + r),
                  smart=smart, record=False)
        hits += res["had_stockout"]
        lost += res["lost"]

    # runway = days until stock would hit zero if you never restocked
    horizon = min(len(lam), 60)
    cum = np.cumsum(lam[:horizon], axis=0)
    out = []
    for p, prod in enumerate(products):
        reached = np.where(cum[:, p] >= config.CAPACITY)[0]
        out.append({
            "id": p,
            "name": prod["name"],
            "risk": round(float(hits[p] / runs), 3),
            "avg_lost": round(float(lost[p] / runs), 1),
            "runway_days": int(reached[0] + 1) if len(reached) else None,
        })
    out.sort(key=lambda r: (-r["risk"], r["runway_days"] or 999))
    return out
