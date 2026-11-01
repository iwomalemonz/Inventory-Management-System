"""Daily demand forecast = Random Forest x Naive Bayes nudge x calendar assumptions.

    expected units = RF units
                     x clip(P_high(NB) / base_rate, 0.85, 1.20)
                     x calendar prior (weekday, payday, season, festival)
                     x DEMAND_SCALE

Each model's influence is weighted by how much it beat a plain average on the
held-out test days (its "skill"). A model with no skill gets weight 0, so it
cannot inject noise into the forecast. On this dataset both weights come out
near 0 because the data has almost no calendar pattern; with real shop data
the weights rise automatically.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from . import config
from .calendar_features import active_event, busy_reason, calendar_prior
from .models import DemandModels


class Forecaster:
    def __init__(self, models: DemandModels):
        self.models = models
        self.products = models.products
        self._baseline = np.array([p["mean_units"] for p in self.products]) * config.DEMAND_SCALE
        self._mean_units = np.array([p["mean_units"] for p in self.products])
        m = models.metrics
        self.nb_weight = float(np.clip((m["naive_bayes"]["auc"] - 0.5) * 2, 0, 1))
        self.rf_weight = float(np.clip(m["random_forest"]["r2"], 0, 1))
        self._cache: dict = {}

    def forecast(self, start: date, n_days: int):
        """Returns (lam, infos).

        lam   - array (n_days, n_products) of expected units per day
        infos - list of dicts describing each day (date, event, headline, level)
        """
        key = (start, n_days)
        if key in self._cache:
            return self._cache[key]

        dates = [start + timedelta(days=i) for i in range(n_days)]
        rf_units, p_high = self.models.predict(dates)
        raw_nb = np.clip(p_high / self.models.base_rate[None, :], 0.85, 1.20)
        nb_factor = 1.0 + self.nb_weight * (raw_nb - 1.0)
        rf_units = self._mean_units[None, :] + self.rf_weight * (rf_units - self._mean_units[None, :])

        prior = np.ones_like(rf_units)
        infos = []
        for i, d in enumerate(dates):
            event = active_event(d)
            for j, prod in enumerate(self.products):
                prior[i, j] = calendar_prior(d, prod["category"], event)
            infos.append({"date": d, "event": event})

        lam = rf_units * nb_factor * prior * config.DEMAND_SCALE
        index = (lam / self._baseline[None, :]).mean(axis=1)   # 1.0 = a normal day

        for i, info in enumerate(infos):
            info.update(self._describe(info["date"], info["event"], float(index[i])))
            info.pop("date")   # keep infos JSON-friendly (iso string is in there)

        self._cache[key] = (lam, infos)
        return lam, infos

    @staticmethod
    def _describe(d: date, event, index: float) -> dict:
        if index >= config.HIGH_LEVEL:
            level, level_text = "high", "High demand"
        elif index < config.LOW_LEVEL:
            level, level_text = "low", "Low demand"
        else:
            level, level_text = "normal", "Normal"

        if event is not None and event.kind == "day":
            headline = f"It's {event.name}! {level_text}" if level == "high" else f"{event.name} · {level_text}"
        elif event is not None:
            when = "tomorrow" if event.days_until == 1 else f"in {event.days_until} days"
            headline = f"{event.name} {when} · {level_text}"
        elif level == "high":
            headline = f"{busy_reason(d)} · High demand"
        else:
            headline = level_text

        return {
            "weekday": d.strftime("%A"),
            "date_label": d.strftime("%d %b %Y").lstrip("0"),
            "iso": d.isoformat(),
            "level": level,
            "headline": headline,
            "event": event.name if event else None,
            "index": round(index, 2),
        }
