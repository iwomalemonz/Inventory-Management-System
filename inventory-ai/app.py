"""Run with:  python app.py   then open http://127.0.0.1:5000"""
from __future__ import annotations

import random
from datetime import date

import numpy as np
from flask import Flask, jsonify, render_template, request

from inventory import config, simulator
from inventory.calendar_features import upcoming_festivals
from inventory.forecaster import Forecaster
from inventory.models import DemandModels

app = Flask(__name__)

MODELS = DemandModels.load_or_train()
FORECASTER = Forecaster(MODELS)
LOOKAHEAD = config.LEAD_TIME_RANGE[1] + config.SAFETY_DAYS + 2   # extra forecast days
RISK_CACHE: dict = {}


def _clamp_days(value) -> int:
    try:
        days = int(value)
    except (TypeError, ValueError):
        days = 7
    return max(config.MIN_DAYS, min(config.MAX_DAYS, days))


def _truthy(value, default=True) -> bool:
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "yes", "on")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/bootstrap")
def bootstrap():
    start = date.today()
    return jsonify({
        "start": start.isoformat(),
        "capacity": config.CAPACITY,
        "seconds_per_day": config.SECONDS_PER_DAY,
        "min_days": config.MIN_DAYS,
        "max_days": config.MAX_DAYS,
        "risk_runs": config.RISK_RUNS,
        "calendar_priors": config.USE_CALENDAR_PRIORS,
        "products": [
            {"id": p["id"], "name": p["name"], "category": p["category"],
             "tier": p["tier"], "price": p["price"]}
            for p in MODELS.products
        ],
        "metrics": MODELS.metrics,
    })


@app.route("/api/forecast")
def forecast():
    days = _clamp_days(request.args.get("days"))
    smart = _truthy(request.args.get("smart"))
    start = date.today()

    lam, infos = FORECASTER.forecast(start, 60 + LOOKAHEAD)
    key = (start, days, smart)
    if key not in RISK_CACHE:
        RISK_CACHE.clear() if len(RISK_CACHE) > 200 else None
        RISK_CACHE[key] = simulator.stockout_risk(
            lam[: days + LOOKAHEAD], infos, MODELS.products, days, smart,
            config.RISK_RUNS, seed=1000)

    return jsonify({
        "days": [
            {
                "iso": infos[i]["iso"],
                "weekday": infos[i]["weekday"],
                "date_label": infos[i]["date_label"],
                "headline": infos[i]["headline"],
                "level": infos[i]["level"],
                "event": infos[i]["event"],
                "index": infos[i]["index"],
                "units": round(float(lam[i].sum()), 1),
            }
            for i in range(days)
        ],
        "risk": RISK_CACHE[key],
        "festivals": upcoming_festivals(start, days),
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    body = request.get_json(silent=True) or {}
    days = _clamp_days(body.get("days"))
    smart = _truthy(body.get("smart"))
    seed = int(body.get("seed", random.randrange(1, 10**9)))
    start = date.today()

    lam, infos = FORECASTER.forecast(start, 60 + LOOKAHEAD)
    result = simulator.run(lam[: days + LOOKAHEAD], infos[:days], MODELS.products,
                           days, np.random.default_rng(seed), smart=smart, record=True)
    return jsonify({
        "days": result["days"],
        "seed": seed,
        "smart": smart,
        "totals": {
            "sold": int(result["sold"].sum()),
            "lost": int(result["lost"].sum()),
            "restocked": int(result["restocked"].sum()),
        },
    })


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
