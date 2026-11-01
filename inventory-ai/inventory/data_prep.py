"""Turn the raw CSV into one row per (day, product) for training.

The dataset has 5 product categories and no SKU column. To get 10 products
for the app, each category is split into Budget and Premium using the
category's median price (price >= median -> Premium).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CATEGORY_ORDER = ["Electronics", "Clothing", "Groceries", "Toys", "Furniture"]
TIER_ORDER = ["Budget", "Premium"]


def load_daily() -> pd.DataFrame:
    """Daily units sold per product: columns date, product, category, tier, units, avg_price."""
    if not config.DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.DATA_PATH}. "
            "Put unified_sales_dataset.csv inside the data/ folder."
        )
    df = pd.read_csv(
        config.DATA_PATH,
        usecols=["date", "category", "price", "quantity", "source"],
    )
    df = df[df["source"] == config.DATA_SOURCE].dropna(subset=["price", "quantity"]).copy()
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df[df["category"].isin(CATEGORY_ORDER)]

    median_price = df.groupby("category")["price"].transform("median")
    df["tier"] = np.where(df["price"] >= median_price, "Premium", "Budget")
    df["product"] = df["tier"] + " " + df["category"]

    daily = (
        df.groupby(["date", "product", "category", "tier"], as_index=False)
        .agg(units=("quantity", "sum"), avg_price=("price", "mean"))
        .sort_values(["date", "product"])
        .reset_index(drop=True)
    )
    return daily


def build_products(daily: pd.DataFrame) -> list[dict]:
    """Stable, ordered list of the 10 products with their average price."""
    products = []
    for cat in CATEGORY_ORDER:
        for tier in TIER_ORDER:
            name = f"{tier} {cat}"
            rows = daily[daily["product"] == name]
            if rows.empty:
                continue
            products.append(
                {
                    "id": len(products),
                    "name": name,
                    "category": cat,
                    "tier": tier,
                    "price": round(float(rows["avg_price"].mean()), 2),
                    "mean_units": float(rows["units"].mean()),
                }
            )
    return products
