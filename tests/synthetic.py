"""
Synthetic transaction generator matching the train.csv schema (DATA_DICTIONARY.md).

Used by the test-suite to run the *real* label_rules.py and the full pipeline end-to-end
without the confidential competition data. Behaviour is deliberately heterogeneous
(churners, late joiners, category adoption, seasonality) so every head has signal.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORY_ITEMS = {
    "Beverages": ["COKE 500ML", "FANTA ORANGE 440ML", "VALPRE STILL WATER 1L", "RED BULL 250ML"],
    "Snacking": ["SIMBA CHIPS SALT 120G", "BILTONG 80G", "NIKNAKS CHEESE 135G"],
    "Confectionary": ["CADBURY DAIRY MILK 80G", "BAR ONE 55G", "PS CHOCOLATE 48G"],
    "Food & beverage": ["CHICKEN PIE", "TOASTED CHEESE SANDWICH", "BOEREWORS ROLL"],
    "Coffee": ["CAPPUCCINO REGULAR", "AMERICANO LARGE"],
    "Groceries": ["CLOVER MILK 2L", "WHITE SUGAR 1KG", "SUNFLOWER OIL 750ML"],
    "Breads & rolls": ["ALBANY WHITE BREAD 700G", "HOTDOG ROLLS 6PK"],
    "Tobacco": ["PETER STUYVESANT 20", "LIGHTER CLIPPER"],
    "Ice-cream": ["MAGNUM CLASSIC", "CORNETTO VANILLA"],
    "Braai & Ice": ["ICE CUBES 2KG", "CHARCOAL 4KG"],
    "Lubricants": ["CASTROL GTX 5W30 1L", "BRAKE FLUID 500ML"],
    "Car Wash": ["CAR WASH EXPRESS"],
    "Automotive Products": ["WIPER BLADE 18IN"],
    "Krispy Kreme": ["KRISPY KREME GLAZED DOZEN"],
    "Carrier Bags": ["CARRIER BAG LARGE"],
    "Value added services": ["AIRTIME R30"],
    "Print media": ["SUNDAY TIMES"],
    "NA": ["MISC ITEM NA"],
}
STORE_CATS = [c for c in CATEGORY_ITEMS if c not in ("Car Wash",)]
FUEL_ITEMS = ["DIESEL 50PPM", "UNLEADED 95"]
ADJUSTMENTS = ["TOTAL/ROUNDING DISCO", "PENSIONER DISCOUNT", "WOW_PROMO"]
COLUMNS = [
    "DateTimeZA", "TransactionId", "LineItemSequence", "LocationId", "ID", "LineItemType",
    "TransactionTotalInclAmount", "StockItemId", "ItemName", "ItemCategoryLevel1",
    "ItemCategoryLevel2", "ItemCategoryLevel3", "Quantity", "TotalExclAmount",
    "TotalModifierAmount", "UnitSellingExclPrice",
]


def generate_transactions(n_customers: int = 400, seed: int = 42,
                          start: str = "2024-01-01", end: str = "2025-12-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    span_days = (t1 - t0).days
    items = {name: i for i, name in enumerate(sorted({x for v in CATEGORY_ITEMS.values() for x in v} | set(FUEL_ITEMS) | set(ADJUSTMENTS)))}
    rows = []
    txn_id = 10_000_000
    for c in range(n_customers):
        cid = str(4001000000000 + c + 1)
        rate = rng.gamma(2.0, 0.6)                          # visits / week
        p_fuel = rng.beta(4, 2)
        p_store = rng.beta(2, 3)
        litres_mu = rng.uniform(15, 55)
        first = int(rng.integers(0, span_days // 2)) if rng.random() < 0.3 else 0
        last = int(rng.integers(span_days // 2, span_days)) if rng.random() < 0.2 else span_days
        fav = rng.choice(STORE_CATS, size=int(rng.integers(1, 5)), replace=False)
        adopt_cat = rng.choice(STORE_CATS)
        adopt_day = int(rng.integers(0, span_days))
        stations = rng.choice(np.arange(100, 140), size=int(rng.integers(1, 4)), replace=False)
        n_visits = max(3, rng.poisson(rate * (last - first) / 7.0))
        days = np.sort(rng.uniform(first, last, size=n_visits))
        for d in days:
            # Seasonality: December/January boost, mid-year dip.
            month = (t0 + pd.Timedelta(days=float(d))).month
            if month in (6, 7) and rng.random() < 0.25:
                continue
            ts = t0 + pd.Timedelta(days=float(d))
            ts = ts.normalize() + pd.Timedelta(hours=int(rng.choice([7, 8, 12, 17, 18, 21])), minutes=int(rng.integers(0, 60)))
            if ts >= t1:
                continue
            txn_id += 1
            loc = str(rng.choice(stations))
            lines = []
            if rng.random() < p_fuel:
                litres = round(max(1.0, rng.normal(litres_mu, 8)), 2)
                item = FUEL_ITEMS[c % 2]
                lines.append(("Sale_Fuel", item, "Fuel", "Fuel", "Liquid fuel", litres, round(litres * 22.5, 2)))
            if rng.random() < 0.05:
                lines.append(("Sale_Item", "CAR WASH EXPRESS", "Services", "Car Wash", "Car Wash", 1.0, 60.0))
            if rng.random() < p_store or not lines:
                pool = list(fav) + ([adopt_cat] if d >= adopt_day else [])
                for _ in range(int(rng.integers(1, 4))):
                    cat = str(rng.choice(pool)) if rng.random() < 0.85 else str(rng.choice(STORE_CATS))
                    item = str(rng.choice(CATEGORY_ITEMS[cat]))
                    qty = float(rng.integers(1, 3))
                    price = round(float(rng.uniform(8, 70)), 2)
                    lines.append(("Sale_Item", item, "Shop", cat, cat + " L3", qty, round(qty * price, 2)))
                    if rng.random() < 0.02:  # same-basket partial return keeps net positive
                        lines.append(("Sale_Item", item, "Shop", cat, cat + " L3", -1.0, -round(price, 2)))
            if rng.random() < 0.08:
                adj = str(rng.choice(ADJUSTMENTS))
                lines.append(("Sale_Item", adj, "Adjustment", "Miscellaneous", "Adjustment", 1.0, -round(float(rng.uniform(0.1, 5)), 2)))
            total = round(sum(l[6] for l in lines) * 1.15, 2)
            for seq, (lt, name, l1, l2, l3, q, amt) in enumerate(lines, 1):
                rows.append((ts.strftime("%Y-%m-%d %H:%M:%S"), str(txn_id), str(seq), loc, cid, lt,
                             f"{total:.2f}", str(items[name]), name, l1, l2, l3,
                             f"{q:.2f}", f"{amt:.2f}", "0.00", f"{(amt / q if q else 0):.4f}"))
    return pd.DataFrame(rows, columns=COLUMNS)


def write_synthetic_raw_dir(dest: Path, real_raw_dir: Path, n_customers: int = 400, seed: int = 42) -> Path:
    """Create a raw dir with synthetic train.csv + test/sample built for the 2025-12-01 cutoff."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("label_rules.py", "label_config.json", "LABEL_RULES.md", "DATA_DICTIONARY.md"):
        shutil.copy(Path(real_raw_dir) / f, dest / f)
    tx = generate_transactions(n_customers=n_customers, seed=seed)
    tx.to_csv(dest / "train.csv", index=False)
    cfg = json.loads((dest / "label_config.json").read_text())
    n = tx.groupby("ID").TransactionId.nunique()
    ids = sorted(n[n >= cfg["minimum_historical_transactions"]].index)
    pd.DataFrame({"ID": ids}).to_csv(dest / "test.csv", index=False)
    pd.DataFrame({"ID": ids, "CLV_fuel": 0, "CLV_nonfuel": 0, "Opportunity": "Stable"}).to_csv(
        dest / "SampleSubmission.csv", index=False)
    return dest
