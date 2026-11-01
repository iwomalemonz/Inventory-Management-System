"""Central settings. Change numbers here, not in the other files."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "unified_sales_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "demand_models.joblib"

# ---- Simulation playback ---------------------------------------------------
SECONDS_PER_DAY = 5          # one simulated day always lasts 5 seconds on screen
MIN_DAYS, MAX_DAYS = 1, 30   # slider range

# ---- Shop size -------------------------------------------------------------
# The dataset logs ~1,300 units/day per product line (a big chain).
# DEMAND_SCALE shrinks that to a single-shop size so the log stays readable.
DEMAND_SCALE = 0.005          # ~6-7 units/day per product on a normal day
CAPACITY = 120                # units each product holds when "full"
AVG_ORDER_SIZE = 3.0          # average units per purchase line in the log

# ---- Restocking ------------------------------------------------------------
LEAD_TIME_RANGE = (1, 4)      # supplier delivers 1-4 days after an order
SAFETY_DAYS = 2               # extra days of cover in forecast-aware mode
SAFETY_FACTOR = 1.2           # multiply forecast demand by this when ordering
BASIC_REORDER_FRACTION = 0.30 # "basic" mode: reorder when stock < 30% of capacity
MIN_ORDER = 10                # don't place orders smaller than this
ORDER_ROUNDING = 5            # order sizes are rounded to multiples of 5

# ---- Randomness in the simulation -----------------------------------------
DAY_NOISE_SIGMA = 0.18        # whole-shop good/bad day
PRODUCT_NOISE_SIGMA = 0.10    # per-product wobble

# ---- Demand labels shown in the corner banner ------------------------------
HIGH_LEVEL = 1.35             # demand index >= this  -> "High demand"
LOW_LEVEL = 0.95              # demand index <  this  -> "Low demand"

# ---- Machine learning ------------------------------------------------------
DATA_SOURCE = "Retail Store Inventory (Kaggle)"   # the 2-year daily sales rows
HIGH_DEMAND_QUANTILE = 0.70   # Naive Bayes target: day in the top 30% for that product
TEST_FRACTION = 0.20          # last 20% of days are held out to measure accuracy

# ---- Risk estimate ---------------------------------------------------------
RISK_RUNS = 120               # simulated runs used for the stock-out risk chart

# ---- Calendar assumptions --------------------------------------------------
# The uploaded dataset shows almost no weekday / month / holiday pattern
# (see README). If True, the editable assumptions in calendar_features.py
# (weekend lift, payday lift, festivals, seasons) are applied on top of the
# ML forecast. Set to False to run on the ML models alone.
USE_CALENDAR_PRIORS = True
