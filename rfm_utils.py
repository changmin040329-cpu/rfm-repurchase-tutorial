"""Shared helpers for the "Does RFM really predict the future?" tutorial.

Everyone (A, B, C) imports this module, so the three methods are compared on
exactly the same customers, snapshot dates, labels and score format.

Pipeline covered here (steps 0-3 of the project plan):
    download_raw -> load_raw -> clean_transactions -> invoice_table
        -> build_features(snapshot) / build_labels(snapshot) -> make_dataset(snapshot)
    plus the shared score format and the lift@k metric used for H4.

Rule of thumb that must never be broken (data leakage):
    FEATURES use only invoices strictly BEFORE the snapshot date.
    LABELS use only invoices in [snapshot, snapshot + LABEL_DAYS).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants the whole team shares. Change them here, nowhere else.
# --------------------------------------------------------------------------- #
SEED = 42
LABEL_DAYS = 90  # repurchase window after the snapshot

# Candidate snapshots (checked on 2026-11-16: every one has 4,500+ customers
# with history and its 90-day label window lies inside the data).
SNAPSHOTS: Dict[str, str] = {
    "S1": "2011-03-01",
    "S2": "2011-06-01",
    "S3": "2011-09-01",
}

DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"  # git-ignored: raw data is never committed
OUTPUT_DIR = PROJECT_DIR / "outputs"   # score files handed from B/C to A (committed)
RAW_XLSX = "online_retail_II.xlsx"
CACHE_CSV = "online_retail_II_raw.csv.gz"

# Columns of the cleaned invoice table (the team-wide schema).
INVOICE_COLUMNS = ["customer_id", "invoice", "invoice_date", "amount"]
# Columns of the shared score table every method must output.
SCORE_COLUMNS = ["customer_id", "snapshot", "method", "score"]

_RENAME = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


# --------------------------------------------------------------------------- #
# Step 0: get the data (nobody commits the raw file; they run this instead)
# --------------------------------------------------------------------------- #
def download_raw(data_dir: Path = DATA_DIR) -> Path:
    """Download the UCI Online Retail II zip and extract the xlsx (about 46 MB).

    Check the dataset's license on the UCI page before redistributing anything.
    Returns the path of the extracted xlsx; does nothing if it already exists.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / RAW_XLSX
    if target.exists():
        return target
    print(f"Downloading {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL, timeout=300) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        # Extract only the one file we expect (never trust member paths).
        zf.extract(RAW_XLSX, path=data_dir)
    return target


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return the raw transactions (1,067,371 rows, both sheets stacked).

    Reading the xlsx takes ~30 s, so the first call caches a gzip CSV and every
    later call reads that cache.
    """
    data_dir = Path(data_dir)
    cache = data_dir / CACHE_CSV
    if cache.exists():
        raw = pd.read_csv(
            cache, parse_dates=["InvoiceDate"], dtype={"Invoice": str, "StockCode": str}
        )
        return raw
    xlsx = download_raw(data_dir)
    sheets = pd.read_excel(xlsx, sheet_name=None, engine="openpyxl")
    raw = pd.concat(sheets.values(), ignore_index=True)
    raw["Invoice"] = raw["Invoice"].astype(str)
    raw["StockCode"] = raw["StockCode"].astype(str)
    raw.to_csv(cache, index=False)
    return raw


# --------------------------------------------------------------------------- #
# Step 1: cleaning (rules are documented because they change the results)
# --------------------------------------------------------------------------- #
# StockCodes that are not goods: fees, postage, manual entries, adjustments, tests.
# Found by listing every code that is not "5 digits + optional letters" (2026-11-26).
# SP1002 (a real chalkboard product) is deliberately NOT on this list.
NON_PRODUCT_CODES = frozenset({
    "POST", "DOT", "C2",            # postage / carriage
    "M",                            # manual entry: what was bought is unknown
    "ADJUST", "ADJUST2", "D",       # bookkeeping adjustments, discount
    "BANK CHARGES", "AMAZONFEE",    # fees
    "TEST001", "TEST002",           # test rows
    "PADS",                         # price ~0, not a sale
    "B", "S", "CRUK",               # bad debt / samples / charity (removed earlier, listed for safety)
})


def clean_transactions(
    raw: pd.DataFrame, drop_duplicates: bool = True, drop_non_product: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw rows and return (clean_rows, report).

    Rules, applied in this order (the report counts the rows each one removes):
      1. cancellations: invoice numbers starting with "C" are returns/cancels
      2. missing Customer ID: we cannot tell who bought, so RFM is impossible
      3. non-positive quantity or price: stock adjustments, not real purchases
      4. non-product lines (optional, default on): StockCodes in NON_PRODUCT_CODES.
         An invoice made only of such lines (e.g. postage only) disappears, so it
         is not counted as an order or as a repurchase.
      5. exact duplicate rows (optional, default on): treated as double entries.
         They could also be a customer scanning the same item twice at the same
         second, so the team should look at the report before deciding.
    Adds an `amount` column (quantity x price) and a clean integer customer_id.
    """
    df = raw.rename(columns=_RENAME).copy()
    report = [("raw rows", len(df))]

    is_cancel = df["invoice"].astype(str).str.startswith("C")
    report.append(("removed: cancellations (invoice starts with C)", int(is_cancel.sum())))
    df = df[~is_cancel]

    missing_id = df["customer_id"].isna()
    report.append(("removed: missing customer id", int(missing_id.sum())))
    df = df[~missing_id]

    bad = (df["quantity"] <= 0) | (df["price"] <= 0)
    report.append(("removed: quantity <= 0 or price <= 0", int(bad.sum())))
    df = df[~bad]

    if drop_non_product:
        non_product = df["stock_code"].astype(str).str.strip().str.upper().isin(NON_PRODUCT_CODES)
        report.append(("removed: non-product lines (postage, fees, manual)", int(non_product.sum())))
        df = df[~non_product]

    if drop_duplicates:
        dup = df.duplicated()
        report.append(("removed: exact duplicate rows", int(dup.sum())))
        df = df[~dup]

    df = df.assign(
        customer_id=df["customer_id"].astype("int64"),
        amount=df["quantity"] * df["price"],
    ).reset_index(drop=True)
    report.append(("clean rows", len(df)))
    return df, pd.DataFrame(report, columns=["step", "rows"])


def invoice_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Collapse line items to one row per (customer, invoice).

    This is the team-wide base table: columns == INVOICE_COLUMNS. Frequency (F)
    counts invoices, not line items, so everything downstream starts here.
    """
    inv = (
        clean.groupby(["customer_id", "invoice"], as_index=False)
        .agg(invoice_date=("invoice_date", "min"), amount=("amount", "sum"))
        .sort_values(["customer_id", "invoice_date"])
        .reset_index(drop=True)
    )
    return inv[INVOICE_COLUMNS]


# --------------------------------------------------------------------------- #
# Step 3: snapshot split (features from the past, labels from the future)
# --------------------------------------------------------------------------- #
def _ts(snapshot) -> pd.Timestamp:
    return pd.Timestamp(snapshot)


def build_features(inv: pd.DataFrame, snapshot) -> pd.DataFrame:
    """Per-customer features from invoices strictly BEFORE `snapshot`.

    Index: customer_id (only customers with at least one invoice before the date).
    Columns:
      R                 days since the last invoice (smaller = more recent)
      F                 number of invoices
      M                 total spend
      avg_order_value   M / F
      T                 days since the first invoice (customer "age")
      recency_bgnbd     days between first and last invoice  (BG/NBD "recency";
                        NOT the same as R - see the plan, section 3-7)
      frequency_bgnbd   F - 1, i.e. repeat purchases only
      repeat_avg_amount mean invoice amount excluding the first invoice (0 if none);
                        the Gamma-Gamma model uses this
    """
    snap = _ts(snapshot)
    past = inv[inv["invoice_date"] < snap]
    g = past.groupby("customer_id")
    feats = g.agg(
        first=("invoice_date", "min"),
        last=("invoice_date", "max"),
        F=("invoice", "nunique"),
        M=("amount", "sum"),
    )
    first_amount = (
        past.sort_values("invoice_date").groupby("customer_id")["amount"].first()
    )
    out = pd.DataFrame(index=feats.index)
    out["R"] = (snap - feats["last"]).dt.days
    out["F"] = feats["F"]
    out["M"] = feats["M"]
    out["avg_order_value"] = feats["M"] / feats["F"]
    out["T"] = (snap - feats["first"]).dt.days
    out["recency_bgnbd"] = (feats["last"] - feats["first"]).dt.days
    out["frequency_bgnbd"] = feats["F"] - 1
    repeats = (feats["F"] - 1).replace(0, np.nan)
    out["repeat_avg_amount"] = ((feats["M"] - first_amount) / repeats).fillna(0.0)
    out.index.name = "customer_id"
    return out


def build_labels(
    inv: pd.DataFrame, snapshot, days: int = LABEL_DAYS
) -> pd.DataFrame:
    """Per-customer outcome from invoices in [snapshot, snapshot + days).

    Returns only customers who bought in the window, with columns
    future_orders and future_revenue. Use make_dataset() to align this with the
    features (customers who did not buy get 0 / repurchase = 0 there).

    Raises ValueError if the window extends past the end of the data: the label
    would silently count "no purchase" for days we never observed.
    """
    snap = _ts(snapshot)
    end = snap + pd.Timedelta(days=days)
    last_day = inv["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    if end > last_day:
        raise ValueError(
            f"Label window [{snap.date()}, {end.date()}) goes past the data end "
            f"({inv['invoice_date'].max()}). Choose an earlier snapshot."
        )
    future = inv[(inv["invoice_date"] >= snap) & (inv["invoice_date"] < end)]
    lab = future.groupby("customer_id").agg(
        future_orders=("invoice", "nunique"), future_revenue=("amount", "sum")
    )
    lab.index.name = "customer_id"
    return lab


def make_dataset(inv: pd.DataFrame, snapshot, days: int = LABEL_DAYS) -> pd.DataFrame:
    """Features + labels for one snapshot: the table every method starts from.

    One row per customer who had history before the snapshot. Extra columns:
      repurchase       1 if the customer bought at least once in the window
      future_orders    number of invoices in the window (0 if none)
      future_revenue   spend in the window (0 if none); use THIS, not M, for H2
      snapshot         the snapshot date (Timestamp)
    """
    feats = build_features(inv, snapshot)
    labels = build_labels(inv, snapshot, days).reindex(feats.index)
    labels = labels.fillna(0.0)
    data = feats.join(labels)
    data["future_orders"] = data["future_orders"].astype(int)
    data["repurchase"] = (data["future_orders"] > 0).astype(int)
    data["snapshot"] = _ts(snapshot)
    return data


def assert_no_leakage(inv: pd.DataFrame, data: pd.DataFrame, snapshot) -> None:
    """Sanity checks; call this in every notebook after make_dataset().

    Re-derives R and F from scratch using only pre-snapshot invoices and
    compares them with the dataset, and confirms the label never uses a date
    before the snapshot.
    """
    snap = _ts(snapshot)
    past = inv[inv["invoice_date"] < snap]
    assert set(data.index) == set(past["customer_id"]), "customer set != pre-snapshot buyers"
    check = past.groupby("customer_id").agg(
        last=("invoice_date", "max"), F=("invoice", "nunique")
    )
    assert (data["F"] == check["F"].reindex(data.index)).all(), "F uses a wrong window"
    r_expected = (snap - check["last"]).dt.days.reindex(data.index)
    assert (data["R"] == r_expected).all(), "R uses a wrong window"
    assert (data["R"] >= 0).all(), "negative R means a future invoice leaked into features"
    assert data["repurchase"].isin([0, 1]).all()


# --------------------------------------------------------------------------- #
# Shared score format (all three methods output this) and the H4 metric
# --------------------------------------------------------------------------- #
def to_score_frame(customer_ids, scores, snapshot, method: str) -> pd.DataFrame:
    """Put any method's output in the team format. Higher score = more likely to
    repurchase. Rule-based segments must be turned into numbers first (for
    example the sum of the R, F and M quintile scores)."""
    return pd.DataFrame(
        {
            "customer_id": np.asarray(customer_ids),
            "snapshot": _ts(snapshot),
            "method": method,
            "score": np.asarray(scores, dtype=float),
        }
    )[SCORE_COLUMNS]


def validate_score_frame(scores: pd.DataFrame, data: pd.DataFrame) -> None:
    """Raise AssertionError unless `scores` covers exactly the dataset's customers."""
    assert list(scores.columns) == SCORE_COLUMNS, f"columns must be {SCORE_COLUMNS}"
    assert scores["customer_id"].is_unique, "duplicate customer_id in one score frame"
    assert set(scores["customer_id"]) == set(data.index), "customers differ from the dataset"
    assert scores["score"].notna().all(), "NaN scores"
    assert scores["method"].nunique() == 1 and scores["snapshot"].nunique() == 1


def baseline_recency_score(data: pd.DataFrame) -> pd.Series:
    """The baseline every method must beat (H3): "bought recently" only."""
    return -data["R"].astype(float)


def lift_at_k(y_true, score, k: float = 0.10, seed: int = SEED) -> Dict[str, float]:
    """Targeting efficiency of a score when only the top k share is contacted.

    Returns
      capture   share of all repurchasers found inside the top-k group
      precision repurchase rate inside the top-k group
      lift      precision / overall repurchase rate (1.0 = no better than random)
    Ties (common for rule-based scores) are broken at random with a fixed seed so
    that a score with many equal values is not judged by row order.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    n = len(y)
    n_top = max(1, int(round(n * k)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    ranked = order[np.argsort(-s[order], kind="stable")]
    top = y[ranked[:n_top]]
    base = y.mean()
    precision = top.mean()
    return {
        "capture": float(top.sum() / y.sum()) if y.sum() > 0 else float("nan"),
        "precision": float(precision),
        "lift": float(precision / base) if base > 0 else float("nan"),
    }


def evaluate_scores(y_true, score, k: float = 0.10) -> Dict[str, float]:
    """One shared report card: AUC, PR-AUC, lift@k, capture@k, precision@k.

    Every method (rules, ML, BG/NBD) is judged with this same function so the
    numbers in the final comparison table are comparable. Needs scikit-learn.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    out = {
        "auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "base_rate": float(y.mean()),
    }
    out.update({f"{name}@{int(k * 100)}": value for name, value in lift_at_k(y, s, k).items()})
    return out


def save_score_frame(scores: pd.DataFrame, method: str, out_dir: Path = OUTPUT_DIR) -> Path:
    """Write one method's scores (all snapshots) to outputs/scores_<method>.csv."""
    assert list(scores.columns) == SCORE_COLUMNS, f"columns must be {SCORE_COLUMNS}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"scores_{method}.csv"
    scores.to_csv(path, index=False)
    return path


def load_score_frames(out_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Read every outputs/scores_*.csv into one long frame (A uses this to compare methods)."""
    files = sorted(Path(out_dir).glob("scores_*.csv"))
    if not files:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    return pd.concat([pd.read_csv(f, parse_dates=["snapshot"]) for f in files], ignore_index=True)
