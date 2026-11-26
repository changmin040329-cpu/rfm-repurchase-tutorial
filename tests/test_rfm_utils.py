"""Leakage and format tests on a tiny hand-made dataset (no download needed).

Run from the project folder:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rfm_utils as ru  # noqa: E402

SNAP = "2011-06-01"


def _raw():
    """Raw-format rows covering every cleaning rule and the snapshot boundaries."""
    rows = [
        # customer 1: bought twice before the snapshot, once in the window
        ("100", "A", "x", 2, "2011-01-10 10:00", 5.0, 1.0, "UK"),
        ("101", "A", "x", 1, "2011-05-20 10:00", 10.0, 1.0, "UK"),
        ("102", "A", "x", 1, "2011-07-01 10:00", 7.0, 1.0, "UK"),
        # customer 2: one purchase before; buys exactly AT the snapshot (counts as future)
        ("200", "A", "x", 1, "2011-04-01 09:00", 20.0, 2.0, "UK"),
        ("201", "A", "x", 1, "2011-06-01 00:00", 3.0, 2.0, "UK"),
        # customer 3: buys exactly 90 days after the snapshot (window is half open: excluded)
        ("300", "A", "x", 1, "2011-02-01 09:00", 8.0, 3.0, "UK"),
        ("301", "A", "x", 1, "2011-08-30 00:00", 9.0, 3.0, "UK"),  # 2011-06-01 + 90d = 2011-08-30
        # customer 4: only a cancellation and a bad row -> must not appear at all
        ("C400", "A", "x", -1, "2011-03-01 09:00", 5.0, 4.0, "UK"),
        ("401", "A", "x", -2, "2011-03-02 09:00", 5.0, 4.0, "UK"),
        # missing customer id -> dropped
        ("500", "A", "x", 1, "2011-03-03 09:00", 5.0, np.nan, "UK"),
        # exact duplicate of the first row -> dropped when drop_duplicates=True
        ("100", "A", "x", 2, "2011-01-10 10:00", 5.0, 1.0, "UK"),
        # postage line on customer 1's first invoice -> dropped, amount unchanged
        ("100", "POST", "POSTAGE", 1, "2011-01-10 10:00", 18.0, 1.0, "UK"),
        # customer 3: a postage-only invoice inside the label window -> not a repurchase
        ("302", "POST", "POSTAGE", 1, "2011-07-01 10:00", 18.0, 3.0, "UK"),
        # filler invoice so the data extends well past the label window
        ("900", "A", "x", 1, "2011-09-30 10:00", 1.0, 9.0, "UK"),
    ]
    return pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
                 "Price", "Customer ID", "Country"],
    ).assign(InvoiceDate=lambda d: pd.to_datetime(d["InvoiceDate"]))


@pytest.fixture()
def inv():
    clean, report = ru.clean_transactions(_raw())
    return ru.invoice_table(clean)


def test_cleaning_rules_and_report():
    clean, report = ru.clean_transactions(_raw())
    steps = dict(zip(report["step"], report["rows"]))
    assert steps["removed: cancellations (invoice starts with C)"] == 1
    assert steps["removed: missing customer id"] == 1
    assert steps["removed: quantity <= 0 or price <= 0"] == 1
    assert steps["removed: non-product lines (postage, fees, manual)"] == 2
    assert steps["removed: exact duplicate rows"] == 1
    assert not clean["stock_code"].isin(ru.NON_PRODUCT_CODES).any()
    assert 4 not in set(clean["customer_id"])
    assert clean["customer_id"].dtype == "int64"
    assert (clean["amount"] > 0).all()


def test_keep_duplicates_option():
    clean, _ = ru.clean_transactions(_raw(), drop_duplicates=False)
    assert len(clean) == len(ru.clean_transactions(_raw())[0]) + 1


def test_invoice_table_schema(inv):
    assert list(inv.columns) == ru.INVOICE_COLUMNS
    # customer 1, invoice 100: 2 x 5.0 = 10.0 (duplicate removed)
    row = inv[(inv.customer_id == 1) & (inv.invoice == "100")].iloc[0]
    assert row["amount"] == 10.0


def test_features_use_only_the_past(inv):
    feats = ru.build_features(inv, SNAP)
    assert set(feats.index) == {1, 2, 3}  # customer 9 first buys after the snapshot
    c1 = feats.loc[1]
    assert c1["F"] == 2 and c1["M"] == 20.0
    assert c1["R"] == (pd.Timestamp(SNAP) - pd.Timestamp("2011-05-20 10:00")).days
    assert c1["frequency_bgnbd"] == 1
    assert c1["repeat_avg_amount"] == 10.0  # the 2nd invoice only
    assert feats.loc[2, "F"] == 1  # the invoice AT the snapshot must not count
    assert feats.loc[2, "repeat_avg_amount"] == 0.0
    assert (feats["R"] >= 0).all()


def test_label_window_is_half_open(inv):
    data = ru.make_dataset(inv, SNAP)
    assert data.loc[1, "repurchase"] == 1  # bought inside the window
    assert data.loc[2, "repurchase"] == 1  # bought exactly at the snapshot
    assert data.loc[3, "repurchase"] == 0  # bought exactly snapshot + 90d: outside
    assert data.loc[1, "future_revenue"] == 7.0


def test_label_window_past_data_end_raises(inv):
    with pytest.raises(ValueError):
        ru.build_labels(inv, "2011-09-15")  # window runs to mid-December, data ends 09-30


def test_assert_no_leakage_passes_and_catches_tampering(inv):
    data = ru.make_dataset(inv, SNAP)
    ru.assert_no_leakage(inv, data, SNAP)
    broken = data.copy()
    broken.loc[1, "F"] = 99
    with pytest.raises(AssertionError):
        ru.assert_no_leakage(inv, broken, SNAP)


def test_score_frame_format(inv):
    data = ru.make_dataset(inv, SNAP)
    sf = ru.to_score_frame(data.index, ru.baseline_recency_score(data), SNAP, "recency_only")
    ru.validate_score_frame(sf, data)
    with pytest.raises(AssertionError):
        ru.validate_score_frame(sf.iloc[:-1], data)  # missing a customer


def test_lift_at_k_perfect_and_random():
    y = np.array([1] * 10 + [0] * 90)
    perfect = ru.lift_at_k(y, y.astype(float), k=0.10)
    assert perfect["capture"] == 1.0 and perfect["lift"] == pytest.approx(10.0)
    constant = ru.lift_at_k(y, np.zeros(100), k=0.10)  # all ties: behaves like random
    assert constant["lift"] == pytest.approx(constant["precision"] / 0.10)
    assert constant["lift"] < 5  # nowhere near perfect


def test_evaluate_scores_perfect_and_keys():
    y = np.array([1] * 10 + [0] * 90)
    res = ru.evaluate_scores(y, y.astype(float))
    assert res["auc"] == 1.0 and res["pr_auc"] == 1.0
    assert res["lift@10"] == pytest.approx(10.0)
    assert set(res) >= {"auc", "pr_auc", "base_rate", "capture@10", "precision@10", "lift@10"}


def test_save_and_load_score_frames(inv, tmp_path):
    data = ru.make_dataset(inv, SNAP)
    sf = ru.to_score_frame(data.index, ru.baseline_recency_score(data), SNAP, "recency_only")
    ru.save_score_frame(sf, "recency_only", out_dir=tmp_path)
    back = ru.load_score_frames(out_dir=tmp_path)
    assert list(back.columns) == ru.SCORE_COLUMNS and len(back) == len(sf)
    with pytest.raises(AssertionError):
        ru.save_score_frame(sf.rename(columns={"score": "s"}), "bad", out_dir=tmp_path)


def test_postage_only_invoice_is_not_a_repurchase():
    data = ru.make_dataset(ru.invoice_table(ru.clean_transactions(_raw())[0]), SNAP)
    assert data.loc[3, "repurchase"] == 0
    kept, _ = ru.clean_transactions(_raw(), drop_non_product=False)
    data_kept = ru.make_dataset(ru.invoice_table(kept), SNAP)
    assert data_kept.loc[3, "repurchase"] == 1  # what would happen without the rule
