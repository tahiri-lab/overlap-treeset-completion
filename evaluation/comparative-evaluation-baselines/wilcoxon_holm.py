#!/usr/bin/env python3
# ------------------------------------------------------------
# Wilcoxon signed-rank tests (one-sided) + Holm correction
# using the saved per-subset summary file: df_subsets.csv
#
# Tests (per species group, per metric):
#   H1: Proposed < Baseline  (smaller distance is better)
# Holm correction is applied across the 4 baseline comparisons
# within each (group, metric).
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd

# SciPy is used for Wilcoxon signed-rank tests
from scipy.stats import wilcoxon


# CONFIG

SUBSETS_CSV_CANDIDATES = [
    "df_subsets.csv",
    os.path.join("results", "df_subsets.csv")
]

PROPOSED_KEY = "original"
BASELINES = ["at_root", "nearest_leaf_parent", "no_mcs", "supertree"]

METRICS = {
    "normalized RF (median)": "median_normalized_rf",
    "normalized BSD (median)": "median_bsd",
}


# HELPERS
def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find df_subsets.csv. Tried:\n  - " + "\n  - ".join(candidates)
    )

def holm_adjust(pvals, alpha=0.05):
    """
    Holm step-down adjustment.
    Input: dict {name: p_raw}
    Output: dict {name: p_holm}
    """
    names = list(pvals.keys())
    raw = np.array([pvals[n] for n in names], dtype=float)

    m = len(raw)
    order = np.argsort(raw)  # ascending
    adj_sorted = np.empty(m, dtype=float)

    # Holm: p_(i) * (m - i), with step-down monotonicity (cummax)
    for rank, idx in enumerate(order):
        multiplier = m - rank
        adj_sorted[rank] = raw[idx] * multiplier

    # enforce non-decreasing adjusted p-values along the sorted order
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    # map back to original names
    p_holm = {}
    for rank, idx in enumerate(order):
        p_holm[names[idx]] = float(adj_sorted[rank])
    return p_holm

def sci_wilcoxon_less(x, y):
    """
    One-sided Wilcoxon signed-rank test, H1: x < y.
    """
    # Drop NaN pairs
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan, 0

    d = x - y
    # If all differences are exactly 0, Wilcoxon is undefined; treat as p=1
    if np.allclose(d, 0.0):
        return 1.0, len(x)

    # Prefer exact method if available
    try:
        res = wilcoxon(
            x, y,
            alternative="less",
            zero_method="wilcox",
            correction=False,
            method="exact",
        )
        return float(res.pvalue), len(x)
    except TypeError:
        # Older SciPy
        pass

    try:
        res = wilcoxon(
            x, y,
            alternative="less",
            zero_method="wilcox",
            correction=False,
        )
        return float(res.pvalue), len(x)
    except TypeError:
        # Very old SciPy: only two-sided; convert to one-sided.
        # Rule: If the observed effect is in the wrong direction, p_one=1.
        res2 = wilcoxon(x, y, zero_method="wilcox", correction=False)
        p_two = float(res2.pvalue)
        # If median(x - y) < 0, effect is toward x<y; one-sided p = p_two/2
        # else one-sided p = 1 - (p_two/2)
        if np.nanmedian(d) < 0:
            return p_two / 2.0, len(x)
        else:
            return 1.0 - (p_two / 2.0), len(x)

def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NaN"
    return f"{p:.3e}"

def count_wins_ties_losses(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) == 0:
        return 0, 0, 0, 0
    wins = int(np.sum(x < y))
    ties = int(np.sum(np.isclose(x, y)))
    losses = int(np.sum(x > y))
    return len(x), wins, ties, losses


# MAIN
def main():
    path = find_existing_path(SUBSETS_CSV_CANDIDATES)
    df = pd.read_csv(path)

    required_cols = {"group", "subset", "method_key"}
    required_cols |= set(METRICS.values())
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"df_subsets.csv is missing required columns: {missing}")

    # Ensure consistent ordering within group by subset index
    df = df.sort_values(["group", "subset", "method_key"]).reset_index(drop=True)

    groups = list(df["group"].dropna().unique())
    print(f"Loaded: {path}")
    print(f"Groups found: {groups}\n")

    # Print results
    for metric_name, col in METRICS.items():
        print("=" * 80)
        print(f"{metric_name}  (column: {col})")
        print("=" * 80)

        for group in groups:
            sub = df[df["group"] == group]

            # Proposed vector indexed by subset
            prop = sub[sub["method_key"] == PROPOSED_KEY][["subset", col]].rename(columns={col: "proposed"})
            if prop.empty:
                print(f"\n[{group}] No proposed rows found; skipping.")
                continue

            # Run comparisons vs each baseline
            raw_p = {}
            details = {}

            for b in BASELINES:
                base = sub[sub["method_key"] == b][["subset", col]].rename(columns={col: "baseline"})
                if base.empty:
                    raw_p[b] = np.nan
                    details[b] = (0, 0, 0, 0)
                    continue

                merged = prop.merge(base, on="subset", how="inner")
                x = merged["proposed"].values
                y = merged["baseline"].values

                p, n_pairs = sci_wilcoxon_less(x, y)
                raw_p[b] = p
                details[b] = count_wins_ties_losses(x, y)

            # Holm adjust across the 4 baselines for this (group, metric)
            raw_nonan = {k: v for k, v in raw_p.items() if np.isfinite(v)}
            holm = holm_adjust(raw_nonan) if raw_nonan else {}

            print(f"\n[{group}]")
            print("baseline_key            n   wins ties loss   p_raw       p_holm")
            print("-" * 76)
            for b in BASELINES:
                n, wins, ties, losses = details.get(b, (0, 0, 0, 0))
                p_raw = raw_p.get(b, np.nan)
                p_holm = holm.get(b, np.nan) if np.isfinite(p_raw) else np.nan
                print(f"{b:<20} {n:>3}  {wins:>3}  {ties:>3}  {losses:>3}   {fmt_p(p_raw):>9}   {fmt_p(p_holm):>9}")

        print()

if __name__ == "__main__":
    main()
