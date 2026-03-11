#!/usr/bin/env python3
"""
Build 20 base species lists per group (e.g., Amphibians/Mammals/Sharks/Squamates) from a
wide CSV where each column is a group and rows are species names.

"""

import argparse
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


def parse_sizes(spec: str) -> List[int]:
    """
    Parse sizes like '50:145:5' -> [50,55,...,145]
    Also supports a comma list like '50,60,75'.
    """
    if ":" in spec:
        start, end, step = map(int, spec.split(":"))
        return list(range(start, end + 1, step))
    return [int(x) for x in spec.split(",")]


def read_wide_species_csv(path: str, groups: List[str]) -> Dict[str, List[str]]:
    df = pd.read_csv(path)
    available = {}
    for g in groups:
        if g not in df.columns:
            raise ValueError(f"Group '{g}' not found in CSV columns {list(df.columns)}")
        # unique, keep order, drop blanks
        col = df[g].dropna().astype(str).tolist()
        seen = set()
        uniq = []
        for s in col:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                uniq.append(s)
        available[g] = uniq
    return available


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return 0.0 if not sa and not sb else len(sa & sb) / len(sa | sb)


def jaccard_matrix(bases: List[List[str]]) -> pd.DataFrame:
    n = len(bases)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            M[i, j] = 1.0 if i == j else jaccard(bases[i], bases[j])
    idx = [f"Base_{i+1:02d}" for i in range(n)]
    return pd.DataFrame(M, index=idx, columns=idx)


def weighted_choice_without_replacement(
    candidates: List[str], weights: np.ndarray, k: int, rng: np.random.Generator
) -> List[str]:
    if k <= 0:
        return []
    w = np.asarray(weights, dtype=float)
    w[w < 0] = 0
    if w.sum() == 0:
        idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
        return [candidates[i] for i in idx]
    p = w / w.sum()
    idx = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False, p=p)
    return [candidates[i] for i in idx]


def build_bases_for_group(
    species_list: List[str],
    sizes: List[int],
    seed: int = 0,
    max_per_species: int = 6,
    reuse_fraction: float = 0.15,
) -> Tuple[List[List[str]], Dict[str, int]]:
    """
    Build a sequence of bases with a size ladder and controlled overlap.

    Strategy:
      - Base 1: sample by underuse (all 0 initially).
      - Base i>1: pick ~reuse_fraction * size taxa from a random previous base,
        then fill remaining slots with underuse-weighted sampling subject to a
        per-species reuse cap.
    """
    rng = np.random.default_rng(seed)
    # unique + preserve order
    universe = []
    seen = set()
    for s in species_list:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            universe.append(s)

    used_count = defaultdict(int)
    bases: List[List[str]] = []

    for i, size in enumerate(sizes, start=1):
        if i == 1 or not bases:
            # First base: favor underused species (all equal here)
            usage = np.array([used_count[s] for s in universe], dtype=float)
            weights = 1.0 / (1.0 + usage)
            chosen = weighted_choice_without_replacement(universe, weights, size, rng)
        else:
            # Seed some taxa from a random previous base to keep non-zero overlap
            prev = bases[int(rng.integers(0, len(bases)))]
            seed_cnt = max(0, min(size - 1, int(round(reuse_fraction * min(size, len(prev))))))
            seed_taxa = rng.choice(prev, size=seed_cnt, replace=False).tolist() if seed_cnt > 0 else []

            # Fill the rest via underuse weights, respecting a soft cap
            cap = max_per_species + (1 if i > 0.75 * len(sizes) else 0)
            remaining = [s for s in universe if s not in set(seed_taxa) and used_count[s] < cap]
            usage = np.array([used_count[s] for s in remaining], dtype=float)
            weights = 1.0 / (1.0 + usage)
            fill = weighted_choice_without_replacement(remaining, weights, size - len(seed_taxa), rng)
            chosen = seed_taxa + fill

            # If still short (due to caps), top up uniformly from what’s left
            if len(chosen) < size:
                rest = [s for s in universe if s not in set(chosen)]
                extra = weighted_choice_without_replacement(rest, np.ones(len(rest)), size - len(chosen), rng)
                chosen.extend(extra)

        bases.append(chosen)
        for s in chosen:
            used_count[s] += 1

    return bases, used_count


def to_wide_df(bases: List[List[str]]) -> pd.DataFrame:
    max_len = max(len(b) for b in bases)
    data = {f"Base_{i+1:02d}": b + [""] * (max_len - len(b)) for i, b in enumerate(bases)}
    return pd.DataFrame(data)


def write_excel(out_path: str, results: Dict[str, List[List[str]]], metas: Dict[str, dict]) -> None:
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for g, bases in results.items():
            to_wide_df(bases).to_excel(writer, sheet_name=g, index=False)
            metas[g]["jaccard"].to_excel(writer, sheet_name=f"{g}_Jaccard")
            metas[g]["reuse_summary"].to_excel(writer, sheet_name=f"{g}_Reuse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to wide CSV with columns per group")
    ap.add_argument("--out", required=True, help="Output Excel path")
    ap.add_argument("--groups", nargs="+", default=["Amphibians", "Squamates", "Mammals", "Sharks"])
    ap.add_argument("--sizes", default="50:145:5", help="Sizes as start:end:step or comma list")
    ap.add_argument("--reuse-fraction", type=float, default=0.15, help="Fraction of taxa to reuse from a previous base")
    ap.add_argument("--max-per-species", type=int, default=6, help="Per-species reuse cap across the 20 bases")
    ap.add_argument("--seed", type=int, default=20251104, help="Base RNG seed")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    species = read_wide_species_csv(args.csv, args.groups)

    results = {}
    metas = {}
    for g_idx, g in enumerate(args.groups):
        bases, used = build_bases_for_group(
            species[g],
            sizes=sizes,
            seed=args.seed + g_idx,
            max_per_species=args.max_per_species,
            reuse_fraction=args.reuse_fraction,
        )
        results[g] = bases
        J = jaccard_matrix(bases)
        reuse_summary = pd.Series(used).describe().to_frame("reuse_count")
        metas[g] = {"jaccard": J, "reuse_summary": reuse_summary}

    write_excel(args.out, results, metas)

    # Console summary
    for g in args.groups:
        M = metas[g]["jaccard"].values
        upper = M[np.triu_indices_from(M, k=1)]
        print(
            f"{g}: sizes={list(map(len, results[g]))} | "
            f"Jaccard min={upper.min():.3f}, median={np.median(upper):.3f}, max={upper.max():.3f}"
        )


if __name__ == "__main__":
    main()
