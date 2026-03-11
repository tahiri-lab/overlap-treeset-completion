#!/usr/bin/env python3
"""
Phylogenetically stratified base-taxa sampler

Goal: build 20 base species lists whose pruned base trees have more comparable
overall scales (root→tip height), by ensuring each list spans the deep structure
of the full phylogeny instead of accidentally sampling a single shallow clade.

Dependencies:
- ete3 (for Tree)
- pandas, numpy
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ete3 import Tree


# Utilities


def parse_sizes(spec: str) -> List[int]:
    """Parse sizes like '50:145:5' -> [50,55,...,145]. Or comma list: '50,60,75'."""
    spec = spec.strip()
    if ":" in spec:
        start, end, step = map(int, spec.split(":"))
        return list(range(start, end + 1, step))
    return [int(x) for x in spec.split(",") if x.strip()]


def load_first_newick(path: str) -> str:
    """Load the first Newick tree from a file that may contain multiple trees."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    # take first tree by splitting on ';'
    parts = [p.strip() for p in txt.split(";") if p.strip()]
    if not parts:
        raise ValueError(f"No Newick trees found in {path}")
    return parts[0] + ";"


def normalize_name_to_tree(name: str) -> str:
    """Convert 'Genus species' -> 'Genus_species' and normalize whitespace."""
    name = str(name).strip()
    name = " ".join(name.split())
    return name.replace(" ", "_")


def display_name_from_tree(name: str) -> str:
    """Convert 'Genus_species' -> 'Genus species' for nicer Excel display."""
    return str(name).replace("_", " ").strip()


def weighted_choice_without_replacement(
    candidates: List[str], weights: np.ndarray, k: int, rng: np.random.Generator
) -> List[str]:
    """Sample k distinct items from candidates according to weights (non-negative)."""
    if k <= 0 or not candidates:
        return []
    w = np.asarray(weights, dtype=float)
    w[w < 0] = 0
    k = min(k, len(candidates))
    if w.sum() <= 0:
        idx = rng.choice(len(candidates), size=k, replace=False)
        return [candidates[i] for i in idx]
    p = w / w.sum()
    idx = rng.choice(len(candidates), size=k, replace=False, p=p)
    return [candidates[i] for i in idx]


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


def to_wide_df(bases: List[List[str]], display: bool = True) -> pd.DataFrame:
    """Bases -> columns Base_01..Base_20, padded with blanks."""
    max_len = max(len(b) for b in bases) if bases else 0
    data = {}
    for i, b in enumerate(bases):
        col = [display_name_from_tree(x) for x in b] if display else list(b)
        data[f"Base_{i+1:02d}"] = col + [""] * (max_len - len(col))
    return pd.DataFrame(data)


# Strata construction


def make_strata(tree: Tree, k: int) -> List[Tree]:
    """
    Greedy top-down partition into k clade-strata.
    Assumes binary splits (we resolve polytomies beforehand).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    # Work on a copy
    t = tree.copy(method="deepcopy")

    # Ensure every internal node is at most bifurcating so each split adds exactly +1 bin.
    # This does not change leaf sets; it only adds 0-length structure if needed.
    t.resolve_polytomy(recursive=True)

    bins: List[Tree] = [t]
    while len(bins) < k:
        splittable = [n for n in bins if (not n.is_leaf()) and len(n.children) >= 2 and len(n) > 1]
        if not splittable:
            break
        # pick the bin with the most leaves
        node = max(splittable, key=lambda n: len(n))
        bins.remove(node)
        # binary after resolve_polytomy
        bins.extend(node.children)

    if len(bins) < k:
        raise ValueError(
            f"Could only create {len(bins)} strata (requested {k}). "
            f"Try smaller --k-strata."
        )
   
    return bins[:k]


def strata_leaf_sets(strata: List[Tree]) -> List[List[str]]:
    """Return leaf name lists for each stratum."""
    return [s.get_leaf_names() for s in strata]


def build_stratum_index(strata_leaves: List[List[str]]) -> Dict[str, int]:
    """Map each species name -> stratum index."""
    idx = {}
    for i, leaves in enumerate(strata_leaves):
        for sp in leaves:
            idx[sp] = i
    return idx


# Sampling


def allocate_across_strata_equal(
    size: int, k: int, strata_sizes: List[int], ensure_min1: bool = True
) -> List[int]:
    """
    Allocate 'size' picks across k strata:
    - roughly equal
    - optionally ensure >=1 per stratum if size>=k
    - remainder goes first to larger strata (reduces chance of running out)
    """
    if k <= 0:
        return []

    if ensure_min1 and size >= k:
        alloc = [1] * k
        remaining = size - k
    else:
        alloc = [0] * k
        remaining = size

    if remaining <= 0:
        return alloc

    base = remaining // k
    rem = remaining % k
    alloc = [a + base for a in alloc]

    # distribute remainder to the biggest strata first
    order = sorted(range(k), key=lambda i: strata_sizes[i], reverse=True)
    for i in order[:rem]:
        alloc[i] += 1
    return alloc


def pick_from_stratum(
    stratum_species: List[str],
    need: int,
    chosen: set,
    used_count: Dict[str, int],
    cap: int,
    rng: np.random.Generator,
) -> List[str]:
    """Pick 'need' taxa from a stratum, prioritizing underused and respecting cap when possible."""
    if need <= 0:
        return []

    # Prefer taxa below cap
    eligible = [s for s in stratum_species if s not in chosen and used_count[s] < cap]
    if len(eligible) < need:
        # fall back to any unused-in-this-base taxa (ignoring cap) to avoid short bases
        eligible = [s for s in stratum_species if s not in chosen]

    usage = np.array([used_count[s] for s in eligible], dtype=float)
    weights = 1.0 / (1.0 + usage)
    pick = weighted_choice_without_replacement(eligible, weights, need, rng)
    return pick


def build_phylo_stratified_bases(
    universe: List[str],
    strata_leaves: List[List[str]],
    sizes: List[int],
    seed: int = 0,
    max_per_species: int = 6,
    reuse_fraction: float = 0.15,
) -> Tuple[List[List[str]], Dict[str, int], pd.DataFrame]:
    """
    Build bases of varying sizes using phylogenetic strata, controlled overlap,
    and underuse-weighted sampling with a soft reuse cap.
    """
    rng = np.random.default_rng(seed)

    # de-dup universe while keeping order
    seen = set()
    uni = []
    for s in universe:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            uni.append(s)

    strata = []
    for leaves in strata_leaves:
        # intersect stratum leaves with universe to be safe (and keep stable ordering using uni)
        leafset = set(leaves)
        strata.append([s for s in uni if s in leafset])

    k = len(strata)
    strata_sizes = [len(x) for x in strata]
    sp_to_stratum = build_stratum_index(strata)

    used_count = defaultdict(int)
    bases: List[List[str]] = []

    # Track composition per base
    counts_rows = []

    for base_i, size in enumerate(sizes, start=1):
        # Softly relax cap late in the ladder
        cap = max_per_species + (1 if base_i > 0.75 * len(sizes) else 0)

        chosen: List[str] = []
        chosen_set = set()

        # Optional seeding from a previous base for overlap
        if base_i > 1 and bases and reuse_fraction > 0:
            prev = bases[int(rng.integers(0, len(bases)))]
            seed_cnt = max(0, min(size - 1, int(round(reuse_fraction * min(size, len(prev))))))
            if seed_cnt > 0:
                seed_taxa = rng.choice(prev, size=seed_cnt, replace=False).tolist()
                # ensure seeds exist in our universe
                seed_taxa = [s for s in seed_taxa if s in seen]
                chosen.extend(seed_taxa)
                chosen_set.update(seed_taxa)

        # Allocate target counts per stratum
        alloc = allocate_across_strata_equal(size=size, k=k, strata_sizes=strata_sizes, ensure_min1=True)

        # subtract what seeding already contributed per stratum
        seeded_counts = [0] * k
        for s in chosen:
            si = sp_to_stratum.get(s, None)
            if si is not None:
                seeded_counts[si] += 1
        need_per = [max(0, alloc[i] - seeded_counts[i]) for i in range(k)]

        # Pick within each stratum
        deficit = 0
        for i in range(k):
            take = pick_from_stratum(strata[i], need_per[i], chosen_set, used_count, cap, rng)
            chosen.extend(take)
            chosen_set.update(take)
            deficit += max(0, need_per[i] - len(take))

        # If any strata were short, redistribute deficit across all strata
        if deficit > 0:
            pool = [s for s in uni if s not in chosen_set and used_count[s] < cap]
            usage = np.array([used_count[s] for s in pool], dtype=float)
            weights = 1.0 / (1.0 + usage)
            extra = weighted_choice_without_replacement(pool, weights, deficit, rng)
            chosen.extend(extra)
            chosen_set.update(extra)

        # If still short (caps/coverage), top up from anything left
        if len(chosen) < size:
            pool = [s for s in uni if s not in chosen_set]
            extra = weighted_choice_without_replacement(pool, np.ones(len(pool)), size - len(chosen), rng)
            chosen.extend(extra)
            chosen_set.update(extra)

        # final trim (rarely)
        chosen = chosen[:size]

        bases.append(chosen)
        for s in chosen:
            used_count[s] += 1

        # composition row
        row = {"Base": f"Base_{base_i:02d}", "Size": len(chosen)}
        comp = [0] * k
        for s in chosen:
            si = sp_to_stratum.get(s, None)
            if si is not None:
                comp[si] += 1
        for i in range(k):
            row[f"Stratum_{i+1:02d}"] = comp[i]
        counts_rows.append(row)

    strata_counts_df = pd.DataFrame(counts_rows)
    return bases, used_count, strata_counts_df


# Main


def strata_to_wide_df(strata_leaves: List[List[str]]) -> pd.DataFrame:
    """Write strata as columns Stratum_01..Stratum_K."""
    max_len = max(len(x) for x in strata_leaves) if strata_leaves else 0
    data = {}
    for i, leaves in enumerate(strata_leaves, start=1):
        col = [display_name_from_tree(x) for x in leaves]
        data[f"Stratum_{i:02d}"] = col + [""] * (max_len - len(col))
    return pd.DataFrame(data)

# example for squamates
def read_wide_species_csv_squamates(path: str, group_col: str = "Squamates") -> List[str]:
    """
    Read the Squamates column from the wide CSV (one column per group, rows are species names).
    Returns normalized-to-tree names (underscores).
    """
    df = pd.read_csv(path)
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in {path}. Columns: {list(df.columns)}")
    col = df[group_col].dropna().astype(str).tolist()
    out = []
    seen = set()
    for s in col:
        nm = normalize_name_to_tree(s)
        if nm and nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="Path to Newick file for the FULL Squamates tree (single tree recommended)")
    ap.add_argument("--out", required=True, help="Output Excel path")
    ap.add_argument("--sizes", default="50:145:5", help="Sizes as start:end:step or comma list")
    ap.add_argument("--k-strata", type=int, default=12, help="Number of phylogenetic strata (clade bins)")
    ap.add_argument("--seed", type=int, default=20251104, help="RNG seed")

    # overlap / reuse controls
    ap.add_argument("--reuse-fraction", type=float, default=0.15, help="Fraction of taxa to reuse from a previous base")
    ap.add_argument("--max-per-species", type=int, default=6, help="Soft per-species reuse cap across all bases")

    # optional: also load a Squamates list from the wide CSV and intersect with tree taxa
    ap.add_argument("--csv", default=None, help="Optional wide CSV with a Squamates column to define allowed taxa")
    ap.add_argument("--squamates-col", default="Squamates", help="Column name for Squamates in the wide CSV (if --csv is used)")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)

    newick = load_first_newick(args.tree)
    full_tree = Tree(newick, format=1)

    # universe from tree leaves
    tree_leaves = full_tree.get_leaf_names()
    tree_leaf_set = set(tree_leaves)

    # optional universe restriction using CSV
    if args.csv:
        csv_squamates = read_wide_species_csv_squamates(args.csv, group_col=args.squamates_col)
        universe = [s for s in csv_squamates if s in tree_leaf_set]
        if len(universe) < 2:
            raise ValueError("After intersecting CSV Squamates list with tree leaves, <2 taxa remain.")
    else:
        universe = tree_leaves[:]  # keep tree order

    # build strata on the FULL tree (but strata membership will be intersected with universe later)
    strata_nodes = make_strata(full_tree, args.k_strata)
    strata_leaves = strata_leaf_sets(strata_nodes)

    bases, used_count, strata_counts_df = build_phylo_stratified_bases(
        universe=universe,
        strata_leaves=strata_leaves,
        sizes=sizes,
        seed=args.seed,
        max_per_species=args.max_per_species,
        reuse_fraction=args.reuse_fraction,
    )

    # diagnostics
    J = jaccard_matrix(bases)
    reuse_summary = pd.Series(dict(used_count)).describe().to_frame("reuse_count")

    # write workbook
    with pd.ExcelWriter(args.out, engine="xlsxwriter") as writer:
        to_wide_df(bases, display=True).to_excel(writer, sheet_name="Squamates", index=False)
        J.to_excel(writer, sheet_name="Squamates_Jaccard")
        reuse_summary.to_excel(writer, sheet_name="Squamates_Reuse")
        strata_to_wide_df(strata_leaves).to_excel(writer, sheet_name="Squamates_Strata", index=False)
        strata_counts_df.to_excel(writer, sheet_name="Squamates_StrataCounts", index=False)

    # console summary
    M = J.values
    upper = M[np.triu_indices_from(M, k=1)]
    print(
        f"[DONE] Wrote {len(bases)} bases to {args.out}\n"
        f"Sizes: {list(map(len, bases))}\n"
        f"Jaccard min={upper.min():.3f}, median={np.median(upper):.3f}, max={upper.max():.3f}\n"
        f"Strata (K) = {len(strata_leaves)}"
    )


if __name__ == "__main__":
    main()
