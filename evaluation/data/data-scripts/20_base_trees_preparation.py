% Script to prepare 20 base trees

import os, re, math, textwrap
import pandas as pd
from ete3 import Tree

excel_path = "base_tree_species.xlsx"
input_files = ["amphibians_trees20.txt"]

def find_group_sheet(xl: pd.ExcelFile, key: str) -> str:
    """
    Find the sheet whose name contains the group key (case-insensitive).
    E.g., key='amphib' will match 'Amphibians' or 'amphibians_base'.
    """
    key = key.lower()
    cands = [s for s in xl.sheet_names if key in s.lower()]
    if not cands:
        # fallback: exact case-insensitive match or raise
        cands = [s for s in xl.sheet_names if s.lower() == key]
    if not cands:
        raise ValueError(f"No sheet found for key '{key}'. Sheets present: {xl.sheet_names}")
    return cands[0]

_base_col_pat = re.compile(r"^base[_\s-]?(\d+)$", re.IGNORECASE)

def extract_base_sets(df: pd.DataFrame):
    """
    From a sheet DataFrame, find columns named like Base_01, Base-2, BASE 03, etc.
    Return a list of (index, [taxa...]) sorted by index (1..20).
    Taxa values are cleaned (stripped) and converted from 'Genus species' to 'Genus_species'.
    """
    cols = []
    for col in df.columns:
        m = _base_col_pat.match(str(col).strip())
        if m:
            idx = int(m.group(1))
            cols.append((idx, col))
    if not cols:
        raise ValueError(
            "No base columns found. Expected columns like 'Base_01', 'Base 02', 'BASE-03'. "
            f"Got columns: {list(df.columns)}"
        )
    cols.sort(key=lambda x: x[0])  # sort by numeric index
    base_sets = []
    for idx, col in cols:
        raw = df[col].dropna().astype(str).tolist()
        cleaned = []
        for name in raw:
            name = name.strip()
            if not name:
                continue
            # Excel has spaces; trees use underscores
            name = re.sub(r"\s+", "_", name)
            cleaned.append(name)
        base_sets.append((idx, cleaned))
    return base_sets

def load_newick_trees(path: str):
    """
    Load one or more Newick trees from a file. Splits on semicolons.
    Keeps branch lengths/support if present.
    Returns list of raw Newick strings (each ending with ';').
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = [p.strip() for p in re.split(r";\s*", content) if p.strip()]
    trees = [p + ";" for p in parts]
    return trees
    
def prune_tree_to_keep_list(newick_str: str, keep_taxa: list) -> str:
    """
    Given a Newick string and a list of taxa to retain, return a pruned Newick string.
    - Intersects keep_taxa with actually present leaf names
    - Preserves branch lengths
    Returns None if fewer than 2 taxa remain after intersection.
    """
    t = Tree(newick_str, format=1)  # format=1: keep support/lengths if present
    present = {leaf.name for leaf in t.iter_leaves()}
    keep = [x for x in keep_taxa if x in present]
    if len(keep) < 2:
        return None  # can't write a meaningful tree with <2 leaves
    t.prune(keep, preserve_branch_length=True)
    # format=5 keeps internal supports + branch lengths
    return t.write(format=5)

def process_group(excel_path: str, input_path: str, output_path: str, sheet_key_guess: str):
    xl = pd.ExcelFile(excel_path)
    sheet_name = find_group_sheet(xl, sheet_key_guess)
    df = xl.parse(sheet_name)
    base_sets = extract_base_sets(df)  # list of (idx, [taxa...]) sorted by idx

    trees = load_newick_trees(input_path)

    n_bases = len(base_sets)
    n_trees = len(trees)
    n = min(n_bases, n_trees)
    if n_bases != n_trees:
        print(f"[WARN] {os.path.basename(input_path)}: {n_trees} trees but {n_bases} base columns. Processing first {n}.")

    pruned_out = []
    skipped = 0
    for i in range(n):
        base_idx, keep_list = base_sets[i]
        newick = trees[i]
        pruned = prune_tree_to_keep_list(newick, keep_list)
        if pruned is None:
            skipped += 1
            # Write an empty line as a placeholder to preserve indexing, or skip entirely:
            # pruned_out.append("")  # uncomment to insert blank line placeholders
            continue
        pruned_out.append(pruned)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pruned_out) + ("\n" if pruned_out else ""))

    print(f"[DONE] Wrote {len(pruned_out)} pruned trees to {output_path} (skipped {skipped}).")
    
# Map rough key → file stems. Keys are used to find the right Excel sheet.
jobs = [
    
    ("amphibians",   "amphibians_trees20.txt",      "amphibians_base_trees20.txt")
    
]

for key, infile, outfile in jobs:
    if not os.path.exists(infile):
        print(f"[WARN] Missing input file: {infile} (skipping)")
        continue
    process_group(excel_path, infile, outfile, key)