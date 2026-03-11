# --- Build 30 overlapping trees PER base tree -> 20 files per group ---
# Prereqs:
# !pip install -q ete3 pyyaml

import os, re, sys, random
from pathlib import Path
from typing import List
from ete3 import Tree

sys.path.append(str(Path.cwd()))
import gen_pruned_trees as gp  # uses gp.generate_sets_of_leaves, gp.prune_to_leaves, gp.resolve_defaults, gp.get_leaf_names


def load_newick_list(path: str) -> List[str]:
    """Split a text file into individual Newick strings by semicolon."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    parts = [p.strip() for p in re.split(r";\s*", txt) if p.strip()]
    return [p + ";" for p in parts]


# Generator configuration
CFG_BASE = {
    "random_seed": 7,                 # we offset this by base-tree index for variety + reproducibility
    "n_trees": 30,                    # exactly 30 overlapping subsets per base tree

    # overlap controls
    "overlap_metric": "jaccard",
    "pairwise_overlap_range": [0.30, 0.70],
    "min_shared_leaves_per_pair": 2,
    "per_leaf_min_coverage": 1,
    "min_leaves_per_tree": 50,        # will be overwritten per base tree via target_tree_size_range
    "enforce_full_coverage": True,

    # subset construction
    "prune_mode": "mixed",            # "leaves" | "clades" | "mixed"
    "clade_selection_bias": 0.6,
    "anchor_taxa_count": 10,           # keep a small stable anchor set
    "clade_size_pref": "by_size",
    "leaf_prune_blockiness": 0.3,
    "contract_degree2": True,

    # Topology noise
    "topology_noise": "nni",          # none | swap_labels | nni
    "nni_moves": 3,                   # recommended starting point for 50–145 taxa
    "swap_fraction": 0.0,             # used only if topology_noise = swap_labels
    "protect_anchors_in_noise": True,

    # Branch-length noise
    "length_scaling": "global_uniform",     # none | global | global_uniform | edgewise_lognormal | ...
    "length_scale_params": {"low": 1.003, "high": 1.009},
    "renormalize_root_height": "to_base",   # keep overall tree height comparable across pruned trees

    "base_tree_path": "__inline__",
    "out_dir": "__inline__",
}

JOBS = [
    ("amphibians", "amphibians_base_trees20.txt"),
    ("squamates",  "squamates_base_trees20.txt"),
    ("mammals",    "mammals_base_trees20.txt"),
    ("sharks",     "sharks_base_trees20.txt"),
]

for group, base_file in JOBS:
    if not os.path.exists(base_file):
        print(f"[WARN] Missing {base_file}; skipping {group}")
        continue

    base_newicks = load_newick_list(base_file)
    n_base = len(base_newicks)
    print(f"\n=== {group.upper()} ===  ({n_base} base trees)")

    out_dir = f"{group}_input_multisets"
    os.makedirs(out_dir, exist_ok=True)

    for i, nwk in enumerate(base_newicks, start=1):
        # Parse base tree and ensure tiny positive branch lengths
        base_tree = Tree(nwk, format=1)
        for n in base_tree.traverse():
            try:
                n.dist = max(float(getattr(n, "dist", 0.0)), 1e-8)
            except Exception:
                n.dist = 1e-8

        leaves = gp.get_leaf_names(base_tree)
        base_n = len(leaves)

        # Per-base-tree config + seed
        cfg = gp.resolve_defaults(dict(CFG_BASE))
        seed_here = int(cfg["random_seed"]) + 1000 * i
        cfg["random_seed"] = seed_here
        random.seed(seed_here)

        # Per-base-tree subset size regime: target sizes around 60%–90% of base tree
       
        lo = max(int(cfg["min_shared_leaves_per_pair"]) + int(cfg["anchor_taxa_count"]), int(0.50 * base_n))
        hi = max(lo, int(0.75 * base_n))
        cfg["target_tree_size_range"] = [lo, hi]
        cfg["min_leaves_per_tree"] = lo

        # Generate 30 overlapping leaf sets from this base tree
        leaf_sets, _notes = gp.generate_sets_of_leaves(base_tree, cfg, leaves)

        anchors = set(cfg.get("anchors_explicit", []))
        base_h = gp.root_height(base_tree)

        # Turn each leaf set into a pruned Newick; collect 30 lines
        K = int(cfg["n_trees"])
        lines = []
        for k in range(K):
            keep = leaf_sets[k]
            pruned = gp.prune_to_leaves(
                base_tree,
                keep,
                contract_degree2=bool(cfg.get("contract_degree2", True))
            )

            # APPLY NOISE
            gp.apply_topology_noise(pruned, cfg, anchors)
            gp.apply_length_scaling(pruned, cfg, anchors, base_h)

            newick = pruned.write(format=1).strip()
            if not newick.endswith(";"):
                newick += ";"
            lines.append(newick)

        # Write one file per base tree: multiset_i.txt (30 lines)
        out_path = os.path.join(out_dir, f"multiset_{i}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        if i % 5 == 0 or i == n_base:
            print(f"  wrote {out_path}  [{K} trees]  (subset size range: {lo}-{hi}, anchors: {len(anchors)})")

    print(f"[DONE] Wrote {n_base} multiset files under ./{out_dir}")