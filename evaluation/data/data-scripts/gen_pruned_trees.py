#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate overlapping pruned phylogenetic trees from a single base tree (ETE3).

Usage:
    python gen_pruned_trees.py --config config.yaml

YAML example (minimal):
    base_tree_path: "/path/to/base.nwk"
    out_dir: "./sim_pruned_trees"
    random_seed: 7
    n_trees: 40
    pairwise_overlap_range: [0.10, 0.90]
    min_shared_leaves_per_pair: 2
    per_leaf_min_coverage: 1
    min_leaves_per_tree: 4
    enforce_full_coverage: true

    prune_mode: mixed           # leaves | clades | mixed
    clade_selection_bias: 0.6   # only used if prune_mode = mixed
    clade_size_pref: by_size    # uniform | by_size | small_first | large_first
    leaf_prune_blockiness: 0.3  # 0..1, more -> contiguous subsets

    overlap_metric: jaccard     # jaccard | base_fraction
    overlap_strategy: sample_target_sizes  # (implemented)
    overlap_distribution: uniform

    length_scaling: none        # none | global | global_uniform | edgewise_lognormal | pendant_only_jitter | internal_only_jitter | clade_local
    length_scale_params: {mu: 0.0, sigma: 0.2}
    # If length_scaling = global_uniform: use {low: 1.003, high: 1.009}

    # Optional realism noise (applied after pruning each tree)
    topology_noise: none        # none | swap_labels | nni
    swap_fraction: 0.0          # for swap_labels; fraction of leaves participating (evened)
    nni_moves: 0                # for nni; number of NNI operations to apply
    protect_anchors_in_noise: true  # avoid moving anchor taxa / clades containing anchors
    renormalize_root_height: none  # none | to_base
    anchor_taxa_count: 0
    preserve_paths_between_anchors: false

    write_formats: [newick]
    filename_template: "sim_{:03d}.nwk"
    emit_metadata: true
    zip_output: false

Requirements:
    - ete3
    - pyyaml
"""

from __future__ import annotations
import argparse
import os
import sys
import math
import random
import json
import csv
import shutil
from typing import List, Dict, Set, Tuple, Optional

try:
    import yaml
except Exception as e:
    print("ERROR: pyyaml is required. Install with: pip install pyyaml", file=sys.stderr)
    raise

try:
    from ete3 import Tree
except Exception as e:
    print("ERROR: ete3 is required. Install with: pip install ete3", file=sys.stderr)
    raise


# Utilities

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def copy_if_needed(src: str, dst: str):
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)


def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def write_jsonl(rows: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_csv(header: List[str], rows: List[List], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def positive(x: Optional[float], eps: float = 1e-8) -> float:
    if x is None or x <= 0:
        return eps
    return x


# Tree helpers

def load_tree_with_positive_lengths(path: str, eps: float = 1e-8) -> Tree:
    # format=1 expects internal node names ignored; branch lengths respected
    t = Tree(path, format=1)
    for n in t.traverse():
        n.dist = positive(getattr(n, "dist", None), eps)
    return t


def get_leaf_names(t: Tree) -> List[str]:
    return [lf.name for lf in t.iter_leaves()]


def root_height(t: Tree) -> float:
    # max root-to-leaf path length
    return max(t.get_distance(leaf) for leaf in t.iter_leaves())


def scale_all_edges(t: Tree, factor: float):
    for n in t.traverse():
        n.dist = positive(n.dist * factor)


def is_internal(n) -> bool:
    return not n.is_leaf()


def collect_internal_clades(t: Tree) -> List[Tuple[Tree, Set[str]]]:
    out = []
    for n in t.traverse("postorder"):
        if is_internal(n):
            leaves = {lf.name for lf in n.iter_leaves()}
            out.append((n, leaves))
    return out


def contract_unary_nodes(t: Tree):
    """
    Contract internal nodes with exactly one child.
    """
    changed = True
    while changed:
        changed = False
        for n in list(t.traverse("postorder")):
            if n.is_root():
                continue
            if not n.is_leaf() and len(n.children) == 1:
                child = n.children[0]
                # move child's edge length up
                child.dist = positive(child.dist + n.dist)
                parent = n.up
                # reattach child to parent
                child.detach()
                parent.add_child(child)
                n.detach()
                changed = True
    return t


def prune_to_leaves(base: Tree, keep_leaves: Set[str], contract_degree2: bool = True) -> Tree:
    t = base.copy(method="deepcopy")
    # ETE3 prune keeps listed leaves
    t.prune(list(keep_leaves), preserve_branch_length=True)
    if contract_degree2:
        contract_unary_nodes(t)
    return t


# Overlap metrics

def jaccard(a: Set[str], b: Set[str]) -> float:
    inter = len(a & b)
    union = len(a | b)
    return 0.0 if union == 0 else inter / union


def base_fraction(a: Set[str], b: Set[str], base_size: int) -> float:
    return len(a & b) / float(base_size)


def pairwise_overlap_matrix(sets: List[Set[str]], metric: str, base_size: int) -> List[List[float]]:
    m = len(sets)
    M = [[1.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if metric == "jaccard":
                val = jaccard(sets[i], sets[j])
            else:
                val = base_fraction(sets[i], sets[j], base_size)
            M[i][j] = M[j][i] = val
    return M


# Candidate set construction

def sample_target_sizes(n_trees: int, n_base: int, min_leaves_per_tree: int,
                        target_tree_size_range: Optional[Tuple[int, int]] = None) -> List[int]:
    lo, hi = target_tree_size_range if target_tree_size_range else (min_leaves_per_tree, n_base - 1)
    lo = max(min_leaves_per_tree, lo)
    hi = min(n_base, hi)
    if lo > hi:
        lo = min_leaves_per_tree
        hi = n_base - 1
    sizes = [random.randint(lo, hi) for _ in range(n_trees)]
    return sizes


def choose_anchor_taxa(base_leaves: List[str], k: int) -> Set[str]:
    if k <= 0:
        return set()
    if k > len(base_leaves):
        k = len(base_leaves)
    return set(random.sample(base_leaves, k))


def build_set_by_random_leaves(base_leaves: List[str], target_size: int,
                               blockiness: float, base_tree: Tree) -> Set[str]:
    """
    If blockiness>0, bias toward contiguous subsets by sampling a random internal node
    and taking many leaves from that clade.
    """
    target_size = max(1, min(target_size, len(base_leaves)))
    if target_size == len(base_leaves):
        return set(base_leaves)

    if blockiness <= 0.0 or random.random() > blockiness:
        return set(random.sample(base_leaves, target_size))

    # Cluster-biased: pick an internal clade, take as many leaves as possible from it
    internal = [n for n in base_tree.traverse() if is_internal(n)]
    if not internal:
        return set(random.sample(base_leaves, target_size))
    node = random.choice(internal)
    clade_leaves = [lf.name for lf in node.iter_leaves()]
    clade_take = min(target_size, len(clade_leaves))
    chosen = set(random.sample(clade_leaves, clade_take))
    # Fill remaining from outside
    remaining = [x for x in base_leaves if x not in chosen]
    if len(chosen) < target_size:
        chosen.update(random.sample(remaining, target_size - len(chosen)))
    return chosen


def sort_clades_by_pref(clades: List[Tuple[Tree, Set[str]]], pref: str) -> List[Tuple[Tree, Set[str]]]:
    if pref == "uniform":
        random.shuffle(clades)
    else:
        sized = [(node, leaves, len(leaves)) for (node, leaves) in clades]
        if pref == "by_size":
            random.shuffle(sized)
            sized.sort(key=lambda x: x[2])  # small to large, but shuffled within sizes
        elif pref == "small_first":
            sized.sort(key=lambda x: x[2])
        elif pref == "large_first":
            sized.sort(key=lambda x: -x[2])
        clades = [(node, leaves) for (node, leaves, _) in sized]
    return clades


def build_set_by_union_of_clades(base_tree: Tree, target_size: int, clade_size_pref: str) -> Set[str]:
    """
    Greedily choose disjoint clades to approach target_size; fill remainder randomly.
    """
    all_leaves = {lf.name for lf in base_tree.iter_leaves()}
    clades = collect_internal_clades(base_tree)
    # Avoid trivial root when too large
    clades = [(n, L) for (n, L) in clades if len(L) <= max(target_size, 2 * target_size)]
    clades = sort_clades_by_pref(clades, clade_size_pref)

    chosen: Set[str] = set()
    used: Set[str] = set()
    for _, L in clades:
        if len(chosen) >= target_size:
            break
        if L & used:
            continue
        if len(chosen) + len(L) <= target_size:
            chosen.update(L)
            used.update(L)

    # If we couldn't reach the target exactly, fill uniformly from remaining leaves
    if len(chosen) < target_size:
        remaining = list(all_leaves - chosen)
        need = target_size - len(chosen)
        if need > 0 and remaining:
            need = min(need, len(remaining))
            chosen.update(random.sample(remaining, need))
    # If we have gone over the limit (very rare given checks), trim back randomly
    if len(chosen) > target_size:
        chosen = set(random.sample(list(chosen), target_size))
    return chosen


def overlaps_ok(candidate: Set[str], prev_sets: List[Set[str]], cfg: dict,
                base_size: int) -> Tuple[bool, Optional[str]]:
    min_pair = cfg["min_shared_leaves_per_pair"]
    lo, hi = cfg["pairwise_overlap_range"]
    metric = cfg["overlap_metric"]
    for s in prev_sets:
        inter = len(candidate & s)
        if inter < min_pair:
            return False, f"min_shared {inter}<{min_pair}"
        if metric == "jaccard":
            val = jaccard(candidate, s)
        else:
            val = base_fraction(candidate, s, base_size)
        if not (lo <= val <= hi):
            return False, f"overlap {val:.3f} not in [{lo},{hi}]"
    return True, None


def repair_for_full_coverage(base_leaves: Set[str], sets: List[Set[str]], cfg: dict) -> None:
    """
    Ensure union equals base set and each leaf has at least per_leaf_min_coverage.
    Best effort greedy fixes that try not to break overlaps too much.
    """
    base_size = len(base_leaves)
    per_leaf_min = cfg.get("per_leaf_min_coverage", 0)
    metric = cfg["overlap_metric"]
    lo, hi = cfg["pairwise_overlap_range"]
    min_pair = cfg["min_shared_leaves_per_pair"]
    anchors = set(cfg.get("anchors_explicit", []))

    # Union coverage
    missing = list(base_leaves - set().union(*sets))
    random.shuffle(missing)
    for taxon in missing:
        # Choose the tree with smallest size to add
        idx = min(range(len(sets)), key=lambda i: len(sets[i]))
        # Try adding without removal. If violates badly, swap with a redundant leaf
        trial = sets[idx] | {taxon}
        ok, _ = overlaps_ok(trial, [sets[j] for j in range(len(sets)) if j != idx], cfg, base_size)
        if ok:
            sets[idx] = trial
        else:
            # Try swapping with a leaf that appears many times
            counts = leaf_counts(sets)
            removable = sorted([x for x in sets[idx] if x not in anchors],
                               key=lambda x: counts.get(x, 0), reverse=True)
            swapped = False
            for r in removable:
                if r == taxon:
                    continue
                trial = (sets[idx] - {r}) | {taxon}
                ok, _ = overlaps_ok(trial, [sets[j] for j in range(len(sets)) if j != idx], cfg, base_size)
                if ok:
                    sets[idx] = trial
                    swapped = True
                    break
            if not swapped:
                
                sets[idx] = sets[idx] | {taxon}

    # Per-leaf minimum coverage
    if per_leaf_min > 0:
        counts = leaf_counts(sets)
        deficit = [(leaf, per_leaf_min - counts.get(leaf, 0)) for leaf in base_leaves]
        needs = [leaf for leaf, d in deficit if d > 0]
        random.shuffle(needs)
        for leaf in needs:
            # Add to trees where it is absent, preferring smaller trees
            order = sorted(range(len(sets)), key=lambda i: (leaf not in sets[i], len(sets[i])), reverse=True)
            for i in order:
                if leaf in sets[i]:
                    continue
                trial = sets[i] | {leaf}
                ok, _ = overlaps_ok(trial, [sets[j] for j in range(len(sets)) if j != i], cfg, base_size)
                if ok:
                    sets[i] = trial
                    counts[leaf] = counts.get(leaf, 0) + 1
                    if counts[leaf] >= per_leaf_min:
                        break


def leaf_counts(sets: List[Set[str]]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for s in sets:
        for x in s:
            c[x] = c.get(x, 0) + 1
    return c


# Topology noise (optional)

def apply_topology_noise(t: Tree, cfg: dict, anchors: Set[str]):
    """Apply optional topology perturbations after pruning.

    Modes:
      - none: no change
      - swap_labels: swap leaf labels AND pendant lengths for a fraction of leaves
      - nni: apply a small number of rooted NNI moves
    """
    mode = str(cfg.get("topology_noise", "none") or "none").lower()
    if mode == "none":
        return

    protect = bool(cfg.get("protect_anchors_in_noise", True))

    if mode == "swap_labels":
        frac = float(cfg.get("swap_fraction", 0.0))
        swap_leaf_labels_with_lengths(t, frac=frac, anchors=anchors, protect_anchors=protect)
        return

    if mode == "nni":
        moves = int(cfg.get("nni_moves", 0))
        nni_perturb(t, moves=moves, anchors=anchors, protect_anchors=protect)
        return

    raise ValueError(f"Unknown topology_noise mode: {mode}")


def swap_leaf_labels_with_lengths(t: Tree, frac: float, anchors: Set[str], protect_anchors: bool = True):
    """Swap leaf labels and their pendant branch lengths for a fraction of leaves.

    This keeps the topology fixed but perturbs the taxon-to-tip mapping, which can be
    used as a simple topology-noise proxy.
    """
    if frac <= 0.0:
        return

    leaves = list(t.iter_leaves())
    if protect_anchors and anchors:
        leaves = [lf for lf in leaves if lf.name not in anchors]

    n = len(leaves)
    if n < 2:
        return

    k = int(round(frac * n))
    k = max(2, min(k, n))
    if k % 2 == 1:
        k -= 1
    if k < 2:
        return

    chosen = random.sample(leaves, k)
    random.shuffle(chosen)

    for i in range(0, k, 2):
        a = chosen[i]
        b = chosen[i + 1]
        a.name, b.name = b.name, a.name
        a.dist, b.dist = b.dist, a.dist


def nni_perturb(t: Tree, moves: int, anchors: Set[str], protect_anchors: bool = True):
    """Apply a small number of rooted NNI moves.

    Pattern:
      parent u has children [v, a], and v has children [b, c].
      Swap a with either b or c.
    """
    if moves <= 0:
        return

    def contains_anchor(node: Tree) -> bool:
        return any((lf.name in anchors) for lf in node.iter_leaves())

    for _ in range(moves):
        candidates = []
        for v in t.traverse("preorder"):
            u = v.up
            if u is None:
                continue
            # rooted binary requirement locally
            if len(u.children) != 2 or len(v.children) != 2:
                continue

            # sibling 'a' is the other child of u
            a = u.children[0] if u.children[1] is v else u.children[1]
            b, c = v.children[0], v.children[1]

            if protect_anchors and anchors:
                if contains_anchor(a) or contains_anchor(b) or contains_anchor(c):
                    continue

            candidates.append((u, v, a, b, c))

        if not candidates:
            return

        u, v, a, b, c = random.choice(candidates)
        swap_with = b if random.random() < 0.5 else c

        # Preserve parent-edge lengths by swapping the stored .dist values
       
        a_dist = float(getattr(a, "dist", 0.0) or 0.0)
        sw_dist = float(getattr(swap_with, "dist", 0.0) or 0.0)

        # Detach and swap
        a.detach()
        swap_with.detach()
        u.add_child(swap_with)
        v.add_child(a)

        swap_with.dist = positive(a_dist)
        a.dist = positive(sw_dist)


# Length scaling

def apply_length_scaling(t: Tree, cfg: dict, anchors: Set[str], base_height: float):
    mode = cfg.get("length_scaling", "none")
    if mode == "none":
        return

    mu = float(cfg.get("length_scale_params", {}).get("mu", 0.0))
    sigma = float(cfg.get("length_scale_params", {}).get("sigma", 0.2))
    preserve_paths = bool(cfg.get("preserve_paths_between_anchors", False))

    def is_pendant(n) -> bool:
        return n.is_leaf()

    def is_internal_edge(n) -> bool:
        return not n.is_leaf() and not n.is_root()

    protected_nodes = set()
    if preserve_paths and len(anchors) >= 2:
        # Collect minimal connecting subtree of anchors
        anchor_leaves = [x for x in t.iter_leaves() if x.name in anchors]
        if anchor_leaves:
            # Mark all nodes on all anchor-to-anchor paths as protected
            for i in range(len(anchor_leaves)):
                for j in range(i + 1, len(anchor_leaves)):
                    path = t.get_common_ancestor(anchor_leaves[i], anchor_leaves[j]).traverse()
                    for n in path:
                        protected_nodes.add(n)

    def lognormal_scale() -> float:
        # Multiplicative factor
        return math.exp(random.gauss(mu, sigma))

    if mode == "global":
        s = lognormal_scale()
        scale_all_edges(t, s)

    elif mode == "global_uniform":
        # Draw a single global multiplicative factor uniformly from [low, high].
        params = cfg.get("length_scale_params", {}) or {}
        low = float(params.get("low", 1.0))
        high = float(params.get("high", 1.0))
        if high < low:
            low, high = high, low
        s = random.uniform(low, high)
        scale_all_edges(t, s)

    elif mode == "edgewise_lognormal":
        for n in t.traverse():
            if preserve_paths and n in protected_nodes:
                continue
            n.dist = positive(n.dist * lognormal_scale())

    elif mode == "pendant_only_jitter":
        for n in t.iter_leaves():
            if preserve_paths and n in protected_nodes:
                continue
            n.dist = positive(n.dist * lognormal_scale())

    elif mode == "internal_only_jitter":
        for n in t.traverse():
            if is_internal_edge(n):
                if preserve_paths and n in protected_nodes:
                    continue
                n.dist = positive(n.dist * lognormal_scale())

    elif mode == "clade_local":
        # Pick 1-3 random clades to scale
        clades = collect_internal_clades(t)
        random.shuffle(clades)
        k = random.randint(1, min(3, max(1, len(clades))))
        chosen = clades[:k]
        for node, _ in chosen:
            s = lognormal_scale()
            for n in node.traverse():
                if preserve_paths and n in protected_nodes:
                    continue
                n.dist = positive(n.dist * s)

    # Optional renormalization of root height
    renorm = cfg.get("renormalize_root_height", "none")
    if renorm == "to_base":
        h = root_height(t)
        if h > 0:
            scale_all_edges(t, base_height / h)


# Main generator

def generate_sets_of_leaves(base_tree: Tree, cfg: dict, base_leaves: List[str]) -> Tuple[List[Set[str]], List[dict]]:
    """
    Build n_trees leaf sets satisfying overlap constraints as best as possible.
    Returns sets and per-tree notes (e.g., if constraints were relaxed).
    """
    n_trees = int(cfg["n_trees"])
    base_size = len(base_leaves)
    anchors = choose_anchor_taxa(base_leaves, int(cfg.get("anchor_taxa_count", 0)))
    cfg["anchors_explicit"] = list(anchors)

    # Target sizes
    size_range = cfg.get("target_tree_size_range", None)
    if size_range is not None:
        size_range = (int(size_range[0]), int(size_range[1]))
    sizes = sample_target_sizes(n_trees, base_size, int(cfg["min_leaves_per_tree"]), size_range)

    prune_mode = cfg.get("prune_mode", "mixed")
    clade_bias = float(cfg.get("clade_selection_bias", 0.6))
    clade_size_pref = cfg.get("clade_size_pref", "by_size")
    blockiness = float(cfg.get("leaf_prune_blockiness", 0.3))
    max_retry = int(cfg.get("max_retry_per_tree", 200))
    strict = bool(cfg.get("strict_constraints", True))
    lo, hi = cfg["pairwise_overlap_range"]

    leaf_sets: List[Set[str]] = []
    notes: List[dict] = []

    for i in range(n_trees):
        target_size = sizes[i]
        # Anchors must be included
        target_size = max(target_size, len(anchors) + int(cfg["min_shared_leaves_per_pair"]))
        reason = None
        accepted = False
        # Progressive relaxation if necessary
        lo_i, hi_i = lo, hi

        for attempt in range(max_retry):
            # Choose construction flavor
            use_clade = (prune_mode == "clades") or (prune_mode == "mixed" and random.random() < clade_bias)
            if use_clade:
                cand = build_set_by_union_of_clades(base_tree, target_size, clade_size_pref)
            else:
                cand = build_set_by_random_leaves(base_leaves, target_size, blockiness, base_tree)
            # Enforce anchors
            if anchors:
                cand = set(cand) | set(anchors)
                if len(cand) > target_size:
                    # Prune extra (but never remove anchors)
                    extras = [x for x in cand if x not in anchors]
                    need_to_drop = len(cand) - target_size
                    if need_to_drop > 0 and len(extras) >= need_to_drop:
                        cand = set(random.sample(extras, len(extras) - need_to_drop)) | set(anchors)

            # Check overlaps
            ok, reason = overlaps_ok(cand, leaf_sets, {**cfg, "pairwise_overlap_range": (lo_i, hi_i)}, base_size)
            if ok:
                leaf_sets.append(cand)
                notes.append({
                    "tree_index": i,
                    "target_size": target_size,
                    "construction": "clade_union" if use_clade else "random_leaves",
                    "relaxed_bounds": None
                })
                accepted = True
                break

            # Try next attempt
            continue

        if not accepted:
            if strict:
                # Final relaxation try: widen bounds and accept the best found
                relax_step = float(cfg.get("relax_step", 0.05))
                lo_i = max(0.0, lo - relax_step)
                hi_i = min(1.0, hi + relax_step)
                # One more sweep with relaxed bounds
                for attempt in range(max_retry):
                    use_clade = (prune_mode == "clades") or (prune_mode == "mixed" and random.random() < clade_bias)
                    if use_clade:
                        cand = build_set_by_union_of_clades(base_tree, target_size, clade_size_pref)
                    else:
                        cand = build_set_by_random_leaves(base_leaves, target_size, blockiness, base_tree)
                    if anchors:
                        cand |= set(anchors)
                        if len(cand) > target_size:
                            extras = [x for x in cand if x not in anchors]
                            need_to_drop = len(cand) - target_size
                            if need_to_drop > 0 and len(extras) >= need_to_drop:
                                cand = set(random.sample(extras, len(extras) - need_to_drop)) | set(anchors)
                    ok, _ = overlaps_ok(cand, leaf_sets, {**cfg, "pairwise_overlap_range": (lo_i, hi_i)}, base_size)
                    if ok:
                        leaf_sets.append(cand)
                        notes.append({
                            "tree_index": i,
                            "target_size": target_size,
                            "construction": "clade_union" if use_clade else "random_leaves",
                            "relaxed_bounds": [lo_i, hi_i]
                        })
                        accepted = True
                        break
            if not accepted:
                # Give up constraints for this tree (record)
                use_clade = (prune_mode == "clades") or (prune_mode == "mixed" and random.random() < clade_bias)
                if use_clade:
                    cand = build_set_by_union_of_clades(base_tree, target_size, clade_size_pref)
                else:
                    cand = build_set_by_random_leaves(base_leaves, target_size, blockiness, base_tree)
                cand |= set(anchors)
                if len(cand) > target_size:
                    extras = [x for x in cand if x not in anchors]
                    need_to_drop = len(cand) - target_size
                    if need_to_drop > 0 and len(extras) >= need_to_drop:
                        cand = set(random.sample(extras, len(extras) - need_to_drop)) | set(anchors)
                leaf_sets.append(cand)
                notes.append({
                    "tree_index": i,
                    "target_size": target_size,
                    "construction": "clade_union" if use_clade else "random_leaves",
                    "relaxed_bounds": "failed_all"
                })

    # Post-fixes: coverage + per-leaf coverage
    if cfg.get("enforce_full_coverage", True) or cfg.get("per_leaf_min_coverage", 0) > 0:
        repair_for_full_coverage(set(base_leaves), leaf_sets, cfg)

    return leaf_sets, notes


def write_outputs(base_tree_path: str, base_tree: Tree, leaf_sets: List[Set[str]], notes: List[dict], cfg: dict):
    out_dir = cfg["out_dir"]
    trees_dir = os.path.join(out_dir, "trees")
    meta_dir = os.path.join(out_dir, "metadata")
    ensure_dir(out_dir)
    ensure_dir(trees_dir)
    ensure_dir(meta_dir)

    # Copy base tree for convenience and reproducibility
    copy_if_needed(base_tree_path, os.path.join(out_dir, "base_tree.nwk"))

    # Dump the resolved config actually used
    write_yaml(cfg, os.path.join(meta_dir, "config.yaml"))

    # Prepare metadata accumulators
    removed_rows = []  # jsonl
    summary_rows = []  # csv
    m = len(leaf_sets)
    base_leaves = set(get_leaf_names(base_tree))
    base_size = len(base_leaves)
    base_h = root_height(base_tree)

    # Generate and write trees
    fn_template = cfg.get("filename_template", "sim_{:03d}.nwk")
    contract_degree2 = bool(cfg.get("contract_degree2", True))
    write_formats = cfg.get("write_formats", ["newick"])
    anchors = set(cfg.get("anchors_explicit", []))  # just in case we later pass anchors explicitly

    # Collect Newick strings (trees) for combined output
    combined_newicks: List[str] = []

    for i, keep in enumerate(leaf_sets):
        pruned = prune_to_leaves(base_tree, keep, contract_degree2=contract_degree2)
        # Optional topology noise (applied after pruning)
        apply_topology_noise(pruned, cfg, anchors)
        # Optional length scaling/jitter (applied after topology noise)
        apply_length_scaling(pruned, cfg, anchors, base_h)

        # Get Newick as a string
        nwk_str = pruned.write(format=1)

        # Write tree
        fname = fn_template.format(i + 1)
        out_path = os.path.join(trees_dir, fname)
        # Newick is the default
        pruned.write(format=1, outfile=out_path)

        # Accumulate for combined output
        combined_newicks.append(nwk_str.strip())

        # Record metadata rows
        removed = sorted(list(base_leaves - keep))
        kept = sorted(list(keep))
        removed_rows.append({
            "tree_file": fname,
            "kept_leaves": kept,
            "removed_leaves": removed,
            "construction": notes[i]["construction"],
            "relaxed_bounds": notes[i]["relaxed_bounds"],
        })
        summary_rows.append([
            fname,
            len(keep),
            len(keep) / base_size,  # fraction of base leaves kept
            notes[i]["construction"],
            json.dumps(notes[i]["relaxed_bounds"])
        ])

    # Write the combined file
    combined_path = cfg.get("combined_output_file")
    if combined_path:
        combined_path = os.path.join(out_dir, combined_path)
        if cfg.get("combined_output_format", "newick").lower() == "nexus":
            with open(combined_path, "w", encoding="utf-8") as f:
                f.write("#NEXUS\nBegin trees;\n")
                for i, nwk in enumerate(combined_newicks, 1):
                    # tree names follow your per-file template numbering
                    tree_name = os.path.splitext(fn_template.format(i))[0]
                    f.write(f"  Tree {tree_name} = {nwk}\n")
                f.write("End;\n")
        else:
            # Newick. One tree per line
            with open(combined_path, "w", encoding="utf-8") as f:
                for nwk in combined_newicks:
                    f.write(nwk if nwk.endswith("\n") else nwk + "\n")

    # Pairwise overlap matrix
    M = pairwise_overlap_matrix(leaf_sets, cfg["overlap_metric"], base_size)
    header = ["tree"] + [fn_template.format(i + 1) for i in range(m)]
    rows = []
    for i in range(m):
        rows.append([fn_template.format(i + 1)] + [f"{M[i][j]:.6f}" for j in range(m)])

    write_csv(["tree_file", "n_leaves", "frac_of_base_kept", "construction", "relaxed_bounds"], summary_rows,
              os.path.join(meta_dir, "summary.csv"))
    write_csv(header, rows, os.path.join(meta_dir, "pairwise_overlap.csv"))

    # Leaf coverage
    counts = leaf_counts(leaf_sets)
    cov_rows = [[leaf, counts.get(leaf, 0)] for leaf in sorted(base_leaves)]
    write_csv(["leaf", "count"], cov_rows, os.path.join(meta_dir, "leaf_coverage.csv"))

    # Removed items JSONL
    write_jsonl(removed_rows, os.path.join(meta_dir, "removed_items.jsonl"))

    # Optional zip
    if bool(cfg.get("zip_output", False)):
        zip_base = os.path.abspath(out_dir).rstrip(os.sep)
        shutil.make_archive(zip_base, "zip", out_dir)


def resolve_defaults(cfg_in: dict) -> dict:
    """Apply sane defaults and cast types."""
    cfg = dict(cfg_in)  # shallow copy
    # Required
    if "base_tree_path" not in cfg:
        raise ValueError("config missing 'base_tree_path'")
    if "out_dir" not in cfg:
        raise ValueError("config missing 'out_dir'")
    cfg.setdefault("random_seed", 7)
    cfg.setdefault("n_trees", 40)
    cfg.setdefault("overlap_metric", "jaccard")  # or base_fraction
    cfg.setdefault("pairwise_overlap_range", [0.10, 0.90])
    cfg.setdefault("min_shared_leaves_per_pair", 2)
    cfg.setdefault("per_leaf_min_coverage", 1)
    cfg.setdefault("min_leaves_per_tree", 4)
    cfg.setdefault("enforce_full_coverage", True)
    cfg.setdefault("anchor_taxa_count", 0)

    cfg.setdefault("prune_mode", "mixed")  # leaves | clades | mixed
    cfg.setdefault("clade_selection_bias", 0.6)
    cfg.setdefault("clade_size_pref", "by_size")  # uniform | by_size | small_first | large_first
    cfg.setdefault("leaf_prune_blockiness", 0.3)
    cfg.setdefault("contract_degree2", True)

    cfg.setdefault("overlap_strategy", "sample_target_sizes")
    cfg.setdefault("overlap_distribution", "uniform")

    cfg.setdefault("length_scaling", "none")
    cfg.setdefault("length_scale_params", {"mu": 0.0, "sigma": 0.2})
    cfg.setdefault("renormalize_root_height", "none")
    cfg.setdefault("preserve_paths_between_anchors", False)

    # Optional realism noise (applied after pruning each tree)
    cfg.setdefault("topology_noise", "none")          # none | swap_labels | nni
    cfg.setdefault("swap_fraction", 0.0)              # for swap_labels
    cfg.setdefault("nni_moves", 0)                    # for nni
    cfg.setdefault("protect_anchors_in_noise", True)  # don't move anchors / anchor clades

    cfg.setdefault("write_formats", ["newick"])
    cfg.setdefault("filename_template", "sim_{:03d}.nwk")
    cfg.setdefault("emit_metadata", True)
    cfg.setdefault("zip_output", False)
    cfg.setdefault("combined_output_file", None)         # e.g., "all_trees.nwk" or "all_trees.nexus"
    cfg.setdefault("combined_output_format", "newick")   # "newick" or "nexus"

    cfg.setdefault("max_retry_per_tree", 200)
    cfg.setdefault("strict_constraints", True)
    cfg.setdefault("relax_step", 0.05)

    # Type casting & sanity
    pr = cfg["pairwise_overlap_range"]
    if not (isinstance(pr, (list, tuple)) and len(pr) == 2):
        raise ValueError("pairwise_overlap_range must be [min, max]")
    cfg["pairwise_overlap_range"] = [float(pr[0]), float(pr[1])]
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Generate overlapping pruned trees from a base tree.")
    ap.add_argument("--config", required=True, help="Path to YAML config file.")
    args = ap.parse_args()

    cfg_raw = read_yaml(args.config)
    cfg = resolve_defaults(cfg_raw)

    random.seed(int(cfg["random_seed"]))

    # Prepare output dirs
    ensure_dir(cfg["out_dir"])
    ensure_dir(os.path.join(cfg["out_dir"], "trees"))
    ensure_dir(os.path.join(cfg["out_dir"], "metadata"))

    # Load base tree
    base_tree_path = cfg["base_tree_path"]
    base_tree = load_tree_with_positive_lengths(base_tree_path, eps=1e-8)
    base_leaves = get_leaf_names(base_tree)
    if len(base_leaves) < max(int(cfg["min_leaves_per_tree"]), 4):
        raise ValueError(f"Base tree has too few leaves ({len(base_leaves)}).")

    # Build leaf sets
    leaf_sets, notes = generate_sets_of_leaves(base_tree, cfg, base_leaves)

    # generate_sets_of_leaves already populated cfg["anchors_explicit"] with the anchors it used (if any).
    cfg.setdefault("anchors_explicit", [])

    # Emit outputs
    write_outputs(base_tree_path, base_tree, leaf_sets, notes, cfg)

    print(f"Done. Wrote {len(leaf_sets)} trees to: {os.path.join(cfg['out_dir'], 'trees')}")
    if cfg.get("emit_metadata", True):
        print(f"Metadata written to: {os.path.join(cfg['out_dir'], 'metadata')}")


if __name__ == "__main__":
    main()
