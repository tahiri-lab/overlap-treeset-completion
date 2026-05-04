#!/usr/bin/env python3

"""
Independent per-tree branch-length scaling helper for tree-set completion.

This script is designed for multiset/tree-set completion workflows where one
input file contains multiple Newick trees, typically one tree per line. Each
input tree receives its own scale factor. The scale factors are saved in a JSON
file and can later be used to unscale the corresponding completed trees.

Suggested workflow when branch length scaling is needed:
    1. Scale every tree in one multiset/tree-set file.
    2. Run tree-set completion on the scaled multiset.
    3. Unscale the completed multiset using the saved per-tree factors.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

NUMBER_RE = re.compile(
    r"(?P<space>\s*)"
    r"(?P<num>[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?)"
)

@dataclass(eq=False)
class Node:
    children: List["Node"] = field(default_factory=list)
    suffix: str = ""
    parent: Optional["Node"] = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

def split_newick_records(text: str) -> List[str]:
    """Split text into Newick records using semicolons outside quotes."""
    records: List[str] = []
    buf: List[str] = []
    in_quote = False
    in_comment = False
    i = 0

    while i < len(text):
        ch = text[i]
        buf.append(ch)

        if in_comment:
            if ch == "]":
                in_comment = False

        elif in_quote:
            if ch == "'":
                # Newick escaped quote inside quoted label: ''
                if i + 1 < len(text) and text[i + 1] == "'":
                    i += 1
                    buf.append(text[i])
                else:
                    in_quote = False

        else:
            if ch == "[":
                in_comment = True
            elif ch == "'":
                in_quote = True
            elif ch == ";":
                record = "".join(buf).strip()
                if record:
                    records.append(record)
                buf = []

        i += 1

    leftover = "".join(buf).strip()
    if leftover:
        raise ValueError("Input contains a Newick tree without a terminating semicolon.")

    return records

class NewickParser:
    def __init__(self, text: str):
        self.text = text.strip()
        self.pos = 0

    def parse(self) -> Node:
        root = self.parse_subtree()
        self.skip_spaces()

        if self.pos >= len(self.text) or self.text[self.pos] != ";":
            raise ValueError("Expected semicolon at end of Newick tree.")

        self.pos += 1
        self.skip_spaces()

        if self.pos != len(self.text):
            raise ValueError("Unexpected text after final semicolon.")

        return root

    def parse_subtree(self) -> Node:
        self.skip_spaces()

        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of Newick string.")

        if self.text[self.pos] == "(":
            self.pos += 1
            children: List[Node] = []

            while True:
                child = self.parse_subtree()
                children.append(child)
                self.skip_spaces()

                if self.pos >= len(self.text):
                    raise ValueError("Unexpected end inside internal node.")

                ch = self.text[self.pos]

                if ch == ",":
                    self.pos += 1
                    continue

                if ch == ")":
                    self.pos += 1
                    break

                raise ValueError(f"Unexpected character inside child list: {ch!r}")

            suffix = self.parse_suffix()
            node = Node(children=children, suffix=suffix)

            for child in children:
                child.parent = node

            return node

        suffix = self.parse_suffix()
        if not suffix:
            raise ValueError("Found an empty leaf label.")

        return Node(children=[], suffix=suffix)

    def parse_suffix(self) -> str:
        """Parse node label/comment/branch-length text until ',', ')' or ';'."""
        start = self.pos
        in_quote = False
        in_comment = False

        while self.pos < len(self.text):
            ch = self.text[self.pos]

            if in_comment:
                if ch == "]":
                    in_comment = False

            elif in_quote:
                if ch == "'":
                    if self.pos + 1 < len(self.text) and self.text[self.pos + 1] == "'":
                        self.pos += 1
                    else:
                        in_quote = False

            else:
                if ch == "[":
                    in_comment = True
                elif ch == "'":
                    in_quote = True
                elif ch in ",);":
                    break

            self.pos += 1

        return self.text[start:self.pos].strip()

    def skip_spaces(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

def traverse(root: Node) -> Iterable[Node]:
    yield root
    for child in root.children:
        yield from traverse(child)

def leaves(root: Node) -> Iterable[Node]:
    for node in traverse(root):
        if node.is_leaf():
            yield node

def find_branch_colon(suffix: str) -> Optional[int]:
    """Return index of branch-length colon outside quotes/comments, if present."""
    in_quote = False
    in_comment = False
    i = 0

    while i < len(suffix):
        ch = suffix[i]

        if in_comment:
            if ch == "]":
                in_comment = False

        elif in_quote:
            if ch == "'":
                if i + 1 < len(suffix) and suffix[i + 1] == "'":
                    i += 1
                else:
                    in_quote = False

        else:
            if ch == "[":
                in_comment = True
            elif ch == "'":
                in_quote = True
            elif ch == ":":
                return i

        i += 1

    return None

def branch_length_span(suffix: str) -> Optional[Tuple[int, int, float]]:
    """
    Return (start, end, value) for the numeric branch length inside suffix.

    Example:
        A:0.123[comment] -> start/end span the text '0.123'.
    """
    colon = find_branch_colon(suffix)
    if colon is None:
        return None

    m = NUMBER_RE.match(suffix[colon + 1 :])
    if not m:
        raise ValueError(f"Could not parse branch length in suffix: {suffix!r}")

    start = colon + 1 + m.start("num")
    end = colon + 1 + m.end("num")
    value = float(m.group("num"))

    if not math.isfinite(value):
        raise ValueError(f"Non-finite branch length in suffix: {suffix!r}")

    return start, end, value

def get_branch_length(node: Node) -> Optional[float]:
    span = branch_length_span(node.suffix)
    if span is None:
        return None
    return span[2]

def format_length(x: float) -> str:
    if not math.isfinite(x):
        raise ValueError(f"Non-finite branch length: {x}")
    return f"{x:.12g}"

def scale_all_branch_lengths(root: Node, factor: float) -> None:
    """Multiply every branch length present in the tree by factor."""
    if factor <= 0 or not math.isfinite(factor):
        raise ValueError(f"Scale factor must be a positive finite number; got {factor!r}.")

    for node in traverse(root):
        span = branch_length_span(node.suffix)
        if span is not None:
            start, end, value = span
            node.suffix = node.suffix[:start] + format_length(value * factor) + node.suffix[end:]

def leaf_name_from_suffix(suffix: str) -> str:
    """Extract a leaf label from a Newick leaf suffix."""
    suffix = suffix.strip()
    if not suffix:
        raise ValueError("Unnamed leaf found.")

    if suffix.startswith("'"):
        chars: List[str] = []
        i = 1

        while i < len(suffix):
            ch = suffix[i]

            if ch == "'":
                if i + 1 < len(suffix) and suffix[i + 1] == "'":
                    chars.append("'")
                    i += 2
                    continue

                name = "".join(chars)
                if not name:
                    raise ValueError("Unnamed quoted leaf found.")
                return name

            chars.append(ch)
            i += 1

        raise ValueError(f"Unterminated quoted leaf label in suffix: {suffix!r}")

    chars: List[str] = []
    in_comment = False

    for ch in suffix:
        if in_comment:
            if ch == "]":
                in_comment = False
            continue

        if ch == "[":
            in_comment = True
            continue

        if ch == ":":
            break

        chars.append(ch)

    name = "".join(chars).strip()
    if not name:
        raise ValueError(f"Unnamed leaf found in suffix: {suffix!r}")

    return name

def validate_leaf_names(root: Node) -> None:
    """
    Require leaf names to be non-empty.

    Duplicate leaf names are not forbidden here because the scaling methods only
    need leaf nodes and branch lengths, not a leaf-name dictionary. However,
    duplicate leaf names may still be problematic for subsequent tree-completion
    algorithms.
    """
    for leaf in leaves(root):
        leaf_name_from_suffix(leaf.suffix)


def validate_branch_lengths(root: Node, tree_label: str) -> None:
    """
    Require all non-root branches to have finite non-negative branch lengths.

    Root branch length, if present, is also required to be finite and non-negative.
    """
    missing: List[str] = []

    for node in traverse(root):
        length = get_branch_length(node)

        if node.parent is not None and length is None:
            if node.is_leaf():
                missing.append(leaf_name_from_suffix(node.suffix))
            else:
                missing.append("<internal node>")

            if len(missing) >= 5:
                break

        if length is not None and length < 0:
            raise ValueError(f"{tree_label} contains a negative branch length: {length}")

    if missing:
        preview = ", ".join(missing)
        raise ValueError(
            f"{tree_label} has non-root branches without branch lengths. "
            f"Examples: {preview}. Branch-length scaling requires branch lengths."
        )


def distance_between_leaves(a: Node, b: Node) -> float:
    def distances_to_ancestors(node: Node) -> Dict[Node, float]:
        d: Dict[Node, float] = {node: 0.0}
        cur = node
        accum = 0.0

        while cur.parent is not None:
            edge = get_branch_length(cur)
            if edge is None:
                raise ValueError("Found a non-root branch without a branch length.")

            accum += edge
            cur = cur.parent
            d[cur] = accum

        return d

    da = distances_to_ancestors(a)
    db = distances_to_ancestors(b)
    common = set(da).intersection(db)

    if not common:
        raise ValueError("Internal error: leaves have no common ancestor.")

    return min(da[x] + db[x] for x in common)


def all_pairwise_leaf_distances(root: Node) -> List[float]:
    leaf_nodes = list(leaves(root))

    if len(leaf_nodes) < 2:
        raise ValueError("At least two leaves are required for pairwise-distance scaling.")

    distances = [distance_between_leaves(a, b) for a, b in combinations(leaf_nodes, 2)]

    if any(d < 0 for d in distances):
        raise ValueError("Negative pairwise distance encountered.")

    return distances


def total_tree_length(root: Node) -> float:
    """Return the sum of all non-root branch lengths."""
    total = 0.0

    for node in traverse(root):
        if node.parent is None:
            continue

        length = get_branch_length(node)
        if length is None:
            raise ValueError("Found a non-root branch without a branch length.")

        total += length

    return total


def estimate_independent_factor(root: Node, method: str) -> Tuple[float, dict]:
    """
    Return (scale_factor, details) for one tree.

    scaled_branch_length = original_branch_length * scale_factor
    """
    leaf_count = sum(1 for _ in leaves(root))

    if method == "median-all-pairwise":
        distances = all_pairwise_leaf_distances(root)
        value = median(distances)

        if value <= 0:
            raise ValueError(
                "Median pairwise leaf distance is zero. Cannot use median-all-pairwise. "
                "Try mean-all-pairwise or total-tree-length if appropriate."
            )

        return 1.0 / value, {
            "leaf_count": leaf_count,
            "pair_count": len(distances),
            "normalization_value": value,
            "normalization_value_mean_pairwise": mean(distances),
            "normalization_value_median_pairwise": value,
        }

    if method == "mean-all-pairwise":
        distances = all_pairwise_leaf_distances(root)
        value = mean(distances)

        if value <= 0:
            raise ValueError("Mean pairwise leaf distance is zero. Cannot scale this tree.")

        return 1.0 / value, {
            "leaf_count": leaf_count,
            "pair_count": len(distances),
            "normalization_value": value,
            "normalization_value_mean_pairwise": value,
            "normalization_value_median_pairwise": median(distances),
        }

    if method == "total-tree-length":
        value = total_tree_length(root)

        if value <= 0:
            raise ValueError("Total tree length is zero. Cannot scale this tree.")

        return 1.0 / value, {
            "leaf_count": leaf_count,
            "pair_count": None,
            "normalization_value": value,
            "normalization_value_total_tree_length": value,
        }

    raise ValueError(f"Unknown scaling method: {method}")

def write_newick(node: Node) -> str:
    if node.is_leaf():
        return node.suffix

    return "(" + ",".join(write_newick(child) for child in node.children) + ")" + node.suffix

def parse_trees_from_file(path: str) -> List[Node]:
    text = Path(path).read_text(encoding="utf-8")
    records = split_newick_records(text)

    if not records:
        raise ValueError(f"No Newick trees found in {path!r}.")

    return [NewickParser(record).parse() for record in records]

def write_trees_to_file(path: str, trees: List[Node]) -> None:
    text = "\n".join(write_newick(tree) + ";" for tree in trees) + "\n"
    Path(path).write_text(text, encoding="utf-8")

def make_factor_file(args: argparse.Namespace, tree_entries: List[dict]) -> dict:
    return {
        "version": 3,
        "mode": "tree-set-independent-per-tree",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "meaning": "scaled_branch_length = original_branch_length * scale_factor",
        "input_multiset_file": args.input,
        "scaled_multiset_file": args.output,
        "tree_count": len(tree_entries),
        "trees": tree_entries,
    }

def cmd_scale(args: argparse.Namespace) -> int:
    trees = parse_trees_from_file(args.input)
    factor_entries: List[dict] = []

    for idx, tree in enumerate(trees, start=1):
        label = f"tree {idx}"

        validate_branch_lengths(tree, label)
        validate_leaf_names(tree)

        factor, details = estimate_independent_factor(tree, args.method)
        scale_all_branch_lengths(tree, factor)

        factor_entries.append(
            {
                "tree_index": idx,
                "scale_factor": factor,
                "unscale_factor": 1.0 / factor,
                "details": details,
            }
        )

        sys.stderr.write(
            f"Tree {idx}: scale_factor={factor:.12g}, "
            f"unscale_factor={1.0 / factor:.12g}\n"
        )

    write_trees_to_file(args.output, trees)

    factor_data = make_factor_file(args, factor_entries)
    Path(args.factors).write_text(
        json.dumps(factor_data, indent=2) + "\n",
        encoding="utf-8",
    )

    sys.stderr.write("Scaling complete.\n")
    sys.stderr.write(f"  trees scaled: {len(trees)}\n")
    sys.stderr.write(f"  method: {args.method}\n")
    sys.stderr.write(f"  scaled multiset: {args.output}\n")
    sys.stderr.write(f"  factors file: {args.factors}\n")

    return 0

def parse_tree_indices(value: Optional[str], tree_count: int, factor_count: int) -> List[int]:
    if value is None:
        if tree_count != factor_count:
            raise ValueError(
                f"The unscale input contains {tree_count} tree(s), but the factor file "
                f"contains {factor_count} tree(s). Provide --tree-indices to specify "
                "which original scale factor applies to each completed tree."
            )

        return list(range(1, tree_count + 1))

    parts = [x.strip() for x in value.split(",") if x.strip()]

    if not parts:
        raise ValueError("--tree-indices was provided but no indices were found.")

    try:
        indices = [int(x) for x in parts]
    except ValueError as exc:
        raise ValueError(
            "--tree-indices must be a comma-separated list of integers, e.g. 1,2,5."
        ) from exc

    if len(indices) != tree_count:
        raise ValueError(
            f"--tree-indices gives {len(indices)} index/indices, but the unscale input "
            f"contains {tree_count} tree(s). These counts must match."
        )

    return indices

def cmd_unscale(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.factors).read_text(encoding="utf-8"))

    entries = data.get("trees")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Could not read tree factors from {args.factors!r}.")

    factors_by_index: Dict[int, float] = {}

    for entry in entries:
        idx = int(entry["tree_index"])
        factors_by_index[idx] = float(entry["unscale_factor"])

    trees = parse_trees_from_file(args.input)
    indices = parse_tree_indices(args.tree_indices, len(trees), len(entries))

    for output_position, (tree, original_index) in enumerate(zip(trees, indices), start=1):
        if original_index not in factors_by_index:
            raise ValueError(
                f"No factor found for original tree index {original_index}. "
                f"Available indices: {sorted(factors_by_index)}"
            )

        factor = factors_by_index[original_index]
        scale_all_branch_lengths(tree, factor)

        sys.stderr.write(
            f"Output tree {output_position}: using original tree index {original_index}, "
            f"unscale_factor={factor:.12g}\n"
        )

    write_trees_to_file(args.output, trees)

    sys.stderr.write("Unscaling complete.\n")
    sys.stderr.write(f"  trees unscaled: {len(trees)}\n")
    sys.stderr.write(f"  output file: {args.output}\n")

    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent per-tree branch-length scaling for tree-set completion "
            "multiset files."
        )
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_scale = sub.add_parser(
        "scale",
        help="Scale every Newick tree in one multiset/tree-set input file.",
    )
    p_scale.add_argument(
        "--input",
        required=True,
        help="Input multiset Newick file containing one or more trees.",
    )
    p_scale.add_argument(
        "--output",
        required=True,
        help="Output multiset Newick file for scaled trees.",
    )
    p_scale.add_argument(
        "--factors",
        required=True,
        help="JSON file for per-tree scaling/unscaling factors.",
    )
    p_scale.add_argument(
        "--method",
        choices=["median-all-pairwise", "mean-all-pairwise", "total-tree-length"],
        default="median-all-pairwise",
        help="Independent per-tree scaling method. Default: median-all-pairwise.",
    )
    p_scale.set_defaults(func=cmd_scale)

    p_unscale = sub.add_parser(
        "unscale",
        help="Apply saved inverse scale factors to completed/scaled multiset trees.",
    )
    p_unscale.add_argument(
        "--input",
        required=True,
        help="Input Newick file containing scaled completed tree(s).",
    )
    p_unscale.add_argument(
        "--output",
        required=True,
        help="Output Newick file for unscaled completed tree(s).",
    )
    p_unscale.add_argument(
        "--factors",
        required=True,
        help="JSON factor file produced by the scale command.",
    )
    p_unscale.add_argument(
        "--tree-indices",
        default=None,
        help=(
            "Comma-separated original tree indices whose factors should be used for the "
            "input trees, e.g. 3,7. If omitted, the input must contain the same number "
            "of trees as the factor file and factors are applied by position."
        ),
    )
    p_unscale.set_defaults(func=cmd_unscale)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return args.func(args)
    except Exception as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
