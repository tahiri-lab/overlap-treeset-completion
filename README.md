# Overlapping phylogenetic tree set completion

This repository contains a Python implementation of a **set-wide phylogenetic tree completion** algorithm for collections of rooted trees with **partially overlapping taxon sets**.

For each target tree, the algorithm identifies maximal completion subtrees (MCSs) supported by the other trees, builds a weighted majority-rule consensus for each selected subtree, rescales branch lengths using distances derived from common leaves, and inserts the consensus subtree at the position that minimizes a quadratic objective function on the original target tree branches.

## What the implementation does

- Completes every tree in a multiset using overlap information from the other trees.
- Uses consensus maximal completion subtrees to infer missing taxa groups.
- Aggregates branch-length information across source trees.
- Supports optional post-processing to enforce a binary topology.
- Processes all matching input files in batch mode.

## Input format

Place one tree set per file in `input_multisets/`, using filenames of the form:

`multiset_X.txt`

Each line of a file must contain one rooted phylogenetic tree in Newick format. Branch lengths should be included.

## Running the script

Basic run:

```bash
python tree_set_completion.py
````

Optional arguments:

```bash
python tree_set_completion.py --k 3 --binary
```

* `--k` limits the number of common leaves used during insertion-distance estimation.
* `--binary` post-processes each completed tree to resolve multifurcations into a binary topology.

The script reads every `multiset_*.txt` file from `input_multisets/` and writes the corresponding completed multiset to:

`completed_multisets/completed_multiset_X.txt`

## Example dataset

A small toy dataset with 10 overlapping input trees is included for testing and demonstration:

* `input_multisets/multiset_1.txt`

This example uses a shared taxa across all trees plus partially overlapping extra taxa.

## Output format

For each input multiset, the output file contains one completed tree per line in Newick format:

* input: `input_multisets/multiset_X.txt`
* output: `completed_multisets/completed_multiset_X.txt`

## Dependencies

* Python 3.6+
* `ete3`
* `numpy`

Install them with:

```bash
pip install ete3 numpy
```

## Citation

If you use this repository in academic work, please cite the associated paper describing the overlapping tree set completion algorithm (in progress).

## License

This project is licensed under the [MIT License](LICENSE).
