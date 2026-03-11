# Dataset preparation scripts

This folder contains the Python scripts used to prepare species subsets and phylogenetic tree inputs for the evaluation analyses.

## Overview

The workflow starts from a species list (`all_species_lists.csv`) containing all available taxa for each species group. From this file, 20 base species subsets are generated for each group, either with a standard sampling approach (`make_base_species_lists.py`) or with a phylogenetically stratified approach (`make_base_species_lists_phylo_stratified.py`). The standard script builds 20 base lists across groups from a wide CSV, while the stratified version uses a full phylogeny to distribute sampled taxa across deeper clades.

These base species lists are then used to prune full phylogenetic trees downloaded separately from the VertLife project website. The script `20_base_trees_preparation.py` matches each base subset to a corresponding full tree and writes 20 pruned “base trees” (for example, `*_base_trees20.txt`).

From each of these 20 base trees, overlapping tree sets are generated in two steps. `overlap_subset_creation.py` loops over each base tree and creates multiple overlapping subsets per tree, while `gen_pruned_trees.py` provides the core functions for generating overlapping leaf sets, pruning trees, and optionally adding topology or branch-length perturbations.

## Folder structure

- `all_species_lists.csv`  
  List of taxa for all species groups.

- `make_base_species_lists.py`  
  Creates 20 base species subsets per group from the species list.

- `make_base_species_lists_phylo_stratified.py`  
  Alternative base-subset generator that uses a full phylogeny to sample taxa in a stratified way.

- `20_base_trees_preparation.py`  
  Prunes the full downloaded trees to match the 20 selected base subsets, producing 20 base trees per group.

- `overlap_subset_creation.py`  
  Uses each base tree to generate overlapping tree sets.

- `gen_pruned_trees.py`  
  Helper module used by the overlap-generation step for subset construction, pruning, overlap control, and tree perturbation options.

## Suggested execution order

1. Prepare the species list in `all_species_lists.csv`.
2. Generate 20 base species subsets with either:
   - `make_base_species_lists.py`, or
   - `make_base_species_lists_phylo_stratified.py`
3. Download the full phylogenetic trees for the relevant group(s) from the VertLife project website.
4. Run `20_base_trees_preparation.py` to prune the full trees into 20 base trees.
5. Run `overlap_subset_creation.py` (which uses `gen_pruned_trees.py`) to create overlapping tree sets from the 20 base trees.

## Notes

- Full phylogenetic trees are not included in this repository and must be downloaded manually.
- Some scripts are group-specific or require small path/input edits before running.
- Output file names in the scripts can be adjusted depending on the species group being processed.
