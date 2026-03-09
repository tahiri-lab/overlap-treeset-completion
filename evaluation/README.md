# Evaluation

This folder contains the data and scripts used to evaluate the overlapping phylogenetic tree set completion method and its baselines.

## Contents

- `data/` — evaluation input tree sets and completed sets.
- evaluation scripts — code for running completion experiments, computing distance measures, and generating result summaries and plots.
- This folder also contain saved intermediate or final evaluation outputs in `.pkl` and/or `.csv` format. These files are included for reproducibility and faster reuse of previously computed results. They can be used directly to regenerate tables and plots without rerunning the full evaluation pipeline. Unless stated otherwise in the corresponding script, saved result files are derived from the datasets in `data/` and the evaluation procedures described in the paper.


## Evaluation setup

The evaluation uses four species groups:

- Amphibians
- Mammals
- Sharks
- Squamates

For each group, 20 reference subsets were generated with sizes from 50 to 145 taxa (step size 5).  
Each reference tree was used to construct an overlapping tree set of 30 trees with controlled taxon overlap, full taxon coverage, fixed anchor taxa, minor topological perturbations, and branch-length scaling noise.

The union of taxa in each overlapping tree set defines the completion target, and the corresponding unpruned reference tree is used as ground truth.

## Compared methods

The experiments compare the proposed method against four baselines:

- Root Attach
- Nearest Parent
- No MCS
- Supertree-based completion

## Evaluation metrics

Completed trees are compared to their reference trees using:

- Normalized Robinson-Foulds (RF) distance for topology
- Normalized Branch Score Distance (BSD) for branch lengths

For each subset and method, distances are computed for all completed trees and summarized using the median.

## Analyses included

The scripts in this folder support the main analyses reported in the paper, including:

- comparative evaluation across methods
- subset size trend analysis
- majority-rule consensus evaluation
- statistical testing across subsets

## Notes

- The refined binary version of completed trees is used for RF-based evaluation.
- BSD is unaffected by multifurcation resolution because pairwise leaf-to-leaf distances are preserved.
- Some scripts may require external tools or additional dependencies depending on the baseline being reproduced.

---
For details on dataset construction, baseline definitions, and evaluation methodology, see the associated paper.
