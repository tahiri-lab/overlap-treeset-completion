# Branch-length scaling for tree-set completion

`branch_scaler.py` is an optional preprocessing and postprocessing utility for tree-set / multiset completion workflows.

It is useful when a multiset of input phylogenetic trees contains branch lengths on very different numerical scales. The script scales each tree in the multiset independently, writes the scaled trees to a new Newick file, and saves the corresponding scale/unscale factors in a JSON file.

The scaled multiset can then be used directly as input to tree-set completion. After completion, the completed trees can be returned to their original branch-length scales using the saved JSON factor file.

## Basic idea

The script applies multiplicative branch-length scaling.

For each tree:

```text
scaled_branch_length = original_branch_length * scale_factor
````

Later, to return the completed tree to its original scale:

```text
original_scale_branch_length = scaled_branch_length * unscale_factor
```

where:

```text
unscale_factor = 1 / scale_factor
```

The script does not change tree topology. It only multiplies branch lengths.

## Per-tree scaling

In tree-set completion, one input file contains a set of trees. Sometimes these trees have very different branch-length scales, especially if they were estimated from different genes, datasets, models, or pipelines.

Using one scaling factor for the entire multiset may not be appropriate if each tree has its own numerical scale. Therefore, this script assigns one independent scaling factor to each tree.

For example:

```text
tree 1 median pairwise distance = 0.01
tree 2 median pairwise distance = 20
tree 3 median pairwise distance = 3
```

After `median-all-pairwise` scaling:

```text
tree 1 median pairwise distance = 1
tree 2 median pairwise distance = 1
tree 3 median pairwise distance = 1
```

This places all trees into a comparable normalized branch-length space before tree-set completion.

## Requirements

Python 3. No external Python packages are required.

The input file should contain one or more Newick trees. Each tree must end with a semicolon.

Example:

```text
((A:1,B:1):1,C:2);
((A:10,B:10):10,C:20);
(A:1,B:2,C:3,D:4);
```

All non-root branches must have branch lengths.

Negative branch lengths are not allowed.

## Commands

The script has two subcommands:

```bash
python branch_scaler.py scale   ...
python branch_scaler.py unscale ...
```

Use `scale` before tree-set completion.

Use `unscale` after tree-set completion.

## Scaling methods

The script supports three independent per-tree scaling methods.

### 1. `median-all-pairwise`

This is the recommended default.

For each tree, the script computes all pairwise leaf-to-leaf distances and scales the tree by:

```text
scale_factor = 1 / median(all pairwise leaf-to-leaf distances)
```
Use this method when you want a robust default. It is less sensitive to outlier taxa and unusually long branches than the mean.

Recommended for most tree-set completion preprocessing workflows.

### 2. `mean-all-pairwise`

For each tree, the script computes all pairwise leaf-to-leaf distances and scales the tree by:

```text
scale_factor = 1 / mean(all pairwise leaf-to-leaf distances)
```
Use this method when branch lengths are relatively homogeneous and you do not expect strong outliers.

This method uses the whole distance structure of the tree, but it is more sensitive to long branches than `median-all-pairwise`.

### 3. `total-tree-length`

For each tree, the script computes the total non-root branch length and scales the tree by:

```text
scale_factor = 1 / total_tree_length
```

This method is simple and fast. However, it is sensitive to the number of taxa and the amount of tree resolution. Trees with more taxa usually have more branches, so total tree length may reflect taxon sampling as well as branch-length scale.

Use this method only when total-length normalization is specifically desired.

## Recommendations

Use `median-all-pairwise` as the default choice.

Use `mean-all-pairwise` when the trees have no extreme branch-length outliers and you want the average leaf-to-leaf distance to be normalized.

Use `total-tree-length` when you want simple whole-tree normalization, but be cautious if trees differ strongly in taxon number or sampling density.

Do not use branch-length scaling to force together trees whose branch lengths have incompatible meanings. For example, if one tree has branch lengths in substitutions/site and another tree has branch lengths in absolute time, a simple multiplicative scaling may not be biologically meaningful.

## When is external scaling useful

External per-tree scaling is useful when:

* trees in the multiset have very different numerical branch-length scales
* the same multiset contains trees estimated from heterogeneous sources
* some trees are globally much longer or shorter than others
* branch lengths differ by more than about 10x across trees

External scaling is usually less important when all trees were estimated using the same data type, model, pipeline, and branch-length units, and when global branch-length differences are biologically meaningful.

### Step 1: Scale all trees in a multiset file

Example:

```bash
python branch_scaler.py scale \
  --input input_multisets/multiset_1.txt \
  --output input_multisets_scaled/multiset_1.txt \
  --factors scaling_factors/multiset_1.factors.json \
  --method median-all-pairwise
```

This writes:

```text
input_multisets_scaled/multiset_1.txt
scaling_factors/multiset_1.factors.json
```

The scaled Newick file contains the same number of trees as the input file, in the same order.

The factor file records one scale factor and one unscale factor per tree.

### Example factor file

The JSON factor file has entries like this:

```json
{
  "version": 3,
  "mode": "tree-set-independent-per-tree",
  "method": "median-all-pairwise",
  "meaning": "scaled_branch_length = original_branch_length * scale_factor",
  "input_multiset_file": "input_multisets/multiset_1.txt",
  "scaled_multiset_file": "input_multisets_scaled/multiset_1.txt",
  "tree_count": 3,
  "trees": [
    {
      "tree_index": 1,
      "scale_factor": 0.25,
      "unscale_factor": 4.0,
      "details": {
        "leaf_count": 3,
        "pair_count": 3,
        "normalization_value": 4.0
      }
    },
    {
      "tree_index": 2,
      "scale_factor": 0.025,
      "unscale_factor": 40.0,
      "details": {
        "leaf_count": 3,
        "pair_count": 3,
        "normalization_value": 40.0
      }
    },
    {
      "tree_index": 3,
      "scale_factor": 0.2,
      "unscale_factor": 5.0,
      "details": {
        "leaf_count": 4,
        "pair_count": 6,
        "normalization_value": 5.0
      }
    }
  ]
}
```

Tree indices are 1-based.

### Step 2: Run tree-set completion on the scaled multiset

Use the scaled multiset file as input to tree-set completion.

Example scaled input:

```text
input_multisets_scaled/multiset_1.txt
```

After completion, suppose the completed scaled output is:

```text
completed_multisets/completed_multiset_1.txt
```

The completed trees are still on the scaled branch-length scale.

### Step 3: Unscale the completed multiset

Use the same factor file created during the `scale` step.

If the completed file contains the same number of trees in the same order as the original multiset, run:

```bash
python branch_scaler.py unscale \
  --input completed_multisets/completed_multiset_1.txt \
  --output completed_multisets_unscaled/completed_multiset_1.txt \
  --factors scaling_factors/multiset_1.factors.json
```

This applies:

```text
completed tree 1 -> unscale factor from original tree 1
completed tree 2 -> unscale factor from original tree 2
completed tree 3 -> unscale factor from original tree 3
...
```

The output file contains the completed trees returned to their original respective branch-length scales.

### Unscaling selected completed trees

If the completed file contains only selected trees from the original multiset, provide their original tree indices with `--tree-indices`.

Example:

```bash
python branch_scaler.py unscale \
  --input completed_selected_scaled.txt \
  --output completed_selected_unscaled.txt \
  --factors scaling_factors/multiset_1.factors.json \
  --tree-indices 3,7
```

This means:

```text
first tree in completed_selected_scaled.txt  -> use unscale factor from original tree 3
second tree in completed_selected_scaled.txt -> use unscale factor from original tree 7
```

The number of indices in `--tree-indices` must match the number of trees in the input file being unscaled.

## Full recommended workflow

```text
1. Start with one multiset file containing multiple original Newick trees.

2. Scale every tree independently:

   python branch_scaler.py scale \
     --input input_multisets/multiset_1.txt \
     --output input_multisets_scaled/multiset_1.txt \
     --factors scaling_factors/multiset_1.factors.json \
     --method median-all-pairwise

3. Run tree-set completion on:

   input_multisets_scaled/multiset_1.txt

4. Unscale the completed output:

   python branch_scaler.py unscale \
     --input completed_multisets/completed_multiset_1.txt \
     --output completed_multisets_unscaled/completed_multiset_1.txt \
     --factors scaling_factors/multiset_1.factors.json
```

## Practical diagnostic before scaling

Before deciding whether scaling is needed, compare simple branch-length summaries across trees in the multiset:

```text
median all-leaf pairwise distance
mean all-leaf pairwise distance
total tree length
```

As a practical rule of thumb:

```text
scale ratio < 2-5x:
    external scaling is usually optional

scale ratio around 5-10x:
    external scaling is reasonable

scale ratio > 10x:
    external scaling is recommended

scale ratio varies strongly depending on which summary is used:
    scaling may still help numerically, but completed branch lengths should be interpreted cautiously
```

These thresholds are practical guidelines (empirical), not strict mathematical rules.

## What scaling does and does not do

Scaling preserves relative branch lengths within each tree.

For example, if a tree has branch lengths:

```text
1, 2, 5, 10
```

and is scaled by `0.1`, the scaled branch lengths become:

```text
0.1, 0.2, 0.5, 1.0
```

The branch-length ratios are unchanged.

However, scaling does not make all branches fall between 0 and 1. The pairwise methods normalize a tree-level distance summary, not every individual branch.

## Additional notes

* The script preserves tree topology and leaf labels.

* The script rewrites Newick output, so exact whitespace and line wrapping are not preserved. This means the tree content is preserved, but the exact text formatting of the original file may change.

* Numeric notation may change. For example, `1e-1` may be written as `0.1`.

* All non-root branches must have branch lengths during scaling.

* The script scales every branch length it finds during both scaling and unscaling. Therefore, completed tree-set outputs should include branch lengths on inserted branches too.

* The script validates that leaf names are non-empty. Duplicate leaf names are not rejected by the scaler itself, but duplicate taxa may be problematic for subsequent tree-completion algorithms.

* Scaling makes branch-length magnitudes more comparable numerically, but it does not guarantee that branch lengths from different biological sources have the same meaning.

## Important notes

External scaling should not be used to force together branch lengths with incompatible biological interpretations.

For example:

```text
Tree 1 branch lengths = substitutions/site
Tree 2 branch lengths = absolute time
```

A single multiplicative scale factor may not make such trees biologically comparable.

Therefore, for branch-length-aware completion, all input trees should preferably have branch lengths with the same biological interpretation, such as all substitutions/site or all absolute time. Mixing different branch-length types, such as substitutions/site and absolute time, is not recommended because a simple multiplicative scaling factor cannot make those quantities biologically equivalent.

