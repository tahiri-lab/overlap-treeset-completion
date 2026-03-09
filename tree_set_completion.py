#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phylogenetic tree set completion algorithm.

 – Reads each multiset input_multisets/multiset_*.txt
 - Constructs consensus Maximal completion subtrees (MCS) with distict leaves
 – Inserts each consensus MCS to complete a tree
 – Writes completed sets to completed_multisets/completed_multiset_i.txt
"""

# import
import os, re, math, argparse, glob
from collections import defaultdict
import numpy as np
from ete3 import Tree
from itertools import count

# These are used by compute_weights_globally() and some helpers
global_T_set = []          # store the current list of host trees
global_T_i   = None        # store the current target tree

# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=None,
                help="k for the number of common leaves (default = |common|)")
ap.add_argument("--binary", action="store_true",
                help="post-process each completed tree to enforce a binary topology")
args = ap.parse_args()
K_USER = args.k
FORCE_BINARY = args.binary


# Section 1. Utility functions

def preprocess_newick(s):            # spaces to underscores inside labels
    return re.sub(r'(?<=\w) (?=\w)', '_', s)

def get_leaf_set_ete(t):             # returns set of leaf names
    return set([lf.name for lf in t.iter_leaves()])

def clear_internal_node_names(t):
    for n in t.traverse():
        if not n.is_leaf():
            n.name = ""

def common_leaves_overall(T_set):
    """Common leaf set across all trees in T_set."""
    if not T_set: return set()
    result = get_leaf_set_ete(T_set[0])
    for T in T_set[1:]:
        result &= get_leaf_set_ete(T)
    return result

def all_leaves_overall(T_set):
    result = set()
    for T in T_set:
        result |= get_leaf_set_ete(T)
    return result

def get_common_leaves(host_list):
    if not host_list: return set()
    S = get_leaf_set_ete(host_list[0])
    for h in host_list[1:]:
        S &= get_leaf_set_ete(h)
    return S

def remove_common_leaves_copy(tree, common):
    """
    Return a copy of the tree with leaves in 'common' removed.
    Then simplify by removing degree-2 internal nodes, but keep the same
    root as in the input tree.
    """
    t = tree.copy("deepcopy")
    for lf in t.iter_leaves():
        if lf.name in common:
            lf.detach()

    # prune internal deg-2 nodes (suppress)
    changed = True
    while changed:
        changed = False
        for n in list(t.traverse("postorder")):
            if n.is_leaf():
                continue
            if n.up is None:
                continue
            if len(n.children) == 1:
                c = n.children[0]
                d = n.dist
                # reattach c to parent of n
                p = n.up
                c.detach()
                p.add_child(c)
                c.dist += d
                n.detach()
                changed = True
    return t

def leafset_hash(leaf_names):
    """Hashable key for a set of leaves."""
    return tuple(sorted(leaf_names))

def subtree_rooted_at_lca(tree, leafset):
    """
    Return the subtree of `tree` rooted at the LCA of all leaves in leafset.
    If leafset has size 1, it's the single leaf.
    """
    leaves = [tree&nm for nm in leafset]
    if not leaves:
        return None
    if len(leaves) == 1:
        return leaves[0]
    return tree.get_common_ancestor(leaves)

def mark_original_nodes(t):
    """
    Mark and enumerate the edges of the original target tree before any insertion.
    We store an edge-level flag/index on the child node:
      child.is_original_edge == True means the edge (child.up -> child) is original,
      child.edge_index is its fixed DFS order index.
    """
    # Assign flags first
    for n in t.traverse():
        if n.up is not None:
            n.add_feature("is_original_edge", True)
        else:
            n.add_feature("is_original_edge", False)

    # Depth-first enumeration of original edges
    idx = 1
    for n in t.traverse("preorder"):
        if n.up is not None:
            # this is an original edge (before any insertion)
            n.add_feature("edge_index", idx)
            idx += 1
        else:
            n.add_feature("edge_index", -1)  # root has no incoming edge

def binarize_multifurcations(t, eps=0.0, name_prefix="_bin"):
    """
    Deterministically refine any multifurcating internal node into a binary tree.

    - Any node with > 2 children is resolved by repeatedly grouping together
      the two children whose subtrees have the lexicographically smallest
      leaf-name minima.
    - New internal nodes get branch length `eps` and temporary names like
      '_bin0', '_bin1', ... to avoid anonymous ':0' artifacts.

    Properties:
      * Leaf set (names and count) is unchanged.
      * No new leaves are created; only internal nodes are added.
      * If eps = 0.0, all leaf–leaf and root–leaf path distances are preserved.
      * Deterministic: depends only on the tree structure and leaf names,
        not on child insertion order.

    The tree is modified in-place and also returned.
    """
    counter = count()

    def min_leaf_name(nd):
        # Deterministic key based on leaf labels below this node
        # This assumes leaf names are unique and comparable (strings).
        return min(leaf.name for leaf in nd.iter_leaves())

    for nd in t.traverse("postorder"):
        # While this node has more than 2 children, resolve the polytomy
        while len(nd.children) > 2:
            # Create a deterministic ordering of children based on their
            # lexicographically minimal leaf name
            children = list(nd.children)
            children_sorted = sorted(children, key=min_leaf_name)

            # Take the two lexicographically smallest subtrees
            c1, c2 = children_sorted[0], children_sorted[1]

            # Detach them from nd
            c1.detach()
            c2.detach()

            # New internal node with eps-length edge from nd
            new = nd.add_child(
                name=f"{name_prefix}{next(counter)}",
                dist=eps,
            )

            # Attach the chosen children under the new internal node
            new.add_child(c1)
            new.add_child(c2)

    return t
    
def drop_anonymous_leaves(t):
    # Remove ghost leaves with empty names like ':0'
    for lf in list(t.iter_leaves()):
        if not lf.name:    # empty string or None
            lf.detach()


# Section 2a. Distance oracle & optimization helpers

from functools import lru_cache
from math       import floor, log2

class DistOracle:
    """O(1) distance queries between any two nodes after O(n log n) build."""
    __slots__ = ("dist_to_root","_euler","_depth","_first","_st","_log2",
                 "_dist_cached")

    def __init__(self, tree: Tree, cache_size=20_000):
        self.dist_to_root = {}
        self._annotate_depth(tree)
        self._build_lca_struct(tree)
        @lru_cache(maxsize=cache_size)
        def _dc(a, b):
            lca = self._lca(a, b)
            return (self.dist_to_root[a] +
                    self.dist_to_root[b] -
                    2 * self.dist_to_root[lca])
        self._dist_cached = _dc

    def dist(self, a, b): return self._dist_cached(a, b)
    def dist_leaf_to_node(self, leaf, node): return self.dist(leaf, node)

    def _annotate_depth(self, tree):
        for n in tree.traverse("preorder"):
            self.dist_to_root[n] = 0.0 if n.up is None else \
                                   self.dist_to_root[n.up] + n.dist
    def _build_lca_struct(self, tree):
        eul, dep, first = [], [], {}
        def dfs(v,d):
            first.setdefault(v, len(eul))
            eul.append(v); dep.append(d)
            for ch in v.children:
                dfs(ch,d+1); eul.append(v); dep.append(d)
        dfs(tree,0)
        self._euler,self._depth,self._first=eul,dep,first
        m=len(eul); kmax=floor(log2(m))+1
        st=[[0]*m for _ in range(kmax)]
        st[0]=list(range(m))
        for k in range(1,kmax):
            half=1<<(k-1)
            for i in range(m-(1<<k)+1):
                a,b=st[k-1][i],st[k-1][i+half]
                st[k][i]= a if dep[a]<dep[b] else b
        self._st=st
        self._log2=[0]*(m+1)
        for i in range(2,m+1):
            self._log2[i]=self._log2[i>>1]+1
    def _lca(self,a,b):
        ia,ib=self._first[a],self._first[b]
        if ia>ib: ia,ib=ib,ia
        span=ib-ia+1
        k=self._log2[span]
        left=self._st[k][ia]
        right=self._st[k][ib-(1<<k)+1]
        return self._euler[left] if self._depth[left]<self._depth[right] \
               else self._euler[right]


def scale_subtree(root, factor):
    for n in root.traverse(): n.dist *= factor

def find_distinct_leaves(T, common):       # leaves unique to tree T
    return get_leaf_set_ete(T) - common


# Section 2b.  Objective function

def find_optimal_insertion_point(target_tree, Dtgt,
                                 subtree_root,
                                 ncl_names, d_p,
                                 dl_names,
                                 min_terminal=1e-3,
                                 eps=1e-6):
    """
    Objective function quadratic optimization over original edges only.
    An edge is eligible iff child.is_original_edge == True.
    We skip edges that lie fully inside a previously inserted subtree (child.inInserted).

    Tie-breaking (to ensure uniqueness), applied when objective values are equal
    within a small tolerance:
      1) prefer smaller distance from the (original) root to the candidate v(x);
      2) if still tied, prefer the candidate on the original edge with the
         smallest fixed DFS index (edge_index).
    """
    best_edge, best_x, best_val = None, 0.0, float("inf")
    best_root_dist = float("inf")
    best_edge_idx  = float("inf")

    ncl_set = set(ncl_names)
    root = target_tree  # ete3 Tree is already the root node

    TOL = 1e-12
    TOL2 = 1e-12

    # Precompute distance from root to each node, via Dtgt
    # We already have dist_to_root in Dtgt.
    for child in target_tree.traverse("preorder"):
        if not getattr(child, "is_original_edge", False):
            # skip edges that are not original
            continue
        if getattr(child, "inInserted", False):
            # skip edges fully inside already inserted subtree
            continue

        parent = child.up
        edge_len = child.dist
        if parent is None:
            continue  # root has no incoming edge

        # For each leaf in ncl_names, we need d(l_c,u) and whether l_c is in subtree rooted at child
        # We'll define epsilon_c based on that.
        # We must solve argmin_x f(x) = sum over c ( d(l_c,u) + eps_c x|e| - target_c )^2

        # Precompute these  for c in ncl_names
        A = 0.0
        B = 0.0

        # Distances from the leaf to parent u
        for lc_name in ncl_names:
            lc = target_tree&lc_name
            d_lc_u = Dtgt.dist_leaf_to_node(lc, parent)
            # we need to know if lc is in subtree rooted at child
            # if LCA(lc, child) == child, then lc is in descendant subtree.
            # Ete3: child.get_common_ancestor(lc) == child
            lca = target_tree.get_common_ancestor(lc, child)
            eps_c = -1 if lca == child else +1

            # residual expression: d(l_c,u) + eps_c x|e| - t_c
            # derivative wrt x => 2 * sum_c ( d(l_c,u) + eps_c x|e| - t_c )*(eps_c |e|)
            # => set to 0 => x * ( 2 |e|^2 * (#c) ) + ... we do simpler formula:
            # we can expand: f(x) = sum (a_c + b_c x)^2 with b_c = eps_c * |e|
            # => derivative = 2 sum b_c (a_c + b_c x) = 0 => x = - (sum b_c a_c)/(sum b_c^2)
            t_c = d_p[lc_name]  # target distance
            a_c = d_lc_u - t_c
            b_c = eps_c * edge_len
            A += b_c * a_c
            B += b_c * b_c

        if B < 1e-15:      # no curvature => skip
            continue
        x_opt = -A / B
        # clamp x_opt to [0,1]
        if x_opt < 0.0:   x_opt = 0.0
        if x_opt > 1.0:   x_opt = 1.0

        # evaluate objective at x_opt
        val = 0.0
        for lc_name in ncl_names:
            lc = target_tree&lc_name
            d_lc_u = Dtgt.dist_leaf_to_node(lc, parent)
            lca = target_tree.get_common_ancestor(lc, child)
            eps_c = -1 if lca == child else +1
            t_c   = d_p[lc_name]

            pred = d_lc_u + eps_c * x_opt * edge_len
            val += (pred - t_c)**2

        # compute distance from original root to v(x_opt)
        dist_root_u = Dtgt.dist_to_root[parent]
        dist_root_v = dist_root_u + x_opt * edge_len

        edge_idx = getattr(child, "edge_index", float("inf"))

        if (val + TOL < best_val or
            (abs(val - best_val) <= TOL2 and dist_root_v + TOL < best_root_dist) or
            (abs(val - best_val) <= TOL2 and abs(dist_root_v - best_root_dist) <= TOL2 and edge_idx < best_edge_idx)):
            best_val       = val
            best_edge      = (parent, child)
            best_x         = x_opt
            best_root_dist = dist_root_v
            best_edge_idx  = edge_idx

    return best_edge, best_x, best_val


def insert_subtree_at_point(target_tree, edge, x_opt, subtree_copy,
                            eps=1e-6, min_terminal=1e-9):
    """
    Insert subtree_copy into target_tree along 'edge' at fraction x_opt
    from the parent side. We handle three cases:
      1) x_opt ~ 0 => attach subtree as another child of parent (no edge split)
      2) x_opt ~ 1 => attach subtree as another child of child (if child is internal)
      3) otherwise => split edge with new internal node.

    The new edges used for the subtree are marked child.is_original_edge = False,
    so they won't be used for future anchor edges.
    """
    parent, child = edge
    orig_len = child.dist

    # 1) Attach almost at the parent
    if x_opt <= eps:
        # parent has new child subtree_copy
        subtree_copy.add_feature("is_original_edge", False)
        subtree_copy.add_feature("edge_index", -1)
        parent.add_child(subtree_copy)
        return

    # 2) Attach almost at the child
    if x_opt >= 1 - eps:
        # If child is not a leaf (has children), we attach as child of "child".
        if not child.is_leaf():
            subtree_copy.add_feature("is_original_edge", False)
            subtree_copy.add_feature("edge_index", -1)
            child.add_child(subtree_copy)
            return
        # If child is a leaf, we still split, but with a very small top edge or bottom edge.
        # We'll let bottom be extremely tiny:
        d_up   = orig_len - min_terminal
        d_down = min_terminal

        mid = Tree()
        mid.dist = d_up
        mid.add_feature("is_original_edge", True)
        mid.add_feature("edge_index", child.edge_index)

        # rewire
        parent.remove_child(child)
        parent.add_child(mid)
        child.dist = d_down
        mid.add_child(child)

        subtree_copy.dist = 0.0
        subtree_copy.add_feature("is_original_edge", False)
        subtree_copy.add_feature("edge_index", -1)
        mid.add_child(subtree_copy)
        return

    # 3) Genuine split
    d_up   = max(x_opt * orig_len,       min_terminal)
    d_down = max((1 - x_opt) * orig_len, min_terminal)

    mid = Tree()
    mid.dist = d_up
    mid.add_feature("is_original_edge", True)
    mid.add_feature("edge_index", child.edge_index)

    parent.remove_child(child)
    parent.add_child(mid)
    child.dist = d_down
    mid.add_child(child)

    subtree_copy.dist = 0.0
    subtree_copy.add_feature("is_original_edge", False)
    subtree_copy.add_feature("edge_index", -1)
    mid.add_child(subtree_copy)


# Section 3. Weighted majority-rule consensus and scaling

def compute_weights_globally(T_set, T_i):
    """
    Compute weights w_j = |L(T_j) cap L(T_i)| / |L(T_i)| for all source trees T_j.
    This matches Eq. (9) in the text.
    We reuse global_T_set and global_T_i.
    """
    Li = get_leaf_set_ete(T_i)
    denom = len(Li) if Li else 1
    weights = []
    for Tj in T_set:
        Lj = get_leaf_set_ete(Tj)
        wj = len(Li & Lj)/denom
        weights.append(wj)
    return weights

def build_consensus_mcs(mcs_groups):
    """
    Given mcs_groups (list of (leafset_key, [subtrees_with_same_leafset], [host_indices])),
    perform weighted majority-rule consensus on each group, returning:
      consensus_subtrees: list of new consensus subtree roots
      host_index_groups:  list of host-index subsets that contributed
    """
    consensus_subtrees = []
    host_index_groups  = []

    for leafset_key, subtrees, host_idxs in mcs_groups:
        # Let L(M_t) = leafset_key
        if len(leafset_key) == 0:
            continue
        if len(leafset_key) == 1:
            # single leaf: just consensus leaf
            leaf_name = leafset_key[0]
            root = Tree(name=leaf_name)
            root.dist = 0.0
            consensus_subtrees.append(root)
            host_index_groups.append(host_idxs)
            continue

        # Need to do split-based majority-rule with weights.
        # We'll treat each subtree S_j as coming from T_j in the current host_list.
        # global_T_set is the list of source trees in this iteration.
        global global_T_set, global_T_i
        if not global_T_set or global_T_i is None:
            raise RuntimeError("global_T_set / global_T_i not defined, please set before calling build_consensus_mcs")

        weights = compute_weights_globally(global_T_set, global_T_i)

        # Collect all splits
        split_lengths = defaultdict(list)  # sp -> list of (weight, edge_length)
        total_weight = 0.0

        for st, j_index in zip(subtrees, host_idxs):
            # st is the subtree in T_j with leafset leafset_key.
            wj = weights[j_index]
            if wj <= 0:  # skip zero weight
                continue
            total_weight += wj
            # get splits: for each internal edge, define bipartition
            for nd in st.traverse("postorder"):
                if nd.is_leaf(): continue
                # removing nd.dist edge from its parent (if any) => bipart
                # But we define splits by each internal edge below nd's parent
                # For clarity, we treat each child-edge in st as an internal edge.
                for child in nd.children:
                    # root side vs subtree side
                    subtree_leaves = set([lf.name for lf in child.iter_leaves()])
                    # Represent split as (frozenset(A), frozenset(B)) with A smaller
                    A = frozenset(subtree_leaves)
                    B = frozenset(leafset_key) - A
                    sp = (A, B) if len(A) <= len(B) else (B, A)
                    split_lengths[sp].append((wj, child.dist))

            # Also connecting branch from parent( r_S ) to LCA(leafset_key) in T_j
            rS = subtree_rooted_at_lca(global_T_set[j_index], leafset_key)
            if rS is not None and rS.up is not None:
                cbranch_len = rS.dist
                # We treat the "connecting branch" as a special pseudo-split
                # with negative key (just store separately).
                sp_conn = ("CONNECTING_BRANCH",)
                split_lengths[sp_conn].append((wj, cbranch_len))

        if total_weight <= 0:
            # no contribution?
            continue

        # Weighted majority-rule: include sp if
        # f(sp) = sum_{j} w_j I{sp in SP_j} / sum_j w_j  > 0.5
        # But we only stored lengths for splits that do appear.
        final_splits = []
        final_lengths = {}

        for sp, wlen_list in split_lengths.items():
            w_sum = sum(w for (w,_) in wlen_list)
            if sp == ("CONNECTING_BRANCH",):
                # connecting branch always used if any weight
                if w_sum > 0:
                    length = sum(w*ell for (w,ell) in wlen_list)/w_sum
                    final_lengths[sp] = length
                continue
            f_sp = w_sum / total_weight
            if f_sp > 0.5:
                final_splits.append(sp)
                length = sum(w*ell for (w,ell) in wlen_list)/w_sum
                final_lengths[sp] = length

        # Build consensus topology via a simple greedy split-addition approach:
        # start from a star, then refine using final_splits.
        # For small leafset_key, we can do a naive approach:
        leaf2node = {nm: Tree(name=nm) for nm in leafset_key}
        current_root = Tree()
        for nm, lf in leaf2node.items():
            current_root.add_child(lf)

        # Sort splits by size to refine smaller first
        final_splits.sort(key=lambda sp: min(len(sp[0]), len(sp[1])))

        for sp in final_splits:
            A, B = sp
            # We want an internal node that has leaves in A in one side, B in other.
            # We'll refine current_root to introduce that split if not present.
            # A simpler but heavier approach:
            #  1) find the minimal subtree covering leaves in A
            #  2) if it already forms a clade, do nothing
            #  3) else restructure.
            # In interest of time, we do "clade refine" that only works nicely
            # if the splits are compatible; majority-rule with only the
            # chosen splits is typically compatible, so we proceed.

            # 1) gather all leaves in A
            nodesA = [leaf2node[nm] for nm in A]
            lcaA = current_root.get_common_ancestor(nodesA)

            # Check if lcaA is clean: all its leaves are subset of A
            leaves_lcaA = set([lf.name for lf in lcaA.iter_leaves()])
            if leaves_lcaA == A:
                # Already a clade, do nothing to topology
                continue

            # Otherwise, we restructure: we create a new internal node for A.
            # Steps:
            #  - all leaves in A are collected under new node A_int
            #  - reattach them so that they form a child clade somewhere
            A_int = Tree()
            # we detach those leaves and reattach under A_int
            for nm in A:
                lf = leaf2node[nm]
                lf.detach()
                A_int.add_child(lf)

            # attach A_int to lcaA
            lcaA.add_child(A_int)

        # Now we have a consensus topology
        # with the correct leafset; assign branch lengths using final_lengths:
        for nd in current_root.traverse("postorder"):
            if nd.up is None:
                continue
            # identify the split induced by nd's branch:
            leaves_here = set([lf.name for lf in nd.iter_leaves()])
            A = frozenset(leaves_here)
            B = frozenset(leafset_key) - A
            sp = (A,B) if len(A)<=len(B) else (B,A)
            if sp in final_lengths:
                nd.dist = final_lengths[sp]
            else:
                # fallback
                nd.dist = 0.0

        # connecting branch length
        conn_len = final_lengths.get(("CONNECTING_BRANCH",), 0.0)
        # We'll set the root dist = conn_len, leaving actual hooking to the host tree
        current_root.dist = conn_len

        consensus_subtrees.append(current_root)
        host_index_groups.append(host_idxs)

    return consensus_subtrees, host_index_groups


# Section 4. Insert subtree using common-leaves distances and scaling

def mean_distance_across_hosts(hosts, leafA, leafB):
    """
    Mean distance between leafA and leafB across the list of host trees.
    If a host doesn't contain both leaves, skip it.
    """
    dvals = []
    for T in hosts:
        leaves = get_leaf_set_ete(T)
        if leafA not in leaves or leafB not in leaves:
            continue
        nA, nB = T&leafA, T&leafB
        dvals.append(T.get_distance(nA, nB))
    if not dvals:
        return None
    return sum(dvals)/len(dvals)

def compute_global_scale_factor(orig_tree, hosts, common_leaves):
    """
    tau(T_i, T^S*).  We calculate the mean distance
    over host trees for each pair in common_leaves, then form the ratio.
    """
    if len(common_leaves) < 2:
        return 1.0
    common_list = sorted(common_leaves)
    num = 0.0
    den = 0.0
    for i in range(len(common_list)):
        for j in range(i+1, len(common_list)):
            la, lb = common_list[i], common_list[j]
            # distance in T_i
            na, nb = orig_tree&la, orig_tree&lb
            d_orig = orig_tree.get_distance(na, nb)
            d_host_mean = mean_distance_across_hosts(hosts, la, lb)
            if d_host_mean is None or d_host_mean <= 0:
                continue
            num += d_orig
            den += d_host_mean
    if den <= 0:
        return 1.0
    return num/den

def compute_leaf_based_rate(orig_tree, hosts, common_leaves, lc):
    """
    tau^{(l_c)}(T_i, T^S*).
    """
    others = [x for x in common_leaves if x != lc]
    if not others:
        return 1.0
    num = 0.0
    den = 0.0

    for la in others:
        # distance in T_i
        nlc, nla = orig_tree&lc, orig_tree&la
        d_orig = orig_tree.get_distance(nlc, nla)
        d_host_mean = mean_distance_across_hosts(hosts, lc, la)
        if d_host_mean is None or d_host_mean <= 0:
            continue
        num += d_orig
        den += d_host_mean

    if den <= 0:
        return 1.0
    return num/den


def compute_target_distances_for_insertion(orig_tree, hosts, S_star, common_leaves):
    """
    For each common leaf l_c, compute
       d^{(T^S*)}(l_c, S^*) = mean over hosts of dist(l_c, att(S^*)) * tau^{(l_c)}.
    We identify att(S^*) in host T as the parent of the subtree root's LCA
    that covers leafset of S^*.
    """
    leafset_S = [lf.name for lf in S_star.iter_leaves()]
    d_p = {}   # dictionary leaf_name -> target distance

    for lc in common_leaves:
        tau_lc = compute_leaf_based_rate(orig_tree, hosts, common_leaves, lc)
        if tau_lc <= 0:
            tau_lc = 1.0

        dvals = []
        for T in hosts:
            leaves = get_leaf_set_ete(T)
            if lc not in leaves:
                continue
            # LCA of leafset_S in T
            # We'll restrict to those leaves that exist in T
            leafset_in_T = [x for x in leafset_S if x in leaves]
            if not leafset_in_T:
                continue
            rS = subtree_rooted_at_lca(T, leafset_in_T)
            if rS is None or rS.up is None:
                continue
            att_node = rS.up
            nlc = T&lc
            dvals.append(T.get_distance(nlc, att_node))

        if not dvals:
            d_p[lc] = 0.0   # default
        else:
            d_p[lc] = (sum(dvals)/len(dvals)) * tau_lc

    return d_p


def insert_subtree_kncl(target_tree, orig_tree,
                        S_star, hosts,
                        anchor_leaves_original,
                        k_user=None):
    """
    Insert S_star into target_tree using the distances as in the 
    "Leaf-based" method: we first compute global scale factor, then
    leaf-based adjustments. We then solve the quadratic objective.

    anchor_leaves_original is the set CL(T_i, T^S*). In the implementation,
    that is the intersection of orig_tree's leaves and all host trees.
    """
    # Determine the common leaves between orig_tree and hosts that
    # exist in S_star's leafset or at least relate to them.
    # We'll use anchor_leaves_original, possibly truncated to k_user if set.

    common_leaves = set(anchor_leaves_original)
    if not common_leaves:
        # nothing to align with, skip insertion
        return False

    if k_user is not None and k_user > 0 and len(common_leaves) > k_user:
        # just take first k_user leaves in the sorted order
        common_leaves = set(sorted(common_leaves)[:k_user])

    # Global scale factor
    tau_global = compute_global_scale_factor(orig_tree, hosts, common_leaves)
    if tau_global <= 0:
        tau_global = 1.0

    S_copy = S_star.copy("deepcopy")
    scale_subtree(S_copy, tau_global)

    # Dist oracle for target_tree
    Dtgt = DistOracle(target_tree)

    # Compute target distances d_p for each leaf in common_leaves
    d_p = compute_target_distances_for_insertion(orig_tree, hosts,
                                                 S_copy, common_leaves)

    # Solve objective to find best insertion edge / position
    best_edge, best_x, best_val = find_optimal_insertion_point(
        target_tree, Dtgt, S_copy,
        sorted(common_leaves), d_p,
        dl_names=[lf.name for lf in S_copy.iter_leaves()]
    )
    if best_edge is None:
        return False

    # Perform the actual insertion
    insert_subtree_at_point(target_tree, best_edge, best_x, S_copy)
    # Mark edges inside S_copy as not original
    for n in S_copy.traverse("preorder"):
        if n.up is not None:
            n.add_feature("is_original_edge", False)
            n.add_feature("edge_index", -1)
        else:
            n.add_feature("is_original_edge", False)
            n.add_feature("edge_index", -1)
        n.add_feature("inInserted", True)

    return True


# Section 5. Selection of MCS


def selection_of_mcs(target_tree, host_list, U, p, multi_leaf=True):
    """
    Implements the MCS selection step for a single iteration, at threshold p.

    - target_tree:    currently partially-completed T_i
    - host_list:      the list of source trees
    - U:              set of uncovered taxa (subset of overall leaf set minus L(T_i))
    - p:              threshold in (0,1] for frequency
    - multi_leaf:     if True, only consider subtrees with >= 2 leaves.
                      if False, only consider subtrees with exactly 1 leaf.
    Returns:
       a list of (leafset_key, [subtrees], [host_indices]).
    """
    overall_candidates = defaultdict(list)  # leafset_key -> list[(subtree, host_idx)]
    num_hosts = len(host_list)
    if num_hosts == 0:
        return []

    # 1) For each host, gather candidate subtrees S with L(S) subset of U
    for j, H in enumerate(host_list):
        # repeated: find nodes whose leaves subset of U
        for nd in H.traverse("postorder"):
            if nd.is_leaf():
                if nd.name in U:
                    leafset = {nd.name}
                else:
                    leafset = set()
            else:
                child_sets = [set([lf.name for lf in c.iter_leaves()]) for c in nd.children]
                leafset = set().union(*child_sets)

            if not leafset:
                continue
            if not (leafset <= U):
                continue
            if multi_leaf and len(leafset) < 2:
                continue
            if (not multi_leaf) and len(leafset) != 1:
                continue

            # add candidate
            key = leafset_hash(leafset)
            overall_candidates[key].append((nd, j))

    if not overall_candidates:
        return []

    # 2) filter by frequency threshold p
    # For each leafset key, find how many distinct host trees contain it.
    freq_filtered = {}
    for key, lst in overall_candidates.items():
        hosts_with_key = set([hj for (nd, hj) in lst])
        freq = len(hosts_with_key) / num_hosts
        if freq >= p:
            freq_filtered[key] = lst

    if not freq_filtered:
        return []

    # 3) group by leafset and keep the group that maximizes leaf coverage L(G)
    best_groups = []
    best_size   = 0
    for key, lst in freq_filtered.items():
        leafset = set(key)
        size    = len(leafset)
        if size > best_size:
            best_groups = [(key, lst)]
            best_size   = size
        elif size == best_size:
            best_groups.append((key, lst))

    if not best_groups:
        return []

    # tie-break lexicographically by sorted leafset
    best_groups.sort(key=lambda x: list(x[0]))
    chosen_key, chosen_lst = best_groups[0]

    host_to_subtrees = defaultdict(list)
    for (nd, hj) in chosen_lst:
        host_to_subtrees[hj].append(nd)

    final_subtrees = []
    final_hosts    = []

    leafset_chosen = set(chosen_key)
    for hj, nodes in host_to_subtrees.items():
        H = host_list[hj]
        # get subtree rooted at LCA of leafset in H
        rS = subtree_rooted_at_lca(H, leafset_chosen)
        if rS is None:
            continue
        final_subtrees.append(rS)
        final_hosts.append(hj)

    if not final_subtrees:
        return []

    return [(chosen_key, final_subtrees, final_hosts)]


# Section 6. Main part for one multiset


def _ms_label(fname):
    """Helper to parse 'multiset_X.txt' -> integer X for sorting, logs, etc."""
    base = os.path.basename(fname)
    m = re.search(r"multiset_(\d+)\.txt", base)
    if not m:
        return 0
    return int(m.group(1))

def complete_all_multisets():
    in_dir  = "input_multisets"
    out_dir = "completed_multisets"
    os.makedirs(out_dir, exist_ok=True)

    # Find all files  input_multisets/multiset_*.txt
    pattern    = os.path.join(in_dir, "multiset_*.txt")
    file_list  = sorted(glob.glob(pattern), key=_ms_label)

    for fin in file_list:
        ms_i  = _ms_label(fin)          # for log / output file name
        fout  = os.path.join(out_dir, f"completed_multiset_{ms_i}.txt")
        print(f"Processing multiset {ms_i} …")

        #  Read trees
        with open(fin) as f:
            lines = [line.strip() for line in f if line.strip()]
        all_trees = []
        for line in lines:
            nwk = preprocess_newick(line)
            t   = Tree(nwk, format=1)
            all_trees.append(t)

        if not all_trees:
            print("  [WARN] no trees found.")
            continue

        # Precompute overall leaf sets
        overall_leaves = all_leaves_overall(all_trees)
        common_all     = common_leaves_overall(all_trees)
        print(f"  overall leaves: {len(overall_leaves)}, common: {len(common_all)}")

        completed = []

        for idx, orig in enumerate(all_trees, 1):
            print(f"  completing tree {idx}/{len(all_trees)}")
            tgt_updated = orig.copy("deepcopy")
            # Mark original edges so later we only anchor on original edges,
            # and split parts remain original; assign fixed edge indices (DFS).
            mark_original_nodes(tgt_updated)

            host_list = [all_trees[j].copy("deepcopy")
                         for j in range(len(all_trees)) if j != idx - 1]

            global global_T_set, global_T_i
            global_T_set = host_list
            global_T_i   = tgt_updated

            anchor_leaves_original = sorted(
                get_leaf_set_ete(orig) & get_common_leaves(host_list)
            )
            U = overall_leaves - get_leaf_set_ete(tgt_updated)
            inserted_leaves = set()

            p = 0.5
            while U and p >= 0:
                # Multi-leaf MCS first
                selected = selection_of_mcs(
                    tgt_updated, host_list, U, p, multi_leaf=True)
                if selected:
                    c_subtrees, host_idxs = build_consensus_mcs(selected)
                    if c_subtrees:
                        before = len(U)
                        for S_star, hlist in zip(c_subtrees, host_idxs):
                            hosts = [host_list[h] for h in hlist]
                            ok = insert_subtree_kncl(
                                tgt_updated, orig, S_star, hosts,
                                anchor_leaves_original,
                                K_USER
                            )
                            if ok:
                                inserted_leaves |= get_leaf_set_ete(S_star)
                                U -= get_leaf_set_ete(S_star)
                        if len(U) < before:
                            continue

                # Single-leaf MCS
                selected_single = selection_of_mcs(
                    tgt_updated, host_list, U, p, multi_leaf=False)
                if selected_single:
                    c_subtrees, host_idxs = build_consensus_mcs(selected_single)
                    if c_subtrees:
                        before = len(U)
                        for S_star, hlist in zip(c_subtrees, host_idxs):
                            hosts = [host_list[h] for h in hlist]
                            ok = insert_subtree_kncl(
                                tgt_updated, orig, S_star, hosts,
                                anchor_leaves_original,
                                K_USER
                            )
                            if ok:
                                inserted_leaves |= get_leaf_set_ete(S_star)
                                U -= get_leaf_set_ete(S_star)
                        if len(U) < before:
                            continue

                # Relax the MCS frequency threshold if nothing inserted
                p -= 0.05

            leftover = U & (overall_leaves - get_leaf_set_ete(tgt_updated))
            if leftover:
                print(f"   [WARN] still missing {len(leftover)} leaves {leftover}")
            else:
                print("   done.")

            if FORCE_BINARY:
                # Post-process the completed tree to enforce a binary topology.
                # This only introduces additional internal nodes and does not
                # change leaf–leaf distances when eps = 0.0.
                drop_anonymous_leaves(tgt_updated)
                binarize_multifurcations(tgt_updated, eps=0.0)
                

            completed.append(tgt_updated)

        # Write output
        with open(fout, "w") as fo:
            for t in completed:
                clear_internal_node_names(t)
                fo.write(t.write(format=1) + "\n")
    print("All multisets done.")

# Main
if __name__ == "__main__":
    complete_all_multisets()
