#!/usr/bin/env python3
"""
Prototype experiment harness for Undirected Edge Geography (UEG).

This script is intended as a supplementary validation artifact for the dynamic
programming algorithm described in Section 5 of the paper. It provides:

1. Random instance generation together with a rooted tree partition of bounded width.
2. A brute-force minimax solver for UEG.
3. A tree-partition-based exact solver inspired by the XP algorithm in the paper.
4. Verification and benchmarking utilities, including an optional runtime plot.

Important notes
---------------
* This is an experimental correctness / benchmarking tool, not a production
  implementation. The decomposition-based solver is extremely inefficient.
* For arbitrary graphs, computing a good rooted tree partition is difficult.
  Accordingly, the preferred mode is to generate graphs together with a rooted
  tree partition. A trivial one-bag partition is also provided for debugging.
* The solver implemented here follows the same high-level design as Section 5:
  each child subtree is summarized by a finite type, children of the same type
  are interchangeable, and bag states are compressed to multiplicities of child
  types.

Example commands
----------------
python ueg_tree_partition_experiment.py demo --n 10 --width 3 --seed 1
python ueg_tree_partition_experiment.py verify --trials 50 --n 10 --width 3 --seed 2
python ueg_tree_partition_experiment.py benchmark --sizes 6,7,8,9,10,11,12 --width 3 \
    --trials-per-size 5 --seed 3 --progress --plot benchmark.png
python ueg_tree_partition_experiment.py surface --sizes 6,8,10 --widths 2,3,4 \
    --trials-per-point 3 --solver both --progress --plot-prefix surface
"""

from __future__ import annotations

import argparse
import csv
import functools
import itertools
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _fmt_seconds(sec: float) -> str:
    """Return a compact human-readable duration string."""
    if sec < 60:
        return f"{sec:.1f}s"
    minutes, seconds = divmod(sec, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(rem_minutes)}m {seconds:.0f}s"


def parity_flip(p: int) -> int:
    """Flip parity: same=0 <-> opp=1."""
    return p ^ 1


def parity_compose(p: int, c: int) -> int:
    """Compose parities. With same=0 and opp=1, composition is xor."""
    return p ^ c


def bit_count(x: int) -> int:
    """Return the number of set bits in x."""
    return x.bit_count()


def edge_key(u: int, v: int) -> Tuple[int, int]:
    """Return the canonical ordered representation of an undirected edge."""
    return (u, v) if u < v else (v, u)


def powerset_masks(size: int) -> Iterable[int]:
    """Yield all bitmasks of length ``size``."""
    for mask in range(1 << size):
        yield mask


def tuple_counter_add(vec: Tuple[int, ...], idx: int, delta: int) -> Tuple[int, ...]:
    """Return ``vec`` with ``delta`` added at coordinate ``idx``."""
    lst = list(vec)
    lst[idx] += delta
    if lst[idx] < 0:
        raise ValueError("negative multiplicity")
    return tuple(lst)


def parse_int_list(spec: str) -> List[int]:
    """Parse a comma-separated list like ``"6,8,10"`` into integers."""
    values = [int(x.strip()) for x in spec.split(',') if x.strip()]
    if not values:
        raise ValueError('expected at least one integer')
    return values


# ---------------------------------------------------------------------------
# Basic graph / partition data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Port:
    """A parent-child cut edge, together with local labels on both sides."""

    edge_index: int
    parent_vertex: int
    child_vertex: int
    parent_label: int
    child_label: int


@dataclass
class NodeData:
    """All information attached to one bag of the rooted tree partition."""

    node_id: int
    vertices: Tuple[int, ...]
    parent: Optional[int]
    children: Tuple[int, ...]
    label_of_vertex: Dict[int, int]
    vertex_of_label: Dict[int, int]
    internal_edge_indices: Tuple[int, ...]
    ports: Tuple[Port, ...]


@dataclass
class GraphWithPartition:
    """A graph together with a rooted tree partition and a start vertex."""

    n: int
    edges: Tuple[Tuple[int, int], ...]
    adjacency: Tuple[Tuple[int, int], ...]
    edge_to_index: Dict[Tuple[int, int], int]
    bag_of_vertex: Tuple[int, ...]
    nodes: Dict[int, NodeData]
    root: int
    start: int


# ---------------------------------------------------------------------------
# Type system for the DP solver
# ---------------------------------------------------------------------------

# ExitLabel = (exit_port_abs_index, successor_type_id, parity_bit)
# parity_bit: 0 = same, 1 = opposite
ExitLabel = Tuple[int, int, int]


@dataclass(frozen=True)
class EntrySemantics:
    """The semantics of a type when entered through one abstract port."""

    exits: Tuple[ExitLabel, ...]
    truth_table: int


@dataclass(frozen=True)
class TypeSignature:
    """Canonical encoding of one realizable interface type."""

    labels: Tuple[int, ...]
    entries: Tuple[EntrySemantics, ...]


class TypeRegistry:
    """Intern canonical type signatures and assign integer ids."""

    def __init__(self) -> None:
        self._sig_to_id: Dict[TypeSignature, int] = {}
        self._id_to_sig: Dict[int, TypeSignature] = {}
        self._rank: Dict[int, int] = {}
        self.bot_id = self.intern(TypeSignature(labels=(), entries=()))

    def intern(self, sig: TypeSignature) -> int:
        if sig in self._sig_to_id:
            return self._sig_to_id[sig]
        tid = len(self._sig_to_id)
        self._sig_to_id[sig] = tid
        self._id_to_sig[tid] = sig
        self._rank[tid] = len(sig.labels)
        return tid

    def signature(self, tid: int) -> TypeSignature:
        return self._id_to_sig[tid]

    def rank(self, tid: int) -> int:
        return self._rank[tid]

    def eval_entry(self, tid: int, entry_idx: int, assignment: Dict[ExitLabel, bool]) -> bool:
        """
        Evaluate the Boolean function stored at entry ``entry_idx`` of type ``tid``.

        ``assignment`` maps exit labels to truth values from the viewpoint of the
        player who entered the child.
        """
        sem = self._id_to_sig[tid].entries[entry_idx]
        idx = 0
        for i, ex in enumerate(sem.exits):
            if assignment.get(ex, False):
                idx |= 1 << i
        return ((sem.truth_table >> idx) & 1) == 1


# ---------------------------------------------------------------------------
# Random instance generation
# ---------------------------------------------------------------------------

def random_partitioned_instance(
    n_vertices: int,
    width: int,
    seed: int = 0,
    p_internal: float = 0.35,
    p_cut: float = 0.45,
    force_connected_ports: bool = True,
) -> GraphWithPartition:
    """
    Generate a random graph together with a rooted tree partition of width at most ``width``.

    Construction scheme:
    * First generate a random rooted bag tree.
    * Then assign vertices to bags.
    * Finally add edges only inside bags or across one parent-child cut.

    This guarantees that the returned graph is compatible with the generated tree partition.
    """
    rnd = random.Random(seed)
    if n_vertices <= 0:
        raise ValueError("n_vertices must be positive")
    if width <= 0:
        raise ValueError("width must be positive")

    # 1) Random bag sizes summing to n_vertices, each in [1, width].
    bag_sizes: List[int] = []
    remaining = n_vertices
    while remaining > 0:
        max_size = min(width, remaining)
        size = rnd.randint(1, max_size)
        bag_sizes.append(size)
        remaining -= size

    num_bags = len(bag_sizes)

    # 2) Random rooted tree on bags: bag 0 is root, each new bag chooses a parent among previous bags.
    parent: List[Optional[int]] = [None]
    children: Dict[int, List[int]] = defaultdict(list)
    for i in range(1, num_bags):
        p = rnd.randrange(0, i)
        parent.append(p)
        children[p].append(i)

    # 3) Assign vertices to bags.
    bags: Dict[int, List[int]] = {}
    bag_of_vertex = [-1] * n_vertices
    next_vertex = 0
    for i, size in enumerate(bag_sizes):
        verts = list(range(next_vertex, next_vertex + size))
        for v in verts:
            bag_of_vertex[v] = i
        bags[i] = verts
        next_vertex += size

    # 4) Local labels 0, ..., |bag|-1.
    label_of_vertex_by_bag: Dict[int, Dict[int, int]] = {}
    vertex_of_label_by_bag: Dict[int, Dict[int, int]] = {}
    for i, verts in bags.items():
        label_of_vertex_by_bag[i] = {v: idx for idx, v in enumerate(verts)}
        vertex_of_label_by_bag[i] = {idx: v for idx, v in enumerate(verts)}

    # 5) Add edges: internal or parent-child only.
    edges_set: Set[Tuple[int, int]] = set()

    # Internal edges.
    for i, verts in bags.items():
        for u, v in itertools.combinations(verts, 2):
            if rnd.random() < p_internal:
                edges_set.add(edge_key(u, v))

    # Parent-child cut edges.
    for i in range(1, num_bags):
        p = parent[i]
        assert p is not None
        cut_edges_added = 0
        for u in bags[p]:
            for v in bags[i]:
                if rnd.random() < p_cut:
                    edges_set.add(edge_key(u, v))
                    cut_edges_added += 1
        if force_connected_ports and cut_edges_added == 0:
            u = rnd.choice(bags[p])
            v = rnd.choice(bags[i])
            edges_set.add(edge_key(u, v))

    edges = tuple(sorted(edges_set))
    edge_to_index = {e: idx for idx, e in enumerate(edges)}

    adjacency_lists: List[List[int]] = [[] for _ in range(n_vertices)]
    for idx, (u, v) in enumerate(edges):
        adjacency_lists[u].append(idx)
        adjacency_lists[v].append(idx)

    # Pick the start among movable vertices if possible.
    movable = [v for v in range(n_vertices) if adjacency_lists[v]]
    start = rnd.choice(movable) if movable else 0

    # Re-root the bag tree at the bag containing the start vertex.
    root_bag = bag_of_vertex[start]
    undirected_tree_neighbors: Dict[int, List[int]] = defaultdict(list)
    for i in range(1, num_bags):
        p = parent[i]
        assert p is not None
        undirected_tree_neighbors[i].append(p)
        undirected_tree_neighbors[p].append(i)

    new_parent: List[Optional[int]] = [None] * num_bags
    new_children: Dict[int, List[int]] = defaultdict(list)
    stack = [root_bag]
    seen = {root_bag}
    while stack:
        cur = stack.pop()
        for nxt in undirected_tree_neighbors[cur]:
            if nxt in seen:
                continue
            seen.add(nxt)
            new_parent[nxt] = cur
            new_children[cur].append(nxt)
            stack.append(nxt)

    # 6) Build node metadata.
    nodes: Dict[int, NodeData] = {}
    for i in range(num_bags):
        verts = tuple(bags[i])

        internal_edge_indices: List[int] = []
        for u, v in itertools.combinations(verts, 2):
            ek = edge_key(u, v)
            if ek in edge_to_index:
                internal_edge_indices.append(edge_to_index[ek])

        ports: List[Port] = []
        if new_parent[i] is not None:
            p = new_parent[i]
            assert p is not None
            for u in bags[p]:
                for v in bags[i]:
                    ek = edge_key(u, v)
                    if ek in edge_to_index:
                        ports.append(
                            Port(
                                edge_index=edge_to_index[ek],
                                parent_vertex=u,
                                child_vertex=v,
                                parent_label=label_of_vertex_by_bag[p][u],
                                child_label=label_of_vertex_by_bag[i][v],
                            )
                        )

        nodes[i] = NodeData(
            node_id=i,
            vertices=verts,
            parent=new_parent[i],
            children=tuple(new_children[i]),
            label_of_vertex=dict(label_of_vertex_by_bag[i]),
            vertex_of_label=dict(vertex_of_label_by_bag[i]),
            internal_edge_indices=tuple(sorted(internal_edge_indices)),
            ports=tuple(ports),
        )

    return GraphWithPartition(
        n=n_vertices,
        edges=edges,
        adjacency=tuple(tuple(lst) for lst in adjacency_lists),
        edge_to_index=edge_to_index,
        bag_of_vertex=tuple(bag_of_vertex),
        nodes=nodes,
        root=root_bag,
        start=start,
    )


def trivial_single_bag_partition(
    n: int,
    edges: Sequence[Tuple[int, int]],
    start: int = 0,
) -> GraphWithPartition:
    """Fallback partition for debugging: all vertices are placed in one bag."""
    edges = tuple(sorted(edge_key(u, v) for u, v in edges))
    edge_to_index = {e: idx for idx, e in enumerate(edges)}

    adjacency_lists: List[List[int]] = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adjacency_lists[u].append(idx)
        adjacency_lists[v].append(idx)

    label_of_vertex = {v: v for v in range(n)}
    vertex_of_label = {v: v for v in range(n)}
    node = NodeData(
        node_id=0,
        vertices=tuple(range(n)),
        parent=None,
        children=(),
        label_of_vertex=label_of_vertex,
        vertex_of_label=vertex_of_label,
        internal_edge_indices=tuple(range(len(edges))),
        ports=(),
    )
    return GraphWithPartition(
        n=n,
        edges=edges,
        adjacency=tuple(tuple(lst) for lst in adjacency_lists),
        edge_to_index=edge_to_index,
        bag_of_vertex=tuple(0 for _ in range(n)),
        nodes={0: node},
        root=0,
        start=start,
    )


# ---------------------------------------------------------------------------
# Brute-force UEG solver
# ---------------------------------------------------------------------------

def brute_force_ueg_winner(instance: GraphWithPartition) -> bool:
    """Return True iff Player 1 wins from the start by memoized minimax."""

    @functools.lru_cache(maxsize=None)
    def win(current_vertex: int, used_mask: int) -> bool:
        legal_successors: List[Tuple[int, int]] = []
        for eidx in instance.adjacency[current_vertex]:
            if (used_mask >> eidx) & 1:
                continue
            u, v = instance.edges[eidx]
            nxt = v if current_vertex == u else u
            legal_successors.append((nxt, used_mask | (1 << eidx)))
        if not legal_successors:
            return False
        return any(not win(nxt, new_mask) for (nxt, new_mask) in legal_successors)

    return win(instance.start, 0)


# ---------------------------------------------------------------------------
# Tree-partition-based solver
# ---------------------------------------------------------------------------

@dataclass
class NodeSolveResult:
    """Summary information computed for one node in the tree partition."""

    realizable_types: Set[int]
    initial_type: int


class TreePartitionUEGSolver:
    """
    Prototype solver implementing the type-compression dynamic program.

    The code mirrors the paper at a high level:
    * process bags bottom-up,
    * summarize each child subtree by a realizable type,
    * compress multisets of child subinstances to multiplicity vectors,
    * evaluate local games by reverse induction on an acyclic state graph.
    """

    def __init__(self, instance: GraphWithPartition):
        self.instance = instance
        self.registry = TypeRegistry()
        self.node_results: Dict[int, NodeSolveResult] = {}

    def solve(self) -> bool:
        """Return True iff Player 1 wins from the start vertex."""
        self._process_node(self.instance.root)
        return self._solve_root()

    def _process_node(self, node_id: int) -> NodeSolveResult:
        if node_id in self.node_results:
            return self.node_results[node_id]

        node = self.instance.nodes[node_id]

        # Process children first.
        child_results: Dict[int, NodeSolveResult] = {}
        for child in node.children:
            child_results[child] = self._process_node(child)

        # The root bag is solved separately at the end.
        if node.parent is None:
            result = NodeSolveResult(realizable_types=set(), initial_type=self.registry.bot_id)
            self.node_results[node_id] = result
            return result

        # Local type universe contributed by the already processed children.
        child_type_options: List[Set[int]] = [child_results[c].realizable_types for c in node.children]
        child_initial_types: List[int] = [child_results[c].initial_type for c in node.children]
        local_type_ids = sorted(set().union(*child_type_options) if child_type_options else set())
        local_type_index = {tid: idx for idx, tid in enumerate(local_type_ids)}

        possible_m_vectors = self._possible_multiplicity_vectors(child_type_options, local_type_index)

        # Full initial multiplicity vector for the original residual configuration at this node.
        initial_counter = [0] * len(local_type_ids)
        for tid in child_initial_types:
            if tid in local_type_index:
                initial_counter[local_type_index[tid]] += 1
        initial_m = tuple(initial_counter)

        port_count = len(node.ports)
        int_edge_count = len(node.internal_edge_indices)

        configs: List[Tuple[int, int, Tuple[int, ...]]] = []
        for U_mask in powerset_masks(port_count):
            for F_mask in powerset_masks(int_edge_count):
                for m_vec in possible_m_vectors:
                    configs.append((U_mask, F_mask, m_vec))

        def cfg_measure(cfg: Tuple[int, int, Tuple[int, ...]]) -> int:
            U_mask, F_mask, m_vec = cfg
            total = bit_count(U_mask) + bit_count(F_mask)
            for idx, cnt in enumerate(m_vec):
                if cnt:
                    total += cnt * self.registry.rank(local_type_ids[idx])
            return total

        configs.sort(key=cfg_measure)

        cfg_to_type: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
        realizable: Set[int] = set()

        for cfg in configs:
            U_mask, _, _ = cfg
            if bit_count(U_mask) == 0:
                tid = self.registry.bot_id
            else:
                tid = self._compute_config_type(node_id, cfg, local_type_ids, local_type_index, cfg_to_type)
            cfg_to_type[cfg] = tid
            realizable.add(tid)

        full_U = (1 << port_count) - 1
        full_F = (1 << int_edge_count) - 1
        initial_cfg = (full_U, full_F, initial_m)
        result = NodeSolveResult(realizable_types=realizable, initial_type=cfg_to_type[initial_cfg])
        self.node_results[node_id] = result
        return result

    def _possible_multiplicity_vectors(
        self,
        child_type_options: List[Set[int]],
        local_type_index: Dict[int, int],
    ) -> Set[Tuple[int, ...]]:
        """
        Enumerate multiplicity vectors realizable by choosing one realizable type per child.
        """
        zero = tuple(0 for _ in range(len(local_type_index)))
        vectors: Set[Tuple[int, ...]] = {zero}
        for options in child_type_options:
            new_vectors: Set[Tuple[int, ...]] = set()
            for vec in vectors:
                for tid in options:
                    if tid not in local_type_index:
                        continue
                    idx = local_type_index[tid]
                    new_vectors.add(tuple_counter_add(vec, idx, 1))
            vectors = new_vectors
        if not child_type_options:
            return {zero}
        return vectors

    def _compute_config_type(
        self,
        node_id: int,
        cfg: Tuple[int, int, Tuple[int, ...]],
        local_type_ids: List[int],
        local_type_index: Dict[int, int],
        cfg_to_type: Dict[Tuple[int, int, Tuple[int, ...]], int],
    ) -> int:
        """
        Compute the type of one compressed residual configuration at a non-root node.
        """
        node = self.instance.nodes[node_id]
        U_mask, F_mask_full, m_full = cfg

        # Canonically relabel the active ports by 0, ..., |U|-1.
        actual_ports = [pid for pid in range(len(node.ports)) if (U_mask >> pid) & 1]
        abs_of_actual = {pid: idx for idx, pid in enumerate(actual_ports)}
        labels = tuple(node.ports[pid].parent_label for pid in actual_ports)

        entries: List[EntrySemantics] = []

        # Precompute per-bag data for active ports.
        port_child_vertex = {pid: node.ports[pid].child_vertex for pid in actual_ports}
        port_child_label = {pid: node.ports[pid].child_label for pid in actual_ports}

        for entry_actual in actual_ports:
            U_after_entry_mask = U_mask & ~(1 << entry_actual)
            start_vertex = port_child_vertex[entry_actual]
            start_state = (start_vertex, F_mask_full, m_full, 1)  # parity=opp

            @functools.lru_cache(maxsize=None)
            def reachable_exits(state: Tuple[int, int, Tuple[int, ...], int]) -> FrozenSet[ExitLabel]:
                z, F_mask, m_vec, parity = state
                exits: Set[ExitLabel] = set()

                # Direct exit moves.
                for q in actual_ports:
                    if q == entry_actual:
                        continue
                    if not ((U_after_entry_mask >> q) & 1):
                        continue
                    if port_child_label[q] != node.label_of_vertex[z]:
                        continue
                    succ_cfg = (U_after_entry_mask & ~(1 << q), F_mask, m_vec)
                    succ_type = cfg_to_type[succ_cfg]
                    ex = (abs_of_actual[q], succ_type, parity_flip(parity))
                    exits.add(ex)

                # Internal moves.
                for local_e_idx, global_e_idx in enumerate(node.internal_edge_indices):
                    if not ((F_mask >> local_e_idx) & 1):
                        continue
                    u, v = self.instance.edges[global_e_idx]
                    if z not in (u, v):
                        continue
                    z2 = v if z == u else u
                    succ_state = (z2, F_mask & ~(1 << local_e_idx), m_vec, parity_flip(parity))
                    exits.update(reachable_exits(succ_state))

                # Child excursions.
                z_label = node.label_of_vertex[z]
                for sigma_idx, cnt in enumerate(m_vec):
                    if cnt == 0:
                        continue
                    sigma_tid = local_type_ids[sigma_idx]
                    sig = self.registry.signature(sigma_tid)
                    for b_idx, sem in enumerate(sig.entries):
                        if sig.labels[b_idx] != z_label:
                            continue
                        for ex in sem.exits:
                            q_abs_child, tau_tid, c = ex
                            return_label = sig.labels[q_abs_child]
                            z2 = node.vertex_of_label[return_label]
                            new_m = list(m_vec)
                            new_m[sigma_idx] -= 1
                            if tau_tid in local_type_index:
                                new_m[local_type_index[tau_tid]] += 1
                            else:
                                # The only absent successor type allowed locally is the null type.
                                if self.registry.rank(tau_tid) != 0:
                                    raise RuntimeError("Non-null successor type missing from local universe")
                            succ_state = (z2, F_mask, tuple(new_m), parity_compose(parity, c))
                            exits.update(reachable_exits(succ_state))

                return frozenset(exits)

            attainable = tuple(sorted(reachable_exits(start_state)))
            exit_index = {ex: i for i, ex in enumerate(attainable)}

            @functools.lru_cache(maxsize=None)
            def win(state: Tuple[int, int, Tuple[int, ...], int], valuation_bits: int) -> bool:
                z, F_mask, m_vec, parity = state
                move_values: List[bool] = []

                # Exit moves.
                for q in actual_ports:
                    if q == entry_actual:
                        continue
                    if not ((U_after_entry_mask >> q) & 1):
                        continue
                    if port_child_label[q] != node.label_of_vertex[z]:
                        continue
                    succ_cfg = (U_after_entry_mask & ~(1 << q), F_mask, m_vec)
                    succ_type = cfg_to_type[succ_cfg]
                    ex = (abs_of_actual[q], succ_type, parity_flip(parity))
                    idx = exit_index[ex]
                    move_values.append(((valuation_bits >> idx) & 1) == 1)

                # Internal moves.
                for local_e_idx, global_e_idx in enumerate(node.internal_edge_indices):
                    if not ((F_mask >> local_e_idx) & 1):
                        continue
                    u, v = self.instance.edges[global_e_idx]
                    if z not in (u, v):
                        continue
                    z2 = v if z == u else u
                    succ_state = (z2, F_mask & ~(1 << local_e_idx), m_vec, parity_flip(parity))
                    move_values.append(win(succ_state, valuation_bits))

                # Child excursions.
                z_label = node.label_of_vertex[z]
                for sigma_idx, cnt in enumerate(m_vec):
                    if cnt == 0:
                        continue
                    sigma_tid = local_type_ids[sigma_idx]
                    sig = self.registry.signature(sigma_tid)
                    for b_idx, sem in enumerate(sig.entries):
                        if sig.labels[b_idx] != z_label:
                            continue
                        assignment: Dict[ExitLabel, bool] = {}
                        for ex in sem.exits:
                            q_abs_child, tau_tid, c = ex
                            return_label = sig.labels[q_abs_child]
                            z2 = node.vertex_of_label[return_label]
                            new_m = list(m_vec)
                            new_m[sigma_idx] -= 1
                            if tau_tid in local_type_index:
                                new_m[local_type_index[tau_tid]] += 1
                            else:
                                if self.registry.rank(tau_tid) != 0:
                                    raise RuntimeError("Non-null successor type missing from local universe")
                            succ_state = (z2, F_mask, tuple(new_m), parity_compose(parity, c))
                            outer_win = win(succ_state, valuation_bits)
                            assignment[ex] = outer_win if parity == 0 else (not outer_win)
                        child_entrant_wins = self.registry.eval_entry(sigma_tid, b_idx, assignment)
                        move_values.append(child_entrant_wins if parity == 0 else (not child_entrant_wins))

                if not move_values:
                    # Terminal position: current player loses. The entrant wins iff current
                    # player is the opponent of the entrant.
                    return parity == 1
                if parity == 0:
                    return any(move_values)
                return all(move_values)

            table_bits = 0
            for valuation_bits in range(1 << len(attainable)):
                if win(start_state, valuation_bits):
                    table_bits |= 1 << valuation_bits

            entries.append(EntrySemantics(exits=attainable, truth_table=table_bits))

        sig = TypeSignature(labels=labels, entries=tuple(entries))
        return self.registry.intern(sig)

    def _solve_root(self) -> bool:
        """Solve the compressed root game once all child summaries are available."""
        root = self.instance.nodes[self.instance.root]
        if root.parent is not None:
            raise ValueError("Root node expected")

        child_initial_types: List[int] = [self.node_results[c].initial_type for c in root.children]
        child_type_options: List[Set[int]] = [self.node_results[c].realizable_types for c in root.children]

        local_type_ids = sorted(set().union(*child_type_options) if child_type_options else set())
        local_type_index = {tid: idx for idx, tid in enumerate(local_type_ids)}

        m0 = [0] * len(local_type_ids)
        for tid in child_initial_types:
            if tid in local_type_index:
                m0[local_type_index[tid]] += 1
        m0 = tuple(m0)

        full_F = (1 << len(root.internal_edge_indices)) - 1
        start_state = (self.instance.start, full_F, m0, 0)  # same = Player 1 to move

        @functools.lru_cache(maxsize=None)
        def win(state: Tuple[int, int, Tuple[int, ...], int]) -> bool:
            z, F_mask, m_vec, parity = state
            move_values: List[bool] = []

            # Internal moves.
            for local_e_idx, global_e_idx in enumerate(root.internal_edge_indices):
                if not ((F_mask >> local_e_idx) & 1):
                    continue
                u, v = self.instance.edges[global_e_idx]
                if z not in (u, v):
                    continue
                z2 = v if z == u else u
                move_values.append(win((z2, F_mask & ~(1 << local_e_idx), m_vec, parity_flip(parity))))

            # Child excursions.
            z_label = root.label_of_vertex[z]
            for sigma_idx, cnt in enumerate(m_vec):
                if cnt == 0:
                    continue
                sigma_tid = local_type_ids[sigma_idx]
                sig = self.registry.signature(sigma_tid)
                for a_idx, sem in enumerate(sig.entries):
                    if sig.labels[a_idx] != z_label:
                        continue
                    assignment: Dict[ExitLabel, bool] = {}
                    for ex in sem.exits:
                        q_abs_child, tau_tid, c = ex
                        return_label = sig.labels[q_abs_child]
                        z2 = root.vertex_of_label[return_label]
                        new_m = list(m_vec)
                        new_m[sigma_idx] -= 1
                        if tau_tid in local_type_index:
                            new_m[local_type_index[tau_tid]] += 1
                        else:
                            if self.registry.rank(tau_tid) != 0:
                                raise RuntimeError("Non-null successor type missing from root universe")
                        outer_win = win((z2, F_mask, tuple(new_m), parity_compose(parity, c)))
                        assignment[ex] = outer_win if parity == 0 else (not outer_win)
                    child_entrant_wins = self.registry.eval_entry(sigma_tid, a_idx, assignment)
                    move_values.append(child_entrant_wins if parity == 0 else (not child_entrant_wins))

            if not move_values:
                return parity == 1
            if parity == 0:
                return any(move_values)
            return all(move_values)

        return win(start_state)


# ---------------------------------------------------------------------------
# Verification / benchmarking helpers
# ---------------------------------------------------------------------------

def summarize_instance(instance: GraphWithPartition) -> str:
    """Return a human-readable summary of the generated instance."""
    lines = [
        f"n={instance.n}, m={len(instance.edges)}, start={instance.start}, root={instance.root}",
        "bags:",
    ]
    for nid, node in sorted(instance.nodes.items()):
        lines.append(
            "  bag "
            f"{nid}: verts={list(node.vertices)} parent={node.parent} children={list(node.children)} "
            f"ports={len(node.ports)} internal_edges={len(node.internal_edge_indices)}"
        )
    return "\n".join(lines)


def verify_random_instances(
    trials: int,
    n_vertices: int,
    width: int,
    seed: int = 0,
    p_internal: float = 0.35,
    p_cut: float = 0.45,
    verbose: bool = False,
    progress: bool = False,
) -> bool:
    """Compare the brute-force solver and the DP solver on random instances."""
    rnd = random.Random(seed)
    t_verify0 = time.perf_counter()

    for t in range(trials):
        trial_t0 = time.perf_counter()
        inst_seed = rnd.randrange(10**9)
        inst = random_partitioned_instance(
            n_vertices=n_vertices,
            width=width,
            seed=inst_seed,
            p_internal=p_internal,
            p_cut=p_cut,
        )
        brute = brute_force_ueg_winner(inst)
        tp = TreePartitionUEGSolver(inst).solve()
        trial_dt = time.perf_counter() - trial_t0

        if verbose:
            print(
                f"trial {t+1}/{trials}: brute={brute} tp={tp} n={inst.n} m={len(inst.edges)} "
                f"time={trial_dt:.3f}s"
            )
        elif progress:
            elapsed = time.perf_counter() - t_verify0
            avg = elapsed / (t + 1)
            eta = avg * (trials - (t + 1))
            print(
                f"[verify] trial {t+1}/{trials} finished in {_fmt_seconds(trial_dt)}; "
                f"elapsed={_fmt_seconds(elapsed)}; eta~{_fmt_seconds(eta)}",
                flush=True,
            )

        if brute != tp:
            print("Mismatch found!")
            print(f"seed={inst_seed}, n={inst.n}, m={len(inst.edges)}, start={inst.start}")
            print(f"edges={inst.edges}")
            raise AssertionError("Tree-partition solver disagrees with brute-force minimax")

    return True


def benchmark(
    sizes: Sequence[int],
    width: int,
    trials_per_size: int,
    seed: int = 0,
    p_internal: float = 0.35,
    p_cut: float = 0.45,
    progress: bool = False,
) -> Dict[str, List[float]]:
    """Benchmark both solvers on random instances of varying size."""
    rnd = random.Random(seed)
    brute_times: List[float] = []
    tp_times: List[float] = []

    total_trials = len(sizes) * trials_per_size
    completed_trials = 0
    t_bench0 = time.perf_counter()

    for n in sizes:
        brute_acc = 0.0
        tp_acc = 0.0
        if progress:
            print(f"[benchmark] starting n={n} ({trials_per_size} trial(s))", flush=True)

        for trial_idx in range(trials_per_size):
            trial_t0 = time.perf_counter()
            inst_seed = rnd.randrange(10**9)
            inst = random_partitioned_instance(
                n_vertices=n,
                width=width,
                seed=inst_seed,
                p_internal=p_internal,
                p_cut=p_cut,
            )

            t0 = time.perf_counter()
            brute = brute_force_ueg_winner(inst)
            t1 = time.perf_counter()

            tp = TreePartitionUEGSolver(inst).solve()
            t2 = time.perf_counter()

            if brute != tp:
                raise AssertionError(f"Mismatch during benchmark at seed={inst_seed}, n={n}")

            brute_dt = t1 - t0
            tp_dt = t2 - t1
            brute_acc += brute_dt
            tp_acc += tp_dt

            completed_trials += 1
            if progress:
                elapsed = time.perf_counter() - t_bench0
                avg = elapsed / completed_trials
                eta = avg * (total_trials - completed_trials)
                total_dt = time.perf_counter() - trial_t0
                print(
                    f"[benchmark] n={n} trial {trial_idx+1}/{trials_per_size} "
                    f"done in {_fmt_seconds(total_dt)} "
                    f"(brute={_fmt_seconds(brute_dt)}, tp={_fmt_seconds(tp_dt)}); "
                    f"overall {completed_trials}/{total_trials}; "
                    f"elapsed={_fmt_seconds(elapsed)}; eta~{_fmt_seconds(eta)}",
                    flush=True,
                )

        brute_times.append(brute_acc / trials_per_size)
        tp_times.append(tp_acc / trials_per_size)
        if progress:
            print(
                f"[benchmark] finished n={n}: avg brute={_fmt_seconds(brute_times[-1])}, "
                f"avg tp={_fmt_seconds(tp_times[-1])}",
                flush=True,
            )

    return {
        "sizes": list(sizes),
        "brute_force_seconds": brute_times,
        "tree_partition_seconds": tp_times,
    }


def save_benchmark_plot(results: Dict[str, List[float]], outpath: Path) -> None:
    """Save a runtime comparison plot."""
    if plt is None:
        raise RuntimeError("matplotlib is not available")

    sizes = results["sizes"]
    brute = results["brute_force_seconds"]
    tp = results["tree_partition_seconds"]

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(sizes, brute, marker="o", label="Brute-force minimax")
    plt.plot(sizes, tp, marker="o", label="Tree-partition DP")
    plt.xlabel("Number of vertices")
    plt.ylabel("Average runtime (seconds)")
    plt.title("UEG runtime comparison on random tree-partitioned graphs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def benchmark_surface(
    sizes: Sequence[int],
    widths: Sequence[int],
    trials_per_point: int,
    seed: int = 0,
    p_internal: float = 0.35,
    p_cut: float = 0.45,
    solver: str = "tree_partition",
    progress: bool = False,
) -> Dict[str, object]:
    """
    Benchmark one solver, or both solvers, on a 2D grid of (n, width) values.

    Returns a dictionary containing the grid and one runtime matrix per requested
    solver. Each matrix is indexed by width first, then by size.
    """
    valid_solvers = {"tree_partition", "brute_force", "both"}
    if solver not in valid_solvers:
        raise ValueError(f"solver must be one of {sorted(valid_solvers)}")

    rnd = random.Random(seed)
    total_trials = len(widths) * len(sizes) * trials_per_point
    completed_trials = 0
    t0_global = time.perf_counter()

    brute_grid: List[List[float]] = []
    tp_grid: List[List[float]] = []

    for width in widths:
        brute_row: List[float] = []
        tp_row: List[float] = []
        for n in sizes:
            brute_acc = 0.0
            tp_acc = 0.0
            if progress:
                print(f"[surface] starting width={width}, n={n} ({trials_per_point} trial(s))", flush=True)

            for trial_idx in range(trials_per_point):
                trial_t0 = time.perf_counter()
                inst_seed = rnd.randrange(10**9)
                inst = random_partitioned_instance(
                    n_vertices=n,
                    width=width,
                    seed=inst_seed,
                    p_internal=p_internal,
                    p_cut=p_cut,
                )

                brute = None
                tp = None

                if solver in {"brute_force", "both"}:
                    t0 = time.perf_counter()
                    brute = brute_force_ueg_winner(inst)
                    t1 = time.perf_counter()
                    brute_acc += t1 - t0
                else:
                    t1 = time.perf_counter()

                if solver in {"tree_partition", "both"}:
                    t2 = time.perf_counter()
                    tp = TreePartitionUEGSolver(inst).solve()
                    t3 = time.perf_counter()
                    tp_acc += t3 - t2
                else:
                    t3 = time.perf_counter()

                if solver == "both" and brute != tp:
                    raise AssertionError(
                        f"Mismatch during surface benchmark at seed={inst_seed}, n={n}, width={width}"
                    )

                completed_trials += 1
                if progress:
                    elapsed = time.perf_counter() - t0_global
                    avg = elapsed / completed_trials
                    eta = avg * (total_trials - completed_trials)
                    total_dt = time.perf_counter() - trial_t0
                    print(
                        f"[surface] width={width} n={n} trial {trial_idx+1}/{trials_per_point} "
                        f"done in {_fmt_seconds(total_dt)}; overall {completed_trials}/{total_trials}; "
                        f"elapsed={_fmt_seconds(elapsed)}; eta~{_fmt_seconds(eta)}",
                        flush=True,
                    )

            if solver in {"brute_force", "both"}:
                brute_row.append(brute_acc / trials_per_point)
            if solver in {"tree_partition", "both"}:
                tp_row.append(tp_acc / trials_per_point)

        if solver in {"brute_force", "both"}:
            brute_grid.append(brute_row)
        if solver in {"tree_partition", "both"}:
            tp_grid.append(tp_row)

    results: Dict[str, object] = {
        "sizes": list(sizes),
        "widths": list(widths),
        "solver": solver,
    }
    if solver in {"brute_force", "both"}:
        results["brute_force_seconds"] = brute_grid
    if solver in {"tree_partition", "both"}:
        results["tree_partition_seconds"] = tp_grid
    return results


def save_surface_plot(
    results: Dict[str, object],
    outpath: Path,
    metric_key: str = "tree_partition_seconds",
    zlim: Optional[Tuple[float, float]] = None,
    log_z: bool = False,
) -> None:
    """Save a 3D runtime surface plot for one runtime matrix in ``results``.

    Parameters
    ----------
    results:
        Output of :func:`benchmark_surface`.
    outpath:
        Destination PNG path.
    metric_key:
        Which runtime matrix to plot.
    zlim:
        Optional shared z-axis limits. This is useful when plotting multiple
        surfaces that should be visually comparable.
    log_z:
        If true, plot ``log10(runtime_seconds)`` instead of raw runtime.
    """
    if plt is None:
        raise RuntimeError("matplotlib is not available")
    if metric_key not in results:
        raise KeyError(f"metric {metric_key!r} not present in results")

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np

    sizes = results["sizes"]
    widths = results["widths"]
    raw_z = np.array(results[metric_key], dtype=float)
    z = np.log10(np.maximum(raw_z, 1e-12)) if log_z else raw_z
    x, y = np.meshgrid(sizes, widths)

    if metric_key == "tree_partition_seconds":
        title = "UEG runtime surface: tree-partition DP"
        cmap = "Blues"
    else:
        title = "UEG runtime surface: brute-force minimax"
        cmap = "Reds"

    fig = plt.figure(figsize=(8.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=True)
    ax.view_init(elev=28, azim=-135)
    ax.set_xlabel("n")
    ax.set_ylabel("width k")
    ax.set_zlabel("log10 runtime (seconds)" if log_z else "runtime (seconds)")
    if zlim is not None:
        zmin, zmax = zlim
        if zmin == zmax:
            pad = 1e-9 if not log_z else 1e-3
            zmin -= pad
            zmax += pad
        ax.set_zlim(zmin, zmax)
    ax.set_title(title)
    fig.colorbar(surface, ax=ax, shrink=0.7, pad=0.1)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _shared_surface_zlim(results: Dict[str, object], metric_keys: Sequence[str], log_z: bool) -> Optional[Tuple[float, float]]:
    """Return a shared z-axis range for the given surface metrics."""
    import numpy as np

    values: List[float] = []
    for metric_key in metric_keys:
        if metric_key not in results:
            continue
        arr = np.array(results[metric_key], dtype=float)
        if log_z:
            arr = np.log10(np.maximum(arr, 1e-12))
        values.extend(arr.ravel().tolist())
    if not values:
        return None
    return (min(values), max(values))


def save_surface_csv(results: Dict[str, object], outpath: Path) -> None:
    """Save surface benchmark results as CSV in long format."""
    sizes = results["sizes"]
    widths = results["widths"]
    solver = results["solver"]

    with outpath.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["solver", "width", "n", "avg_runtime_seconds"])

        if solver in {"brute_force", "both"}:
            grid = results["brute_force_seconds"]
            for width, row in zip(widths, grid):
                for n, value in zip(sizes, row):
                    writer.writerow(["brute_force", width, n, value])

        if solver in {"tree_partition", "both"}:
            grid = results["tree_partition_seconds"]
            for width, row in zip(widths, grid):
                for n, value in zip(sizes, row):
                    writer.writerow(["tree_partition", width, n, value])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UEG correctness / benchmark prototype")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="verify the DP solver against brute-force on random instances")
    p_verify.add_argument("--trials", type=int, default=20)
    p_verify.add_argument("--n", type=int, default=10)
    p_verify.add_argument("--width", type=int, default=3)
    p_verify.add_argument("--seed", type=int, default=0)
    p_verify.add_argument("--p-internal", type=float, default=0.35)
    p_verify.add_argument("--p-cut", type=float, default=0.45)
    p_verify.add_argument("--verbose", action="store_true")
    p_verify.add_argument("--progress", action="store_true")

    p_bench = sub.add_parser("benchmark", help="benchmark both solvers and optionally save a plot")
    p_bench.add_argument("--sizes", type=str, default="6,7,8,9,10,11")
    p_bench.add_argument("--width", type=int, default=3)
    p_bench.add_argument("--trials-per-size", type=int, default=5)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("--p-internal", type=float, default=0.35)
    p_bench.add_argument("--p-cut", type=float, default=0.45)
    p_bench.add_argument("--plot", type=str, default="")
    p_bench.add_argument("--progress", action="store_true")

    p_demo = sub.add_parser("demo", help="generate one instance and solve it both ways")
    p_demo.add_argument("--n", type=int, default=10)
    p_demo.add_argument("--width", type=int, default=3)
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--p-internal", type=float, default=0.35)
    p_demo.add_argument("--p-cut", type=float, default=0.45)
    p_demo.add_argument("--show-edges", action="store_true", help="print the edge list in the demo output")

    p_surface = sub.add_parser("surface", help="benchmark on a grid of (n, width) values and optionally save 3D plots")
    p_surface.add_argument("--sizes", type=str, default="6,8,10")
    p_surface.add_argument("--widths", type=str, default="2,3,4")
    p_surface.add_argument("--trials-per-point", type=int, default=3)
    p_surface.add_argument("--seed", type=int, default=0)
    p_surface.add_argument("--p-internal", type=float, default=0.35)
    p_surface.add_argument("--p-cut", type=float, default=0.45)
    p_surface.add_argument("--solver", choices=["tree_partition", "brute_force", "both"], default="tree_partition")
    p_surface.add_argument("--plot-prefix", type=str, default="")
    p_surface.add_argument("--csv", type=str, default="")
    p_surface.add_argument(
        "--log-z",
        action="store_true",
        help="plot log10(runtime) on the z-axis for easier visual comparison",
    )
    p_surface.add_argument(
        "--independent-z-scale",
        action="store_true",
        help="do not force a shared z-axis range when saving multiple surface plots",
    )
    p_surface.add_argument("--progress", action="store_true")

    args = parser.parse_args()

    if args.cmd == "verify":
        ok = verify_random_instances(
            trials=args.trials,
            n_vertices=args.n,
            width=args.width,
            seed=args.seed,
            p_internal=args.p_internal,
            p_cut=args.p_cut,
            verbose=args.verbose,
            progress=args.progress,
        )
        print(f"verification passed: {ok}")
        return

    if args.cmd == "benchmark":
        sizes = parse_int_list(args.sizes)
        results = benchmark(
            sizes=sizes,
            width=args.width,
            trials_per_size=args.trials_per_size,
            seed=args.seed,
            p_internal=args.p_internal,
            p_cut=args.p_cut,
            progress=args.progress,
        )
        print("sizes                 :", results["sizes"])
        print("brute_force_seconds   :", [round(x, 6) for x in results["brute_force_seconds"]])
        print("tree_partition_seconds:", [round(x, 6) for x in results["tree_partition_seconds"]])
        if args.plot:
            outpath = Path(args.plot)
            save_benchmark_plot(results, outpath)
            print(f"saved plot to {outpath}")
        return

    if args.cmd == "surface":
        sizes = parse_int_list(args.sizes)
        widths = parse_int_list(args.widths)
        results = benchmark_surface(
            sizes=sizes,
            widths=widths,
            trials_per_point=args.trials_per_point,
            seed=args.seed,
            p_internal=args.p_internal,
            p_cut=args.p_cut,
            solver=args.solver,
            progress=args.progress,
        )
        print("sizes   :", results["sizes"])
        print("widths  :", results["widths"])
        print("solver  :", results["solver"])
        if args.solver in {"brute_force", "both"}:
            print("brute_force_seconds   :", [[round(x, 6) for x in row] for row in results["brute_force_seconds"]])
        if args.solver in {"tree_partition", "both"}:
            print("tree_partition_seconds:", [[round(x, 6) for x in row] for row in results["tree_partition_seconds"]])
        if args.csv:
            csv_path = Path(args.csv)
            save_surface_csv(results, csv_path)
            print(f"saved csv to {csv_path}")
        if args.plot_prefix:
            prefix = Path(args.plot_prefix)
            metric_keys: List[str] = []
            if args.solver in {"tree_partition", "both"}:
                metric_keys.append("tree_partition_seconds")
            if args.solver in {"brute_force", "both"}:
                metric_keys.append("brute_force_seconds")

            shared_zlim = None
            if len(metric_keys) > 1 and not args.independent_z_scale:
                shared_zlim = _shared_surface_zlim(results, metric_keys, log_z=args.log_z)

            if args.solver in {"tree_partition", "both"}:
                tp_plot = prefix.with_name(prefix.name + "_tree_partition.png")
                save_surface_plot(
                    results,
                    tp_plot,
                    metric_key="tree_partition_seconds",
                    zlim=shared_zlim,
                    log_z=args.log_z,
                )
                print(f"saved tree-partition surface to {tp_plot}")
            if args.solver in {"brute_force", "both"}:
                brute_plot = prefix.with_name(prefix.name + "_brute_force.png")
                save_surface_plot(
                    results,
                    brute_plot,
                    metric_key="brute_force_seconds",
                    zlim=shared_zlim,
                    log_z=args.log_z,
                )
                print(f"saved brute-force surface to {brute_plot}")
        return

    if args.cmd == "demo":
        inst = random_partitioned_instance(
            n_vertices=args.n,
            width=args.width,
            seed=args.seed,
            p_internal=args.p_internal,
            p_cut=args.p_cut,
        )
        print(summarize_instance(inst))
        if args.show_edges:
            print("edges:", inst.edges)
        brute = brute_force_ueg_winner(inst)
        tp = TreePartitionUEGSolver(inst).solve()
        print(f"brute-force winner    : {'P1' if brute else 'P2'}")
        print(f"tree-partition winner : {'P1' if tp else 'P2'}")
        return

    raise RuntimeError("unknown command")


if __name__ == "__main__":
    main()
