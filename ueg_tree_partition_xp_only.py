#!/usr/bin/env python3
"""
Minimal implementation of the XP algorithm for Undirected Edge Geography (UEG)
on a graph given together with a rooted tree partition.

This module intentionally omits:
- brute-force minimax,
- random instance generation,
- benchmarking,
- plotting,
- verification harnesses.

It contains only the core data structures and the dynamic programming solver.

Notes
-----
- The input partition must be a *valid rooted tree partition* of the graph.
- For arbitrary graphs, computing a good tree partition is difficult.
- A trivial one-bag partition helper is provided for debugging/small instances.
"""

from __future__ import annotations

import functools
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


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
    adjacency: Tuple[Tuple[int, ...], ...]
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
# Instance construction helpers
# ---------------------------------------------------------------------------


def build_graph_with_partition(
    n: int,
    edges: Sequence[Tuple[int, int]],
    bags: Sequence[Sequence[int]],
    parent: Sequence[Optional[int]],
    start: int,
) -> GraphWithPartition:
    """
    Build a ``GraphWithPartition`` from an explicit graph and rooted tree partition.

    Parameters
    ----------
    n:
        Number of vertices. Vertices are assumed to be ``0, ..., n-1``.
    edges:
        Undirected edges of the graph.
    bags:
        ``bags[i]`` is the list of vertices in bag ``i``.
    parent:
        ``parent[i]`` is the parent of bag ``i`` in the rooted tree partition,
        or ``None`` for the root.
    start:
        Start vertex of the UEG instance.

    Returns
    -------
    GraphWithPartition

    Raises
    ------
    ValueError
        If the supplied data do not define a valid rooted tree partition.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= start < n):
        raise ValueError("start vertex out of range")
    if len(bags) != len(parent):
        raise ValueError("bags and parent must have the same length")
    if not bags:
        raise ValueError("at least one bag is required")

    # Normalize / validate edges.
    norm_edges = tuple(sorted({edge_key(u, v) for (u, v) in edges}))
    for u, v in norm_edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge endpoint out of range")
        if u == v:
            raise ValueError("loops are not supported")

    edge_to_index = {e: idx for idx, e in enumerate(norm_edges)}
    adjacency_lists: List[List[int]] = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(norm_edges):
        adjacency_lists[u].append(idx)
        adjacency_lists[v].append(idx)

    num_bags = len(bags)

    # Validate the rooted bag tree.
    roots = [i for i, p in enumerate(parent) if p is None]
    if len(roots) != 1:
        raise ValueError("parent must specify exactly one root")
    root = roots[0]
    children: Dict[int, List[int]] = defaultdict(list)
    for i, p in enumerate(parent):
        if p is None:
            continue
        if not (0 <= p < num_bags):
            raise ValueError("parent index out of range")
        if p == i:
            raise ValueError("a bag cannot be its own parent")
        children[p].append(i)

    # Check connectivity / acyclicity of the bag tree by DFS from the root.
    seen: Set[int] = set()
    stack = [root]
    while stack:
        cur = stack.pop()
        if cur in seen:
            raise ValueError("parent relation does not define a tree")
        seen.add(cur)
        stack.extend(children[cur])
    if len(seen) != num_bags:
        raise ValueError("parent relation does not span all bags")

    # Validate bag partition of vertices.
    bag_of_vertex = [-1] * n
    for bag_id, bag in enumerate(bags):
        for v in bag:
            if not (0 <= v < n):
                raise ValueError("bag vertex out of range")
            if bag_of_vertex[v] != -1:
                raise ValueError("bags must form a partition of the vertex set")
            bag_of_vertex[v] = bag_id
    if any(x == -1 for x in bag_of_vertex):
        raise ValueError("bags must cover all vertices")

    # Check the tree-partition condition.
    for u, v in norm_edges:
        bu = bag_of_vertex[u]
        bv = bag_of_vertex[v]
        if bu == bv:
            continue
        if parent[bu] != bv and parent[bv] != bu:
            raise ValueError(
                "every edge must be internal to a bag or cross a parent-child cut"
            )

    # Local labelings.
    label_of_vertex_by_bag: Dict[int, Dict[int, int]] = {}
    vertex_of_label_by_bag: Dict[int, Dict[int, int]] = {}
    for i, bag in enumerate(bags):
        ordered_bag = tuple(bag)
        label_of_vertex_by_bag[i] = {v: idx for idx, v in enumerate(ordered_bag)}
        vertex_of_label_by_bag[i] = {idx: v for idx, v in enumerate(ordered_bag)}

    # Build node metadata.
    nodes: Dict[int, NodeData] = {}
    for i, bag in enumerate(bags):
        verts = tuple(bag)

        internal_edge_indices: List[int] = []
        for u, v in itertools.combinations(verts, 2):
            ek = edge_key(u, v)
            if ek in edge_to_index:
                internal_edge_indices.append(edge_to_index[ek])

        ports: List[Port] = []
        p = parent[i]
        if p is not None:
            for u in bags[p]:
                for v in verts:
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
            parent=parent[i],
            children=tuple(children[i]),
            label_of_vertex=dict(label_of_vertex_by_bag[i]),
            vertex_of_label=dict(vertex_of_label_by_bag[i]),
            internal_edge_indices=tuple(sorted(internal_edge_indices)),
            ports=tuple(ports),
        )

    # Root should contain the start vertex, as assumed by the root solver.
    if bag_of_vertex[start] != root:
        raise ValueError(
            "the root bag must contain the start vertex; reroot the partition accordingly"
        )

    return GraphWithPartition(
        n=n,
        edges=norm_edges,
        adjacency=tuple(tuple(lst) for lst in adjacency_lists),
        edge_to_index=edge_to_index,
        bag_of_vertex=tuple(bag_of_vertex),
        nodes=nodes,
        root=root,
        start=start,
    )



def trivial_single_bag_partition(
    n: int,
    edges: Sequence[Tuple[int, int]],
    start: int = 0,
) -> GraphWithPartition:
    """
    Fallback partition for debugging: all vertices are placed in one root bag.

    This is always a valid rooted tree partition, but of width ``n``.
    """
    return build_graph_with_partition(
        n=n,
        edges=edges,
        bags=[list(range(n))],
        parent=[None],
        start=start,
    )



def solve_ueg_with_partition(
    n: int,
    edges: Sequence[Tuple[int, int]],
    bags: Sequence[Sequence[int]],
    parent: Sequence[Optional[int]],
    start: int,
) -> bool:
    """Convenience wrapper: build the instance and solve it."""
    instance = build_graph_with_partition(n=n, edges=edges, bags=bags, parent=parent, start=start)
    return TreePartitionUEGSolver(instance).solve()


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

    Procedure:
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
