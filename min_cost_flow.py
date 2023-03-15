from typing import Any, List, Tuple, Dict
from collections import deque, defaultdict
from heapq import *
import numpy as np
from copy import copy

from numpy import ndarray


class Network:
    def __init__(
            self,
            nodes: List[object],
            edges: List[Tuple[object, object]],
            capacities: List[int],
            costs: List[int],
            supplies: List[int]
    ) -> None:
        # -1: meta source
        # -2: meta sink
        self.V = [*nodes, -1, -2]

        # Update input for meta nodes
        source_indices = [i for i in range(len(nodes)) if supplies[i] > 0]
        sink_indices = [i for i in range(len(nodes)) if supplies[i] < 0]

        source_connections = [(-1, nodes[i]) for i in source_indices]
        sink_connections = [(nodes[i], -2) for i in sink_indices]

        edges = [
            *edges,
            *source_connections,
            *sink_connections,
        ]

        capacities = [
            *capacities,
            *[supplies[i] for i in source_indices],
            *[-supplies[i] for i in sink_indices]
        ]

        costs = [
            *costs,
            *[0 for _, i in source_connections],
            *[0 for _, i in sink_connections]
        ]

        total_supply = sum(np.abs(supplies)) // 2
        supplies = [*np.zeros(len(supplies)), total_supply, -total_supply]
        self.E = [(*edges[i - 1], i) for i in range(1, len(edges) + 1)]
        self.u = dict(zip(self.E, capacities))
        self.c = dict(zip(self.E, costs))
        self.b = dict(zip(self.V, supplies))


def dijkstra(
        S_f: object,
        T_f: object,
        adj: Dict[Tuple[object, object], int]
) -> Tuple[Dict[object, int], List[Tuple[object, object]]]:
    """
    Computes the shortest path distances from [s] to all other nodes 
    and the shortest path from [s] to [t] on the graph with edges/weights 
    given by the entries of [adj]. Implementation of Dijkstra's shortest
    path algorithm for graphs with non-negative edge costs.
    
    Args:
        s: Start node
        t: End node (for path)
        adj: Dictionary containing edges and their respective weights, with
            adj[(i, j)] = w_{ij} for nodes i and j.
    """

    def _format_path(lst):
        return [(lst[i][0], *lst[i + 1]) for i in range(len(lst) - 1)]

    # Generate underlying adjacency lists
    adjacency = defaultdict(list)
    for (i, j, id), c in adj.items():
        if c < 0:
            print(f"cost can't be < 0: {(i, j, id, c)}")
        assert (c >= 0)
        adjacency[i].append((j, id, c))

    # Initialize queue, distances
    queue, seen, distances = [(0, s, 0, []) for s in S_f], set(), {s: 0 for s in S_f}

    while queue:
        cost, v1, id, path = heappop(queue)
        if v1 not in seen:
            seen.add(v1)
            path = [*path, (v1, id)]
            if v1 in T_f:
                out_path = _format_path(path)

            for v2, id, c in adjacency.get(v1, ()):
                if v2 in seen:
                    continue
                prev = distances.get(v2, None)
                nxt = cost + c
                if prev is None or nxt < prev:
                    distances[v2] = nxt
                    heappush(queue, (nxt, v2, id, path))
    return distances, out_path


def BFS(
        S_f: object,
        T_f: object,
        adj: List[Tuple[object, object]]
) -> Tuple[Dict[object, int], List[Tuple[object, object]]]:
    def _format_path(lst):
        return [(lst[i][0], *lst[i + 1]) for i in range(len(lst) - 1)]

    # Generate underlying adjacency lists
    adjacency = defaultdict(list)
    for (i, j, id) in adj:
        adjacency[i].append((j, id))

    # Initialize queue, distances
    queue, seen = (deque([(s, 0, []) for s in S_f]), set())

    while queue:
        v1, id, path = queue[0]

        if v1 not in seen:
            seen.add(v1)
            path = [*path, (v1, id)]
            if v1 in T_f:
                out_path = _format_path(path)
                return out_path

            for v2, id in adjacency.get(v1, ()):
                if v2 in seen:
                    continue
                queue.append((v2, id, path))

        queue.popleft()
    return None


def rev(edge):
    return (edge[1], edge[0], -edge[2])


def reduced_cost(
        N: Network,
        u_f: Dict[Tuple[object, object], int],
        p: Dict[object, int]
) -> Dict[Tuple[object, object], int]:
    """
    Computes reduced costs of the edges in the residual graph [u_f] with respect to edge
    costs [N.c] and node potentials [p].

    Args:
        N: Network representing the problem input
        u_f: Dictionary encoding the residual graph w.r.t the current flow
        p: Current node potentials

    Returns:
        A dictionary which gives the reduced cost for each edge (u, v) according to
        c_p[(u, v)] = c[(u, v)] + p[u] - p[v].
    """
    reduced_costs = {}
    for e in u_f.keys():
        (u, v, _) = e
        if e in N.c:
            reduced_costs[e] = int(N.c[e] + p[u] - p[v])
        else:
            reduced_costs[e] = int(-N.c[rev(e)] + p[u] - p[v])
    return reduced_costs


def excess_nodes(
        N: Network,
        f: Dict[Tuple[object, object], int],
) -> Tuple[List, List, Dict]:
    """
    Compute nodes in the network [N] where flow conservation is violated by at least [K] units
    for the flow [f]. 
    
    Args:
        N: Network object representing the problem input
        f: Potentially infeasible flow
    
    Returns:
        A tuple consisting of a list of nodes where net flow in is greater than [K]
        and a list of nodes where the net flow in is less than -[K].
    """

    def _excess(N, f) -> Dict[object, int]:
        # Initialize excess to be supply
        excess = {v: N.b[v] for v in N.V}
        for (u, v, _), val in f.items():
            excess[u] -= val
            excess[v] += val

        return excess

    e_f = _excess(N, f)
    S_f = [i for (i, val) in e_f.items() if val > 0]
    T_f = [i for (i, val) in e_f.items() if val < 0]
    return S_f, T_f, e_f


def update_potentials(
        p: Dict[object, int],
        distances: Dict[object, int],
        P
) -> None:
    """
    Updates node potentials [p] with the shortest path distances in
    [distances] according to p[i] <- p[i] + distances[i].
    
    Args:
        p: Previous node potentials
        distances: Shortest path distance to each node in graph from a node with
            surplus above the current scaling threshold
        
    """
    t = P[-1][1]

    for i in p.keys():
        if i in distances:
            p[i] += (min(distances[i], distances[t]))
        else:
            p[i] += (distances[t])


def saturate_edges(
        N: Network,
        f: Dict[Tuple[object, object], int],
        u_f: Dict[Tuple[object, object], int],
        edges: List[Tuple[object, object]]
) -> None:
    """
    Updates the flow [f] and residual graph [u_f] by saturating
    all edges in [edges].
    
    Args:
        N: Flow network encoding the problem input
        u_f: The residual graph for the current flow
        f: The current flow
        edges: List of edges to saturate
        
    """

    for e in edges:
        if e in f:
            f[e] = N.u[e]  # Saturate foward edge
            u_f[e] = 0  # Zero forward residual edge
            u_f[rev(e)] = N.u[e]  # Saturate backward residual edge

        else:
            f[rev(e)] = 0  # Zero forward edge
            u_f[e] = 0  # Saturate forward residual edge
            u_f[rev(e)] = N.u[rev(e)]  # Zero backward residual edge


def saturate_neg_cost_admissible(
        N: Network,
        c_p: Dict[Tuple[object, object], int],
        f: Dict[Tuple[object, object], int],
        u_f: Dict[Tuple[object, object], int]
) -> None:
    """
    Updates the current flow [f] and residual graph [u_f] by
    saturating all edges with residual capacity of at least [K]
    and negative reduced cost [c_p] to preserve invariants in the
    algorithm.
    
    Args:
        N: Flow network encoding the problem input
        c_p: Current reduced costs
        f: The current flow
        u_f: The residual graph for the current flow
    """
    neg_cost_admissible = [
        e
        for e, u in u_f.items()
        if c_p[e] < 0 and u > 0
    ]
    print(f"Number of negative cost admissible edges: {len(neg_cost_admissible)}")

    saturate_edges(N, f, u_f, neg_cost_admissible)


def augment_flow_along_path(
        P: List[Tuple[object, object]],
        f: Dict[Tuple[object, object], int],
        u_f: Dict[Tuple[object, object], int],
        e_f
) -> None:
    """ 
    Updates the current flow [f] and residual graph [u_f] by
    pushing [K] units of flow along the directed path P.
    
    Args:
        P: Path of edges to push flow
        f: Current flow
        c_f: Current residual graph
        K: Scaling parameter
    
    """
    s = P[0][0]
    t = P[-1][1]

    min_capacity = min(u_f[e] for e in P)
    delta = min(e_f[s], -e_f[t], min_capacity)

    for e in P:
        if e in f:
            f[e] += delta
            u_f[e] -= delta
            u_f[rev(e)] = u_f.get(rev(e), 0) + delta

        else:
            f[rev(e)] -= delta
            u_f[rev(e)] += delta
            u_f[e] -= delta


def primal_value(N, f):
    return np.sum([f[e] * N.c[e] for e in N.E if e in f])


def dual_value(N, p):
    return -np.sum([p[i] * N.b[i] for i in N.V]) - np.sum([N.u[e] * max(0, p[e[1]] - p[e[0]] - N.c[e]) for e in N.E])


def init_residual_graph(N, f):
    u_f = {}
    for e in N.E:
        if e in f:
            u_f[e] = N.u[e] - f[e]
            u_f[rev(e)] = f[e]
        else:
            u_f[e] = N.u[e]
    return u_f


def compute_feasible_flow(
        N: Network,
        p
):
    f = {e: 0 for e in N.E}
    u_f = init_residual_graph(N, f)
    c_p = reduced_cost(N, u_f, p)
    saturate_neg_cost_admissible(N, c_p, f, u_f)
    N_prime = copy(N)
    N_prime.E = [e for e in N.E if c_p[e] == 0]

    u_f_prime = init_residual_graph(N_prime, f)

    S_f, T_f, e_f = excess_nodes(N_prime, f)

    while len(S_f) > 0:

        # Admissible edges
        adj = [e for (e, u) in u_f_prime.items() if u > 0]
        P = BFS(S_f, T_f, adj)
        if P is None:
            break
        augment_flow_along_path(P, f, u_f_prime, e_f)
        S_f, T_f, e_f = excess_nodes(N_prime, f)

    return f


def successive_shortest_paths(
        N: Network,
        **kwargs
):
    """
    Primal-dual algorithm for computing a minimum-cost flow for the 
    network [N] starting from dual-feasible node potentials [p].
    
    Args:
        N: Flow network encoding the problem input
        p: Initial node potentials for warm start
        
    Returns:
        Minimum cost flow and corresponding optimal node potentials
    """

    # Init zero flow and potentials
    if 'f' not in kwargs:
        f = {e: 0 for e in N.E}
    else:
        f = copy(kwargs['f'])

    if 'p' not in kwargs:
        p = {v: 0 for v in N.V}
        f = {e: 0 for e in N.E}
    else:
        p = copy(kwargs['p'])
        f = compute_feasible_flow(N, p)

    iter_limit = kwargs.get('iter_limit', -1)

    u_f = init_residual_graph(N, f)

    iters = 0

    # Compute reduced costs w.r.t potentials p
    c_p = reduced_cost(N, u_f, p)
    S_f, T_f, e_f = excess_nodes(N, f)

    while len(S_f) > 0:
        # Admissible edges
        adj = {e: c for (e, c) in c_p.items() if u_f[e] > 0}
        D, P = dijkstra(S_f, T_f, adj)
        update_potentials(p, D, P)
        augment_flow_along_path(P, f, u_f, e_f)
        c_p = reduced_cost(N, u_f, p)
        S_f, T_f, e_f = excess_nodes(N, f)

        iters += 1
        if iters == iter_limit:
            if 'iter_limit' in kwargs:
                return False, f, p, np.array([-1])
            else:
                return f, p, np.array([-1])

    assert len({e: c for (e, c) in c_p.items() if u_f[e] > 0 and c < 0}) == 0

    if 'iter_limit' in kwargs:
        return True, f, p, primal_value(N, f)
    else:
        return f, p, primal_value(N, f)
