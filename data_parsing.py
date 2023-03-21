from min_cost_flow import Network, successive_shortest_paths
import networkx as nx
import torch
import matplotlib.pyplot as plt
from typing import Optional
from collections import OrderedDict
import numpy as np


def parse(filename):
    """
    Parses a network file following in DIMACS format

    Some elements of the specification:
    - Lines starting in c are comments
    - Lines starting in p explain what problem to solve (can be ignored,
      we only consider minimum-cost flow problems)
    - Lines starting in n define nodes
    - Lines starting in a define arcs (edges)

    Args:
        filename: name of the file containing the network data

    Returns:
        A dictionary keyed on nodes with supply, and a dictionary
        keyed on edges with capacity and cost.
    """
    # Lines we can ignore
    ignore_list = ['c', 'p']

    file = open(filename, 'r')

    # Nodes is a hashmap from node values to their supply
    nodes = {}
    # Edges is a hashmap from edges to a tuple with their capacity and cost
    edges = {}

    for line in file:
        if len(line) > 0 and line[0] not in ignore_list:
            if line[0] == 'n':
                # Node parsing
                node = [int(elem) for elem in line.split(' ')[1:]]
                nodes[node[0]] = node[1]
            elif line[0] == 'a':
                arc = [int(elem) for elem in line.split(' ')[1:]]
                node1, node2, _, capacity, cost = arc

                # Only nodes with non-zero supply are in a "node line"
                if node1 not in nodes:
                    nodes[node1] = 0
                if node2 not in nodes:
                    nodes[node2] = 0
                if (node1, node2) in edges:
                    old_capacity, old_cost = edges[(node1, node2)]
                    new_cost = old_cost * old_capacity + cost * capacity
                    new_cost /= (old_capacity + capacity)
                    new_cost = int(new_cost)
                    edges[(node1, node2)] = (old_capacity + capacity, new_cost)
                else:
                    edges[(node1, node2)] = (capacity, cost)
    file.close()
    nodes = OrderedDict(sorted(nodes.items(), key = lambda t: t[0]))

    supply = 0
    for node in nodes:
        excess = nodes[node]
        if excess > 0:
            edges[(-1, node)] = (excess, 0)
            supply += excess
        elif excess < 0:
            edges[(node, -2)] = (-excess, 0)

        nodes[node] = 0

    nodes[-1] = supply
    nodes[-2] = -supply

    return nodes, edges


def build_network(nodes, edges):
    capacities, costs = zip(*edges.values())
    return Network(
        list(nodes.keys()),
        list(edges.keys()),
        capacities,
        costs,
        list(nodes.values())
    )


def build_networkx(nodes: dict, edges: dict) -> nx.digraph:
    """
    Converts (nodes, edges) output from parse into a directed
    NetworkX graph.

    Args:
        nodes: dictionary keyed on nodes with supply
        edges: dictionary keyed on edges with capacity and cost

    Returns:
        Directed NetworkX graph representing the flow network.
    """
    G = nx.DiGraph()
    for (n, s) in nodes.items():
        G.add_node(n, demand = -s)

    for (e, (u, c)) in edges.items():
        G.add_edge(e[0], e[1], weight = c, capacity = u)

    return G


def _build_pyg(
        nodes: dict,
        edges: dict,
        opt: float,
        p: dict,
        converged: bool,
        c_p: dict
    ) -> dict:
    """
    Converts (nodes, edges) output from parse into the tensors needed to
    initialize a PyG data object.

    Args:
        nodes: dictionary keyed on nodes with supply
        edges: dictionary keyed on edges with capacity and cost
        opt: the optimal min cost flow value in the network
        p: a set of optimal potentials in the network
        converged: boolean that is True if the min cost flow algorithm converged
            in the desired number of iterations and False otherwise. If False,
            the input is excluded from final dataset.

    Returns:
        Dictionary containing objects needed to initialize PyG dataset:
            1) Edge index tensor
            2) Edge attribute tensor
            3) Node feature tensor
            4) Graph label tensor
    """
    if len(edges.keys()) <= 1e6 and converged:
        index = {node: index for node, index in zip(nodes, range(len(nodes)))}
        x = torch.tensor([supply for supply in nodes.values()]).reshape((-1, 1)).type(torch.FloatTensor)
        edge_index = []
        edge_attr = []
        reduced_cost = []
        y = np.array([[opt for _ in range(len(p))], list(p.values())])
        for e, attr in edges.items():
            edge_index.append([index[e[0]], index[e[1]]])
            edge_attr.append(list(attr))
            reduced_cost.append(c_p[e])
        edge_index = torch.tensor(edge_index).T
        edge_attr = torch.tensor(edge_attr).type(torch.FloatTensor)
        reduced_cost = torch.tensor(reduced_cost)
        y = torch.tensor(y).T
        return {"converged": converged, "x": x, "edge_index": edge_index, "edge_attr": edge_attr, "y": y,
                "reduced_cost": reduced_cost}

    return {"converged": False}


def min_cost_flow(
        nodes: dict,
        edges: dict,
        flow_alg: str,
        debug: bool
    ) -> tuple:
    """
    Given the network defined by [(nodes, edges)], compute a minimum cost
    flow using [flow_alg]. If [debug] is True, print a visualization of the
    network.

    Args:
        nodes: dictionary keyed on nodes with supply
        edges: dictionary keyed on edges with capacity and cost
        flow_alg: one of 'nx' or 'cbn'. The former is networkx's proprietary
            min-cost flow algorithm, the latter calls solvers in
            min_cost_flow.py.

    Returns:
        Whether or not the min-cost flow algorithm converged on the instance. If
        converged, the optimal reduced costs, value, and potentials.
    """
    if flow_alg == 'nx':
        G = build_networkx(nodes, edges)
        if debug:
            nx.draw(G, pos = nx.circular_layout(G), with_labels = True)
            plt.show()
            print(G.nodes(data = True))
            print(G.edges(data = True))
        opt = nx.min_cost_flow_cost(G)
    if flow_alg == 'cbn':
        N = build_network(nodes, edges)
        converged, c_p, _, p, opt = successive_shortest_paths(N, iter_limit=150)
    return converged, c_p, opt, p


def process_file(filename, flow_alg, debug: Optional[bool] = False):
    """
    Given a flow network in DIMACS format in [filename], convert into
    PyG-compatible input tensors, and label the instance according to
    min-cost flow information found by running [flow_alg] on the instance.

    Args:
        filename: name of the file containing the network data
        flow_alg: one of 'nx' or 'cbn'. The former is networkx's proprietary
            min-cost flow algorithm, the latter calls solvers in
            min_cost_flow.py.

    Returns:
        Dictionary containing objects needed to initialize PyG dataset:
            1) Edge index tensor
            2) Edge attribute tensor
            3) Node feature tensor
            4) Graph label tensor
    """
    nodes, edges = parse(filename)
    if len(edges.keys()) <= 1e6:
        converged, c_p, opt, p = min_cost_flow(nodes, edges, flow_alg, debug)
        return _build_pyg(nodes, edges, opt, p, converged, c_p)
    else:
        return {"converged": False}
