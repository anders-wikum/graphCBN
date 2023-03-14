from min_cost_flow import Network, successive_shortest_paths
import networkx as nx
import torch
import matplotlib.pyplot as plt
from typing import Optional

def parse(filename):
    """
    Parses a network file following the DIMACS problem specification
    structure and transforms it into a Network object

    Some elements of the specification:
    - Lines starting in c are comments
    - Lines starting in p explain what problem to solve (can be ignored,
      we only consider minimum-cost flow problems)
    - Lines starting in n define nodes
    - Lines starting in a define arcs (edges)

    Args:
        filename: name of the file containing the network data

    Returns:
        The corresponding Network object
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
                    edges[(node1, node2)] = (old_capacity + capacity, new_cost)
                else:
                    edges[(node1, node2)] = (capacity, cost)
    file.close()
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

def build_networkx(nodes, edges):
    G = nx.DiGraph()
    for (n, s) in nodes.items():
        G.add_node(n, demand=-s)

    for (e, (u, c)) in edges.items():
        G.add_edge(e[0], e[1], weight=c, capacity=u)

    return G

def build_pyg(nodes, edges, opt):
    if (len(edges.keys()) <= 1e6) and (opt is not None):
        index = {node: index for node, index in zip(nodes, range(len(nodes)))}
        x = torch.tensor([supply for supply in nodes.values()]).reshape((-1, 1))
        edge_index = torch.reshape(torch.tensor([[index[e[0]], index[e[1]]] for e in edges]), (2, -1))
        edge_attr = torch.tensor([list(attributes) for attributes in edges.values()])
        y = torch.tensor(opt).reshape(1,1)
        return {"converged": True, "x": x, "edge_index": edge_index, "edge_attr": edge_attr, "y": y}
    
    return {"converged": False}


def min_cost_flow(nodes, edges, flow_alg, debug):
    if flow_alg == 'nx':
        G = build_networkx(nodes, edges)
        if debug:
          nx.draw(G, pos=nx.circular_layout(G), with_labels=True)
          plt.show()
          print(G.nodes(data=True))
          print(G.edges(data=True))
        opt = nx.min_cost_flow_cost(G)
    if flow_alg == 'cbn':
        N = build_network(nodes, edges)
        _, _, _, opt = successive_shortest_paths(N, iter_limit=150)
    return opt


def process_file(filename, flow_alg, debug: Optional[bool] = False):
    nodes, edges = parse(filename)
    if len(edges.keys()) <= 1e6:
        opt = min_cost_flow(nodes, edges, flow_alg, debug)
        return build_pyg(nodes, edges, opt)
    else:
        return {"converged": False}


# def pyg_to_networkx(edge_index, edge_attr):
#     G = nx.DiGraph()
#     nodes = list(set(edge_index[:, 0]).union(set(edge_index[:, 1])))
#     for node in nodes:
#         G.add_node(node)
    
#     for i in range(edge_index.size(0)):
#       G.add_edge(edge_index[i, 0], edge_index[i, 1], weight=edge_attr[i, 0], capacity=[i, 1]) 

