from data_parsing import parse, process_file, build_network, build_networkx
from min_cost_flow import successive_shortest_paths
import networkx as nx
import numpy as np
import pytest
import cvxpy as cp



mcf_files = [
    *[f"./resources/road_flow_01_DC_{i}.txt" for i in ['a', 'b', 'c', 'd', 'e']],
    *[f"./data/raw/netgen_{i}.txt" for i in range(100)]
]

gen_files = [
    *[f"./data/raw/netgen_{i}.txt" for i in range(100)]
]

@pytest.mark.parametrize("file", mcf_files)
def test_mcf(file):
    nodes, edges = parse(file)
    network = build_network(nodes, edges)
    G = build_networkx(nodes, edges)

    converged, _, _, cbn_opt = successive_shortest_paths(network, iter_limit=150)
    nx_opt = nx.min_cost_flow_cost(G)

    if converged:
        print(cbn_opt, nx_opt)
        assert np.isclose(cbn_opt, nx_opt)

@pytest.mark.parametrize("file", gen_files)
def test_pyg_generation(file):
    args = process_file(file, 'nx')
    b = args["x"].numpy().flatten()
    costs = args["edge_attr"][:, 1].numpy().flatten()
    caps = args["edge_attr"][:, 0].numpy().flatten()
    edges = list(zip(*args['edge_index'].numpy()))
    N = len(b)
    y = cp.Variable(N)

    obj = -cp.sum([y[i] * b[i] for i in range(N)]) - cp.sum([caps[i]*cp.maximum(0, -costs[i] - y[edges[i][0]] + y[edges[i][1]]) for i in range(len(edges))])
    prob = cp.Problem(
        cp.Maximize(obj)
    )
    prob.solve()
    assert np.isclose(prob.value, args["y"].item())
