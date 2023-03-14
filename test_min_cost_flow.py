from data_parsing import parse, build_network, build_networkx
from min_cost_flow import successive_shortest_paths
import networkx as nx
import pytest

files = [
    *[f"./resources/road_flow_01_DC_{i}.txt" for i in ['a', 'b', 'c', 'd', 'e']],
    *[f"./data/raw/netgen_{i}.txt" for i in range(100)]
]


@pytest.mark.parametrize("file", files)
def test_mcf(file):
    nodes, edges = parse(file)
    network = build_network(nodes, edges)
    G = build_networkx(nodes, edges)

    converged, _, _, cbn_opt = successive_shortest_paths(network, iter_limit = 150)
    nx_opt = nx.min_cost_flow_cost(G)

    if converged:
        print(cbn_opt, nx_opt)
        assert cbn_opt == nx_opt
