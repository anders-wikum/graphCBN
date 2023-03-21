import os

from min_cost_flow import successive_shortest_paths
from data_parsing import parse, build_network
import numpy as np
import cvxpy as cp


def compute_dataset_metrics(dataset, model, raw_dir):
    """
    Computes metrics for the test dataset. Computes both the improvement in total excess in the graph (ideally, we
    want the total excess to decrease with learned duals) and the speedup in the runtime of the algorithm (as
    measured in number of iterations with warm start and without).
    Args:
        dataset: dataset on which to compute the metrics
        model: model (to get predictions)
        raw_dir: directory in which to find the original graph descriptions (since the algorithm doesn't use the same
        graph input format as PyG)

    Returns:

    """
    speedups = []
    excess_diffs = []
    for data in dataset:
        model.eval()
        preds = model(data.x, data.edge_index, data.edge_attr)
        preds = refine_preds(data, preds)
        excess_diff, speedup = compute_graph_metrics(data, preds, raw_dir)
        speedups.append(speedup)
        excess_diffs.append(excess_diff)

    mean_excess = np.mean(np.array(excess_diffs))
    mean_speedup = np.mean(np.array(speedups))

    print(
        f"Metrics: \n     Average speedup with learned duals: {mean_speedup}\n     Mean excess difference: "
        f"{mean_excess}\n")


def compute_graph_metrics(graph, preds, raw_dir):
    """
    Runs the min cost flow algorithm twice, once as it would be run usually, once with learned duals
    (what is called a "warm-start").
    Args:
        graph: graph on which to compute metrics
        preds: predicted/learned duals

    Returns:
         The difference in total excess and the speedup.
    """
    file = graph.filename
    file = os.path.join(raw_dir, file)
    nodes, edges = parse(file)
    N = build_network(nodes, edges)
    p = dict(zip(N.V, preds))
    orig_excess, orig_nb_iters, _ = successive_shortest_paths(N, metrics = True)
    trained_excess, trained_nb_iters, _ = successive_shortest_paths(N, p = p, metrics = True)

    speedup = (orig_nb_iters - trained_nb_iters) / orig_nb_iters
    # Multiplying by 100 to get speedup in percentage
    speedup *= 100
    print(f"orig iters: {orig_nb_iters}, trained iters: {trained_nb_iters}")
    print(f"orig excess: {orig_excess}, trained_excess: {trained_excess}")
    excess_diff = orig_excess - trained_excess
    return excess_diff, speedup


def refine_preds(data, preds):
    """
    Improves the runtime of the algorithm on learned duals by refining the learned duals such that there remains many 0
    reduced cost edges in the graph.
    Args:
        data: graph on which the predictions were computed
        preds: predicted duals

    Returns:
        Improved duals
    """
    reduced_cost = -preds[data.edge_index[1]].squeeze() + preds[data.edge_index[0]].squeeze() + data.edge_attr[:, 1]
    reduced_cost = reduced_cost.detach().numpy().flatten()
    threshold = np.quantile(reduced_cost[reduced_cost < 0], 0.1)
    neg_edges = list(zip(*data.edge_index[:, reduced_cost < threshold].numpy()))
    if len(neg_edges) == 0:
        # No way to optimize the predictions if there are no negative predicted edges
        return preds.squeeze().detach().numpy().tolist()
    pos_edges = list(zip(*data.edge_index[:, reduced_cost > threshold].numpy()))
    edges = list(zip(*data.edge_index.numpy()))
    costs = dict(zip(edges, list(data.edge_attr[:, 1].numpy())))
    caps = dict(zip(edges, list(data.edge_attr[:, 0].numpy())))
    caps = np.array([caps[(i, j)] for (i, j) in pos_edges])

    N = data.x.size(0)
    y = cp.Variable(N)
    t = cp.Variable(len(neg_edges))
    s = cp.Variable(len(pos_edges))

    prob = cp.Problem(
        cp.Minimize(10000 * cp.sum(-cp.minimum(s, 0)) + 10 * cp.norm(s.T @ caps, 1)),
        [
            *[costs[neg_edges[i]] + y[neg_edges[i][0]] - y[neg_edges[i][1]] <= t[i] for i in range(len(neg_edges))],
            *[costs[pos_edges[j]] + y[pos_edges[j][0]] - y[pos_edges[j][1]] == s[j] for j in range(len(pos_edges))],
            t <= -0.01
        ]
    )

    prob.solve()
    return y.value
