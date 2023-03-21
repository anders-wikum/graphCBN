from pynetgen import netgen_generate
import numpy as np
import os
import shutil


def generate_min_cost_flow_graphs(
        num = 500,
        n = 50,
        s = 2,
        t = 2,
        supply = 250,
        max_cost = 10,
        max_cap = 50,
        seeds = None,
        dst_folder = "./data/raw"
):
    """
    Generates graphs and saves them to the destination folder. The generated graphs are such that minimum cost flows
    exist with the following characteristics:
    Args:
        num: number of graphs to generate
        n: number of nodes in each graph
        s: number of source nodes
        t: number of sink nodes
        supply: total supply
        max_cost: maximum cost of any edge in the graph
        max_cap: maximum capacity of any edge in the graph
        seeds: seed for each graph for reproducibility
        dst_folder: folder in which to save the graphs in the netgen/dimacs convention

    Returns:

    """

    print(f"Generating {num} min cost flow graphs")

    # Remove files in case some were already present
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder)

    if seeds is None:
        seeds = np.arange(num)

    schematic_seq = [{
        'seed':    seeds[i],
        'nodes':   n,
        'sources': s,
        'sinks':   t,
        'density': n ** 1.2,
        'mincost': 1,
        'maxcost': max_cost,
        'supply':  supply,
        'mincap':  1,
        'maxcap':  max_cap,
        'fname':   f'{dst_folder}/netgen_{i}.txt'
    }
        for i in range(num)]

    for schematic in schematic_seq:
        netgen_generate(**schematic)

    print("Done!")
