# GraphCBN: Graph ML for the Min-Cost Flow Problem

## Overview:
This repository contains all functionality necessary to perform graph ML 
on Min-Cost Flow input networks, including
  1. Network generation in `graph_generator.py`
  2. Processing input networks in DIMACS input format into formats expected by
    PyG in `data_parsing.py`
  3. Solving minimum cost flow problems, including from a warm start
    in `min_cost_flow.py
  4. Building a PyG dataset from generated networks in `MinCostDataset.py`
  5. Building warm start input, and comparing run times of cold and warm
    starts in `metrics.py`

All required libraries can be found in `requirements.txt`. We highly recommend
first going through the demo `graphCBN_GNN.ipynb`, which walks through various
steps in building our GNN model and evaluating its results, using functionality
found throughout the repo.
