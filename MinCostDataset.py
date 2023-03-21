import os
import os.path as osp
import re

import torch
from torch_geometric.data import Data, Dataset

from data_parsing import process_file


class MinCostDataset(Dataset):
    def __init__(self, root, transform = None, pre_transform = None, pre_filter = None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """If these files are found in the raw directory, download is skipped, none here since files are not downloaded"""
        return []

    @property
    def processed_file_names(self):
        """If these files are found in the processed directory, processing is skipped"""
        processed_files = []
        path = self.processed_dir
        if not os.path.exists(self.processed_dir):
            return []

        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isdir(file_path) and not file == "pre_filter.pt" and not file == "pre_transform.pt":
                processed_files.append(file)

        return processed_files

    def download(self):
        pass

    def process(self):
        # Create the processed file if it doesn't exist
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        # Start by getting the last file id to avoid overwriting (since we expect
        # filenames to be formatted as "data_{id}.pt")
        idx = 0
        found_file = False
        for file in os.listdir(self.processed_dir):
            prefix_string = "data_"
            if file.startswith(prefix_string):
                found_file = True
                file_id = int(re.findall(r'\d+', file)[0])
                idx = max(idx, file_id)
        if found_file:
            idx += 1

        path = self.raw_dir
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                continue
            # Read data from `raw_path`.
            output = process_file(file_path, 'cbn')
            if output["converged"]:
                x = output["x"]
                edge_index = output["edge_index"]
                edge_attr = output["edge_attr"]
                y = output["y"]
                # Post-process reduced costs: < 0, [1,0], >= 0 -> [0,1]
                reduced_cost = output["reduced_cost"]
                one_hot_reduced_cost = torch.zeros((reduced_cost.shape[0], 2))
                one_hot_reduced_cost[reduced_cost < 0] = torch.tensor([1, 0]).type(torch.FloatTensor)
                one_hot_reduced_cost[reduced_cost >= 0] = torch.tensor([0, 1]).type(torch.FloatTensor)

                data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y, reduced_cost = one_hot_reduced_cost,
                            filename = file)

                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
