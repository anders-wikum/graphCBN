import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class CBN_NNConv(torch.nn.Module):
    """
    GAT to predict node-level embeddings. Then applies a post-message passing layer to transform into the output
    dimension.
    Part of the code definition is inspired by Colab 2:
    https://colab.research.google.com/drive/1xHmpjVO-Z74NK-dH3qoUBTf-tKUPfOKW?usp=sharing

    The main model used for convolutions is NNConv from the "Dynamic Edge-Conditioned Filters in Convolutional Neural
    Networks on Graphs" <https://arxiv.org/abs/1704.02901> paper
    """

    def __init__(self, input_dim, output_dim, edge_feature_dim, args):
        """
        Initializing the GNN
        Args:
            input_dim: dimension of node features
            output_dim: output dimension required
            edge_feature_dim: dimension of the edge features
            args: object containing the rest of the GNN description, including the number of layers, dropout, ...
        """
        super(CBN_NNConv, self).__init__()

        hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        dropout = args.dropout
        self.predict_edges = args.predict_edges

        if self.num_layers > 1:
            conv_modules = [NNConv(input_dim, hidden_dim, nn.Linear(edge_feature_dim, input_dim * hidden_dim))]
            conv_modules.extend(
                [NNConv(hidden_dim, hidden_dim, nn.Linear(edge_feature_dim, hidden_dim * hidden_dim)) for _ in
                 range(self.num_layers - 2)])
            conv_modules.append(NNConv(hidden_dim, output_dim, nn.Linear(edge_feature_dim, hidden_dim * output_dim)))

            self.convs = nn.ModuleList(conv_modules)
        else:
            self.convs = nn.ModuleList(
                [NNConv(input_dim, output_dim, nn.Linear(edge_feature_dim, input_dim * output_dim))])

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(self.num_layers - 1)])

        # Predictor is a post-message passing classification/regression head (depending on the number of output
        # dimensions)
        if self.predict_edges:
            # If we are predicting edge values, they are computed with a linear head that takes the embeddings of both
            # nodes in the edge then outputs a value
            self.predictor = nn.Linear(2 * hidden_dim, output_dim)
        else:
            # If we are predicting node embeddings/values/potentials, we simply apply a regression head to the final
            # node embeddings
            self.predictor = nn.Linear(args.heads * hidden_dim, output_dim)

        # Probability of an element getting zeroed
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        # self.post_mp.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)
        x = self.convs[-1](x, edge_index, edge_attr)

        if self.predict_edges:
            # If predicting edges: predict a value per edge
            edges = torch.concat((x[edge_index[0]], x[edge_index[1]]), dim = -1)
            x = self.predictor(edges)
            x = F.softmax(x, dim = 1)
        else:
            # If predicting node potentials, simply apply the regression head to the node embeddings
            x = F.relu(x)
            x = self.predictor(x)

        return x
